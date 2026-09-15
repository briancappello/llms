#!/usr/bin/env bash
# Phase 2: context ceiling for the three qwen35moe coder candidates.
#
# METHOD. Not a blind descent. results/README.md records that deriving KV cost
# from attention.key_length overestimated Ornith by 2x, and mandates measuring
# via VRAM delta across two contexts. So:
#
#   1. load at CTX_LO and CTX_HI, record VRAM at each
#   2. marginal cost  = (V_hi - V_lo) / (CTX_HI - CTX_LO)     [KiB/token]
#   3. base           = V_lo - CTX_LO * cost                  [weights + cbuf]
#   4. ceiling        = (BUDGET - base) / cost, clamped to the trained 262144
#   5. VERIFY: load at the derived step and actually generate a token
#   6. STRETCH: load one step above it, to bracket the true ceiling
#
# Step 2 deliberately conflates KV growth with compute-buffer growth. That is
# the effective marginal cost, which is exactly what extrapolation needs.
#
# Run in the PRODUCTION shape (-np 2 --kv-unified, f16 KV), not the -np 1
# --no-kv-unified shape used by the older ctx-probe-vulkan.sh, because the output
# of this script is a prod -c value. Numbers are therefore NOT directly
# comparable to the existing ctx-probe.tsv row.
#
# MTP models are probed twice. The MTP layer (blk.40) carries attn_k/attn_v of
# [2048,512] = 2 KV heads x 256, i.e. it is a FULL-ATTENTION layer, so enabling
# speculation should add an 11th cached layer to the 10 trunk ones and push
# cost from ~20 to ~22 KiB/token. This measures whether that is what happens.
#
# Output: /tmp/phase2-ctx.tsv   Log: /tmp/phase2-ctx.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUB="$HOME/.cache/huggingface/hub"
SERVER="$HOME/opt/llama.cpp-vulkan/bin/llama-server"

PORT=8096
TIMEOUT=600
CTX_LO=32768
CTX_HI=131072
VRAM_TOTAL=32624
BUDGET=$(( VRAM_TOTAL - 1800 ))     # same reserve llm uses
TRAINED=262144
STEPS=(262144 229376 196608 180224 163840 131072 98304 65536)

RESULTS=/tmp/phase2-ctx.tsv
LOG=/tmp/phase2-ctx.log
SERVERLOG=/tmp/phase2-server.log
OUTDIR=/tmp/phase2
mkdir -p "$OUTDIR"

# name|path|spec
CONFIGS=(
  "kat-apex|$(ls "$HUB"/models--gbuzhf--KAT-Coder-V2.5-Dev-APEX-MTP-GGUF/snapshots/*/Kwaipilot_KAT-Coder-V2.5-Dev-MTP-UD-Q5_K_XL.gguf 2>/dev/null | head -1)|none"
  "kat-apex|$(ls "$HUB"/models--gbuzhf--KAT-Coder-V2.5-Dev-APEX-MTP-GGUF/snapshots/*/Kwaipilot_KAT-Coder-V2.5-Dev-MTP-UD-Q5_K_XL.gguf 2>/dev/null | head -1)|draft-mtp"
  "kat-q5km|$(ls "$HUB"/models--bartowski--Kwaipilot_KAT-Coder-V2.5-Dev-GGUF/snapshots/*/Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf 2>/dev/null | head -1)|none"
  "qwopus-mtp|$(ls "$HUB"/models--Jackrong--Qwopus3.6-35B-A3B-Coder-MTP-GGUF/snapshots/*/Qwopus3.6-35B-A3B-Coder-MTP-Q5_K_M.gguf 2>/dev/null | head -1)|none"
  "qwopus-mtp|$(ls "$HUB"/models--Jackrong--Qwopus3.6-35B-A3B-Coder-MTP-GGUF/snapshots/*/Qwopus3.6-35B-A3B-Coder-MTP-Q5_K_M.gguf 2>/dev/null | head -1)|draft-mtp"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}'; }
stop() { pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 5; }
trap 'stop; exit 130' INT TERM

: > "$LOG"
printf 'model\tspec\tphase\tctx\tstatus\tvram_mib\tkib_per_tok\tbase_mib\tderived_ceiling\tprod_ctx\n' > "$RESULTS"

# try_load <path> <ctx> <spec>  -> echoes vram on success, empty on failure
try_load() {
    local mpath="$1" ctx="$2" spec="$3"
    local extra=(); [[ "$spec" != none ]] && extra=(--spec-type "$spec")
    : > "$SERVERLOG"
    "$SERVER" -m "$mpath" -dev Vulkan0 -ngl 999 \
        -c "$ctx" -np 2 --kv-unified -fa on \
        --cache-type-k f16 --cache-type-v f16 \
        --jinja --no-warmup "${extra[@]}" \
        --host 127.0.0.1 --port "$PORT" \
        > "$SERVERLOG" 2>&1 &
    local pid=$! s; s=$(date +%s); local el=0
    while :; do
        el=$(( $(date +%s) - s ))
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { vram; return 0; }
        kill -0 $pid 2>/dev/null || return 1
        (( el >= TIMEOUT )) && return 1
        sleep 3
    done
}

# confirm the server can actually run a token, not merely allocate
can_generate() {
    curl -sf --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"say ok"}],"max_tokens":300,"temperature":0}' \
        2>/dev/null | grep -q '"content"'
}

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name mpath spec <<< "$entry"
    [[ -f "$mpath" ]] || { log "SKIP $name (missing)"; continue; }
    tag="$name/$spec"
    log "############ $tag"
    stop

    # ---- step 1: two anchor loads
    v_lo=$(try_load "$mpath" "$CTX_LO" "$spec"); stop
    if [[ -z "$v_lo" ]]; then
        log "  anchor lo FAILED at $CTX_LO -- cannot characterise"
        printf '%s\t%s\tanchor_lo\t%s\tFAIL\t-\t-\t-\t-\t-\n' "$name" "$spec" "$CTX_LO" >> "$RESULTS"
        continue
    fi
    log "  anchor lo  ctx=$CTX_LO  vram=${v_lo} MiB"
    printf '%s\t%s\tanchor_lo\t%s\tOK\t%s\t-\t-\t-\t-\n' "$name" "$spec" "$CTX_LO" "$v_lo" >> "$RESULTS"

    v_hi=$(try_load "$mpath" "$CTX_HI" "$spec"); stop
    if [[ -z "$v_hi" ]]; then
        log "  anchor hi FAILED at $CTX_HI"
        printf '%s\t%s\tanchor_hi\t%s\tFAIL\t-\t-\t-\t-\t-\n' "$name" "$spec" "$CTX_HI" >> "$RESULTS"
        continue
    fi
    log "  anchor hi  ctx=$CTX_HI  vram=${v_hi} MiB"

    # ---- steps 2-4: derive
    read -r cost base ceil prod <<< "$(python3 -c "
v_lo=$v_lo; v_hi=$v_hi; lo=$CTX_LO; hi=$CTX_HI
budget=$BUDGET; trained=$TRAINED
steps=[$(IFS=,; echo "${STEPS[*]}")]
cost=(v_hi-v_lo)*1024.0/(hi-lo)          # KiB per token
base=v_lo-(lo*cost/1024.0)               # MiB
ceil=int((budget-base)*1024.0/cost)
ceil=min(ceil,trained)
prod=next((s for s in steps if s<=ceil), 65536)
print(f'{cost:.2f} {base:.0f} {ceil} {prod}')")"
    log "  derived    ${cost} KiB/token  base ${base} MiB  ceiling ${ceil}  -> step ${prod}"
    printf '%s\t%s\tanchor_hi\t%s\tOK\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$spec" "$CTX_HI" "$v_hi" "$cost" "$base" "$ceil" "$prod" >> "$RESULTS"

    # ---- step 5: verify at the derived step
    v=$(try_load "$mpath" "$prod" "$spec")
    if [[ -n "$v" ]]; then
        if can_generate; then
            log "  VERIFY     ctx=$prod  OK  vram=${v} MiB  (inference OK)"
            printf '%s\t%s\tverify\t%s\tOK\t%s\t%s\t%s\t%s\t%s\n' "$name" "$spec" "$prod" "$v" "$cost" "$base" "$ceil" "$prod" >> "$RESULTS"
        else
            log "  VERIFY     ctx=$prod  LOADED BUT INFERENCE FAILED  vram=${v} MiB"
            printf '%s\t%s\tverify\t%s\tNOGEN\t%s\t%s\t%s\t%s\t%s\n' "$name" "$spec" "$prod" "$v" "$cost" "$base" "$ceil" "$prod" >> "$RESULTS"
        fi
    else
        log "  VERIFY     ctx=$prod  FAILED (derivation optimistic)"
        grep -oiE 'failed to allocate|out of memory|ErrorOutOfDeviceMemory|unable to allocate' "$SERVERLOG" | head -1 | sed 's/^/      /' | tee -a "$LOG"
        printf '%s\t%s\tverify\t%s\tFAIL\t-\t%s\t%s\t%s\t%s\n' "$name" "$spec" "$prod" "$cost" "$base" "$ceil" "$prod" >> "$RESULTS"
    fi
    stop

    # ---- step 6: stretch one step above, to bracket the ceiling
    stretch=""
    for i in "${!STEPS[@]}"; do
        if [[ "${STEPS[$i]}" == "$prod" ]] && (( i > 0 )); then stretch="${STEPS[$((i-1))]}"; fi
    done
    if [[ -n "$stretch" ]]; then
        v=$(try_load "$mpath" "$stretch" "$spec")
        if [[ -n "$v" ]] && can_generate; then
            log "  STRETCH    ctx=$stretch  ALSO OK  vram=${v} MiB  <-- ceiling is higher than derived"
            printf '%s\t%s\tstretch\t%s\tOK\t%s\t%s\t%s\t%s\t%s\n' "$name" "$spec" "$stretch" "$v" "$cost" "$base" "$ceil" "$prod" >> "$RESULTS"
        else
            log "  STRETCH    ctx=$stretch  fails (expected) -- ceiling bracketed in ($prod, $stretch)"
            printf '%s\t%s\tstretch\t%s\tFAIL\t-\t%s\t%s\t%s\t%s\n' "$name" "$spec" "$stretch" "$cost" "$base" "$ceil" "$prod" >> "$RESULTS"
        fi
        stop
    fi
    log ""
done

stop
log ""
column -t -s $'\t' "$RESULTS" | tee -a "$LOG"
log "PHASE2_DONE"
