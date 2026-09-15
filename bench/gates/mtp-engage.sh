#!/usr/bin/env bash
# Phase 1, gate G3 (corrected): does --spec-type draft-mtp actually ENGAGE?
#
# The original G3 in phase1-gates.sh was gated on a log grep for "n_layer_nextn".
# That gate is invalid: llama.cpp only prints n_layer_nextn for LLM_ARCH_BAILINGMOE2
# (src/llama-model.cpp:1921-1930), never for qwen35moe. The MTP tensors were already
# confirmed present by direct GGUF inspection, so the only meaningful test is
# behavioural -- load with speculation and see whether draft_n > 0.
#
# Also captures the baseline TG so we get an early read on whether MTP pays at all.
# This matters most for kat-apex, whose MTP head is GRAFTED from base Qwen3.6-35B-A3B
# rather than trained on KAT's own trunk. Qwopus ships a native head and acts as the
# control: if Qwopus accepts well and KAT does not, the graft is the cause.
#
# Two prompt classes, because acceptance is workload-dependent:
#   copy   - verbatim echo. Copy-heavy agent traffic (diffs, file echoes, JSON).
#            This is MTP's best case.
#   prose  - novel generation. MTP's worst case.
#
# Output: /tmp/phase1-g3.tsv   Log: /tmp/phase1-g3.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUB="$HOME/.cache/huggingface/hub"
SERVER="$HOME/opt/llama.cpp-vulkan/bin/llama-server"

PORT=8095
CTX=32768
TIMEOUT=420
RESULTS=/tmp/phase1-g3.tsv
LOG=/tmp/phase1-g3.log
SERVERLOG=/tmp/phase1-g3-server.log
OUTDIR=/tmp/phase1
mkdir -p "$OUTDIR"

MODELS=(
  "kat-apex|$(ls "$HUB"/models--gbuzhf--KAT-Coder-V2.5-Dev-APEX-MTP-GGUF/snapshots/*/Kwaipilot_KAT-Coder-V2.5-Dev-MTP-UD-Q5_K_XL.gguf 2>/dev/null | head -1)"
  "qwopus-mtp|$(ls "$HUB"/models--Jackrong--Qwopus3.6-35B-A3B-Coder-MTP-GGUF/snapshots/*/Qwopus3.6-35B-A3B-Coder-MTP-Q5_K_M.gguf 2>/dev/null | head -1)"
)

COPY_PROMPT='Repeat the following JSON back to me character for character, with no commentary:\n{"name":"textstat_core","version":"0.1.0","edition":"2021","deps":{"pyo3":{"version":"0.22","features":["extension-module"]}},"lib":{"name":"textstat_core","crate-type":["cdylib"]}}'
PROSE_PROMPT='Explain, in one paragraph of original prose, why speculative decoding helps more when a model is bandwidth-bound than when it is dequantisation-bound.'

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}'; }
stop() { pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 5; }
trap 'stop; exit 130' INT TERM

: > "$LOG"
printf 'model\tspec\tworkload\ttg_tok_s\tdraft_n\tdraft_acc\tacc_pct\tvram_mib\n' > "$RESULTS"

start_server() {
    local mpath="$1"; shift
    : > "$SERVERLOG"
    "$SERVER" -m "$mpath" -dev Vulkan0 -ngl 999 \
        -c "$CTX" -np 1 --kv-unified -fa on \
        --cache-type-k f16 --cache-type-v f16 \
        --jinja --metrics --no-warmup \
        --host 127.0.0.1 --port "$PORT" "$@" \
        > "$SERVERLOG" 2>&1 &
    SRV_PID=$!
    local s; s=$(date +%s); local el=0
    while :; do
        el=$(( $(date +%s) - s ))
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 $SRV_PID 2>/dev/null || return 1
        (( el >= TIMEOUT )) && return 1
        sleep 3
    done
}

# gen <name> <spec> <workload> <prompt>
gen() {
    local name="$1" spec="$2" wl="$3" prompt="$4"
    local out="$OUTDIR/g3-$name-$spec-$wl.json"
    python3 -c "
import json,sys
print(json.dumps({'messages':[{'role':'user','content':sys.argv[1].replace('\\\\n','\n')}],
                  'max_tokens':400,'temperature':0}))" "$prompt" > /tmp/g3-req.json
    curl -sf --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @/tmp/g3-req.json > "$out" 2>/dev/null || { log "    request failed"; return; }
    read -r tg dn da acc <<< "$(python3 - "$out" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
t=d.get("timings",{})
tg=t.get("predicted_per_second",0) or 0
dn=t.get("draft_n"); da=t.get("draft_n_accepted")
if dn is None: print(f"{tg:.2f} - - -")
else: print(f"{tg:.2f} {dn} {da} {(100.0*da/dn if dn else 0):.1f}")
PY
)"
    log "    $wl: TG ${tg} tok/s  draft_n=${dn} accepted=${da} acc=${acc}%"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$spec" "$wl" "$tg" "$dn" "$da" "$acc" "$VRAM" >> "$RESULTS"
}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name mpath <<< "$entry"
    [[ -f "$mpath" ]] || { log "SKIP $name (missing)"; continue; }
    log "############ $name"
    stop

    for spec in none draft-mtp; do
        extra=(); [[ "$spec" != none ]] && extra=(--spec-type draft-mtp)
        log "  --- spec=$spec"
        if ! start_server "$mpath" "${extra[@]}"; then
            log "    LOAD FAILED"
            grep -oiE 'error[^\n]{0,80}|unsupported[^\n]{0,60}|failed to [a-z ]*|nextn[^\n]{0,40}' "$SERVERLOG" \
                | sort -u | head -5 | sed 's/^/      /' | tee -a "$LOG"
            cp "$SERVERLOG" "$OUTDIR/g3-$name-$spec-loadfail.log"
            printf '%s\t%s\t-\t-\t-\t-\t-\tLOAD_FAIL\n' "$name" "$spec" >> "$RESULTS"
            stop; continue
        fi
        VRAM=$(vram)
        log "    loaded, vram ${VRAM} MiB"
        gen "$name" "$spec" copy  "$COPY_PROMPT"
        gen "$name" "$spec" prose "$PROSE_PROMPT"
        stop
    done
    log ""
done

log ""
column -t -s $'\t' "$RESULTS" | tee -a "$LOG"
log "G3_DONE"
