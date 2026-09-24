#!/usr/bin/env bash
# Compare the three from-source llama.cpp backends -- vulkan / hip / hip-wmma --
# on the ACTUAL production model + flags, to decide which one llama-swap should
# launch. All three are built from the same commit (~/opt/llama.cpp-COMMIT) and
# RPATH-isolated, so differences here are backend differences, not version skew.
#
# Method mirrors the serving config: cold-fusion Q4_K_M, f16 K/V, -fa on,
# --spec-type draft-mtp, single stream (-np 1). Prompt-processing at depth is
# the headline number for an agentic coding loop (large system prompt + tools +
# files re-prefilled every turn); TG and MTP acceptance are captured too.
#
# Context is allocated only to ALLOC (max depth + slack), not the full 196608,
# so every backend loads fast and can't OOM on compute buffers -- we are timing
# throughput, not the context ceiling (that is cold-fusion-kv.md's job).
#
# Run with llama-swap STOPPED (it owns the GPU): the wrapper does that.
#   systemctl --user stop llama-swap
#   bench/speed/backend-sweep.sh
#   systemctl --user start llama-swap
#
# Output: results/speed/backend-sweep.tsv   Log: /tmp/backend-sweep.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
LIB="$REPO/bench/lib"
HUB="$HOME/.cache/huggingface/hub"
CORPUS=/tmp/cf-corpus.txt

PORT=8090
ALLOC=73728                     # >= max depth; keeps loads fast, avoids OOM
DEPTHS=(4096 32768 65536)
REPS=2
GEN=128
LOAD_TIMEOUT=600
REQ_TIMEOUT=1800

RESULTS="$REPO/results/speed/backend-sweep.tsv"
LOG=/tmp/backend-sweep.log
SERVERLOG=/tmp/backend-sweep-server.log

MODEL="$(ls "$HUB"/models--DavidAU--Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-MTP-GGUF/snapshots/*/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf 2>/dev/null | head -1)"

# name|prefix|device|extra-env
BACKENDS=(
    "vulkan|$HOME/opt/llama.cpp-vulkan|Vulkan0|MESA_VK_DEVICE_SELECT=1002:7551!"
    "hip|$HOME/opt/llama.cpp-hip|ROCm0|"
    "hip-wmma|$HOME/opt/llama.cpp-hip-wmma|ROCm0|"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}'; }
stop() { pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 4; }
trap 'stop; exit 130' INT TERM

: > "$LOG"
[[ -n "$MODEL" && -f "$MODEL" ]] || { log "model not found"; exit 1; }
[[ -s "$CORPUS" ]] || { log "missing corpus $CORPUS"; exit 1; }
mkdir -p "$(dirname "$RESULTS")"
printf 'backend\tcommit\tdepth\trep\tprompt_tok\tpp_tok_s\ttg_tok_s\tacc_pct\tvram_mib\tgpu_ok\n' > "$RESULTS"
COMMIT="$(cut -d' ' -f1 ~/opt/llama.cpp-COMMIT 2>/dev/null)"

for b in "${BACKENDS[@]}"; do
    IFS='|' read -r name prefix dev env <<< "$b"
    bin="$prefix/bin/llama-server"
    if [[ ! -x "$bin" ]]; then log "SKIP $name (no binary at $bin)"; continue; fi
    log "############ $name ($dev)  $bin"
    stop
    : > "$SERVERLOG"
    env ${env:+$env} "$bin" \
        -m "$MODEL" -dev "$dev" -ngl 999 \
        -c "$ALLOC" -np 1 --kv-unified -fa on \
        --cache-type-k f16 --cache-type-v f16 \
        --jinja --metrics --no-warmup --spec-type draft-mtp \
        --host 127.0.0.1 --port "$PORT" > "$SERVERLOG" 2>&1 &
    pid=$!; ok=0; s=$(date +%s)
    while :; do
        el=$(( $(date +%s) - s ))
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        (( el >= LOAD_TIMEOUT )) && break
        sleep 2
    done
    if (( ok == 0 )); then
        log "  LOAD FAILED"; grep -iE 'error|out of memory|failed|unsupported' "$SERVERLOG" | sort -u | head -5 | sed 's/^/    /' | tee -a "$LOG"
        printf '%s\t%s\t-\t-\t\t\t\t\t\tLOAD_FAIL\n' "$name" "$COMMIT" >> "$RESULTS"; stop; continue
    fi
    # Confirm we landed on the R9700 (gfx1201), not the gfx1036 iGPU.
    gpu_ok=no; grep -qiE 'R9700|gfx1201' "$SERVERLOG" && gpu_ok=yes
    V=$(vram); log "  ready in ${el}s  vram ${V} MiB  gpu_ok=$gpu_ok"

    for d in "${DEPTHS[@]}"; do
        if ! python3 "$LIB/trim_to_tokens.py" "http://127.0.0.1:$PORT/tokenize" "$CORPUS" "$d" "$GEN" \
               > /tmp/bs-raw.json 2>>"$LOG"; then log "  depth $d: prompt build failed"; continue; fi
        python3 -c "import json;d=json.load(open('/tmp/bs-raw.json'));d['cache_prompt']=False;d['temperature']=0.0;json.dump(d,open('/tmp/bs-req.json','w'))"
        for rep in $(seq 1 "$REPS"); do
            if ! curl -sf --max-time "$REQ_TIMEOUT" -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
                   -H 'Content-Type: application/json' -d @/tmp/bs-req.json > /tmp/bs-resp.json 2>>"$LOG"; then
                log "  depth $d rep $rep: request failed"; continue; fi
            read -r pp tg ptok acc <<< "$(python3 - /tmp/bs-resp.json <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); t=d.get("timings",{}); u=d.get("usage",{})
dn,da=t.get("draft_n"),t.get("draft_n_accepted")
acc=f"{100.0*da/dn:.1f}" if dn else "-"
print(f'{t.get("prompt_per_second",0):.1f} {t.get("predicted_per_second",0):.2f} {u.get("prompt_tokens","")} {acc}')
PY
)"
            log "  depth $d rep $rep: ${ptok} tok -> PP ${pp} | TG ${tg} | acc ${acc}%"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$COMMIT" "$d" "$rep" "$ptok" "$pp" "$tg" "$acc" "$V" "$gpu_ok" >> "$RESULTS"
        done
    done
    stop; log ""
done

log "=== RESULTS ==="; column -t -s $'\t' "$RESULTS" | tee -a "$LOG"; log "BACKEND_SWEEP_DONE"
