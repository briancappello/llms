#!/usr/bin/env bash
# VRAM soak test for kat-apex + MTP at deep context.
#
# QUESTION: is 441 MiB of headroom at -c 262144 actually usable, or does it OOM
# once the context is genuinely full?
#
# Theory says it should be fine: llama.cpp allocates both the KV cache and the
# compute graph buffer at LOAD time, sized for worst-case n_ctx and n_batch, so
# VRAM should not grow as context fills. The residual risks are Vulkan driver
# temporaries at large prefill batches, allocator fragmentation across many
# requests, and any external consumer on the same card (it is the VGA device).
# Those are empirical, not theoretical -- hence this test.
#
# METHOD: for each candidate context, load with the LOCKED re-run config, then
# fire a ladder of prompts at increasing fractions of the window, ending with
# three repeats at the deepest point to probe fragmentation. A 1 Hz sampler
# records peak VRAM across the whole run, since the interesting number is the
# transient peak during prefill, not the steady state after load.
#
# Usage: soak-ctx.sh [ctx ...]      default ladder: 262144 253952 245760 237568
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="$HERE/../../lib"
HUB="$HOME/.cache/huggingface/hub"
SERVER="$HOME/opt/llama.cpp-vulkan/bin/llama-server"
M="$(ls "$HUB"/models--gbuzhf--KAT-Coder-V2.5-Dev-APEX-MTP-GGUF/snapshots/*/*MTP-UD-Q5_K_XL.gguf | head -1)"
CORPUS=/tmp/code-corpus.txt
PORT=8097
VRAM_TOTAL=32624
RESULTS=/tmp/soak-ctx.tsv
LOG=/tmp/soak-ctx.log
SERVERLOG=/tmp/soak-server.log

CTXS=("$@"); [[ ${#CTXS[@]} -eq 0 ]] && CTXS=(262144 253952 245760 237568)
FRACS=(0.25 0.50 0.75 0.90 0.94 0.94 0.94)

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
vram(){ rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}'; }
stop(){ pkill -x llama-server 2>/dev/null; sleep 5; }
trap 'stop; kill ${SAMPLER:-0} 2>/dev/null; exit 130' INT TERM

: > "$LOG"
printf 'ctx\tphase\tdepth_tok\tstatus\tpeak_vram_mib\theadroom_mib\tpp_tok_s\ttg_tok_s\n' > "$RESULTS"

for CTX in "${CTXS[@]}"; do
    log "########## ctx=$CTX"
    stop
    : > "$SERVERLOG"
    # LOCKED re-run config
    "$SERVER" -m "$M" -dev Vulkan0 -ngl 999 \
        -c "$CTX" -np 2 --kv-unified -fa on \
        --cache-type-k f16 --cache-type-v f16 \
        --jinja --metrics --spec-type draft-mtp \
        --reasoning-preserve \
        --temp 1.0 --top-p 0.95 --top-k 20 \
        --presence-penalty 1.5 --repeat-last-n -1 \
        --host 127.0.0.1 --port "$PORT" > "$SERVERLOG" 2>&1 &
    pid=$!

    ok=0
    for i in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 3
    done
    if (( ok == 0 )); then
        r=$(grep -oiE "failed to allocate|out of memory|ErrorOutOfDeviceMemory|unable to allocate" "$SERVERLOG" | head -1)
        log "  LOAD FAILED ${r:+($r)}"
        printf '%s\tload\t-\tLOAD_FAIL\t-\t-\t-\t-\n' "$CTX" >> "$RESULTS"
        stop; continue
    fi
    VLOAD=$(vram)
    log "  loaded: ${VLOAD} MiB, headroom $((VRAM_TOTAL - VLOAD)) MiB"

    # 1 Hz peak sampler
    PEAKF=/tmp/soak-peak-$CTX.txt; : > "$PEAKF"
    ( while :; do vram >> "$PEAKF"; sleep 1; done ) &
    SAMPLER=$!

    failed=0
    for i in "${!FRACS[@]}"; do
        f="${FRACS[$i]}"
        D=$(python3 -c "print(int($CTX*$f))")
        printf '  depth %6d (%s) ... ' "$D" "$f" | tee -a "$LOG"
        if ! python3 "$LIB/trim_to_tokens.py" "http://127.0.0.1:$PORT/tokenize" \
               "$CORPUS" "$D" 256 > /tmp/soak-req-raw.json 2>/dev/null; then
            echo "prompt build FAILED" | tee -a "$LOG"; failed=1; break
        fi
        python3 -c "
import json
d=json.load(open('/tmp/soak-req-raw.json')); d['cache_prompt']=False
json.dump(d,open('/tmp/soak-req.json','w'))"
        if ! curl -sf --max-time 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
               -H 'Content-Type: application/json' -d @/tmp/soak-req.json \
               > /tmp/soak-resp.json 2>/dev/null; then
            echo "REQUEST FAILED" | tee -a "$LOG"
            grep -oiE "out of memory|failed to allocate|ErrorOutOfDeviceMemory" "$SERVERLOG" | tail -1 | sed 's/^/      /' | tee -a "$LOG"
            failed=1; break
        fi
        read -r pp tg ptok <<< "$(python3 -c "
import json
d=json.load(open('/tmp/soak-resp.json')); t=d.get('timings',{}); u=d.get('usage',{})
print(f\"{t.get('prompt_per_second',0):.0f} {t.get('predicted_per_second',0):.1f} {u.get('prompt_tokens',0)}\")")"
        PEAK=$(sort -n "$PEAKF" | tail -1)
        echo "ok  ptok=${ptok} PP=${pp} TG=${tg}  peak=${PEAK} MiB" | tee -a "$LOG"
        printf '%s\treq%d\t%s\tOK\t%s\t%s\t%s\t%s\n' \
            "$CTX" "$i" "$ptok" "$PEAK" "$((VRAM_TOTAL - PEAK))" "$pp" "$tg" >> "$RESULTS"
    done

    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
    PEAK=$(sort -n "$PEAKF" | tail -1)
    if (( failed )); then
        log "  ==> ctx=$CTX UNSTABLE (peak ${PEAK} MiB, headroom $((VRAM_TOTAL-PEAK)))"
        printf '%s\tverdict\t-\tUNSTABLE\t%s\t%s\t-\t-\n' "$CTX" "$PEAK" "$((VRAM_TOTAL-PEAK))" >> "$RESULTS"
        stop
    else
        log "  ==> ctx=$CTX STABLE (peak ${PEAK} MiB, headroom $((VRAM_TOTAL-PEAK)) MiB)"
        printf '%s\tverdict\t-\tSTABLE\t%s\t%s\t-\t-\n' "$CTX" "$PEAK" "$((VRAM_TOTAL-PEAK))" >> "$RESULTS"
        stop
        log "  first stable context found; stopping ladder"
        break
    fi
done

log ""
column -t -s $'\t' "$RESULTS" | tee -a "$LOG"
log "SOAK_DONE"
