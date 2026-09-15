#!/usr/bin/env bash
# Close the bracket left open by phase2-ctx-probe.sh for kat-apex + draft-mtp.
# The stretch step only tests ONE step above the derived value, so that config
# was only bounded as ">= 196608". Test 229376 and 262144 directly.
#
# NOTE: use `pkill -x llama-server` (match on process NAME), never
# `pkill -f llama-server`, which also matches any shell whose command line
# contains the string -- including the caller.
set -uo pipefail

SERVER="$HOME/opt/llama.cpp-vulkan/bin/llama-server"
M="$(ls "$HOME"/.cache/huggingface/hub/models--gbuzhf--KAT-Coder-V2.5-Dev-APEX-MTP-GGUF/snapshots/*/*Q5_K_XL.gguf | head -1)"
PORT=8098
OUT=/tmp/phase2-gap.tsv

printf 'model\tspec\tctx\tstatus\tvram_mib\tgen_ok\n' > "$OUT"

for ctx in 229376 262144; do
    pkill -x llama-server 2>/dev/null; sleep 5
    LOG=/tmp/gap-$ctx.log
    : > "$LOG"
    "$SERVER" -m "$M" -dev Vulkan0 -ngl 999 \
        -c "$ctx" -np 2 --kv-unified -fa on \
        --cache-type-k f16 --cache-type-v f16 \
        --jinja --no-warmup --spec-type draft-mtp \
        --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
    pid=$!
    ok=0
    for i in $(seq 1 100); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 3
    done
    if (( ok == 1 )); then
        v=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}')
        g=$(curl -sf --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
              -H 'Content-Type: application/json' \
              -d '{"messages":[{"role":"user","content":"say ok"}],"max_tokens":300,"temperature":0}' \
              2>/dev/null | grep -c '"content"')
        echo "ctx=$ctx LOADED vram=${v} MiB gen_ok=${g}"
        printf 'kat-apex\tdraft-mtp\t%s\tOK\t%s\t%s\n' "$ctx" "$v" "$g" >> "$OUT"
    else
        r=$(grep -oiE 'failed to allocate|out of memory|ErrorOutOfDeviceMemory|unable to allocate' "$LOG" | head -1)
        echo "ctx=$ctx FAILED ${r:-（see $LOG)}"
        printf 'kat-apex\tdraft-mtp\t%s\tFAIL\t-\t-\n' "$ctx" >> "$OUT"
    fi
    pkill -x llama-server 2>/dev/null; sleep 5
done

column -t -s $'\t' "$OUT"
echo GAP_DONE
