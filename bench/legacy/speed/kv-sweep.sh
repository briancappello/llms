#!/usr/bin/env bash
# Fable-Fusion: does quantising the V cache buy back the full 262144 window
# without costing throughput?
#
# Fable is 27B DENSE, 64 layers, 4 KV heads -- every layer caches KV, so it
# pays 68.4 KiB/token at f16+MTP versus ~24 for the A3B hybrids. That makes
# 262144 physically impossible at f16/f16 (34,710 MiB needed, 32,624 available).
#
# Two arms:
#   A  f16 K + q8_0 V @ 262144   full window, ~2.0 GB predicted headroom
#   B  f16 K + f16  V @ 212992   safe bump from today's 180224, ~1.2 GB headroom
#
# WHY THIS ISN'T ALREADY ANSWERED: config.header.yaml records "f16 is both
# higher precision AND faster here (PP 2174 vs 1322 at 64k)". That was measured
# by ctx-deep-test.sh, which loads ORNITH, not fable. Ornith caches 10 layers;
# fable caches 64, and reads ~6x more KV per token. The penalty does not
# transfer, in either direction, and has never been measured on fable.
#
# Also note arm A quantises only V. K is what every query is scored against;
# V is only the weighted sum afterwards, so V is the more forgiving half.
#
# SCOPE: throughput and VRAM only. Quantised KV can degrade long-context
# RECALL, which PP/TG cannot see -- gate arm A on niah.py before shipping it.
#
# Output: /tmp/fable-kv.tsv (all samples), /tmp/fable-kv-summary.tsv (medians)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="$HERE/../../lib"
HUB="$HOME/.cache/huggingface/hub"
SERVER="$HOME/opt/llama.cpp-vulkan/bin/llama-server"
D="$HUB/models--DavidAU--Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF/snapshots"
M="$(ls "$D"/*/*NEO-MTP-IQ4_NL.gguf 2>/dev/null | head -1)"
MMPROJ="$(ls "$D"/*/mmproj-F16.gguf 2>/dev/null | head -1)"

PORT=8094
GEN=192
REPEATS=3
CORPUS=/tmp/code-corpus.txt
VRAM_TOTAL=32624
TIMEOUT=600
RESULTS=/tmp/fable-kv.tsv
SUMMARY=/tmp/fable-kv-summary.tsv
LOG=/tmp/fable-kv.log
SERVERLOG=/tmp/fable-kv-server.log

# arm|ctx|cache_k|cache_v|depths
ARMS=(
  "A-q8V|262144|f16|q8_0|4096 16384 65536 131072 196608 245760"
  "B-f16|212992|f16|f16|4096 16384 65536 131072 196608"
)

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
vram(){ rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}'; }
stop(){ pkill -x llama-server 2>/dev/null; sleep 5; }
trap 'stop; exit 130' INT TERM

: > "$LOG"
[[ -f "$M" ]] || { echo "fable weights not found" >&2; exit 1; }
[[ -f "$CORPUS" ]] || { echo "missing $CORPUS" >&2; exit 1; }
printf 'arm\tctx\tcache_k\tcache_v\tdepth\trep\tprompt_tok\tpp_tok_s\ttg_tok_s\tvram_mib\theadroom_mib\n' > "$RESULTS"

# llama-swap holds fable at -c 180224; it must not compete for VRAM
systemctl --user stop llama-swap 2>/dev/null; sleep 3
stop

for spec in "${ARMS[@]}"; do
    IFS='|' read -r ARM CTX CK CV DEPTHS <<< "$spec"
    log "########## arm $ARM : -c $CTX  K=$CK V=$CV"
    stop
    : > "$SERVERLOG"
    "$SERVER" -m "$M" --mmproj "$MMPROJ" -dev Vulkan0 -ngl 999 \
        -c "$CTX" -np 2 --kv-unified -fa on \
        --cache-type-k "$CK" --cache-type-v "$CV" \
        --jinja --metrics --no-warmup --spec-type draft-mtp \
        --host 127.0.0.1 --port "$PORT" > "$SERVERLOG" 2>&1 &
    pid=$!; ok=0
    for i in $(seq 1 $((TIMEOUT/3))); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 3
    done
    if (( ok == 0 )); then
        log "  LOAD FAILED"
        grep -oiE "out of memory|failed to allocate|unsupported|ErrorOutOfDeviceMemory|not supported" "$SERVERLOG" | sort -u | head -3 | sed 's/^/    /' | tee -a "$LOG"
        printf '%s\t%s\t%s\t%s\t-\t-\t-\t-\t-\tLOAD_FAIL\t-\n' "$ARM" "$CTX" "$CK" "$CV" >> "$RESULTS"
        stop; continue
    fi
    V=$(vram)
    log "  loaded: ${V} MiB, headroom $((VRAM_TOTAL-V)) MiB"

    for d in $DEPTHS; do
        if ! python3 "$LIB/trim_to_tokens.py" "http://127.0.0.1:$PORT/tokenize" \
               "$CORPUS" "$d" "$GEN" > /tmp/fkv-raw.json 2>/dev/null; then
            log "  depth $d: prompt build failed"; continue
        fi
        python3 -c "
import json
d=json.load(open('/tmp/fkv-raw.json')); d['cache_prompt']=False
json.dump(d,open('/tmp/fkv-req.json','w'))"
        for rep in $(seq 1 $REPEATS); do
            if ! curl -sf --max-time 3600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
                   -H 'Content-Type: application/json' -d @/tmp/fkv-req.json \
                   > /tmp/fkv-resp.json 2>/dev/null; then
                log "  depth $d rep $rep: request FAILED"; continue
            fi
            read -r pp tg ptok <<< "$(python3 -c "
import json
d=json.load(open('/tmp/fkv-resp.json')); t=d.get('timings',{}); u=d.get('usage',{})
print(f\"{t.get('prompt_per_second',0):.2f} {t.get('predicted_per_second',0):.2f} {u.get('prompt_tokens',0)}\")")"
            PK=$(vram)
            log "  d=$d rep$rep: ${ptok} tok  PP=${pp}  TG=${tg}  vram=${PK}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$ARM" "$CTX" "$CK" "$CV" "$d" "$rep" "$ptok" "$pp" "$tg" "$PK" "$((VRAM_TOTAL-PK))" >> "$RESULTS"
        done
    done
    stop
    log ""
done

python3 - "$RESULTS" "$SUMMARY" <<'PY'
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t')
      if r['pp_tok_s'] not in ('','-','LOAD_FAIL')]
cells={}
for r in rows: cells.setdefault((r['arm'],r['ctx'],int(r['depth'])),[]).append(r)
with open(sys.argv[2],'w') as f:
    w=csv.writer(f,delimiter='\t')
    w.writerow(['arm','ctx','depth','n','pp_med','tg_med','pp_min','pp_max','tg_min','tg_max','vram_mib','headroom'])
    for k in sorted(cells,key=lambda x:(x[0],x[2])):
        v=cells[k]
        pp=[float(x['pp_tok_s']) for x in v]; tg=[float(x['tg_tok_s']) for x in v]
        w.writerow([k[0],k[1],k[2],len(v),f"{st.median(pp):.1f}",f"{st.median(tg):.2f}",
                    f"{min(pp):.1f}",f"{max(pp):.1f}",f"{min(tg):.2f}",f"{max(tg):.2f}",
                    v[0]['vram_mib'],v[0]['headroom_mib']])
PY

log ""
column -t -s $'\t' "$SUMMARY" | tee -a "$LOG"
log "FABLE_KV_DONE"
