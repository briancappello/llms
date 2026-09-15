#!/usr/bin/env bash
# Phase 3: depth sweep for the three qwen35moe coder candidates, with and
# without MTP self-speculation.
#
# Answers:
#   1. PP/TG for kat-apex vs kat-q5km vs qwopus-mtp at real depths.
#   2. Does the high ACTIVE bpw of kat-apex (7.827, vs 6.556 for APEX classic)
#      buy TG on this GPU? results/README.md records that A3B here is
#      dequant-bound rather than bandwidth-bound -- Ornith's larger UD-Q5_K_XL
#      generated FASTER than its MXFP4. kat-apex is the extreme version of that
#      bet, so it should win TG if the finding generalises.
#   3. How does MTP acceptance move with depth? G3 measured it only at ~0 depth.
#
# METHOD NOTE -- cache_prompt is forced OFF, but it turns out not to matter.
#   llama-server defaults cache_prompt=true, and trim_to_tokens.py builds prompts
#   as corpus[:n] + QUESTION, so the depth ladder does share a growing common
#   prefix. The initial worry was that reused prefix would inflate PP at the
#   deeper rungs, making these numbers incomparable to bench-quants.tsv.
#
#   That worry was unfounded. llama.cpp computes prompt_per_second over the
#   tokens it ACTUALLY evaluated (n_prompt_tokens_processed), not over
#   usage.prompt_tokens, so a cache hit reduces prompt_ms and the processed
#   count together and leaves the rate honest. Confirmed empirically: these
#   cache-off numbers land within ~3% of bench-quants.tsv's cache-on numbers for
#   Ornith q5_k_xl at matched depths (PP 2587/2502/1940 here vs 2508/2524/1874
#   there, on identical prompt_tok of 4178/16466/65617).
#
#   Kept off anyway, because it removes a variable for free. Absolute PP IS
#   comparable to bench-quants.tsv and fable-quants.tsv.
#
# REPEATS=3 per cell, reported as median with min-max spread. The existing
# scripts are n=1; the G3 results had no error bars and that was a real gap.
#
# Output: /tmp/kat-quants.tsv (every sample)
#         /tmp/kat-quants-summary.tsv (median per cell)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="$HERE/../lib"
HUB="$HOME/.cache/huggingface/hub"
SERVER="$HOME/opt/llama.cpp-vulkan/bin/llama-server"

PORT=8099
CTX=131072              # every config was verified to hold this in phase 2
GEN=192
DEPTHS=(4096 16384 65536)
REPEATS=3
CORPUS=/tmp/code-corpus.txt
TIMEOUT=600

RESULTS=/tmp/kat-quants.tsv
SUMMARY=/tmp/kat-quants-summary.tsv
LOG=/tmp/kat-quants.log
SERVERLOG=/tmp/kat-quants-server.log

KAT_APEX="$(ls "$HUB"/models--gbuzhf--KAT-Coder-V2.5-Dev-APEX-MTP-GGUF/snapshots/*/Kwaipilot_KAT-Coder-V2.5-Dev-MTP-UD-Q5_K_XL.gguf 2>/dev/null | head -1)"
KAT_Q5KM="$(ls "$HUB"/models--bartowski--Kwaipilot_KAT-Coder-V2.5-Dev-GGUF/snapshots/*/Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf 2>/dev/null | head -1)"
QWOPUS="$(ls "$HUB"/models--Jackrong--Qwopus3.6-35B-A3B-Coder-MTP-GGUF/snapshots/*/Qwopus3.6-35B-A3B-Coder-MTP-Q5_K_M.gguf 2>/dev/null | head -1)"

# name|path|spec
CONFIGS=(
  "kat-apex|$KAT_APEX|none"
  "kat-apex|$KAT_APEX|draft-mtp"
  "kat-q5km|$KAT_Q5KM|none"
  "qwopus-mtp|$QWOPUS|none"
  "qwopus-mtp|$QWOPUS|draft-mtp"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}'; }
# match on process NAME: `pkill -f llama-server` would also kill the caller
stop() { pkill -x llama-server 2>/dev/null; sleep 5; }
trap 'stop; exit 130' INT TERM

: > "$LOG"
[[ -f "$CORPUS" ]] || { echo "missing $CORPUS (build with lib/build_code_corpus.py)" >&2; exit 1; }
printf 'model\tspec\tdepth\trep\tprompt_tok\tcompl_tok\tpp_tok_s\ttg_tok_s\tdraft_n\tdraft_acc\tacc_pct\tvram_mib\n' > "$RESULTS"

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name mpath spec <<< "$entry"
    [[ -n "$mpath" && -f "$mpath" ]] || { log "SKIP $name/$spec (missing file)"; continue; }
    extra=(); [[ "$spec" != none ]] && extra=(--spec-type "$spec")

    log "############ $name / spec=$spec"
    stop
    : > "$SERVERLOG"
    "$SERVER" -m "$mpath" -dev Vulkan0 -ngl 999 \
        -c "$CTX" -np 1 --kv-unified -fa on \
        --cache-type-k f16 --cache-type-v f16 \
        --jinja --metrics --no-warmup "${extra[@]}" \
        --host 127.0.0.1 --port "$PORT" > "$SERVERLOG" 2>&1 &
    pid=$!; ok=0; s=$(date +%s)
    while :; do
        el=$(( $(date +%s) - s ))
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        (( el >= TIMEOUT )) && break
        sleep 3
    done
    if (( ok == 0 )); then
        log "  LOAD FAILED"
        grep -oiE 'error|out of memory|failed to [a-z ]*' "$SERVERLOG" | sort -u | head -3 | sed 's/^/    /' | tee -a "$LOG"
        printf '%s\t%s\t-\t-\t\t\t\t\t\t\t\tLOAD_FAIL\n' "$name" "$spec" >> "$RESULTS"
        stop; continue
    fi
    V=$(vram)
    log "  loaded, vram ${V} MiB"

    for d in "${DEPTHS[@]}"; do
        # build once per depth, then replay it REPEATS times
        if ! python3 "$LIB/trim_to_tokens.py" "http://127.0.0.1:$PORT/tokenize" \
               "$CORPUS" "$d" "$GEN" > /tmp/kq-req-raw.json 2>/dev/null; then
            log "  depth $d: prompt build failed"; continue
        fi
        python3 -c "
import json
d=json.load(open('/tmp/kq-req-raw.json'))
d['cache_prompt']=False
json.dump(d,open('/tmp/kq-req.json','w'))"

        for rep in $(seq 1 "$REPEATS"); do
            if ! curl -sf --max-time 1800 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
                   -H 'Content-Type: application/json' -d @/tmp/kq-req.json \
                   > /tmp/kq-resp.json 2>/dev/null; then
                log "  depth $d rep $rep: request failed"; continue
            fi
            read -r pp tg ptok ctok dn da acc <<< "$(python3 - /tmp/kq-resp.json <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
t=d.get("timings",{}); u=d.get("usage",{})
pp=t.get("prompt_per_second",0) or 0
tg=t.get("predicted_per_second",0) or 0
dn=t.get("draft_n"); da=t.get("draft_n_accepted")
if dn is None: dn=da="-"; acc="-"
else: acc=f"{100.0*da/dn:.1f}" if dn else "0.0"
print(f'{pp:.2f} {tg:.2f} {u.get("prompt_tokens","")} {u.get("completion_tokens","")} {dn} {da} {acc}')
PY
)"
            log "  depth $d rep $rep: ${ptok} tok -> PP ${pp} | TG ${tg} | acc ${acc}%"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$name" "$spec" "$d" "$rep" "$ptok" "$ctok" "$pp" "$tg" "$dn" "$da" "$acc" "$V" >> "$RESULTS"
        done
    done
    stop
    log ""
done

stop

# ---------------------------------------------------------------- summarise
python3 - "$RESULTS" "$SUMMARY" <<'PY'
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t') if r.get('pp_tok_s') and r['pp_tok_s'] not in ('','-')]
key=lambda r:(r['model'],r['spec'],int(r['depth']))
cells={}
for r in rows: cells.setdefault(key(r),[]).append(r)
with open(sys.argv[2],'w') as f:
    w=csv.writer(f,delimiter='\t')
    w.writerow(['model','spec','depth','n','pp_med','pp_min','pp_max','tg_med','tg_min','tg_max','acc_med','vram_mib'])
    for k in sorted(cells):
        v=cells[k]
        pp=[float(x['pp_tok_s']) for x in v]; tg=[float(x['tg_tok_s']) for x in v]
        accs=[float(x['acc_pct']) for x in v if x['acc_pct'] not in ('-','')]
        w.writerow([k[0],k[1],k[2],len(v),
                    f"{st.median(pp):.1f}",f"{min(pp):.1f}",f"{max(pp):.1f}",
                    f"{st.median(tg):.2f}",f"{min(tg):.2f}",f"{max(tg):.2f}",
                    f"{st.median(accs):.1f}" if accs else '-', v[0]['vram_mib']])
PY

echo
column -t -s $'\t' "$SUMMARY" | tee -a "$LOG"
log "KAT_QUANT_BENCH_DONE"
