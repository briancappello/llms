#!/usr/bin/env bash
# Phase 1 capability gates for candidate qwen35moe coder models.
#
# These are GATES, not measurements. Each one can disqualify a model before we
# spend an hour on context probes and quality runs. Ordered cheapest-first.
#
#   G1 load          server reaches /health at a modest context
#   G2 mtp_detect    INFORMATIONAL ONLY -- see note below.
#   G3 mtp_engage    --spec-type draft-mtp loads AND drafts (draft_n > 0)
#
# NOTE on G2: grepping the load log for "n_layer_nextn" does NOT work for this
# architecture. llama.cpp only prints that field for LLM_ARCH_BAILINGMOE2
# (src/llama-model.cpp:1921-1930); qwen35moe never logs it, so the grep always
# fails even when MTP is present and working. MTP presence is established by
# direct GGUF inspection (blk.40.nextn.* tensors + nextn_predict_layers) and
# proven behaviourally by G3. G2 must therefore never gate G3.
# See phase1-g3-mtp.sh for the corrected standalone G3.
#   G4 toolcall      22-scenario suite. KAT/Qwopus emit the XML
#                    <function=><parameter=> format, NOT Qwen JSON, so this
#                    exercises llama.cpp's auto-parser marker extraction.
#                    Ornith and Fable both set the bar at 22/22.
#   G5 reasoning     does it emit reasoning_content? Decides whether it needs
#                    --reasoning-budget, and it is the documented NIAH
#                    max_tokens trap (64 tokens got eaten by reasoning -> 0/12).
#
# G4 is the one that matters. A model that cannot tool-call is useless as an
# opencode agent no matter how fast or how low its KLD is.
#
# Output: /tmp/phase1-gates.tsv   Log: /tmp/phase1-gates.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q="$HERE/../quality"
HUB="$HOME/.cache/huggingface/hub"
SERVER="$HOME/opt/llama.cpp-vulkan/bin/llama-server"

PORT=8094
CTX=32768
TIMEOUT=420
RESULTS=/tmp/phase1-gates.tsv
LOG=/tmp/phase1-gates.log
SERVERLOG=/tmp/phase1-server.log
OUTDIR=/tmp/phase1
mkdir -p "$OUTDIR"

# name|path|has_mtp
MODELS=(
  "kat-apex|$(ls "$HUB"/models--gbuzhf--KAT-Coder-V2.5-Dev-APEX-MTP-GGUF/snapshots/*/Kwaipilot_KAT-Coder-V2.5-Dev-MTP-UD-Q5_K_XL.gguf 2>/dev/null | head -1)|yes"
  "kat-q5km|$(ls "$HUB"/models--bartowski--Kwaipilot_KAT-Coder-V2.5-Dev-GGUF/snapshots/*/Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf 2>/dev/null | head -1)|no"
  "qwopus-mtp|$(ls "$HUB"/models--Jackrong--Qwopus3.6-35B-A3B-Coder-MTP-GGUF/snapshots/*/Qwopus3.6-35B-A3B-Coder-MTP-Q5_K_M.gguf 2>/dev/null | head -1)|yes"
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{print int($NF/1048576); exit}'; }
stop() { pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 5; }
trap 'stop; exit 130' INT TERM

: > "$LOG"
printf 'model\tg1_load\tg2_mtp_detect\tg3_mtp_engage\tg4_toolcall\tg5_reasoning\tvram_mib\tload_s\tnotes\n' > "$RESULTS"

# start_server <model_path> <extra args...>
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
    local s; s=$(date +%s); LOAD_S=0
    while :; do
        LOAD_S=$(( $(date +%s) - s ))
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 $SRV_PID 2>/dev/null || return 1
        (( LOAD_S >= TIMEOUT )) && return 1
        sleep 3
    done
}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name mpath has_mtp <<< "$entry"
    g1=FAIL g2=- g3=- g4=- g5=- v=- ls_=- notes=""

    log "############ $name"
    if [[ -z "$mpath" || ! -f "$mpath" ]]; then
        log "  MISSING FILE"
        printf '%s\tMISSING\t-\t-\t-\t-\t-\t-\tfile not found\n' "$name" >> "$RESULTS"
        continue
    fi
    log "  $(basename "$mpath") ($(awk -v s="$(stat -Lc %s "$mpath")" 'BEGIN{printf "%.0f", s/1048576}') MiB)"
    stop

    # ---------------------------------------------------------------- G1
    if start_server "$mpath"; then
        g1=PASS; ls_=$LOAD_S; v=$(vram)
        log "  G1 load        PASS  (${LOAD_S}s, ${v} MiB)"
    else
        log "  G1 load        FAIL"
        grep -oiE 'error|unsupported|out of memory|failed to [a-z ]*|unknown model architecture' "$SERVERLOG" \
            | sort -u | head -4 | sed 's/^/      /' | tee -a "$LOG"
        cp "$SERVERLOG" "$OUTDIR/$name-loadfail.log"
        printf '%s\tFAIL\t-\t-\t-\t-\t-\t%s\tsee %s-loadfail.log\n' "$name" "$LOAD_S" "$name" >> "$RESULTS"
        stop; continue
    fi

    # ---------------------------------------------------------------- G2
    if [[ "$has_mtp" == "yes" ]]; then
        if grep -qiE 'n_layer_nextn|nextn|nextn_predict' "$SERVERLOG"; then
            g2=logged
            grep -iE 'n_layer_nextn|nextn' "$SERVERLOG" | head -3 | sed 's/^/      /' | tee -a "$LOG"
        else
            # Expected for qwen35moe: the field is simply never printed.
            g2=not-logged
        fi
        log "  G2 mtp_detect  $g2 (informational; G3 is authoritative)"
    else
        g2=NA
        log "  G2 mtp_detect  N/A (no MTP tensors in file)"
    fi

    # ---------------------------------------------------------------- G5
    curl -sf --max-time 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":256,"temperature":0}' \
        > "$OUTDIR/$name-reason.json" 2>/dev/null
    g5=$(python3 - "$OUTDIR/$name-reason.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    m=d["choices"][0]["message"]
    rc=m.get("reasoning_content") or ""
    c=m.get("content") or ""
    if rc.strip():   print("REASONS")
    elif "<think>" in c: print("RAW_THINK")
    else:            print("PLAIN")
except Exception:
    print("ERROR")
PY
)
    log "  G5 reasoning   $g5"

    # ---------------------------------------------------------------- G4
    log "  G4 toolcall    running 22 scenarios..."
    if python3 "$Q/toolcall.py" --url "http://127.0.0.1:$PORT/v1" --model "$name" \
         --out "$OUTDIR/toolcall-$name.json" >> "$LOG" 2>&1; then
        g4=$(python3 -c "
import json;d=json.load(open('$OUTDIR/toolcall-$name.json'))['summary']
print(f\"{d['passed']}/{d['total']}\")" 2>/dev/null || echo ERROR)
    else
        g4=ERROR
    fi
    log "  G4 toolcall    $g4"
    stop

    # ---------------------------------------------------------------- G3
    if [[ "$has_mtp" == "yes" ]]; then
        if start_server "$mpath" --spec-type draft-mtp; then
            curl -sf --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
                -H 'Content-Type: application/json' \
                -d '{"messages":[{"role":"user","content":"Repeat this list verbatim, one per line: alpha bravo charlie delta echo foxtrot golf hotel india juliet"}],"max_tokens":200,"temperature":0}' \
                > "$OUTDIR/$name-spec.json" 2>/dev/null
            curl -sf "http://127.0.0.1:$PORT/metrics" > "$OUTDIR/$name-metrics.txt" 2>/dev/null
            g3=$(python3 - "$OUTDIR/$name-spec.json" "$OUTDIR/$name-metrics.txt" <<'PY'
import json,sys,re
dn=da=None
try:
    t=json.load(open(sys.argv[1])).get("timings",{})
    dn=t.get("draft_n"); da=t.get("draft_n_accepted")
except Exception: pass
if dn is None:
    try:
        m=open(sys.argv[2]).read()
        g=lambda k:(lambda r: float(r.group(1)) if r else None)(re.search(rf'^{k}\s+([0-9.]+)',m,re.M))
        dn=g('llamacpp:n_draft_total'); da=g('llamacpp:n_draft_accepted_total')
    except Exception: pass
if dn is None: print("UNKNOWN")
elif dn == 0:   print("NO_DRAFT")
else:           print(f"{(100.0*(da or 0)/dn):.0f}%acc(n={int(dn)})")
PY
)
            log "  G3 mtp_engage  $g3"
        else
            g3=LOAD_FAIL
            log "  G3 mtp_engage  LOAD_FAIL"
            grep -oiE 'error|unsupported|failed to [a-z ]*|nextn' "$SERVERLOG" | sort -u | head -4 | sed 's/^/      /' | tee -a "$LOG"
            cp "$SERVERLOG" "$OUTDIR/$name-specfail.log"
        fi
        stop
    else
        g3=NA
        log "  G3 mtp_engage  N/A"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$g1" "$g2" "$g3" "$g4" "$g5" "$v" "$ls_" "$notes" >> "$RESULTS"
    log ""
done

stop
log ""
column -t -s $'\t' "$RESULTS" | tee -a "$LOG"
log "PHASE1_DONE"
