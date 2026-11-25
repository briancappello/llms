#!/usr/bin/env bash
# A/B of drafters for a prose model on native Radiance, one session, each arm on its own standalone
# server (the llms registry is not touched): held-out prose prompts at their own thinking levels
# (eval_served.py: per-category tokens/update and t/s, single stream, production sampling) and
# BetterBench quick decode. Logs to results/ab-prose/.
#
# Usage: ab_prose.sh NAME:CONFIG.yaml:MODEL.rad [NAME:CONFIG:MODEL ...]
# Env: PORT (18198), HELDOUT (data/hw_heldout60.jsonl), MAX_TOKENS (2048), SKIP_BB (0), R (results/ab-prose)
set -uo pipefail
cd "$(dirname "$0")"; R=${R:-results/ab-prose}; mkdir -p $R
PORT=${PORT:-18198}; H=${HELDOUT:-data/hw_heldout60.jsonl}; U=http://127.0.0.1:$PORT
say() { echo "$(date +%T) $*" | tee -a $R/ab.log; }
for arm in "$@"; do
  IFS=: read -r NAME CFG MODEL <<< "$arm"
  say "$NAME: start ($CFG, $MODEL)"
  ~/.config/llms/radiance/run-radiance "$CFG" "$MODEL" --host 127.0.0.1 --port "$PORT" > $R/$NAME.serve.log 2>&1 &
  for i in $(seq 120); do curl -sf -m 2 $U/health > /dev/null && break; sleep 2; done
  M=$(curl -s $U/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
  grep -m1 "speculation:" $R/$NAME.serve.log | tee -a $R/ab.log
  curl -s -m 300 $U/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":64}" > /dev/null
  python3 eval_served.py "$H" $R/$NAME.heldout.json --model "$M" --url $U/v1 --metrics $U/metrics \
    --max-tokens "${MAX_TOKENS:-2048}" > $R/$NAME.heldout.log 2>&1 || say "$NAME heldout failed"
  grep -A4 RESULT $R/$NAME.heldout.log | tee -a $R/ab.log
  [ "${SKIP_BB:-0}" = 1 ] || ~/dev/better-bench/.venv/bin/betterbench run --no-update-check --endpoint $U/v1 --model "$M" --quick --decode \
    --max-model-len 262144 --name "hemmingway-$NAME" --note arm=$NAME --out $R/$NAME.bb.json > $R/$NAME.bb.log 2>&1 || say "$NAME betterbench failed"
  grep -E "Combined|^\| (prose|chat|reasoning|summarization) " $R/$NAME.bb.log | tee -a $R/ab.log
  P=$(pgrep -f "[o]pt/radiance/bin/radiance --config .*llms-radiance-$(basename "$CFG" .yaml).yaml")
  [ -n "$P" ] && kill $P; for i in $(seq 30); do [ -z "$(pgrep -f "[o]pt/radiance/bin/radiance --config .*llms-radiance-$(basename "$CFG" .yaml).yaml")" ] && break; sleep 2; done
  say "$NAME done"
done
say "A/B DONE"
