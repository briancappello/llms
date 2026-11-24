#!/usr/bin/env bash
# Final A/B on llama-swap, one session: stock -> ft -> mtp -> stock. Logs to results/ab/.
set -uo pipefail
cd "$(dirname "$0")"; R=results/ab; mkdir -p $R
say() { echo "$(date +%T) $*" | tee -a $R/ab.log; }
warm() {  # first start of a new config compiles and can miss the 262K KV check by ~0.1 GiB: one retry
  for i in 1 2; do
    out=$(curl -s -m 900 http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":50}")
    echo "$out" | grep -q '"choices"' && { say "$1 up (try $i)"; journalctl --user -u llama-swap --since "-15min" --no-pager | grep "GPU KV cache size" | tail -1 | sed 's/.*GPU KV/GPU KV/' | tee -a $R/ab.log; return 0; }
    say "$1 start $i failed: $(echo "$out" | head -c 200)"
  done
  return 1
}
arm() {  # name model [gsm] [heldout]
  local T=$1 M=$2
  warm "$M" || { say "ABORT: $M does not start"; exit 1; }
  bash acc_probe.sh "$M" > $R/$T.probe.log 2>&1; tail -2 $R/$T.probe.log | tee -a $R/ab.log
  ~/dev/better-bench/.venv/bin/betterbench run --endpoint http://127.0.0.1:18080/v1 --model "$M" --quick --config bb-fast.json \
     --max-model-len 262144 --name "$T" --note arm=$T --note image=1.13.4 --out $R/$T.bb.json > $R/$T.bb.log 2>&1 || say "$T betterbench failed"
  grep -E "Combined" $R/$T.bb.log | tee -a $R/ab.log
  [ "${3:-}" = gsm ] && { ./gsm8k.sh "$M" $R/gsm-$T > $R/$T.gsm.log 2>&1; grep flexible $R/$T.gsm.log | tee -a $R/ab.log; }
  [ "${4:-}" = heldout ] && { python3 eval_served.py data/heldout.jsonl $R/$T.heldout.json --model "$M" > $R/$T.heldout.log 2>&1; grep RESULT $R/$T.heldout.log | tee -a $R/ab.log; }
  say "$T done"
}
say "A/B start"
arm a1-stock ultimate-gptq gsm
arm b-ft ultimate-gptq-ft gsm heldout
arm c-mtp ultimate-gptq-mtp gsm heldout
arm a2-stock ultimate-gptq
say "A/B DONE"
