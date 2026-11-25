#!/usr/bin/env bash
# $1 = model name; plain /v1/completions (no chat template), greedy, 3 prompts x 400 tokens
M=$1; U=http://127.0.0.1:18080/upstream/$M
met() { curl -s -m 30 $U/metrics | awk '/^vllm:spec_decode_num_(accepted|draft)_tokens_total/{s[$1]+=$2} /^vllm:spec_decode_num_drafts_total/{d+=$2} END{for(k in s) printf "%s=%s ",k,s[k]; printf "drafts=%s\n",d}'; }
curl -s -m 600 http://127.0.0.1:18080/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"prompt\":\"Hi\",\"max_tokens\":1}" >/dev/null
B=$(met)
for P in "def quicksort(arr):\n    # Sort a list using quicksort.\n" "The history of the Roman Empire begins" "Q: A train travels 120 km in 1.5 hours. What is its average speed in km/h?\nA: Let's think step by step."; do
  curl -s -m 300 http://127.0.0.1:18080/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"prompt\":\"$P\",\"max_tokens\":400,\"temperature\":0}" | python3 -c "import json,sys;t=json.load(sys.stdin)['choices'][0]['text'];print(repr(t[:90]))"
done
A=$(met); echo "before: $B"; echo "after:  $A"
