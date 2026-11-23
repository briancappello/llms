#!/usr/bin/env bash
# GSM8K 200 (cot zero-shot, greedy, chat template) against a llama-swap model. Usage: gsm8k.sh MODEL OUTDIR
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/.venv-eval/bin/lm_eval" run --model local-chat-completions \
  --model_args "model=$1,base_url=http://127.0.0.1:18080/v1/chat/completions,num_concurrent=8,max_retries=3,timeout=600" \
  --tasks gsm8k_cot_zeroshot --limit 200 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=4096 \
  --output_path "$2" --log_samples
