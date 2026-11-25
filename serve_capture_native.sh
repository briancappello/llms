#!/usr/bin/env bash
# serve_capture.sh without a container: the target with a DFlash2 drafter on the host's vLLM
# (~/opt/vllm, the llms companions' ROCm build), capture/dflash_capture.py hooked in, prefix caching
# OFF, the target's own template and production sampling.
#
# Usage: serve_capture_native.sh TARGET_DIR CAPTURE_DIR DRAFTER_DIR [PORT]
# Env: VLLM_PREFIX (~/opt/vllm), MAX_SEQS (8), GPU_UTIL (0.80), MAX_LEN (16384), SERVED (target), KV_DTYPE (fp8),
#      SAMPLING (JSON override-generation-config; default Qwen thinking: temp 1.0, top-p 0.95, top-k 20)
set -euo pipefail
TARGET=$(readlink -f "$1"); CAP=$(readlink -f "$2"); DRAFTER=$(readlink -f "$3"); PORT=${4:-18090}
HERE=$(cd "$(dirname "$0")" && pwd); VP=${VLLM_PREFIX:-$HOME/opt/vllm}
mkdir -p "$CAP/live"
# The hook must load in vLLM's worker processes too: a sitecustomize on PYTHONPATH imports it in
# every interpreter (the venv has no sitecustomize of its own).
SITE=$(mktemp -d /tmp/dflash-capture-site.XXXX)
cp "$HERE/capture/dflash_capture.py" "$SITE/"
echo "import dflash_capture" > "$SITE/sitecustomize.py"
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH
export PATH="$VP/runtime/bin:$PATH" HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} HF_HUB_OFFLINE=1
export DFLASH_CAPTURE_DIR="$CAP/live" DFLASH_CAPTURE_PROMPT_KEEP=${PROMPT_KEEP:-2048} DFLASH_CAPTURE_MAX_TOKENS=${CAP_MAX:-8192}
exec "$VP/runtime/bin/vllm" serve "$TARGET" --host 127.0.0.1 --port "$PORT" --served-model-name "${SERVED:-target}" \
  --max-model-len "${MAX_LEN:-16384}" --gpu-memory-utilization "${GPU_UTIL:-0.80}" --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  --speculative-config "{\"method\": \"dflash\", \"model\": \"$DRAFTER\", \"num_speculative_tokens\": 7}" \
  --max-num-seqs "${MAX_SEQS:-8}" --max-num-batched-tokens 2048 --no-enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template "$TARGET/chat_template.jinja" \
  --override-generation-config "${SAMPLING:-{\"temperature\": 1.0, \"top_p\": 0.95, \"top_k\": 20, \"min_p\": 0.0, \"presence_penalty\": 0.0}}"
