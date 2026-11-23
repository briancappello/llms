#!/usr/bin/env bash
# Serve a target with a DFlash2 drafter on the vllm-sly-radiance image with in-serve capture ON
# (capture/dflash_capture.py), prefix caching OFF, production sampling and template.
#
# Usage: serve_capture.sh TARGET_DIR CAPTURE_DIR [DRAFTER (HF id or dir)] [PORT]
# Env: IMAGE, NAME (container name, default dflash-capture), GPU_UTIL (0.91), MAX_LEN (262144),
#      CHAT_TEMPLATE (default TARGET_DIR/chat_template.jinja), SERVED (served model name, default "target")
set -euo pipefail
TARGET=$(readlink -f "$1"); CAP=$2; DRAFTER=${3:-syvai/Qwen3.8-27B-DFlash2-W4A16}; PORT=${4:-18090}
IMAGE=${IMAGE:-ghcr.io/slybase/vllm-sly-radiance:1.13.4-rocm10.1}; NAME=${NAME:-dflash-capture}
HERE=$(cd "$(dirname "$0")" && pwd); SP=/opt/vllm/lib/python3.12/site-packages
mkdir -p "$CAP"
DMOUNT=(); DARG=$DRAFTER
if [ -d "$DRAFTER" ]; then DMOUNT=(-v "$(readlink -f "$DRAFTER"):/drafter:ro"); DARG=/drafter; fi
exec podman run --rm --replace --name "$NAME" --network host --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --security-opt label=disable --group-add keep-groups --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v radiance-vllm-cache:/root/.cache/vllm -v radiance-triton-cache:/root/.triton -v radiance-aiter-cache:/root/.aiter \
  -v "$TARGET:/models/target:ro" "${DMOUNT[@]}" -v "$CAP:/capture" \
  -v "$HERE/capture/dflash_capture.py:$SP/dflash_capture.py:ro" -v "$HERE/capture/dflash_capture.pth:$SP/dflash_capture.pth:ro" \
  -e DFLASH_CAPTURE_DIR=/capture -e DFLASH_CAPTURE_PROMPT_KEEP=${PROMPT_KEEP:-2048} -e DFLASH_CAPTURE_MAX_TOKENS=${CAP_MAX:-8192} \
  -e HF_HUB_OFFLINE=1 -e RADIANCE_RUN_BWTEST=0 -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=2 \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 -e RADIANCE_MXFP4_DECODE_MAX_M=128 \
  -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 \
  -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 -e RADIANCE_MXFP4_WIDE_MAX_M=192 \
  -e RADIANCE_ADAPTIVE_WIDTH=perseq \
  "$IMAGE" /models/target --host 127.0.0.1 --port "$PORT" --served-model-name "${SERVED:-target}" \
  --max-model-len "${MAX_LEN:-262144}" --gpu-memory-utilization "${GPU_UTIL:-0.91}" --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype bfloat16 --speculative-config.method dflash --speculative-config.model "$DARG" \
  --speculative-config.num_speculative_tokens 7 --speculative-config.draft_sample_method greedy \
  --speculative-config.attention_backend TRITON_ATTN --attention-backend R4D_HYBRID --max-num-seqs 8 \
  --max-num-batched-tokens 2048 --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
  --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' --no-enable-prefix-caching \
  --skip-mm-profiling --enable-mm-embeds --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template "${CHAT_TEMPLATE:-/models/target/chat_template.jinja}" \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}'
