#!/usr/bin/env bash
# $1 = port
exec podman run --rm --replace --name llms-radiance --network host \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --security-opt label=disable \
  --group-add keep-groups --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v radiance-vllm-cache:/root/.cache/vllm -v radiance-triton-cache:/root/.triton -v radiance-aiter-cache:/root/.aiter \
  -e HF_HUB_OFFLINE=1 -e RADIANCE_RUN_BWTEST=0 \
  -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=2 \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
  -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 \
  -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 \
  -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 \
  -e RADIANCE_MXFP4_WIDE_MAX_M=192 -e RADIANCE_ADAPTIVE_WIDTH=perseq \
  ghcr.io/slybase/vllm-sly-radiance:1.13.4-rocm10.1 \
  slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ --served-model-name swift-27b \
  --host 127.0.0.1 --port "$1" \
  --max-model-len 262144 --gpu-memory-utilization 0.91 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config.method dflash \
  --speculative-config.model syvai/Qwen3.8-27B-DFlash2-W4A16 \
  --speculative-config.num_speculative_tokens 7 \
  --speculative-config.draft_sample_method greedy \
  --speculative-config.attention_backend TRITON_ATTN \
  --attention-backend R4D_HYBRID \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
  --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' \
  --enable-prefix-caching --skip-mm-profiling --enable-mm-embeds \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template /opt/qwen-fixed.jinja \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}'
