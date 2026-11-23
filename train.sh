#!/usr/bin/env bash
# Run train_drafter.py on the GPU inside the serving image. Usage: train.sh OUT_NAME [train_drafter args...]
# Paths inside: /cap = /opt/models/dflash-captures/ultimate-gptq, /out = /opt/models/dflash-drafters/work, /drafters = /opt/models/dflash-drafters (ro)
set -euo pipefail
cd "$(dirname "$0")"; OUT=$1; shift
DR=/root/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5
TG=/root/.cache/huggingface/hub/models--DavidAU--Qwen3.8-27B-Ultimate-Heretic-Uncensored/snapshots/74ac9c3ca6c8ac3d24e1668375f908a753c563f4
mkdir -p /opt/models/dflash-drafters/work
exec podman run --rm --name dflash-train --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --security-opt label=disable --group-add keep-groups --ipc=host -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v /opt/models/dflash-captures/ultimate-gptq:/cap:ro -v /opt/models/dflash-drafters/work:/out -v /opt/models/dflash-drafters:/drafters:ro -v "$PWD:/work" -w /work \
  --entrypoint python3 ghcr.io/slybase/vllm-sly-radiance:1.13.4-rocm10.1 -u train_drafter.py \
  --drafter "${DRAFTER:-$DR}" --target "$TG" --out "/out/$OUT" "$@"
