# dflash-scripts

Build a matched DFlash2 drafter (and an MXFP4 GPTQ target) for a Qwen3.8-27B-class model, served on
the `vllm-sly-radiance` image on one AMD R9700. Everything runs inside the serving image
(ROCm torch that works on gfx1201); nothing is installed on the host except `.venv-eval` (lm-eval).

## Pipeline

| step | script | notes |
|---|---|---|
| prompt pool | `build_pool.py data/pool.jsonl` | ~17k prompts, 36 public sources, 8 BetterBench-like categories; GSM8K test is never used |
| split | `split_pool.py data/pool.jsonl data calib=89 val=200 heldout=240 train=5000` | round-robin over sources, disjoint sets |
| target answers | `generate.py PROMPTS OUT --model NAME` | resumable, `X-Request-Id: gen<id>` so captures and answers can be joined |
| GPTQ MXFP4 | `quantize_gptq.py SRC OUT --calib calib_gen.jsonl --prompts calib.jsonl` | Swift-1.5 recipe (MXFP4A16, static act-order, 89 x 2048, sequential); calibration = the model's own answers in its served template; needs `PYTHONPATH=gptq-site CT_ENFORCE_EAGER=1` (see Gotchas) |
| layout check | `check_layout.py REF_DIR NEW_DIR` | tensor names/shapes/dtypes, quant config, ignore list |
| capture | `serve_capture.sh TARGET CAPTURE_DIR [DRAFTER]` + `generate.py` | `capture/dflash_capture.py` hooks the DFlash speculator; prompt trimmed to the last 2,048 positions (the drafter's window) |
| train | `train_drafter.py --capture DIRS --val-dir DIR --drafter BF16 --target BF16_TARGET --out DIR` | DFlash recipe, CPU-offloaded AdamW, anchors on the answer only |
| Hessians | `train_drafter.py ... --hessians H.pt --eval-only` | the drafter's own Linear inputs on captures |
| export | `quant_drafter.py BF16_DIR OUT H.pt` | W4A16 pack-quantized g128 GPTQ, layout identical to syvai/Qwen3.8-27B-DFlash2-W4A16 (1.19 GiB) |
| served eval | `eval_served.py heldout.jsonl out.json --model NAME` | single stream, prod sampling, spec counters per prompt -> per-category acceptance |

## Gotchas (vllm-sly-radiance 1.13.4 image)

- `python3 -m venv` is unavailable in the image: llm-compressor 0.15.0 / compressed-tensors 0.20.1 /
  transformers 5.21.0 are installed `--no-deps --target gptq-site` and put on `PYTHONPATH`.
- llm-compressor's Triton GPTQ kernel calls `libdevice.div_rn`, which the HIP backend lacks:
  set `CT_ENFORCE_EAGER=1` (torch path).
- The image ships no flash / mem-efficient SDPA kernels for gfx1201 (`hipErrorInvalidImage` in the
  first full-attention layer): `quantize_gptq.py` forces the math SDPA path.
- Captures need prefix caching off; the last requests of a run are flushed only when a later
  request arrives, so send one short request after each set.

## Credits

`gptq_utils.py` and `quant_drafter.py` follow syv-ai/qwen38-27b-rtx3090 `drafter/` (Apache-2.0).
`train_drafter.py` and the capture hook derive from vllm-sly-radiance `paroquant/drafter/` and
`radiance_dflash_capture.py`.
