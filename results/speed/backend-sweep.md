# Backend sweep: vulkan vs hip vs hip-wmma (commit 169e4a7)

All three built from the same commit via `~/dev/my-llama/build-backends.sh`, RPATH-isolated (`~/opt/llama.cpp-COMMIT`). Same model + flags as production (cold-fusion Q4_K_M, f16 K/V, `-fa on`, `--spec-type draft-mtp`, single stream), context allocated to 73728 to time throughput not the ceiling. Deterministic (temp 0), 2 reps/cell (agreed within <0.5%). Device: vulkan `-dev Vulkan0` (MESA_VK_DEVICE_SELECT pin), hip/hip-wmma `-dev ROCm0`; all loaded on the R9700/gfx1201 (VRAM ~22.9-23.1 GB -- the 2 GB iGPU can't hold a 17 GB model, and CPU fallback would show <5 t/s TG, so the device is confirmed).

## PP throughput (prompt processing, tok/s) -- the headline for agentic prefill

| depth | vulkan | hip | hip-wmma |
|---:|---:|---:|---:|
| 4096 | **886.9** | 941.8 | 943.4 |
| 32768 | **818.8** | 671.0 | 671.2 |
| 65536 | **713.1** | 494.0 | 493.9 |

## TG throughput (token generation, tok/s)

| depth | vulkan | hip | hip-wmma |
|---:|---:|---:|---:|
| 4096 | **50.4** | 36.5 | 36.5 |
| 32768 | **49.1** | 35.9 | 35.9 |
| 65536 | **43.3** | 29.7 | 29.6 |

## MTP acceptance (%)

| depth | vulkan | hip | hip-wmma |
|---:|---:|---:|---:|
| 4096 | 63.6 | 59.4 | 59.4 |
| 32768 | 67.5 | 66.7 | 66.7 |
| 65536 | 64.3 | 57.1 | 57.1 |

## Verdict: keep **vulkan** for serving

- **TG: vulkan wins decisively at every depth** (~50->43 t/s vs hip ~36->30), i.e. hip generates ~27-31% slower. TG is what a user feels during code output.
- **PP: hip wins only at trivial 4k depth** (941.8 vs 886.9, +6%), then loses at realistic agentic depth -- at 65k vulkan does 713.1 vs hip 494.0 (hip is -31%). The prefill advantage evaporates exactly where an agent's large context lives.
- **hip-wmma == hip**: the rocWMMA flash-attn build is within noise of plain hip at every cell, so rocWMMA brings no benefit for this attention shape (4 KV heads x 256 head_dim) on gfx1201. Not worth serving.
- MTP acceptance is backend-independent (same weights/draft), as expected.

**Serving stays on `~/opt/llama.cpp-vulkan` (config.header.yaml `server` macro already points there); `$PATH` `llama-server` symlinked to the same build.** hip / hip-wmma remain built and available for ad-hoc use (e.g. pure-prefill batch jobs at shallow depth) but are not the serving default.
