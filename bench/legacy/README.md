# Legacy benchmark drivers (Linux / ROCm, taichi)

These shell scripts produced the measurements already recorded in `results/`
and `MODELS.md`. They are kept, unchanged apart from path fixes for the move, so
those results keep their provenance. **Do not use them for new work.**

They are Linux- and host-specific (`rocm-smi`, `~/opt/llama.cpp-vulkan`,
`-dev Vulkan0`, `MESA_VK_DEVICE_SELECT`, HF cache globs, `systemctl`). More
importantly, they start `llama-server` from a copied flag set instead of the
command `llms` renders from the registry, so the copy can drift from what
production serves.

Their successors serve every variant through the production stack
(`bench/lib/stack.py`: an isolated llms instance -> llama-swap -> the entry's
engine). They record the variant patch and the rendered command with every
row, sample memory while the server is generating, and run on Linux and macOS.

| Legacy driver | Produced | Successor |
|---|---|---|
| `speed/depth-sweep.sh` | `results/speed/kat-quants*.tsv` (via /tmp/kat-quants*.tsv) | `bench/speed/depth_sweep.py` |
| `speed/kv-sweep.sh` | `results/speed/fable-kv*.tsv` | `bench/speed/kv_sweep.py` |
| `speed/backend-sweep.sh` | `results/speed/backend-sweep.tsv`, `backend-sweep.md` | `bench/speed/backend_sweep.py` |
| `context/ctx-probe.sh` | `results/context/phase2-ctx.tsv`, `ctx-probe.tsv` | `bench/context/ctx_probe.py` |
| `context/ctx-gap.sh` | `results/context/phase2-gap.tsv` | `bench/context/ctx_probe.py --test CTX,...` |
| `context/soak-ctx.sh` | `results/context/soak-kat-apex-262144.tsv` | `bench/context/soak_ctx.py` |
| `gates/gates.sh` | `results/phase1-gates.tsv` | `bench/gates/gates.py` |
| `gates/mtp-engage.sh` | `results/speed/phase1-g3-mtp.tsv` | `bench/gates/gates.py` (G3) |

Known defect carried by the legacy context drivers: they sample VRAM while the
server is idle. The amdgpu Vulkan driver evicts an idle model to GTT, and VRAM
then reads about 0 (see `results/quality/turbo-735.md`). The successors sample
during generation.

## Cross-check against the new drivers

Task 7.8 of `openspec/changes/add-macos-support` re-runs one recorded cell on
a Linux host with the new driver and compares it here.

The legacy drivers above only run on taichi (Vulkan build, `rocm-smi`, KAT
model paths), so the cross-check ran on p1 (RTX 3000 Ada Laptop 8 GB) instead.
It compares `depth_sweep.py` with an independent recorded run on the same
hardware, `engine_compare.py`'s `results/speed/p1-cyber-tiel-quant.tsv`. The
`iq3-mtp` variant patches the entry's `cmd` to that run's exact rendered command
(ik-cuda, UD-IQ3_XXS, `--n-cpu-moe 34`, MTP n_max=1). Both use 256 generated
tokens and the server's own timings, and 3 reps.

| Depth | Recorded PP med (min-max) | New PP med (min-max) | Recorded TG med (min-max) | New TG med (min-max) |
|---|---|---|---|---|
| 16384 | 123.8 (121.8-148.8) | 136.0 (126.2-137.5) | 22.0 (21.9-23.3) | 21.3 (21.3-22.2) |
| 4096, cold page cache | 161.5 (160.3-198.9) | 95.9 (92.5-99.6) | 26.2 (25.9-26.5) | 15.2 (14.1-19.6) |
| 4096, warm re-run | 161.5 (160.3-198.9) | 135.3 (123.3-139.9) | 26.2 (25.9-26.5) | 21.5 (21.3-22.3) |

**Result: the 16384 cell agrees.** The new PP falls inside the recorded range
and TG overlaps it. The 4096 cell does not, and two host conditions explain it:

- The first run read the IQ3_XXS file cold. With 34 expert layers on the CPU,
  the first requests page expert weights in through mmap, so the speed rises
  rep by rep (TG 14.1 -> 19.6).
- During the warm re-run a qemu VM used more than one core (load average about
  10). CPU experts make the short-prompt cells the most sensitive to this.

Re-run 4096 on an idle host before you use these numbers as a baseline. Files:
`results/speed/p1-depth-sweep-crosscheck*.tsv`.
