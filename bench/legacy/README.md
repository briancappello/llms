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
the Linux host with the new driver and compares it here. **Status: pending
(needs the Linux host).**
