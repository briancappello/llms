# MLX vs GGUF on the M3 Max (Cyber-Tiel-Coder-35B-A3B)

**Question.** On this Mac (M3 Max, 36 GB, Metal budget 28753 MiB), is the MLX build
or the GGUF Q4_K_XL build faster to serve?

**Method.** `bench/speed/engine_compare.py`, through the production stack
(`llms render` -> llama-swap -> each entry's engine). The same prompts go to
every model: the code corpus trimmed to 4k and 16k tokens once, with a unique
leading nonce per request so no server reuses a prefix. Speeds are timed on the
client from the stream, identically for every server:

- prefill = prompt tokens / time to first token
- decode = tokens / remaining time

Median of 2 reps, max_tokens 256, request temperature 0. All four ran in one
session (2026-09-24 08:24-08:31). Swap stayed flat, so the host was not paging.
Raw rows, with the rendered command for each: `mac-engine-compare-omlx.tsv`.

| ctx | serving | prefill tok/s | decode tok/s | TTFT |
|---:|---|---:|---:|---:|
| 4k | GGUF UD-Q4_K_XL, llama.cpp Metal 6b790a9, MTP (production) | 822 | 63.8 | 5.4 s |
| 4k | MLX oQ4e, oMLX 0.7.0.dev4, MTP on | 917 | 57.2 | 4.8 s |
| 4k | MLX oQ4e, oMLX, MTP off | **951** | **66.4** | **4.6 s** |
| 4k | MLX oQ4e, mlx_vlm 0.6.12 | 745 | 19.5 | 5.9 s |
| 16k | GGUF UD-Q4_K_XL, MTP (production) | 572 | **55.7** | 29.1 s |
| 16k | MLX oQ4e, oMLX, MTP on | 1053 | **55.7** | 15.8 s |
| 16k | MLX oQ4e, oMLX, MTP off | **1065** | 54.9 | **15.6 s** |
| 16k | MLX oQ4e, mlx_vlm | 563 | 10.9 | 29.8 s |

**Findings.**

1. **MLX under oMLX is the fastest serving here.** Decode matches the GGUF
   (55-66 tok/s). Prefill is similar at 4k and **1.85x faster at 16k**
   (TTFT 15.6 s vs 29.1 s), so the gap grows with context. That favours oMLX
   for prefill-heavy agentic turns.
2. **The runtime matters more than the format.** The same MLX weights under
   mlx_vlm decode 3-5x slower and fall off with depth (10.9 tok/s at 16k).
   Don't serve MLX through mlx_vlm here.
3. **MTP on oMLX buys nothing on this chip.** Decode is 57.2 vs 66.4 at 4k
   and 55.7 vs 54.9 at 16k, which matches the model card ("in MLX it did
   not"). On llama.cpp, MTP accepted 65-70% of drafts. The earlier same-method
   run measured it at +0-14% decode for -10-15% prefill.
4. Absolute numbers vary between sessions: an earlier run on a busier, hotter
   host measured the GGUF at 490/45 at 4k. Compare rows within one session.

**Not measured here.** oMLX's paged SSD prefix cache, which is its main feature
for agent loops (the nonce deliberately defeats it). Also depths above 16k,
and oMLX's own `timings`, which it does not return, hence the client-side
timing.
