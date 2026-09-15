# cold-fusion vs qwen38-27b -- agentic-coding profile

Both served at **identical request sampling** (temp 0.6, top_p 0.95, top_k 20, min_p 0.0 -- Qwen3.8 'precise coding' preset), same geometry (qwen35 27B-dense, ctx 262144, f16 K / q8_0 V, `--spec-type draft-mtp`, `--reasoning-preserve`), so the only variable is the weights. cold-fusion = DavidAU Cold-Fusion Q4_K_M; qwen38-27b = stock unsloth UD-Q4_K_XL. **Caveat: n=5 single-turn coding prompts, 1 rep, stochastic temp 0.6 -- directional, not a rigorous benchmark.**

## Coding generation profile (bench/quality/coding-profile.py)

| metric | cold-fusion | qwen38-27b | delta |
|---|---:|---:|---:|
| reasoning tokens (total, 5 tasks) | 1772 | 3353 | -47% |
| reasoning tokens (median/task) | 258 | 625 | -59% |
| answer tokens (total) | 934 | 1183 | -21% |
| completion tokens (total) | 2721 | 4551 | -40% |
| think ratio (median) | 0.67 | 0.74 | -9% |
| TG throughput t/s (median) | 54.19 | 60.35 | -10% |
| PP throughput t/s (median) | 185.3 | 200.1 | -7% |
| MTP acceptance % (median) | 68.1 | 65.8 | +3% |
| wall time s (total, 5 tasks) | 53.3 | 76.8 | -31% |

## Tool-calling reliability (bench/quality/toolcall.py, 22 scenarios)

| metric | cold-fusion | qwen38-27b |
|---|---:|---:|
| passed | 22 | 22 |
| pass_rate | 100.0 | 100.0 |
| positive_pass_rate | 100.0 | 100.0 |
| false_positive_rate | 0.0 | 0.0 |
| malformed_json | 0 | 0 |

## Per-task detail (reasoning tok / answer tok / TG t/s / MTP acc%)

| task | cold-fusion | qwen38-27b |
|---|---|---|
| merge_intervals | 204/102/61.0/80.6 | 337/102/57.54/60.9 |
| fix_binsearch | 527/145/48.75/57.3 | 402/193/62.9/69.2 |
| lru_cache | 579/411/58.72/76.3 | 825/410/64.39/72.2 |
| async_refactor | 258/218/54.19/68.1 | 1164/407/60.35/65.8 |
| sql_second_salary | 204/58/48.86/58.1 | 625/71/57.02/60.0 |

## Takeaways

- **Thinking-token reduction (the headline claim): confirmed.** cold-fusion used 53% of qwen38's reasoning tokens in total (47% fewer) and 41% at the median (59% fewer) -- lands in the card's advertised 1/5-1/2 range.
- **Wall-clock: 31% faster** to complete the same 5 tasks (53.3s vs 76.8s), despite slightly lower raw TG (54.19 vs 60.35 t/s) -- the win is from generating fewer tokens, not faster tokens.
- **MTP healthy on both**: acceptance medians 68.1% / 65.8%, both well above the 50% floor where the card says to drop MTP.
- **Tool-calling: tie at 100%** (22/22, 0 false positives, 0 malformed JSON) -- no agentic-reliability regression from the smaller thinking blocks.
