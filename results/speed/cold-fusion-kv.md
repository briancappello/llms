# cold-fusion: f16 vs q8_0 V-cache at 196608 (192Ki)

Controlled A/B via llama-swap: identical weights, ctx (196608), sampling (temp 0.6 preset), `--spec-type draft-mtp`, prompt, and `cache_prompt=false`. Only `--cache-type-v` differs. Deterministic (temp 0), gen=128, 2 reps/cell (reps agreed within <0.5%). qwen35 27B-dense geometry: 17 cached layers x 4 KV heads x 256 head_dim => ~68 KiB/token f16, ~52 KiB/token with q8_0 V.

## VRAM at load (rocm-smi GPU[0] used, 34.2 GB card)

| V cache | used | free |
|---|---:|---:|
| f16  | 33.42 GB | ~0.79 GB |
| q8_0 | 30.40 GB | ~3.81 GB |

f16 V costs +3.0 GB KV at 196608 (exactly the V-half delta, 196608 x 16 KiB). f16 headroom (~786 MiB) matches fable-fusion's accepted 781 MiB default; stable because llama.cpp sizes KV/compute up front (peak grows ~33 MiB to deep ctx, MODELS.md) -- but GPU0 must stay dedicated.

## Speed (median tok/s)

| depth | metric | f16 | q8_0 | f16 advantage |
|---:|---|---:|---:|---:|
| 4096 | PP | 894.1 | 876.9 | +2.0% |
| 4096 | TG | 50.61 | 48.82 | +3.7% |
| 4096 | MTP acc% | 63.6 | 59.6 | +6.7% |
| 65536 | PP | 713.4 | 644.8 | +10.6% |
| 65536 | TG | 43.5 | 40.18 | +8.3% |
| 65536 | MTP acc% | 64.3 | 60.0 | +7.2% |

## Takeaways

- **f16 is faster, and the gap widens with depth** (q8_0 V must dequantise on every KV read, a cost that scales with cached length): PP +2.0% at 4k grows to +10.6% at 64k; TG +3.7% -> +8.3%.
- **MTP acceptance is also ~4 pts higher with f16** (63.6-64.3% vs 59.6-60.0%): q8_0 V's rounding error slightly degrades draft agreement, compounding the TG win.
- **Cost of the choice: 70k of context ceiling and ~3 GB headroom.** f16 caps at 196608 (q8_0 was needed to reach 262144). For an agentic coding loop where prefill of a large-but-<192k context dominates each turn, the ~8-11% prefill speedup at depth is the better trade.
