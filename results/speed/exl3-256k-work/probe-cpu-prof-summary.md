# Decode probe, MTP off, EXL3_MOE_CPU_PROF=1 (2026-10-06 ~19:40-19:47)
Short prompt (48-97 tokens), reasoning_effort low, temp 1.0 / top_p 0.95, one request after a warm-up, exllamav3 rocm-graph-kernarg-leak 9182131, Tabby 2fd6cc7.
CPU job = one MoE layer's CPU-resident experts for one forward pass (1 token here). us/job = sum of the 6 profiled phases (cumulative average incl. warm-up).

| model | split (CPU/total experts) | GPU share | decode tok/s | ms/token | us/CPU job | MoE layers | CPU expert ms/token | rest ms/token |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next 3.05bpw | 300/512 | 41% | 42.5 (1,297 tok) | 23.5 | 195.5 | 48 | 9.4 | 14.1 |
| DeepSeek-V4-Flash 2.52bpw  | 200/256 | 22% | 27.1 (4,368 tok) | 36.9 | 514.2 | 43 | 22.1 | 14.8 |
| GLM-5.3-Flash 2.05bpw      | 234/288 | 19% | 22.7 (1,449 tok) | 44.1 | 645.5 | 42 | 27.1 | 17.0 |

Routed-expert weights read per token (from safetensors): Qwen 0.85 GiB (10 x 1.81 MiB x 48), DeepSeek 1.94 GiB (6 x 7.5 MiB x 43), GLM ~2.0 GiB (8 x 6.04 MiB x 42).
Total routed experts: Qwen 43 GiB, DeepSeek 80.6 GiB (+6.8 MTP), GLM 73 GiB -> GPU holds 41% / 22% / 19% of them.
