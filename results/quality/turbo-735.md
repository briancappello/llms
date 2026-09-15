# Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored NEO-CODER-MAX MTP (Q5_K_M) -- benchmark sweep

DavidAU multi-stage tune/merge of Qwen3.8-27B (Cold-Fusion GAIN + Fable-Fusion 711 DNA, then Heretic'ed/ARA de-censored). `qwen35` geometry, 64L, 262144 trained ctx, native MTP head with MTP tensors at Q8_0, NEO dual-imatrix, output tensor at f16. Weights 19.7 GiB (Q5_K_M) -- the largest quant in the fleet.

**Serving config** (registry `qwen3-8-27b-turbo-fable-cold-fusion-735-882-heretic-uncensored-neo-coder-max-mtp`): Qwen3.8 "precise coding" preset `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0`, presence/repeat penalty left at llama.cpp defaults (0.0 / 1.0) -- the card warns raising rep-pen degrades MTP acceptance and to keep temp <= 1. `-c 131072 -np 2 --kv-unified`, f16 K/V, `--spec-type draft-mtp`, `--reasoning-preserve`, `--jinja`. Backend: llama.cpp-vulkan.

**Sampling was previously unset** (`llm add` scaffold, no sampling flags), so the served config was running llama.cpp's default `min_p=0.05` -- the MODELS.md tail-truncation trap. Fixed before this sweep; `/props` verified `min_p 0.0` post-restart.

## Tool-calling (toolcall.py, 22 scenarios)

- pass **22/22 (100.0%)**, false-positive 0.0%, malformed JSON 0, 45.0s. breakdown: `{'pass': 22}`

Ties cold-fusion / qwen38-27b / kat-apex / fable / ornith / qwopus at the top. Clean gate.

## Agentic generation (coding-profile.py, 5 tasks, temp 0.6)

| metric | value |
|---|---:|
| TG t/s (median, with MTP) | 53.21 |
| MTP acceptance % (median) | **73.4** |
| PP t/s (median) | 198.7 |
| reasoning tok (total, 5 tasks) | 2460 |
| reasoning tok (median/task) | 538 |
| answer tok (total) | 999 |
| think ratio (median) | 0.79 |
| wall s (total) | 66.7 |
| hit 3000-tok cap | 0 |

## Deterministic PP/TG at depth (kv-probe.py, temp 0, 2 reps)

| depth | prompt tok | PP t/s | TG t/s | MTP acc% |
|---:|---:|---:|---:|---:|
| 4096 | 4220 | 855.2 | 54.91 | 78.1 |
| 32768 | 32892 | 790.8 | 45.80 | 67.5 |
| 65536 | 65660 | 694.5 | 43.89 | 73.1 |

Rep-to-rep spread <0.15% on PP at every depth -- the most reproducible run in `results/speed/`.

## Cross-model comparison (coding-profile, all temp 0.6 except muse-glimmer at 1.0)

| tag | reas tok (med) | think% | reas tok (total) | ans tok (total) | TG t/s | acc% |
|---|---:|---:|---:|---:|---:|---:|
| cold-fusion | 258 | 67 | 1772 | 934 | 54.2 | 68.1 |
| cold-fusion-nl | 243 | 67 | 1618 | 915 | 50.9 | 63.0 |
| **turbo-735** | **538** | **79** | **2460** | **999** | **53.2** | **73.4** |
| qwen38-27b (stock) | 625 | 74 | 3353 | 1183 | 60.4 | 65.8 |
| muse-glimmer | 640 | 76 | 3642 | 1476 | 64.5 | 66.3 |
| bigbang-v1 (A3B MoE) | 581 | 63 | 4785 | 1793 | 156.2 | 61.3 |

## Verdict

**Best MTP acceptance in the fleet, but the headline "TURBO" claim does not hold up.**

- **MTP is the standout result.** 73.4% acceptance in generation and 78/68/73% across 4k/32k/64k depth -- the highest of any model measured here (cold-fusion 68%, muse-glimmer 66%, qwen38-27b 66%, bigbang 61%). Critically, acceptance *holds at depth* (73% @64k) where muse-glimmer collapses to 50% and cold-fusion-nl to 52%. The card's rule "switch to normal quants if acceptance <50%" is not close to triggering; the MTP quant is unambiguously the right choice here, and the Q8_0 MTP tensors appear to be doing real work.

- **The thinking-token reduction claim is not supported.** The card claims "1/2 to as high as 1/10" the thinking tokens of regular Qwen3.8. Measured against stock `qwen38-27b` at the same sampling and same default `reasoning_effort=xhigh`: median 538 vs 625 tok (**-14%**), total 2460 vs 3353 (**-27%**). That is a real reduction but an order of magnitude short of the claim's low end.

  Worse for the claim, DavidAU's own *non*-TURBO `cold-fusion` sits at median 258 / total 1772 -- **less than half** the thinking tokens of TURBO-735, at 1.9x lower think-ratio-adjusted cost, while producing a comparable answer length (934 vs 999 tok total). The model named for cutting thinking tokens thinks roughly twice as much as its non-TURBO sibling.

- **Think ratio is the worst in the fleet at 0.79** -- it spends the largest share of its output budget on reasoning rather than answer, and gets no extra answer length for it (999 answer tok, second-lowest in the table). For interactive agent turns this is the metric that hurts.

- **Speed is bottom-tier, as expected for Q5_K_M.** TG 53.2 / PP 199 in generation, essentially tied with cold-fusion (54.2 / 185) despite 2.5 GiB more weights -- the higher MTP acceptance is buying back the quant cost. bigbang-v1's A3B MoE is ~3x faster at 156 t/s. Depth scaling matches the dense qwen35 family (PP -19%, TG -20% from 4k to 64k).

## Context ceiling (measured)

Measured by VRAM delta across two contexts, then verified by loading each config and generating (`results/context/turbo-735-ctx.tsv`).

**KV cost: 70.0 KiB/token at f16/f16** -- derived from the 65536 -> 131072 delta (4480 MiB / 65536 tok), not from metadata. Slightly above the 68.4 predicted from fable's Qwen3.6-27B geometry, despite the Qwen3.8-27B spec confirming identical attention layout (16 gated-attention layers of 64, 4 KV heads, head_dim 256). Fixed footprint is **20847 MiB** (weights 20201 + ~646 compute/graph), leaving **11777 MiB** for KV out of 32624.

| ctx | K | V | KiB/tok | peak VRAM | headroom | status |
|---:|---|---|---:|---:|---:|---|
| 65536 | f16 | f16 | 70.0 | 25327 | 7297 | OK |
| 131072 | f16 | f16 | 70.0 | 29807 | 2817 | OK (previous setting) |
| **163840** | f16 | f16 | 70.0 | 32047 | 577 | **OK -- f16 ceiling, now set** |
| 196608 | f16 | q8_0 | 55.0 | 31405 | 1219 | OK |
| 262144 | q8_0 | q8_0 | 40.0 | 31085 | 1539 | OK -- full trained window |

- **q8_0 is 57% of f16, not 50%** (40.0 vs 70.0 KiB/tok) -- scale factors plus alignment. Halving the estimate overshoots.
- **262144 at q8_0/q8_0 costs about what 131072 at f16/f16 costs** (31085 vs 29807 MiB): doubling context while quantising both halves is close to footprint-neutral here.
- f16 K + q8_0 V tops out near 212992 (predicted 32287 MiB / 337 MiB headroom -- too tight to trust); 196608 is the comfortable stop and matches cold-fusion's window.

## Trap: VRAM reads as ~0 when idle on this Vulkan setup

`mem_info_vram_used` (and `rocm-smi`) report **57 MiB** while a 20 GiB model is loaded and healthy -- the amdgpu driver evicts the whole allocation to GTT when the model is idle (`gtt_used` 30274 MiB, and per-process `drm-total-vram` still shows the 29.1 GiB *allocation*). Under active generation it snaps back to 29807 MiB.

**Every VRAM number in this file was sampled during sustained generation.** This invalidates idle sampling in `bench/context/ctx-probe.sh` and `bench/context/soak-ctx.sh`, which read `rocm-smi --showmeminfo vram` and would silently record near-zero on the Vulkan build. Those scripts need a load-generating thread before they can be trusted here.

## Outstanding

- **NIAH long-context recall not run.** Mandatory before shipping either quantised-KV config (196608 or 262144). The f16/f16 163840 now in the registry carries no such obligation.
- Quantised-KV PP cost not measured on this model. fable measured 9-17% PP above 64k depth for q8_0 V at ~0% TG cost; worth re-measuring here given turbo-735's unusually depth-stable MTP acceptance.
- **`display` name is wrong**: registry says "Qwen3.8 27B Brainwaves NM HERETIC BR LOA1", which belongs to a different model. Scaffolded from GGUF `general.name`; worth correcting by hand.
- The card documents `reasoning_effort` = xhigh (default) / medium / low via jinja injection. Given the 0.79 think ratio, an xhigh-vs-medium A/B is the obvious next experiment -- it may recover the token economy the TURBO branding promises.
- Vision is present in the arch but not wired (no mmproj downloaded).
