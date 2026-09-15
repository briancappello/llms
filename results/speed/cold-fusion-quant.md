# cold-fusion quant A/B: IQ4_NL vs Q4_K_M

Same weights (DavidAU Qwen3.8-27B Cold-Fusion), same serving config (f16 K/V, ctx 196608, temp 0.6 precise-coding preset, --spec-type draft-mtp, --reasoning-preserve). Only the quant differs. MTP tensors are Q8_0 in both (per the card). `cold-fusion` = Q4_K_M (~4.85 bpw); `cold-fusion-nl` = IQ4_NL (~4.25 bpw, non-linear).

## Footprint (measured)

| | Q4_K_M | IQ4_NL | delta |
|---|---:|---:|---:|
| weights on disk | 17.23 GiB | 16.53 GiB | -4.0% |
| VRAM used @ f16/192Ki (rocm GPU[0]) | 33.42 GB | 32.68 GB | -0.74 GB |
| free headroom (34.2 GB card) | ~0.79 GB | ~1.53 GB | +0.74 GB |

## Speed at depth -- deterministic A/B (kv-probe.py, temp 0, 2 reps, <0.5% spread)

| depth | metric | Q4_K_M | IQ4_NL | IQ4_NL vs Q4_K_M |
|---:|---|---:|---:|---:|
| 4096 | PP t/s | 894.1 | 951.9 | +6.5% |
| 4096 | TG t/s | 50.61 | 51.14 | +1.0% |
| 4096 | MTP acc% | 63.6 | 64.8 | +1.9% |
| 65536 | PP t/s | 713.4 | 750.3 | +5.2% |
| 65536 | TG t/s | 43.5 | 37.77 | -13.2% |
| 65536 | MTP acc% | 64.3 | 52.4 | -18.5% |

## Agentic generation (coding-profile.py, 5 tasks, temp 0.6)

| metric | Q4_K_M | IQ4_NL |
|---|---:|---:|
| reasoning tok (total) | 1772 | 1618 |
| reasoning tok (median) | 258 | 243 |
| completion tok (total) | 2721 | 2548 |
| TG t/s (median) | 54.19 | 50.86 |
| MTP acc% (median) | 68.1 | 63.0 |
| wall s (total) | 53.3 | 54.6 |

## Tool-calling (22 scenarios)

| pass_rate | false_positive | malformed_json |
|---|---|---|
| Q4_K_M 100.0% | 0.0% | 0 |
| IQ4_NL 100.0% | 0.0% | 0 |

## Verdict

- **IQ4_NL is smaller (-4.1%) and prefills faster** (PP +5-6% at both depths, from lower weight bandwidth), giving ~0.74 GB more VRAM headroom -- welcome, since Q4_K_M at f16/192Ki has only ~786 MiB free.
- **But IQ4_NL's MTP acceptance collapses at depth: 64.8% @ 4k -> 52.4% @ 64k, vs Q4_K_M's rock-steady 63.6% -> 64.3%.** That drags IQ4_NL TG to -13% at 64k (37.8 vs 43.5 t/s) even though its raw weights are faster. At 4k the two are within ~1% on TG.
- The card's own rule -- *"token acceptance below 50% -> switch to normal quants"* -- puts IQ4_NL right at the edge by 64k; deeper context would likely cross it, i.e. MTP stops paying and normal (non-MTP) IQ4_NL would be faster.
- **Tool-calling and thinking-token behaviour are a wash** (both 22/22, similar reasoning-token counts).
- **Recommendation for agentic coding: keep Q4_K_M (`cold-fusion`).** Agent turns re-prefill large contexts and generate at depth, where Q4_K_M's stable MTP acceptance makes it 13% faster TG. Prefer IQ4_NL (`cold-fusion-nl`) only if you need the extra ~0.74 GB VRAM (e.g. to run something alongside) or your workloads stay near-empty-context; consider dropping --spec-type draft-mtp for IQ4_NL at very deep context.
