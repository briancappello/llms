On ROCm, `convert.py` silently produces a corrupt `lm_head` for large-vocab models. This PR chunks one fp32 GEMM in the quantizer to work around a hipBLASLt bug (https://github.com/ROCm/rocm-libraries/issues/13243).

**Symptom.** Converting Huihui-Ornith-1.5-35B-A3B (Qwen3_5Moe, vocab 248320) at `-b 4.5 -hq` on an R9700 (gfx1201, ROCm 7.2.4, torch 2.14.1+rocm7.2):
- All 40 layers are fine (sqnr 34–42 dB).
- `lm_head` is reported as `proxy_err: 0.722926 ... rfn: 0.997449 cos: 0.920406 sqnr: 0.021167`.
- The job finishes normally, and the model generates `the the the <|audio_pad|> ...`.

**Cause.**
- `preapply_had_r` computes `(x.view(k, -1, 128) @ had)`, an fp32 GEMM with k·n/128 rows. For this lm_head that is 2048·248320/128 ≈ 4M rows.
- On gfx1201, hipBLASLt (reached through rocBLAS for torch fp32 matmuls) returns wrong values from row 2^19 or 2^20 onward, depending on the algorithm. There is a native C++ repro in the linked issue.
- `ldlq` output is correct; the un-rotation afterwards is what corrupts it.
- Bisecting on output width: the error is 0.0003 up to n = 65536, then 0.50 / 0.67 / 0.74 at n = 131072 / 196608 / 248320, which is exactly 1 − 65536/n.

**Fix.** `preapply_had_r` now splits along k so that no GEMM exceeds 2^19 rows. This is exact, so every tensor that fits under the bound is unaffected. Only tensors with in·out > 2^26 take the chunked path.

**Result (same job, tail redone):**
- `lm_head` comes out at `proxy_err 0.000041`, `sqnr 41.27`.
- Output is coherent.
- wikitext-2 PPL over 40 chunks is 8.219, vs 8.184 for the bf16 model via transformers.

Setting `ROCBLAS_USE_HIPBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0` also avoids the bug for every shape, if you'd rather document that than chunk.

**Possible follow-up (not in this PR):** the converter printed a tensor at sqnr 0.02 dB as an ordinary line. A warning or abort for obviously broken tensors, e.g. sqnr < 10 dB on a calibrated module, would have caught this immediately.
