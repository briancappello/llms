# ROCm gfx1201 (R9700) notes: test_encode_ideal fails for mcg/3inst, convert.py RSS ratchet, tests' default devices

Setup: `dev`, R9700 gfx1201, ROCm 7.2.4, torch 2.14.1+rocm7.2. Overall the port works well on RDNA4: 867 inference kernel tests pass, and a full Qwen3.5-MoE conversion lands within 0.4% PPL of bf16 (with the hipBLASLt issue ROCm/rocm-libraries#13243 worked around, see #458).

**1. `test_quant_fn.py::test_encode_ideal` fails for the mcg and 3inst codebooks (all K, 32 cases).**
- mul1 passes, and `test_encode` passes for every codebook.
- The failures are identical with `EXL3_QT_OPTIMIZED=0`.
- Example: 100% of elements mismatched, max abs diff 0.336.
- Possibly the decode path (`ext.decode`), the codebook LUT, or the encoder.
- The module also needs a hardcoded `/mnt/str/models/llama3.1-8b-instruct/hf/` to even import.

**2. convert.py RSS grows ~3.7 GB per layer on Linux/glibc** (6 → 11 → 15 → 19 → 22 GB over 4 layers of a 35B MoE).
- Live CPU tensors stay flat at ~4 GB (the calibration state); this is glibc heap retention, which `malloc_trim()` doesn't release.
- `MALLOC_MMAP_THRESHOLD_=1048576` keeps RSS flat at ~7 GB with no speed change. A `mallopt(M_MMAP_THRESHOLD, 1<<20)` in convert.py would fix it for everyone.
- Side observation: results depend slightly on the allocator setting (layer 0 rfn 0.01244 vs 0.01235, while two runs with the same setting are bit-identical). That suggests something depends on allocation addresses or reads uninitialised memory.

**3. Several tests default to `cuda:1`/`cuda:2`** (test_sampler, test_rope, test_gated_delta_rule, test_reconstruct_had, …), so they fail on single-GPU machines unless `EXL3_TEST_DEVICE=cuda:0` is set. Defaulting to `cuda:0` would be friendlier.

**4. `eval/ppl.py -g` doesn't match llama.cpp for tokenizers that define a BOS but don't add it.**
- `eval_gguf` overwrites each chunk's first token with `tokenizer.bos_token_id` whenever it is set.
- llama.cpp only does that when `add_bos_token` is true.
- For Ornith-1.5 / Qwen3.6-35B-A3B derivatives (bos = `<|endoftext|>`, `add_bos_token = false`) this inflates PPL by ~18%:
  - an EXL3 4.5 bpw quant scores 8.219 with BOS and 6.997 without (40 chunks);
  - the same weights as a Q4_K_XL GGUF score 6.945 in llama-perplexity.
- Qwen3.5-9B has no bos_token_id and was unaffected.
- Suggest following the tokenizer's add-BOS setting.
