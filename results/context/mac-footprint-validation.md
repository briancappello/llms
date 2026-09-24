# macOS footprint validation (task 7.3)

**Question.** On Apple unified memory, does the inference process's
`ri_phys_footprint` (what `bench/lib/mem.py` records on macOS) track the memory
a served model actually uses as context grows?

**Setup.** M3 Max, 36 GB, Metal budget 28753 MiB (`llama-server --list-devices`).
`cyber-tiel` (Cyber-Tiel-Coder-35B-A3B UD-Q4_K_XL, qwen35moe, MTP), served through
the production stack: `llms render` -> llama-swap v243 -> `~/opt/llama.cpp-metal`
(llama.cpp 6b790a9), with the production flags (`-fa on`, f16 K/V, `-np 1
--kv-unified`, `--spec-type draft-mtp`). Variants are the production entry with
only `ctx` patched. Raw rows (including provenance and rendered command):
`mac-footprint-validation.tsv`, produced by `bench/context/footprint_validation.py`.

| ctx | footprint MiB | delta vs previous | KiB/token |
|---:|---:|---:|---:|
| 32768 | 1474 | | |
| 65536 | 2178 | +704 | 22.00 |
| 131072 | 3591 | +1413 | 22.08 |

**Expected KV slope, from the GGUF.** `block_count` 41 = 40 trunk + 1 MTP
(`nextn_predict_layers` 1). `full_attention_interval` 4 gives 10 full-attention
trunk layers; the MTP layer is also full attention, so 11 KV-cached layers.
`head_count_kv` 2, key/value length 256, f16:

    11 layers x (256 + 256) x 2 heads x 2 bytes = 22528 B = 22.0 KiB/token

The measured slope matches to within 0.4%. This also matches the ~22 KiB/token
the Linux ctx-probe recorded for the A3B hybrids with MTP enabled.

**Findings.**

1. The footprint tracks served KV growth. D10's fallback (a Metal
   `currentAllocatedSize` helper) is not needed for context-scaling measurements.
2. The footprint does **not** include the weights. The GGUF is mmap'd and
   file-backed, so its 21696 MiB does not appear in `phys_footprint`. What is
   left at 32k (1474 MiB) is KV (~704 MiB) plus compute buffers and the
   recurrent/SSM state. A capacity figure against the Metal budget is therefore
   `weights_file_mib + footprint_mib`, and the stack drivers record both.
3. llama.cpp at default verbosity no longer logs `llama_kv_cache: size` through
   llama-swap's upstream stream, so the logged-KV column is NA. The analytic
   check above replaces it. Changing log verbosity would change the served
   command, which the benchmark method forbids.
