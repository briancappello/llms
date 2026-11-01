"""
exllamav3 context-depth sweep, the counterpart of `llama-bench -p 512 -n 128 -d <depth>`:
at each depth d, the first d tokens are prefilled (and cached), then the timed job adds 512 new tokens
(prefill rate at depth d) and generates 128 tokens greedily (decode rate at depth d). Text is wikitext-2.

Usage (after `source exl3env.sh`):
  python depth_sweep_exl3.py -m <model> -cs 262144 [-mtp -ndt 2] --depths 0 8192 32768 ...
"""
import sys, os, time, argparse, json
REPO = os.environ.get("REPO", os.path.expanduser("~/dev/exllamav3"))
sys.path.insert(0, REPO)

import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import GreedySampler

def main():
    ap = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(ap, cache = True, add_sampling_args = False, add_draft_model_args = True,
                        default_cache_size = 262144, default_autosplit_max_batch_size = 1)
    ap.add_argument("--depths", type = int, nargs = "+", default = [0, 8192, 32768, 65536, 131072, 200000])
    ap.add_argument("--pp", type = int, default = 512)
    ap.add_argument("--tg", type = int, default = 128)
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--text", default = "/tmp/llama_cpp_ppl_wikitext2/wikitext-2-raw/wiki.test.raw")
    ap.add_argument("--json", default = None)
    args = ap.parse_args()

    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    free, total = torch.cuda.mem_get_info()
    print(f"loaded; cache {cache.max_num_tokens} tokens; VRAM used {(total - free) / 2**30:.2f} GiB of {total / 2**30:.2f}", flush = True)
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft_model,
                    draft_cache = draft_cache, num_draft_tokens = args.num_draft_tokens or 4)
    ids = tokenizer.encode(open(args.text).read() * 2, add_bos = False)
    print(f"text: {ids.shape[-1]} tokens", flush = True)

    def run(prompt, n):
        gen.enqueue(Job(input_ids = prompt, max_new_tokens = n, sampler = GreedySampler(), stop_conditions = []))
        r = None
        while gen.num_remaining_jobs():
            for x in gen.iterate():
                if x.get("eos"): r = x
        return r

    rows = []
    for d in [x // 256 * 256 for x in args.depths]:   # page-aligned, so exactly pp new tokens
        best = None
        for rep in range(args.reps):
            o = rep * 7919 + d % 4096                      # fresh text per rep: no reuse of an earlier rep's pages
            if d: run(ids[:, o:o + d + 1], 1)              # build (and cache) d tokens of context (last token is never cached)
            r = run(ids[:, o:o + d + args.pp], args.tg)    # timed: pp new tokens at depth d, then tg
            new = r["prompt_tokens"] - r["cached_tokens"]
            row = {"depth": d, "pp_tokens": new, "pp_tps": new / r["time_prefill"], "tg_tps": r["new_tokens"] / r["time_generate"]}
            if "accepted_draft_tokens" in r:
                a, j = r["accepted_draft_tokens"], r["rejected_draft_tokens"]
                row["draft_accept"] = a / max(a + j, 1)
            best = row if best is None or row["tg_tps"] > best["tg_tps"] else best
        free, total = torch.cuda.mem_get_info()
        best["vram_gib"] = round((total - free) / 2**30, 2)
        rows.append(best)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in best.items()}), flush = True)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent = 1)

if __name__ == "__main__":
    main()
