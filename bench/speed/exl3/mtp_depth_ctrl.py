"""Controlled MTP acceptance test: same 512-token tail, only the amount of preceding context varies."""
import sys, os, argparse, json
sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dev/exllamav3")))
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import GreedySampler

ap = argparse.ArgumentParser()
model_init.add_args(ap, cache = True, add_sampling_args = False, add_draft_model_args = True,
                    default_cache_size = 65536, default_autosplit_max_batch_size = 1)
ap.add_argument("--depths", type = int, nargs = "+", default = [0, 8192, 32768])
ap.add_argument("--tails", type = int, nargs = "+", default = [100000, 200000, 300000])
ap.add_argument("--tg", type = int, default = 256)
args = ap.parse_args()

model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
gen = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft_model,
                draft_cache = draft_cache, num_draft_tokens = args.num_draft_tokens or 2)
ids = tokenizer.encode(open("/tmp/llama_cpp_ppl_wikitext2/wikitext-2-raw/wiki.test.raw").read() * 2, add_bos = False)
print("loaded", flush = True)

for T in args.tails:
    for d in args.depths:
        gen.enqueue(Job(input_ids = ids[:, T - d:T + 512], max_new_tokens = args.tg, sampler = GreedySampler(), stop_conditions = []))
        r = None
        while gen.num_remaining_jobs():
            for x in gen.iterate():
                if x.get("eos"): r = x
        a, j = r["accepted_draft_tokens"], r["rejected_draft_tokens"]
        print(json.dumps({"tail": T, "depth": d, "accept": round(a / max(a + j, 1), 3), "acc": a, "rej": j,
                          "tg_tps": round(r["new_tokens"] / r["time_generate"], 1)}), flush = True)
