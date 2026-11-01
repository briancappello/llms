"""
llama-server context-depth sweep with the server's own settings (MTP etc.), matching
exl3/depth_sweep_exl3.py token-for-token: at depth d the prompt is ids[o : o + d + 512] (o = d % 4096) of
wikitext-2 tokenized with the model's HF tokenizer, then n_predict tokens greedily. Reports decode tok/s and
draft acceptance from the server's timings; the prompt rate covers the whole prompt (no prompt cache).

Usage: python llama_depth_sweep.py --url http://127.0.0.1:18999 --tokenizer <hf model dir> --depths 0 32768 ...
"""
import argparse, json, time, urllib.request
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--url", default = "http://127.0.0.1:18999")
ap.add_argument("--tokenizer", required = True)
ap.add_argument("--depths", type = int, nargs = "+", default = [0, 8192, 32768, 65536, 131072, 196608, 245760])
ap.add_argument("--pp", type = int, default = 512)
ap.add_argument("--tg", type = int, default = 256)
ap.add_argument("--text", default = "/tmp/llama_cpp_ppl_wikitext2/wikitext-2-raw/wiki.test.raw")
ap.add_argument("--json", default = None)
a = ap.parse_args()

tok = Tokenizer.from_file(f"{a.tokenizer}/tokenizer.json")
ids = tok.encode(open(a.text).read() * 2, add_special_tokens = False).ids
rows = []
for d in [x // 256 * 256 for x in a.depths]:
    o = d % 4096
    body = {"prompt": ids[o:o + d + a.pp], "n_predict": a.tg, "temperature": 0, "top_k": 1, "cache_prompt": False,
            "ignore_eos": True}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(a.url + "/completion", json.dumps(body).encode(),
                                                                 {"Content-Type": "application/json"}), timeout = 7200).read())
    t = r["timings"]
    row = {"depth": d, "prompt_tps": round(t["prompt_per_second"], 1), "tg_tps": round(t["predicted_per_second"], 2),
           "tg_tokens": t["predicted_n"], "wall_s": round(time.time() - t0, 1)}
    if t.get("draft_n"):
        row["draft_accept"] = round(t["draft_n_accepted"] / t["draft_n"], 3)
    rows.append(row); print(json.dumps(row), flush = True)
if a.json:
    json.dump(rows, open(a.json, "w"), indent = 1)
