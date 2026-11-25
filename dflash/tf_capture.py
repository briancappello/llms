"""Teacher-forced captures: the target's hidden states over answers it already wrote.

generate.py writes the answers on the fast server (the served model itself, e.g. a Radiance
container); this replays each prompt + answer through the capture server (serve_capture_native.sh,
a vLLM-servable copy of the target) as ONE prefill with max_tokens=1, so dflash_capture.py records
every position at prefill speed instead of decode speed. Each capture file then gets `prompt_len` =
the answer's first position, which train_drafter.py anchors on, and loses the one decoded token.

The text is rendered as quantize_gptq.py renders calibration samples: the chat template at the
prompt's own reasoning_effort, then reasoning, </think>, the answer and <|im_end|>.

Serve the capture server with PROMPT_KEEP and CAP_MAX at least the longest sequence (e.g. 16384),
or the hook drops early prompt positions / late answer positions.

Usage: tf_capture.py PROMPTS GEN CAPROOT SET --tokenizer DIR [--url URL] [--model NAME] [--conc 8] [--max-len 12288]
Writes CAPROOT/SET/<file>.pt; resumable (skips ids already there).
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("prompts"); ap.add_argument("gen"); ap.add_argument("caproot"); ap.add_argument("set")
ap.add_argument("--tokenizer", required=True); ap.add_argument("--url", default="http://127.0.0.1:18090/v1")
ap.add_argument("--model", default="target"); ap.add_argument("--conc", type=int, default=8)
ap.add_argument("--max-len", type=int, default=12288)
a = ap.parse_args()

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.tokenizer)
tmpl = open(os.path.join(a.tokenizer, "chat_template.jinja")).read()
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(a.prompts)}
live, out = os.path.join(a.caproot, "live"), os.path.join(a.caproot, a.set)
os.makedirs(out, exist_ok=True)
done = {f.split("gen")[1].split("-")[0] for f in os.listdir(out) if "gen" in f}


def render(p, g):
    msgs = p.get("messages") or [{"role": "user", "content": p["prompt"]}]
    eff = p.get("reasoning_effort", "medium")
    kw = {"enable_thinking": False} if eff in ("none", "minimal") else {"reasoning_effort": eff}
    head = tok.apply_chat_template(msgs, chat_template=tmpl, add_generation_prompt=True, tokenize=False, **kw)
    body = ""
    if g.get("reasoning"):
        body = ("" if head.rstrip().endswith("<think>") else "<think>\n") + g["reasoning"].strip()
    if g.get("content"):
        body += ("\n</think>\n\n" if g.get("reasoning") else "") + g["content"].strip() + "<|im_end|>\n"
    h = tok(head, add_special_tokens=False)["input_ids"]
    ids = tok(head + body, add_special_tokens=False)["input_ids"][: a.max_len]
    return ids, len(h)


rows = []
for line in open(a.gen):
    g = json.loads(line)
    if "error" in g or str(g["id"]) in done or not (g.get("content") or g.get("reasoning")):
        continue
    ids, start = render(prompts[g["id"]], g)
    if start < len(ids) - 16:
        rows.append((g["id"], ids, start))
print(f"{a.set}: {len(rows)} to capture ({len(done)} done)", flush=True)


def one(r):
    gid, ids, _ = r
    body = json.dumps({"model": a.model, "prompt": ids, "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(f"{a.url}/completions", body,
                                 {"Content-Type": "application/json", "X-Request-Id": f"gen{gid}"})
    json.load(urllib.request.urlopen(req, timeout=1800))


def flush():   # the hook writes a request's file when a later request arrives
    one((0, [1, 2, 3], 0)); time.sleep(3)


t0 = time.time(); n = 0; toks = 0
starts = {gid: (len(ids), start) for gid, ids, start in rows}
with cf.ThreadPoolExecutor(a.conc) as ex:
    for _ in ex.map(one, rows):
        n += 1
        if n % 50 == 0:
            print(f"{n}/{len(rows)} replayed, {time.time() - t0:.0f}s", flush=True)
flush()
fixed = 0
for gid, (n_ids, start) in starts.items():
    for f in glob.glob(os.path.join(live, f"*gen{gid}-*.pt")) + glob.glob(os.path.join(live, f"*gen{gid}.pt")):
        d = torch.load(f)
        # drop the one decoded token, and the prompt before the drafter's window (2048), as the live
        # hook's PROMPT_KEEP does
        keep = (d["positions"] < n_ids) & (d["positions"] >= start - 2048)
        for k in ("positions", "tokens", "hid_e4m3", "scale"):
            d[k] = d[k][keep]
        if int(d["positions"][0]) > start or len(d["positions"]) < n_ids - int(d["positions"][0]):
            print(f"WARN gen{gid}: capture covers {int(d['positions'][0])}..{int(d['positions'][-1])}, "
                  f"answer {start}..{n_ids - 1}", flush=True)
        d["prompt_len"] = start
        torch.save(d, os.path.join(out, os.path.basename(f)))
        os.remove(f); fixed += 1
for f in glob.glob(os.path.join(live, "*.pt")):   # the flush request and anything else
    os.remove(f)
print(f"{a.set}: {fixed} captures in {out}, {time.time() - t0:.0f}s", flush=True)
