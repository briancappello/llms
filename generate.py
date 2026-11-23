"""Drive a served target over a prompts jsonl (resumable). Server sampling defaults apply unless
--temperature is given. Records the response id and usage so capture files can be joined later.

Usage: generate.py PROMPTS OUT --url http://127.0.0.1:18080/v1 --model NAME [--conc 8] [--max-tokens 1024] [--limit N]
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("prompts"); ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:18080/v1"); ap.add_argument("--model", required=True)
ap.add_argument("--conc", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--timeout", type=int, default=1800)
args = ap.parse_args()

rows = [json.loads(line) for line in open(args.prompts)]
done = set()
try:
    for line in open(args.out):
        d = json.loads(line)
        if "error" not in d:
            done.add(d["id"])
except FileNotFoundError:
    pass
todo = [r for r in rows if r["id"] not in done]
if args.limit:
    todo = todo[: args.limit]
print(f"{len(todo)} prompts to run ({len(done)} done), conc {args.conc}, max_tokens {args.max_tokens}", flush=True)


def one(r):
    msgs = r.get("messages") or [{"role": "user", "content": r["prompt"]}]
    body = json.dumps({"model": args.model, "messages": msgs, "max_tokens": args.max_tokens}).encode()
    req = urllib.request.Request(f"{args.url}/chat/completions", body,
                                 {"Content-Type": "application/json", "X-Request-Id": f"gen{r['id']}"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=args.timeout))
        ch = d["choices"][0]; msg = ch["message"]
        return {"id": r["id"], "source": r["source"], "category": r.get("category"), "resp_id": d.get("id"), "usage": d.get("usage"),
                "finish": ch.get("finish_reason"), "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                "content": msg.get("content"), "secs": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"id": r["id"], "source": r["source"], "error": repr(e)}


t0 = time.time(); n = 0; toks = 0; errs = 0
with open(args.out, "a") as f, cf.ThreadPoolExecutor(args.conc) as ex:
    for res in ex.map(one, todo):
        f.write(json.dumps(res) + "\n"); f.flush(); n += 1
        if "error" in res:
            errs += 1
            print("ERROR", res["id"], res["error"][:300], flush=True)
            if errs >= 3:
                print("3 errors: stopping", flush=True)
                ex.shutdown(wait=False, cancel_futures=True)
                break
        toks += (res.get("usage") or {}).get("completion_tokens", 0)
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"{n}/{len(todo)} done, {toks} completion tokens, {toks / el:.0f} tok/s, {el / 60:.1f} min", flush=True)
print("DONE", n, "responses,", toks, "completion tokens,", errs, "errors", flush=True)
