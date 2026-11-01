"""
exllamav3 decode/prefill benchmark (greedy, via the dynamic Generator).

Takes the standard exllamav3 model_init args (-m, -mcs, -mct, -gs, -cs, -mtp, -ndt, ...) plus:
  -runs N       short-prompt runs (cycles 4 prompts), -ntok N tokens each
  -long A B ..  long-prompt prefill tests at these token counts (text from eval/eval_texts)
  -pin_main C   pin the main thread to CPU C after load
  -spy S        py-spy the process for S seconds during run 1 (-> /tmp/opencode/spy_decode.txt)
  -prof N       torch.profiler over N decode tokens at the end
Prints per-run pp/tg tok/s, MTP draft acceptance and per-step latency percentiles.

Run with exllamav3's venv (REPO env var overrides the checkout path), e.g. on the R9700:
  export CXX=/opt/rocm/llvm/bin/clang++ HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 \
         TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions_exl3dev EXL3_MOE_PINNED_ARENA=1
  ~/dev/exllamav3/.venv/bin/python bench_exl3.py -m ~/models-nvme/Qwen3.8-Flash-Next-exl3-3.05bpw \
      -mcs 300 -gs 30 -cs 16384 -mtp -ndt 4 -runs 6 -ntok 300 -long 4000 8000 14000
"""
import sys, os, time, argparse
REPO = os.environ.get("REPO", os.path.expanduser("~/dev/exllamav3"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "examples"))

import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import GreedySampler

def main():
    global generator, tokenizer
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, cache = True, add_sampling_args = False, add_draft_model_args = True,
                        default_cache_size = 16384, default_autosplit_max_batch_size = 1)
    parser.add_argument("-runs", type = int, default = 3)
    parser.add_argument("-ntok", type = int, default = 256)
    parser.add_argument("-prof", type = int, default = 0, help = "profile N decode tokens")
    parser.add_argument("-spy", type = int, default = 0, help = "py-spy N seconds during run 1")
    parser.add_argument("-pin_main", type = int, default = -1)
    parser.add_argument("-long", type = int, nargs = "*", default = [])
    args = parser.parse_args()

    t0 = time.time()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    print(f"LOAD {time.time() - t0:.1f}s", flush = True)
    if args.pin_main >= 0:
        os.sched_setaffinity(0, {args.pin_main})
        print(f"main thread pinned to cpu {args.pin_main}", flush = True)

    generator = Generator(model = model, cache = cache, tokenizer = tokenizer,
                          draft_model = draft_model, draft_cache = draft_cache,
                          num_draft_tokens = getattr(args, "num_draft_tokens", 4))

    prompts = [
        "Write a short story (about 400 words) about a lighthouse keeper who finds a message in a bottle.",
        "Explain how a CPU branch predictor works, in detail, for a computer science student.",
        "Write a Python function that parses an INI file without using configparser, with tests.",
        "Describe the history of the Silk Road and its economic significance.",
    ]

    def fmt(p):
        return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def run(prompt, ntok):
        ids = tokenizer.encode(fmt(prompt), encode_special_tokens = True)
        job = Job(input_ids = ids, max_new_tokens = ntok, sampler = GreedySampler(), stop_conditions = [])
        generator.enqueue(job)
        r = None
        text = ""
        stamps = []
        while generator.num_remaining_jobs():
            for r_ in generator.iterate():
                text += r_.get("text", "")
                if r_.get("token_ids") is not None:
                    stamps.append((time.perf_counter(), r_["token_ids"].shape[-1]))
                if r_.get("eos"): r = r_
        if len(stamps) > 8:
            d = sorted((b[0] - a[0]) * 1000 for a, b in zip(stamps[1:], stamps[2:]))
            big = [round(x) for x in d if x > 60]
            q = lambda f: d[int(f * (len(d) - 1))]
            print(f"   step ms: p10 {q(0.1):.1f} p50 {q(0.5):.1f} p90 {q(0.9):.1f} max {d[-1]:.1f} | >60ms: {len(big)} steps, sum {sum(big)} ms", flush = True)
        pp = (r["prompt_tokens"] - r["cached_tokens"]) / r["time_prefill"]
        tg = r["new_tokens"] / r["time_generate"]
        extra = ""
        if "accepted_draft_tokens" in r:
            a, j = r["accepted_draft_tokens"], r["rejected_draft_tokens"]
            extra = f" draft {a}/{a+j}"
        return pp, tg, r["new_tokens"], text, extra

    for i in range(args.runs):
        spy = None
        if args.spy and i == 1:
            import ctypes, subprocess
            ctypes.CDLL(None).prctl(0x59616d61, ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF), 0, 0, 0)
            spy = subprocess.Popen([os.path.join(REPO, ".venv/bin/py-spy"), "record", "-p", str(os.getpid()),
                                    "-d", str(args.spy), "-r", "47", "-f", "raw", "--native",
                                    "-o", "/tmp/opencode/spy_decode.txt"])
            time.sleep(2)
        print(f"MARK_START {i} {time.monotonic_ns()}", flush = True)
        pp, tg, n, text, extra = run(prompts[i % len(prompts)], args.ntok)
        print(f"MARK_END {i} {time.monotonic_ns()}", flush = True)
        if spy: spy.wait()
        print(f"RUN {i}: pp {pp:.1f} t/s  tg {tg:.2f} t/s ({n} tok){extra}", flush = True)
        if i == 0: print("   sample:", text[:200].replace("\n", " "), flush = True)

    if args.long:
        text = open(os.path.join(REPO, "eval/eval_texts/variable_man_mod.txt")).read()
        all_ids = tokenizer.encode(text)
        for n in args.long:
            chunk = tokenizer.decode(all_ids[:, :n])[0] if all_ids.dim() == 2 else tokenizer.decode(all_ids[:n])
            p = "Summarize the following story excerpt in five bullet points.\n\n" + chunk
            pp, tg, nn, _, extra = run(p, 128)
            print(f"LONG {n}: pp {pp:.1f} t/s  tg {tg:.2f} t/s ({nn} tok){extra}", flush = True)

    if args.prof:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run(prompts[1], args.prof)
        print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit = 30, max_name_column_width = 70))
        print(prof.key_averages().table(sort_by = "cpu_time_total", row_limit = 30, max_name_column_width = 70))

if __name__ == '__main__':
    main()
