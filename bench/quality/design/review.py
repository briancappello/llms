#!/usr/bin/env python3
"""Blind rubric review of design-task solutions by a judge model (default: Claude Fable 5.1 via opencode).

    review.py [--labels llama-mtp,strata] [--passes 2] [--model anthropic/claude-fable-5-1]

Per task, every sample from every label is pooled, given a random anonymous id, shuffled (a different order per
pass), and sent to the judge in one request together with the task spec, so scores are calibrated against each
other. The judge never sees engine labels, timings or hidden-test results. Output:
results/design/_review/<task>.p<N>.json (raw scores, keyed by anon id) and _review/mapping.json (anon id -> sample).
"""
import argparse, json, random, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from grade import extract  # noqa: E402

CRITERIA = {
    "correctness": "Judging by reading: does it implement the spec exactly, incl. edge cases and error semantics? (5 = no bugs found)",
    "robustness": "Input validation, error handling, failure modes, resource safety, no crash paths",
    "concurrency": "Thread-safety design where the spec requires it (lock scope, races, callbacks outside locks); null if the spec has no concurrency requirement",
    "performance": "Algorithmic complexity vs the spec's requirements and practical efficiency",
    "design": "Structure, cohesion, API clarity, appropriate abstractions, no over/under-engineering",
    "readability": "Naming, clarity, comments/docstrings where they help, type hints",
    "tests": "Quality of the candidate's own test_solution.py: coverage of behaviour, edge cases, concurrency, determinism (1 if missing)",
}

INSTRUCTIONS = """You are a principal engineer doing a blind code review for a hiring-calibre evaluation.
Below is a task specification and {n} independent candidate submissions (ids like C-xxxx). Each submission has
solution.py and (maybe) test_solution.py. Review every candidate carefully against the specification. Look for real
bugs (trace tricky cases mentally), spec deviations, races, complexity problems, and engineering quality.

Do NOT use any tools, do not read or write files, do not run code: answer only from the text in the attached files.

Score each candidate 1-5 on each criterion (integers; concurrency may be null if the spec has no concurrency
requirement):
{criteria}
Also give "overall" 1-10 (would you merge this into a production codebase? 10 = merge as is), "bugs": a list of
concrete defects you are confident about (short, specific: what input/scenario fails and why), and "summary"
(one sentence). Be calibrated and consistent across candidates; differences between candidates should reflect
real differences.

Reply with ONLY a JSON object, between the markers, of the form:
<<<JSON
{{"C-xxxx": {{"correctness": 4, "robustness": 3, "concurrency": null, "performance": 5, "design": 4, "readability": 4,
  "tests": 3, "overall": 7, "bugs": ["..."], "summary": "..."}}, ...}}
JSON>>>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "design"))
    ap.add_argument("--labels", default="llama-mtp,strata")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--model", default="anthropic/claude-fable-5-1")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--per-call", type=int, default=0, help="split each task's candidates into judge calls of this size (0 = all in one)")
    ap.add_argument("--jobs", type=int, default=1, help="judge calls in parallel")
    args = ap.parse_args()
    root = Path(args.results)
    out = root / "_review"
    out.mkdir(exist_ok=True)
    mapping_file = out / "mapping.json"
    mapping = json.loads(mapping_file.read_text()) if mapping_file.exists() else {}
    rng = random.Random(20261004)

    tasks = sorted(p.name for p in (HERE / "tasks").iterdir() if (p / "prompt.md").exists())
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks.split(",")]
    all_jobs = []
    for task in tasks:
        cands = []
        for lab in args.labels.split(","):
            for f in sorted((root / lab).glob(f"{task}.s*.json")):
                if f.name.endswith(".grade.json"):
                    continue
                rec = json.loads(f.read_text())
                sol, tests = extract(rec["content"])
                key = f"{lab}/{f.name}"
                anon = next((a for a, k in mapping.items() if k == key), None)
                if anon is None:
                    while (anon := f"C-{rng.randrange(16**4):04x}") in mapping:
                        pass
                    mapping[anon] = key
                cands.append((anon, sol, tests))
        mapping_file.write_text(json.dumps(mapping, indent=1, sort_keys=True))
        jobs = []
        for p in range(1, args.passes + 1):
            order = cands[:]
            random.Random(f"{task}-{p}").shuffle(order)
            size = args.per_call or len(order)
            groups = [order[k:k + size] for k in range(0, len(order), size)]
            for gi, grp in enumerate(groups, 1):
                suffix = "" if len(groups) == 1 else f".g{gi}"
                jobs.append((task, p, suffix, grp))
        all_jobs.extend(jobs)

    def judge(job):
        task, p, suffix, order = job
        dest = out / f"{task}.p{p}{suffix}.json"
        if dest.exists():
            return
        parts = [("00-task-spec.md", f"# TASK SPECIFICATION\n\n{(HERE / 'tasks' / task / 'prompt.md').read_text()}\n")]
        for i, (anon, sol, tests) in enumerate(order, 1):
            text = (f"# CANDIDATE {anon}\n\n## solution.py\n```python\n{sol or '# (no solution.py delivered)'}\n```\n"
                    f"\n## test_solution.py\n```python\n{tests or '# (no tests delivered)'}\n```\n# END OF CANDIDATE {anon}\n")
            lines = text.splitlines(keepends=True)
            chunks = [lines[k:k + 1200] for k in range(0, len(lines), 1200)]
            for j, ch in enumerate(chunks, 1):
                head = "" if j == 1 else f"# CANDIDATE {anon} (continued, part {j} of {len(chunks)})\n"
                parts.append((f"{i:02d}-{anon}-part{j}of{len(chunks)}.md", head + "".join(ch)))
        crit = "\n".join(f'- "{k}": {v}' for k, v in CRITERIA.items())
        msg = INSTRUCTIONS.format(n=len(order), criteria=crit) + (
            f"\nThe material is in {len(parts)} attached files: " + ", ".join(n for n, _ in parts) +
            ". Read every file completely before scoring; every candidate must appear in your JSON.\n")
        scores, missing = {}, {a for a, _, _ in order}
        for attempt in range(1, 4):  # the judge occasionally returns a partial object; ask again
            with tempfile.TemporaryDirectory() as d:
                files = []
                for name, text in parts:
                    (Path(d) / name).write_text(text)
                    files += ["-f", str(Path(d) / name)]
                r = subprocess.run(["opencode", "run", "-m", args.model, "--dir", d, *files, "--", msg],
                                   capture_output=True, text=True, timeout=3600)
            (out / f"{task}.p{p}{suffix}.a{attempt}.raw.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
            m = re.search(r"<<<JSON\s*(\{.*\})\s*JSON>>>", r.stdout, re.S) or re.search(r"(\{\s*\"C-.*\})", r.stdout, re.S)
            try:
                scores = json.loads(m.group(1)) if m else {}
            except json.JSONDecodeError:
                scores = {}
            missing = {a for a, _, _ in order} - set(scores)
            if not missing:
                break
            print(f"[{task} p{p}{suffix}] attempt {attempt}: missing {sorted(missing)}, retrying", flush=True)
        if not scores:
            return
        dest.write_text(json.dumps({"task": task, "pass": p, "order": [a for a, _, _ in order],
                                    "model": args.model, "scores": scores}, indent=1))
        print(f"[{task} p{p}{suffix}] scored {len(scores)} candidates" + (f", MISSING {sorted(missing)}" if missing else ""),
              flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(judge, all_jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
