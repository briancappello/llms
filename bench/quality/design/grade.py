#!/usr/bin/env python3
"""Grade generated design-task solutions.

    grade.py [--results results/design] [--labels strata,llama] [--jobs 8]

For every results/design/<label>/<task>.s<N>.json: extract solution.py / test_solution.py, then measure
  hidden tests (pass counts per category: basic/edge/prop/conc/stress), the model's own tests run against its own
  code, perf (bench.py time relative to the reference, median of 3), ruff findings, mypy errors, size.
Writes <label>/<task>.s<N>.grade.json and prints a per-label/per-task summary. Candidate code runs in a temp dir
with timeouts (it is model-generated code: run on a machine you are fine running it on).
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, threading, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CATS = ("basic", "edge", "prop", "conc", "stress")
PY = os.environ.get("GRADE_PY", sys.executable)  # a venv with pytest, pytest-timeout, ruff, mypy
BENCH_RUNNER = r"""
import importlib.util, json, statistics, sys
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
sol = load("solution", sys.argv[1]); b = load("task_bench", sys.argv[2])
ts = [b.bench(sol) for _ in range(3)]
print(json.dumps(statistics.median(ts)))
"""


def extract(content: str) -> tuple[str | None, str | None]:
    blocks = list(re.finditer(r"```(?:python|py)?[^\n]*\n(.*?)```", content, re.S))
    sol = tests = None
    for m in blocks:
        head = content[max(0, m.start() - 200):m.start()].lower()
        name = re.findall(r"(test_solution\.py|solution\.py)", head)
        if name and name[-1] == "test_solution.py":
            tests = tests or m.group(1)
        elif name and name[-1] == "solution.py":
            sol = sol or m.group(1)
    if sol is None:  # fallback: biggest non-test block
        cands = [m.group(1) for m in blocks if "def test_" not in m.group(1)]
        sol = max(cands, key=len) if cands else None
    if tests is None:
        cands = [m.group(1) for m in blocks if "def test_" in m.group(1)]
        tests = max(cands, key=len) if cands else None
    return sol, tests


def pytest(d: Path, test_file: str, timeout_s: int) -> dict:
    xml = d / f"{test_file}.xml"
    try:
        p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--timeout=30", f"--junitxml={xml}",
                            test_file], cwd=d, capture_output=True, text=True, timeout=timeout_s)
        tail = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
    except subprocess.TimeoutExpired:
        return {"error": "suite timeout"}
    if not xml.exists():
        return {"error": "no report: " + tail[0][:200]}
    res = {"passed": 0, "failed": 0, "by_cat": {c: [0, 0] for c in CATS}, "failures": [], "tail": tail[0][:200]}
    for tc in ET.parse(xml).getroot().iter("testcase"):
        name = tc.get("name", "")
        bad = tc.find("failure") is not None or tc.find("error") is not None
        skipped = tc.find("skipped") is not None
        if skipped:
            continue
        res["failed" if bad else "passed"] += 1
        cat = next((c for c in CATS if name.startswith(f"test_{c}")), None)
        if cat:
            res["by_cat"][cat][1] += 1
            res["by_cat"][cat][0] += not bad
        if bad:
            res["failures"].append(name)
    return res


def bench(sol: Path, task: Path) -> float | str:
    try:
        p = subprocess.run([PY, "-c", BENCH_RUNNER, str(sol), str(task / "bench.py")], capture_output=True, text=True,
                           timeout=180, cwd=sol.parent)
    except subprocess.TimeoutExpired:
        return "timeout"
    if p.returncode:
        return "error: " + ((p.stderr.strip().splitlines() or [""])[-1])[:200]
    return json.loads(p.stdout.strip().splitlines()[-1])


def lint(sol: Path) -> dict:
    r = subprocess.run([PY, "-m", "ruff", "check", "--no-cache", "--output-format=json", "--select", "E,F,W,B,SIM,UP,PL",
                        "--ignore", "E501,PLR2004,PLR0913,PLR0912,PLR0915,PLR0911", str(sol)],
                       capture_output=True, text=True)
    try:
        findings = json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        findings = []
    serious = [f for f in findings if f["code"][0] in "FB" or f["code"].startswith("PLE")]
    m = subprocess.run([PY, "-m", "mypy", "--ignore-missing-imports", "--no-error-summary", "--no-incremental",
                        "--cache-dir=/dev/null", str(sol)], capture_output=True, text=True, cwd=sol.parent)
    mypy_errors = sum(1 for line in m.stdout.splitlines() if ": error:" in line)
    return {"ruff": len(findings), "ruff_serious": len(serious), "ruff_codes": sorted({f["code"] for f in findings}),
            "mypy_errors": mypy_errors}


_ref_cache: dict[str, float] = {}
_bench_lock = threading.Lock()  # one timing at a time: parallel grading must not skew perf


def ref_time(task: Path) -> float:
    if task.name not in _ref_cache:
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(task / "reference.py", Path(d) / "solution.py")
            _ref_cache[task.name] = bench(Path(d) / "solution.py", task)
    return _ref_cache[task.name]


def grade_one(path: Path) -> dict:
    rec = json.loads(path.read_text())
    task = HERE / "tasks" / rec["task"]
    sol, tests = extract(rec["content"])
    g = {"label": rec["label"], "task": rec["task"], "sample": rec["sample"], "finish_reason": rec["finish_reason"],
         "completion_tokens": (rec.get("usage") or {}).get("completion_tokens"),
         "reasoning_chars": len(rec.get("reasoning") or ""), "wall_s": rec["wall_s"],
         "has_solution": sol is not None, "has_tests": tests is not None}
    if sol is None:
        return g
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "solution.py").write_text(sol)
        g["loc"] = sum(1 for line in sol.splitlines() if line.strip() and not line.strip().startswith("#"))
        shutil.copy(task / "test_hidden.py", d / "test_hidden.py")
        g["hidden"] = pytest(d, "test_hidden.py", 4000)
        if tests:
            (d / "test_solution.py").write_text(tests)
            own = pytest(d, "test_solution.py", 600)
            g["own_tests"] = {k: own.get(k) for k in ("passed", "failed", "error")}
        with _bench_lock:
            t = bench(d / "solution.py", task)
        ref = ref_time(task)
        g["bench_s"] = t
        g["perf_ratio"] = round(t / ref, 2) if isinstance(t, float) and isinstance(ref, float) else None
        g.update(lint(d / "solution.py"))
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "design"))
    ap.add_argument("--labels", default="")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.results)
    labels = args.labels.split(",") if args.labels else sorted(p.name for p in root.iterdir() if p.is_dir())
    todo = [p for lab in labels for p in sorted((root / lab).glob("*.s*.json")) if not p.name.endswith(".grade.json")
            and (args.force or not p.with_suffix(".grade.json").exists())]
    for t in sorted({json.loads(p.read_text())["task"] for p in todo}):
        ref_time(HERE / "tasks" / t)  # serial, before the parallel load, so the baseline is not disturbed

    def work(p: Path):
        g = grade_one(p)
        p.with_suffix(".grade.json").write_text(json.dumps(g, indent=1))
        h = g.get("hidden") or {}
        print(f"graded {p.parent.name}/{p.name}: hidden {h.get('passed')}/{(h.get('passed') or 0) + (h.get('failed') or 0)}"
              f" perf {g.get('perf_ratio')} ruff {g.get('ruff')}", flush=True)

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(work, todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
