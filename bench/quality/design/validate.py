#!/usr/bin/env python3
"""Validate a design task: reference passes all hidden tests, every mutant fails one, bench is in range.

    python3 validate.py <task_id> [<task_id> ...]
"""
import importlib.util, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTEST = ["uv", "run", "-q", "--with", "pytest", "--with", "pytest-timeout", "python", "-m", "pytest", "-q",
          "-p", "no:cacheprovider", "--timeout=30"]


def run_tests(task: Path, module: Path) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(module, Path(d) / "solution.py")
        shutil.copy(task / "test_hidden.py", Path(d) / "test_hidden.py")
        p = subprocess.run(PYTEST + ["test_hidden.py"], cwd=d, capture_output=True, text=True, timeout=900)
        return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1] if (p.stdout + p.stderr).strip() else ""


def bench_time(task: Path, module: Path) -> float:
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    sys.modules.pop("solution", None)
    mod = load("solution", module)
    return load("bench_" + task.name, task / "bench.py").bench(mod)


def main() -> int:
    bad = 0
    for tid in sys.argv[1:]:
        task = HERE / "tasks" / tid
        rc, tail = run_tests(task, task / "reference.py")
        print(f"[{tid}] reference: {'PASS' if rc == 0 else 'FAIL'}  ({tail})")
        bad += rc != 0
        for m in sorted((task / "mutants").glob("*.py")):
            rc, tail = run_tests(task, m)
            print(f"[{tid}]   mutant {m.name}: {'caught' if rc != 0 else 'NOT CAUGHT'}  ({tail})")
            bad += rc == 0
        t = bench_time(task, task / "reference.py")
        ok = 0.2 <= t <= 1.5
        print(f"[{tid}] bench(reference) = {t:.3f}s {'ok' if ok else 'OUT OF RANGE (want 0.2-1.5s)'}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
