#!/usr/bin/env python3
"""Build a deterministic, ordered concatenation of llama.cpp source files
suitable for use as a long-context coding prompt corpus.

Usage:  build_code_corpus.py [target_bytes]
Env:    LLAMA_CPP_SRC (default /opt/src/llama.cpp) — repo to walk.
"""
import os, sys, pathlib

ROOT = os.environ.get("LLAMA_CPP_SRC", "/opt/src/llama.cpp")
SKIP_DIRS = {"build", ".git", "vendor", "tests", "ci", "docs", "scripts", "examples", "models", ".github", "cmake", "common/curl"}
EXTS = {".cpp", ".h", ".hpp", ".c", ".py", ".cu", ".cuh", ".metal", ".comp"}

# Walk source roots in a stable order. Prioritize the core engine
# (ggml/src, src, common, tools/server) for relevance to the prompt question.
PRIORITY = [
    "ggml/src",
    "src",
    "common",
    "tools/server",
    "tools/mtmd",
    "tools",
]

def collect(root):
    paths = []
    base = pathlib.Path(root)
    for pri in PRIORITY:
        d = base / pri
        if not d.exists(): continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in EXTS:
                rel = p.relative_to(base)
                parts = set(rel.parts)
                if parts & SKIP_DIRS: continue
                paths.append(p)
    return paths

paths = collect(ROOT)
out = sys.stdout
total_bytes = 0
target_bytes = int(sys.argv[1]) if len(sys.argv) > 1 else 800_000

for p in paths:
    rel = p.relative_to(ROOT)
    header = f"// ===== {rel} =====\n"
    try:
        body = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    chunk = header + body + "\n\n"
    out.write(chunk)
    total_bytes += len(chunk.encode("utf-8", errors="replace"))
    if total_bytes >= target_bytes:
        break
