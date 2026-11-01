#!/usr/bin/env python3
"""
Watch an exllamav3 convert.py job: progress, ETA, memory, warnings, errors.

Usage: watch_conv.py [--budget SECONDS] [--every SECONDS] [--log PATH] [--layers N]
Exit codes: 0 finished, 1 still running (budget used up), 2 died/errored, 3 stalled.
"""
import argparse, os, re, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--log", default="/tmp/opencode/conv_full.log")
ap.add_argument("--match", default="convert.py -i /opt/models/Huihui")
ap.add_argument("--workdir", default=os.path.expanduser("~/exl3-work/ornith-4.5"))
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--budget", type=int, default=1500)
ap.add_argument("--every", type=int, default=300)
ap.add_argument("--stall", type=int, default=1500, help="seconds without log growth while alive = stalled")
a = ap.parse_args()

ERR = re.compile(r"Traceback|Error|error:|out of memory|Aborted|Segmentation|core dumped|Killed", re.I)
WARN = re.compile(r"^\s*!! (?!--max_module)")
LAYER = re.compile(r"-- Quantized: model\.language_model\.layers\.(\d+)\s.*?bpw:\s*([\d.]+).*?sqnr:\s*([\d.]+)\s+\[([\d.]+) s\]\s+rss:\s*([\d.]+) GB")
PHASES = [("Finished compiling model", "finished"), ("Writing ", "compiling"), ("Quantizing ", "side models"),
          ("Quantized: model.language_model.norm", "head/norm"), ("Loading unquantized module", "layers")]

def alive():
    return subprocess.run(["pgrep", "-f", a.match], capture_output=True).returncode == 0

def vram():
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=20).stdout
        used = re.search(r"GPU\[0\].*Used Memory \(B\):\s*(\d+)", out)
        return f"{int(used.group(1)) / 1e9:.1f} GB" if used else "?"
    except Exception:
        return "?"

def ram_avail():
    with open("/proc/meminfo") as f:
        m = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return m["MemAvailable"] / 1024**2

def fmt(s):
    s = int(max(s, 0)); return f"{s // 3600}h{s % 3600 // 60:02d}m"

seen_lines, warned = 0, set()
start = time.time(); last_report = 0; last_size = -1; last_growth = time.time()

def report(lines, tag="status"):
    layers = [m for m in (LAYER.search(l) for l in lines) if m]
    phase = "starting"
    for l in reversed(lines):
        hit = next((p for k, p in PHASES if k in l), None)
        if hit: phase = hit; break
    done = int(layers[-1].group(1)) + 1 if layers else 0
    try:   # a resumed job's log lacks the earlier layers; the work dir has them
        qt = os.listdir(os.path.join(a.workdir, "qtensors"))
        done = max(done, len([f for f in qt if re.match(r"model\.language_model\.layers\.\d+\.safetensors$", f)]))
    except OSError:
        pass
    rate = sum(float(m.group(4)) for m in layers[-6:]) / max(len(layers[-6:]), 1) if layers else 0
    eta_mine = (a.layers - done) * rate + (600 if phase == "layers" else 0)  # + ~10 min for head/side/compile
    eta_log = next((l.split("time:")[-1].strip() for l in reversed(lines) if "Estimated remaining" in l), "?")
    last = layers[-1] if layers else None
    lst = f"L{last.group(1)} bpw {last.group(2)} sqnr {float(last.group(3)):.1f} {float(last.group(4)):.0f}s rss {last.group(5)}GB" if last else "-"
    pct = 100 * done / a.layers
    bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
    print(f"[{time.strftime('%H:%M')}] {tag:7} {phase:11} [{bar}] {done}/{a.layers} ({pct:4.1f}%) | "
          f"eta ~{fmt(eta_mine) if phase == 'layers' else '-'} (log: {eta_log}) | last {lst} | "
          f"RAM avail {ram_avail():.0f}GB VRAM {vram()} | log age {time.time() - last_growth:.0f}s", flush=True)

while True:
    try:
        lines = open(a.log, errors="replace").read().splitlines()
    except FileNotFoundError:
        print(f"log {a.log} not found"); sys.exit(2)
    size = os.path.getsize(a.log)
    if size != last_size:
        last_size, last_growth = size, time.time()

    for l in lines[seen_lines:]:
        if WARN.search(l) and l.strip() not in warned:
            warned.add(l.strip()); print(f"   WARNING: {l.strip()[:200]}", flush=True)
    seen_lines = len(lines)

    if any("Finished compiling model" in l for l in lines):
        report(lines, "DONE")
        print("Conversion finished. Output:", next(l.split(" to ")[-1] for l in lines if "Finished compiling" in l))
        sys.exit(0)

    errs = [l for l in lines if ERR.search(l) and "Estimated" not in l]
    if errs or not alive():
        time.sleep(3)   # let the log flush
        lines = open(a.log, errors="replace").read().splitlines()
        report(lines, "DIED" if not alive() else "ERROR")
        print("---- last 40 log lines ----"); print("\n".join(lines[-40:]))
        print(f"---- resume: MALLOC_MMAP_THRESHOLD_=1048576 HIP_VISIBLE_DEVICES=0 ... convert.py -w {a.workdir} -r")
        sys.exit(2)

    if time.time() - last_growth > a.stall:
        report(lines, "STALLED"); print("\n".join(lines[-15:])); sys.exit(3)

    now = time.time()
    if now - last_report >= a.every or last_report == 0:
        report(lines); last_report = now
    if now - start > a.budget:
        sys.exit(1)
    time.sleep(20)
