"""Memory used by a served inference process, per platform (design D10).

  Linux  device VRAM in use from rocm-smi or nvidia-smi (how the existing
         results were taken: device-level, not per-process).
  macOS  the process's physical footprint (proc_pid_rusage ri_phys_footprint),
         which counts Metal allocations on unified memory, against the Metal
         recommendedMaxWorkingSetSize budget.

Every reader returns None when it cannot measure. Callers record that as
unavailable ("NA"), never as zero, and never fail the run because of it.
"""

import ctypes
import ctypes.util
import re
import shutil
import subprocess
import sys
import threading

NA = "NA"
RUSAGE_INFO_V4 = 4
# struct rusage_info_v4: uuid[16] then u64 fields; ri_phys_footprint is the
# 8th u64 (user_time, system_time, pkg_idle_wkups, interrupt_wkups, pageins,
# wired_size, resident_size, phys_footprint).
PHYS_FOOTPRINT_OFFSET = 16 + 7 * 8
RUSAGE_INFO_V4_SIZE = 512  # generous upper bound on the struct size


def _run(argv, timeout=10):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def child_pids(pid):
    """Direct children of pid (pgrep -P works on Linux and macOS)."""
    result = _run(["pgrep", "-P", str(pid)])
    if result is None or result.returncode not in (0, 1):
        return []
    return [int(line) for line in result.stdout.split() if line.isdecimal()]


def pid_listening(port):
    """PID listening on a TCP port: lsof on both systems, ss as a Linux fallback."""
    if shutil.which("lsof"):
        result = _run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"])
        if result and result.stdout.split():
            return int(result.stdout.split()[0])
    if shutil.which("ss"):
        result = _run(["ss", "-ltnpH", f"sport = :{port}"])
        match = re.search(r"pid=(\d+)", result.stdout if result else "")
        if match:
            return int(match.group(1))
    return None


def footprint_mib(pid):
    """macOS physical footprint of pid in MiB, or None."""
    if sys.platform != "darwin" or not pid:
        return None
    try:
        libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib", use_errno=True)
        buffer = ctypes.create_string_buffer(RUSAGE_INFO_V4_SIZE)
        if libproc.proc_pid_rusage(int(pid), RUSAGE_INFO_V4, buffer) != 0:
            return None
        footprint = int.from_bytes(buffer.raw[PHYS_FOOTPRINT_OFFSET:PHYS_FOOTPRINT_OFFSET + 8], sys.byteorder)
        return round(footprint / 2**20)
    except OSError:
        return None


def parse_rocm_smi(text):
    match = re.search(r"GPU\[0\].*?Used.*?:\s*(\d+)", text or "")
    return round(int(match.group(1)) / 2**20) if match else None


def parse_nvidia_smi(text):
    first = (text or "").strip().splitlines()[:1]
    return int(first[0].strip()) if first and first[0].strip().isdecimal() else None


def vram_used_mib():
    """Linux device VRAM in use (GPU 0), or None."""
    if not sys.platform.startswith("linux"):
        return None
    if shutil.which("rocm-smi"):
        result = _run(["rocm-smi", "--showmeminfo", "vram"])
        value = parse_rocm_smi(result.stdout if result else "")
        if value is not None:
            return value
    if shutil.which("nvidia-smi"):
        result = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return parse_nvidia_smi(result.stdout if result else "")
    return None


METAL_LOG = re.compile(r"recommendedMaxWorkingSetSize\s*=\s*([\d.]+)\s*MB")
METAL_DEVICES = re.compile(r"MTL\d+:.*?\((\d+) MiB")


def parse_metal_budget(text):
    """Metal budget in MiB from llama.cpp's load log or --list-devices output."""
    match = METAL_LOG.search(text or "")
    if match:
        return round(float(match.group(1)) * 1e6 / 2**20)
    match = METAL_DEVICES.search(text or "")
    return int(match.group(1)) if match else None


def metal_budget_mib(log_text=None, llama_server=None):
    if sys.platform != "darwin":
        return None
    budget = parse_metal_budget(log_text)
    if budget is None and llama_server:
        result = _run([llama_server, "--list-devices"], timeout=30)
        budget = parse_metal_budget((result.stdout + result.stderr) if result else "")
    return budget


def measure(pids):
    """One reading for the served process(es): (mem_mib | None, kind)."""
    if sys.platform == "darwin":
        values = [footprint_mib(pid) for pid in pids]
        values = [v for v in values if v is not None]
        return (sum(values) if values else None), "footprint"
    return vram_used_mib(), "vram"


def cell(value):
    return NA if value is None else value


class PeakSampler:
    """1 Hz background sampler; peak is the interesting number during prefill."""

    def __init__(self, pids_fn, interval=1.0):
        self.pids_fn, self.interval = pids_fn, interval
        self.peak = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            value, _ = measure(self.pids_fn())
            if value is not None:
                self.peak = value if self.peak is None else max(self.peak, value)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        return False
