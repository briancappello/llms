"""Read a live process's exact argv and executable, per platform.

Used only to fail closed: callers treat any OSError as "cannot verify", never
as a match. Argument boundaries must be exact, so ps(1)-style joined command
lines are deliberately not used.

Linux reads /proc. macOS asks the kernel directly through ctypes:
sysctl(KERN_PROCARGS2) for argv (the buffer ps(1) itself parses) and
libproc's proc_pidpath for the executable. No third-party dependency.
"""

import os
from pathlib import Path
import struct
import sys

# <sys/sysctl.h>
CTL_KERN = 1
KERN_ARGMAX = 8
KERN_PROCARGS2 = 49
# <sys/proc_info.h>
PROC_PIDPATHINFO_MAXSIZE = 4096


def _linux_argv(pid):
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    if not raw.endswith(b"\0"):
        raise OSError(f"unterminated /proc/{pid}/cmdline")
    return [os.fsdecode(part) for part in raw[:-1].split(b"\0")]


def _linux_exe(pid):
    return (Path("/proc") / str(pid) / "exe").resolve(strict=True)


def parse_procargs2(buffer):
    """Split a KERN_PROCARGS2 buffer into (exec_path, argv).

    Layout: native int argc, the NUL-terminated exec path, NUL padding, then
    argc NUL-terminated argument strings, then the environment (ignored).
    Anything that does not fit that layout raises OSError.
    """
    size = struct.calcsize("i")
    if len(buffer) < size:
        raise OSError("truncated KERN_PROCARGS2 buffer (no argc)")
    (argc,) = struct.unpack_from("i", buffer, 0)
    if argc < 1:
        raise OSError(f"implausible argc {argc} in KERN_PROCARGS2 buffer")
    end = buffer.find(b"\0", size)
    if end <= size:
        raise OSError("KERN_PROCARGS2 buffer has no executable path")
    exec_path = os.fsdecode(buffer[size:end])
    position = end
    while position < len(buffer) and buffer[position] == 0:
        position += 1
    argv = []
    for _ in range(argc):
        end = buffer.find(b"\0", position)
        if end < 0:
            raise OSError("truncated KERN_PROCARGS2 argument list")
        argv.append(os.fsdecode(buffer[position:end]))
        position = end + 1
    return exec_path, argv


def _libc():
    import ctypes
    import ctypes.util

    return ctypes, ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)


def _darwin_sysctl(mib, size):
    ctypes, libc = _libc()
    array = (ctypes.c_int * len(mib))(*mib)
    length = ctypes.c_size_t(size)
    buffer = ctypes.create_string_buffer(size)
    if libc.sysctl(array, len(mib), buffer, ctypes.byref(length), None, ctypes.c_size_t(0)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"sysctl {mib}: {os.strerror(error)}")
    return buffer.raw[:length.value]


def _darwin_procargs(pid):
    ctypes, _ = _libc()
    argmax = struct.unpack("i", _darwin_sysctl([CTL_KERN, KERN_ARGMAX], ctypes.sizeof(ctypes.c_int)))[0]
    return parse_procargs2(_darwin_sysctl([CTL_KERN, KERN_PROCARGS2, int(pid)], argmax))


def _darwin_exe(pid):
    import ctypes
    import ctypes.util

    libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib", use_errno=True)
    buffer = ctypes.create_string_buffer(PROC_PIDPATHINFO_MAXSIZE)
    length = libproc.proc_pidpath(int(pid), buffer, ctypes.c_uint32(PROC_PIDPATHINFO_MAXSIZE))
    if length <= 0:
        error = ctypes.get_errno()
        raise OSError(error, f"proc_pidpath({pid}): {os.strerror(error)}")
    return Path(os.fsdecode(buffer.raw[:length])).resolve(strict=True)


def argv(pid):
    """Exact argument vector of a live process, or OSError."""
    if sys.platform.startswith("linux"):
        return _linux_argv(pid)
    if sys.platform == "darwin":
        return _darwin_procargs(pid)[1]
    raise OSError(f"live process inspection is not supported on {sys.platform}")


def exe(pid):
    """Resolved executable path of a live process, or OSError."""
    if sys.platform.startswith("linux"):
        return _linux_exe(pid)
    if sys.platform == "darwin":
        return _darwin_exe(pid)
    raise OSError(f"live process inspection is not supported on {sys.platform}")
