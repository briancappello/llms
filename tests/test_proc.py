import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import unittest

from llms import _proc


def procargs(argv, *, exec_path="/bin/sleep", padding=3, env=("HOME=/x", "PATH=/bin")):
    body = exec_path.encode() + b"\0" * (1 + padding)
    body += b"".join(a.encode() + b"\0" for a in argv)
    body += b"".join(e.encode() + b"\0" for e in env)
    return struct.pack("i", len(argv)) + body


class ProcArgsParserTests(unittest.TestCase):
    def test_exact_boundaries_including_spaces_and_empty_arguments(self):
        argv = ["/bin/sleep", "a b", "", "30"]
        self.assertEqual(_proc.parse_procargs2(procargs(argv)), ("/bin/sleep", argv))

    def test_environment_tail_and_padding_are_ignored(self):
        for padding in (0, 1, 7):
            with self.subTest(padding=padding):
                self.assertEqual(_proc.parse_procargs2(procargs(["x"], padding=padding))[1], ["x"])

    def test_malformed_buffers_raise_oserror(self):
        good = procargs(["/bin/sleep", "30"], env=())
        for buffer in (b"", b"\1\0", struct.pack("i", 0) + b"/bin/x\0x\0",
                       struct.pack("i", -3) + b"/bin/x\0x\0", struct.pack("i", 1) + b"\0\0",
                       good[:-3], struct.pack("i", 5) + b"/bin/x\0a\0b\0"):
            with self.subTest(buffer=buffer), self.assertRaises(OSError):
                _proc.parse_procargs2(buffer)

    def test_unsupported_platform_fails_closed(self):
        from unittest.mock import patch

        with patch.object(sys, "platform", "sunos5"):
            for function in (_proc.argv, _proc.exe):
                with self.subTest(function=function.__name__), self.assertRaises(OSError):
                    function(os.getpid())


@unittest.skipUnless(sys.platform in ("darwin", "linux"), "live inspection is Linux/macOS only")
class LiveProcessTests(unittest.TestCase):
    def test_reads_exact_argv_and_executable_of_a_child(self):
        argv = ["/bin/sleep", "30"]
        # sleep(1) rejects odd arguments, so exercise quoting through sh's $0/$@.
        # A compound command keeps the shell from exec'ing into sleep.
        quoted = ["/bin/sh", "-c", "sleep 30; :", "arg with space", ""]
        for command in (argv, quoted):
            with self.subTest(command=command):
                child = subprocess.Popen(command)
                self.addCleanup(child.wait)
                self.addCleanup(child.kill)
                deadline = time.monotonic() + 5
                while True:
                    try:
                        seen = _proc.argv(child.pid)
                        if seen == command:
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        self.fail(f"never observed {command}")
                    time.sleep(0.05)
                self.assertEqual(Path(_proc.exe(child.pid)), Path(command[0]).resolve())
                child.kill()

    def test_missing_process_raises(self):
        child = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
        child.wait()
        with self.assertRaises(OSError):
            _proc.argv(child.pid)
        with self.assertRaises(OSError):
            _proc.exe(child.pid)


if __name__ == "__main__":
    unittest.main()
