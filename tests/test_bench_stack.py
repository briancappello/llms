"""bench/lib/stack.py isolation guarantees, against a stub llama-swap."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "lib"))

import mem  # noqa: E402
from stack import Stack, StackError, Variant, merge_patch  # noqa: E402

STUB = r'''#!/usr/bin/env python3
"""Minimal llama-swap stand-in: /running, unload, chat; spawns one child."""
import http.server, json, subprocess, sys
listen = sys.argv[sys.argv.index("-listen") + 1]
host, port = listen.rsplit(":", 1)
child = subprocess.Popen(["sleep", "300"])
import os; open(os.environ["STUB_CHILD_FILE"], "w").write(str(child.pid))
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, obj):
        body = json.dumps(obj).encode() if obj is not None else b""
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self): self.reply({"running": []})
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path.startswith("/api/models/unload"): return self.reply(None)
        self.reply({"choices": [{"message": {"role": "assistant", "content": "ok"}}], "timings": {"prompt_per_second": 10.0, "predicted_per_second": 5.0, "draft_n": 4, "draft_n_accepted": 3}, "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
http.server.HTTPServer((host, int(port)), H).serve_forever()
'''


def free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MergePatchTests(unittest.TestCase):
    def test_rfc7386(self):
        self.assertEqual(merge_patch({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": None, "e": 4}, "f": [1]}),
                         {"a": 1, "b": {"d": 3, "e": 4}, "f": [1]})
        self.assertEqual(merge_patch({"a": [1, 2]}, {"a": [3]}), {"a": [3]})

    def test_variant_parse(self):
        self.assertEqual(Variant.parse("production").entries, {})
        v = Variant.parse('n1={"engines": {"metal": {"mtp_args": ["--spec-type", "draft-mtp"]}}}')
        self.assertEqual((v.name, v.engines["metal"]["mtp_args"][0]), ("n1", "--spec-type"))
        with self.assertRaises(ValueError):
            Variant.parse('x={"registry": {}}')


class StackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.stub = root / "llama-swap"
        self.stub.write_text(STUB)
        self.stub.chmod(0o755)
        self.prod = root / "prod"
        self.prod.mkdir()
        header = "macros:\n  common: --jinja\n"
        (self.prod / "config.header.yaml").write_text(header)
        (self.prod / "registry.json").write_text(json.dumps({
            "m": {"path": "/models/m.gguf", "capability": "chat", "ctx": 4096, "engine": "e", "mtp": True}}))
        (self.prod / "settings.json").write_text(json.dumps({
            "swap_binary": str(self.stub), "service_manager": "systemd",
            "engines": {"e": {"server": "/opt/e/llama-server", "args": ["-ngl", "999"]}}}))
        (self.prod / "llama-swap.yaml").write_text("rendered: production\n")
        self.files = {p: p.read_bytes() for p in self.prod.iterdir()}
        self.prod_manager = MagicMock()
        self.prod_manager.backend.definition_path.return_value = self.prod / "settings.json"  # exists
        self.prod_manager._verify_service.return_value = True
        self.factory = MagicMock(return_value=self.prod_manager)
        self.env = {"HOME": str(root), "PATH": os.environ["PATH"]}
        self.child_file = root / "stub-child.pid"
        patcher = unittest.mock.patch.dict(os.environ, {"STUB_CHILD_FILE": str(self.child_file)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def stack(self, variant=None):
        return Stack(variant, port=free_port(), production_dir=self.prod, environ=self.env,
                     production_manager=self.factory, log=lambda *_: None)

    def assert_production_intact_and_restored(self):
        self.assertEqual({p: p.read_bytes() for p in self.prod.iterdir()}, self.files)
        calls = [c.args for c in self.prod_manager.service.call_args_list]
        self.assertEqual(calls, [("stop",), ("start",)])
        self.prod_manager.wait_ready.assert_called_once()

    def child_gone(self, _config=None):
        pid = int(self.child_file.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)
        return False

    def test_normal_run_is_isolated_and_restores_production(self):
        variant = Variant("mtp-n1", engines={"e": {"mtp_args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]}})
        with self.stack(variant) as stack:
            config = stack.settings.output
            response = stack.chat({"model": "m", "messages": []})
            self.assertEqual(response["choices"][0]["message"]["content"], "ok")
            record = stack.record("m")
            self.assertEqual(record["variant"], "mtp-n1")
            self.assertIn("--spec-draft-n-max 1", record["rendered_cmd"])
            self.assertEqual(stack.production_state, "active")
            self.assertNotEqual(stack.settings.config_dir, self.prod)
        self.assertTrue(self.child_gone(config))
        self.assert_production_intact_and_restored()

    def test_keyboard_interrupt_still_tears_down_and_restores(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.stack() as stack:
                config = stack.settings.output
                raise KeyboardInterrupt
        self.assertTrue(self.child_gone(config))
        self.assert_production_intact_and_restored()

    def test_sigterm_is_turned_into_teardown(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.stack() as stack:
                config = stack.settings.output
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
        self.assertTrue(self.child_gone(config))
        self.assert_production_intact_and_restored()
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_nohup_ignored_sighup_is_respected(self):
        previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        self.addCleanup(signal.signal, signal.SIGHUP, previous)
        with self.stack():
            self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
            self.assertIsNot(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
        self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)

    def test_unknown_variant_entry_fails_before_touching_production(self):
        with self.assertRaisesRegex(StackError, "unknown registry entry"):
            with self.stack(Variant("bad", entries={"nope": {"ctx": 1}})):
                pass
        self.prod_manager.service.assert_not_called()

    def test_not_installed_production_is_left_alone(self):
        self.prod_manager.backend.definition_path.return_value = self.prod / "missing.service"
        with self.stack() as stack:
            self.assertEqual(stack.production_state, "not-installed")
        self.prod_manager.service.assert_not_called()


class MemTests(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(mem.parse_rocm_smi("GPU[0]\t\t: VRAM Total Used Memory (B): 2147483648\n"), 2048)
        self.assertIsNone(mem.parse_rocm_smi("no gpu"))
        self.assertEqual(mem.parse_nvidia_smi("1234\n"), 1234)
        self.assertIsNone(mem.parse_nvidia_smi("[N/A]"))
        log = "ggml_metal_init: recommendedMaxWorkingSetSize  = 30150.67 MB\n"
        self.assertEqual(mem.parse_metal_budget(log), 28754)
        self.assertEqual(mem.parse_metal_budget("  MTL0: Apple M3 Max (28753 MiB, 28753 MiB free)"), 28753)
        self.assertIsNone(mem.parse_metal_budget(""))

    def test_unavailable_is_na_not_zero(self):
        self.assertEqual(mem.cell(None), "NA")
        self.assertEqual(mem.cell(0), 0)

    @unittest.skipUnless(sys.platform == "darwin", "footprint is macOS-only")
    def test_footprint_of_a_live_process(self):
        child = subprocess.Popen([sys.executable, "-c", "b = bytearray(64 << 20); import time; time.sleep(30)"])
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        time.sleep(1.0)
        value = mem.footprint_mib(child.pid)
        self.assertIsNotNone(value)
        self.assertGreater(value, 60)
        self.assertEqual(mem.measure([child.pid]), (value, "footprint") if value == mem.footprint_mib(child.pid)
                         else mem.measure([child.pid]))

    def test_child_pids(self):
        parent = subprocess.Popen(["/bin/sh", "-c", "sleep 30 & wait"])
        self.addCleanup(parent.wait)
        self.addCleanup(parent.kill)
        for _ in range(50):
            children = mem.child_pids(parent.pid)
            if children:
                break
            time.sleep(0.05)
        self.assertEqual(len(children), 1)
        for pid in children:
            os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
