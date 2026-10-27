"""bench/lib/stack.py isolation guarantees, against a stub llama-swap."""

import io
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
import urllib.request
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
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path.startswith("/api/models/unload"): return self.reply(None)
        if self.path.endswith("/tokenize"): return self.reply({"tokens": [0] * (len(json.loads(body)["content"]) // 4)})
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


class BaseDigestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prod = Path(self.temp.name)
        self.header = {"macros": {"common": "--jinja"}}
        self.registry = {"m": {"path": "/models/m.gguf", "capability": "chat", "ctx": 4096, "engine": "e"},
                         "other": {"path": "/models/o.gguf", "capability": "chat", "ctx": 8192}}
        self.settings_data = {"service_manager": "systemd",
                              "engines": {"e": {"server": "/opt/e/llama-server", "args": ["-ngl", "999"]},
                                          "f": {"server": "/opt/f/llama-server"}}}
        self.write()

    def write(self, indent=None, sort_keys=False):
        import yaml
        (self.prod / "config.header.yaml").write_text(yaml.safe_dump(self.header))
        (self.prod / "registry.json").write_text(json.dumps(self.registry))
        (self.prod / "settings.json").write_text(json.dumps(self.settings_data, indent=indent, sort_keys=sort_keys))

    def digest(self, variant=None):
        from llms import Settings
        from stack import base_digest
        settings = Settings.from_env(self.prod, environ={"HOME": str(self.prod)})
        return base_digest(settings, json.loads((self.prod / "registry.json").read_text()), "m", variant or Variant())

    def test_changes_with_entry_engine_and_macros(self):
        before = self.digest()
        for change in (lambda: self.registry["m"].update(ctx=8192),
                       lambda: self.settings_data["engines"]["e"].update(env={"CUDA_VISIBLE_DEVICES": "1"}),
                       lambda: self.header["macros"].update(common="--jinja -fa on")):
            with self.subTest(change=change):
                self.setUp()
                change()
                self.write()
                self.assertNotEqual(self.digest(), before)

    def test_ignores_formatting_key_order_and_unrelated_models(self):
        before = self.digest()
        self.write(indent=4, sort_keys=True)
        self.assertEqual(self.digest(), before)
        self.registry["other"]["ctx"] = 1
        self.write()
        self.assertEqual(self.digest(), before)

    def test_covers_engines_the_patch_names(self):
        moved = Variant("to-f", entries={"m": {"engine": "f"}})
        patched = Variant("f-args", engines={"f": {"args": ["-ngl", "1"]}})
        before = {v.name: self.digest(v) for v in (moved, patched)}
        self.settings_data["engines"]["f"]["server"] = "/opt/f2/llama-server"
        self.write()
        for variant in (moved, patched):
            self.assertNotEqual(self.digest(variant), before[variant.name])
        self.assertEqual(self.digest(), self.digest(Variant()))


class TSVTests(unittest.TestCase):
    def test_driver_columns_overlapping_provenance_are_written_once(self):
        from common import PROVENANCE, TSV
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.tsv"
            out = TSV(path, ["model", "variant", "depth", *PROVENANCE])
            out.row(model="m", variant="v", depth=1)
            header, row = (line.split("\t") for line in path.read_text().splitlines())
        self.assertEqual(header, ["model", "variant", "depth", *PROVENANCE[1:]])
        self.assertIn("base_digest", PROVENANCE)
        self.assertEqual(row[:3], ["m", "v", "1"])


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
            from llms import Settings
            from stack import base_digest
            production = Settings.from_env(self.prod, environ=self.env)
            expected = base_digest(production, json.loads((self.prod / "registry.json").read_text()), "m", variant)
            self.assertEqual(record["base_digest"], expected)
            isolated = base_digest(stack.settings, stack.registry, "m", variant)
            self.assertNotEqual(isolated, expected)  # the copy carries the patch; the record must not
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

    def run_kv_probe(self, *args):
        corpus = Path(self.temp.name) / "corpus.txt"
        corpus.write_text("int main(void) { return 0; }\n" * 400)
        out = Path(self.temp.name) / "kv.json"
        script = Path(__file__).resolve().parents[1] / "bench" / "speed" / "kv-probe.py"
        result = subprocess.run([sys.executable, str(script), "--model", "m", "--tag", "t", "--depths", "256",
                                 "--reps", "1", "--gen", "4", "--corpus", str(corpus), "--out", str(out), *args],
                                env={**self.env, "STUB_CHILD_FILE": str(self.child_file)},
                                capture_output=True, text=True, timeout=120)
        return result, (json.loads(out.read_text()) if out.exists() else None)

    def test_kv_probe_records_stack_mode_and_base_digest(self):
        result, data = self.run_kv_probe("--config-dir", str(self.prod), "--no-manage-production",
                                         "--port", str(free_port()))
        self.assertEqual(result.returncode, 0, result.stderr)
        from llms import Settings
        from stack import base_digest
        production = Settings.from_env(self.prod, environ=self.env)
        expected = base_digest(production, json.loads((self.prod / "registry.json").read_text()), "m", Variant())
        self.assertEqual(data["provenance"]["mode"], "stack")
        self.assertEqual(data["model"], "m")
        self.assertEqual(data["provenance"]["base_digest"], expected)
        self.assertEqual(data["summary"]["256"]["n"], 1)

    def load_kv_probe(self):
        import importlib.util
        script = Path(__file__).resolve().parents[1] / "bench" / "speed" / "kv-probe.py"
        spec = importlib.util.spec_from_file_location("kv_probe", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_kv_probe_standalone_mode_never_builds_a_stack(self):
        port = free_port()
        server = subprocess.Popen([str(self.stub), "-listen", f"127.0.0.1:{port}"],
                                  env={**self.env, "STUB_CHILD_FILE": str(self.child_file)})
        self.addCleanup(server.wait)
        self.addCleanup(server.kill)
        self.addCleanup(lambda: self.child_gone() or os.kill(int(self.child_file.read_text()), signal.SIGKILL))
        url = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        corpus = Path(self.temp.name) / "corpus.txt"
        corpus.write_text("int main(void) { return 0; }\n" * 400)
        out = Path(self.temp.name) / "standalone.json"
        module = self.load_kv_probe()
        argv = ["kv-probe.py", "--model", "probe", "--tag", "t", "--server", url, "--depths", "256",
                "--reps", "1", "--gen", "4", "--corpus", str(corpus), "--out", str(out)]
        with unittest.mock.patch.object(module, "stack_for", side_effect=AssertionError("stack built")), \
                unittest.mock.patch.object(sys, "argv", argv), \
                unittest.mock.patch("sys.stdout", new_callable=io.StringIO):
            module.main()
        data = json.loads(out.read_text())
        self.assertEqual(data["provenance"], {"mode": "standalone", "server": url})
        self.assertEqual(data["summary"]["256"]["n"], 1)
        self.assertEqual({p: p.read_bytes() for p in self.prod.iterdir()}, self.files)

    def test_kv_probe_server_with_variant_is_a_usage_error(self):
        module = self.load_kv_probe()
        argv = ["kv-probe.py", "--model", "m", "--tag", "t", "--server", "http://127.0.0.1:1",
                "--variant", 'x={"entries": {"m": {"ctx": 1}}}']
        stderr = io.StringIO()
        with unittest.mock.patch.object(module, "stack_for", side_effect=AssertionError("stack built")), \
                unittest.mock.patch.object(sys, "argv", argv), \
                unittest.mock.patch("sys.stderr", stderr), self.assertRaises(SystemExit) as raised:
            module.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--server measures a server outside the stack; --variant does not apply", stderr.getvalue())

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
