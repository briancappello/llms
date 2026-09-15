import contextlib
from dataclasses import replace
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
import urllib.error

import yaml

from llama_swap_manager import (
    ManagerError, ModelManager, Settings, collapse_shards, default_model,
    pick_model_file, probe_gguf, render_config, resolve_source, shard_group, suggest_ctx,
)
from llama_swap_manager.cli import main
from llama_swap_manager.manager import atomic_write


def entry(path="/models/example.gguf", **overrides):
    return {"path": str(path), "capability": "chat", "ctx": 1024, "trained_ctx": 2048,
            "parallel": 1, "extra_args": [], **overrides}


class IsolatedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(self.root / "manager")
        self.opener = Mock(side_effect=AssertionError("unexpected network"))
        self.runner = Mock(side_effect=AssertionError("unexpected service operation"))
        self.probe = Mock(return_value={"general.architecture": "llama", "llama.context_length": 1024})
        self.manager = ModelManager(self.settings, probe=self.probe, opener=self.opener, runner=self.runner)

    def registry(self, registry):
        self.manager.init()
        atomic_write(self.settings.registry, json.dumps(registry))

    def weights(self, name="tiny-Q4_K_M.gguf", size=32, folder=None):
        path = (folder or self.root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def cache_weights(self):
        return self.weights(folder=self.settings.hf_cache / "models--owner--repo" / "snapshots" / ("a" * 40))


class SettingsTests(IsolatedTest):
    def test_direct_instances_are_isolated(self):
        with patch.dict(os.environ, {"LLM_SERVER": "wrong", "LLM_CONFIG_DIR": "/wrong"}):
            first, second = Settings(self.root / "one"), Settings(self.root / "two")
        self.assertEqual(first.server, "llama-server")
        self.assertNotEqual(first.registry, second.registry)
        self.assertTrue(first.output.is_relative_to(first.config_dir))
        self.assertFalse(first.config_dir.exists())

    def test_xdg_and_hf_defaults(self):
        env = {"HOME": str(self.root), "XDG_CONFIG_HOME": str(self.root / "xdg"), "HF_HOME": str(self.root / "hf")}
        settings = Settings.from_env(environ=env)
        self.assertEqual(settings.config_dir, self.root / "xdg" / "llama-swap-manager")
        self.assertEqual(settings.output, self.root / "xdg" / "llama-swap" / "config.yaml")
        self.assertEqual(settings.hf_cache, self.root / "hf" / "hub")

    def test_settings_then_env_then_explicit_config_dir(self):
        self.manager.init()
        atomic_write(self.settings.config_dir / "settings.json", json.dumps({"server": "saved", "output": "nested/out.yaml", "gpu_memory_mib": 6000}))
        settings = Settings.from_env(self.settings.config_dir, environ={"HOME": str(self.root), "LLM_SERVER": "override", "LLM_GPU_MEMORY_MIB": "8000", "LLM_CONFIG_DIR": "/unused"})
        self.assertEqual(settings.server, "override")
        self.assertEqual(settings.gpu_memory_mib, 8000)
        self.assertEqual(settings.output, self.settings.config_dir / "nested" / "out.yaml")

    def test_explicit_config_dir_isolates_default_output(self):
        settings = Settings.from_env(self.root / "isolated", environ={"HOME": str(self.root)})
        self.assertEqual(settings.output, self.root / "isolated" / "llama-swap.yaml")

    def test_invalid_settings_and_secret_not_persisted(self):
        for changes in ({"unit": "../bad.service"}, {"swap_url": "file:///etc/passwd"}, {"gpu_memory_mib": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.settings, **changes)
        self.assertNotIn("hf_token", replace(self.settings, hf_token="secret").to_dict())
        self.assertNotIn("secret", repr(replace(self.settings, hf_token="secret")))
        self.manager.init()
        atomic_write(self.settings.config_dir / "settings.json", '{"typo": true}')
        with self.assertRaises(ValueError):
            Settings.from_env(self.settings.config_dir, environ={})

    def test_binary_paths_and_distinct_managed_files(self):
        settings = replace(self.settings, server="bin/llama-server")
        self.assertEqual(settings.server, str(self.settings.config_dir / "bin" / "llama-server"))
        with self.assertRaisesRegex(ValueError, "distinct"):
            replace(self.settings, output=self.settings.registry)
        with self.assertRaisesRegex(ValueError, "distinct"):
            replace(self.settings, registry=self.settings.config_dir / "settings.json")

    def test_client_and_unit_cannot_alias_any_managed_file(self):
        unit = self.settings.service_dir / self.settings.unit
        for target in (self.settings.config_dir / "settings.json", self.settings.registry,
                       self.settings.header, self.settings.output, unit):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "distinct"):
                replace(self.settings, client_path=target)
        for key in ("registry", "header", "output", "client_path"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "distinct"):
                replace(self.settings, **{key: unit})
        self.manager.init()
        unit.parent.mkdir(parents=True)
        unit.symlink_to(self.settings.config_dir / "settings.json")
        with self.assertRaisesRegex(ValueError, "distinct"):
            replace(self.settings)

    def test_preload_defaults_boolean_validation_and_env_precedence(self):
        self.assertFalse(self.settings.preload)
        self.assertFalse(Settings.from_env(self.root / "new", environ={}).preload)
        self.manager.init()
        settings_path = self.settings.config_dir / "settings.json"
        self.assertIs(json.loads(settings_path.read_text())["preload"], False)
        atomic_write(settings_path, '{"preload": true}')
        self.assertTrue(Settings.from_env(self.settings.config_dir, environ={}).preload)
        for value, expected in (("true", True), ("FALSE", False), ("1", True), ("0", False)):
            with self.subTest(value=value):
                actual = Settings.from_env(self.settings.config_dir, environ={"LLM_PRELOAD": value})
                self.assertIs(actual.preload, expected)
        with self.assertRaisesRegex(ValueError, "LLM_PRELOAD"):
            Settings.from_env(self.settings.config_dir, environ={"LLM_PRELOAD": "yes"})
        for value in ("false", 0, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                replace(self.settings, preload=value)


class PureTests(IsolatedTest):
    def test_yaml_and_command_arguments_are_safe(self):
        model_id = "bad:\n  injected: true"
        path = "/models/it's a $(touch nope); model.gguf"
        registry = {model_id: entry(path, display='quote " and\nnewline', description="x: [y]", aliases=["false", "a: b"])}
        config = yaml.safe_load(render_config(registry, "macros: {}\n", replace(self.settings, preload=True)))
        self.assertEqual(list(config["models"]), [model_id])
        self.assertEqual(config["models"][model_id]["name"], registry[model_id]["display"])
        tokens = shlex.split(config["models"][model_id]["cmd"])
        self.assertEqual(tokens[tokens.index("-m") + 1], path)
        self.assertEqual(config["hooks"]["on_startup"]["preload"], [model_id])
        self.assertEqual(config["models"][model_id]["aliases"], ["false", "a: b"])

    def test_macro_injection_rejected_in_paths(self):
        for path in ("/model/${PORT}.gguf", "/model/one\ntwo.gguf", "nul\0.gguf"):
            with self.subTest(path=path), self.assertRaises(ManagerError):
                render_config({"x": entry(path)}, "{}", self.settings)

    def test_extra_args_fragments_quoted_without_shell_expansion(self):
        config = yaml.safe_load(render_config({"x": entry(extra_args=["--temp 0.6", "--chat-template '$(touch nope)'", "-md '/models/a b.gguf'"])}, "{}", self.settings))
        args = shlex.split(config["models"]["x"]["cmd"])
        self.assertEqual(args[args.index("--chat-template") + 1], "$(touch nope)")
        self.assertEqual(args[args.index("-md") + 1], "/models/a b.gguf")

    def test_custom_commands_and_knobs_preserved(self):
        command = ["bash -c 'exec ./server --port ${PORT}'"]
        custom = entry(cmd=command, proxy="http://localhost:${PORT}", checkEndpoint="/v1/models", useModelName="upstream", unloadTimeout=30, aliases=["alias"])
        before = json.dumps(custom)
        rendered = yaml.safe_load(render_config({"real-id": custom}, "# comment\nmacros: {}", self.settings))
        model = rendered["models"]["real-id"]
        self.assertEqual(model["cmd"], command[0])
        for key in ("proxy", "checkEndpoint", "useModelName", "unloadTimeout", "aliases"):
            self.assertEqual(model[key], custom[key])
        self.assertEqual(json.dumps(custom), before)

    def test_hooks_merged_empty_models_and_nonchat(self):
        text = render_config({"a": entry(), "b": entry(default=True), "e": entry(capability="embedding")}, "hooks:\n  other: retained\n", replace(self.settings, preload=True))
        result = yaml.safe_load(text)
        self.assertEqual(result["hooks"]["on_startup"]["preload"], ["b"])
        self.assertEqual(result["hooks"]["other"], "retained")
        self.assertNotIn("e", result["models"])
        self.assertEqual(yaml.safe_load(render_config({}, "{}", self.settings))["models"], {})
        self.assertIsNone(default_model({}))

    def test_default_is_on_demand_and_explicit_header_preload_wins(self):
        registry = {"a": entry(), "b": entry(default=True)}
        rendered = yaml.safe_load(render_config(registry, "{}", self.settings))
        self.assertNotIn("hooks", rendered)
        for automatic in (False, True):
            settings = replace(self.settings, preload=automatic)
            for explicit in ([], ["a"]):
                with self.subTest(automatic=automatic, explicit=explicit):
                    header = yaml.safe_dump({"hooks": {"on_startup": {"preload": explicit}, "other": "retained"}})
                    rendered = yaml.safe_load(render_config(registry, header, settings))
                    self.assertEqual(rendered["hooks"]["on_startup"]["preload"], explicit)
                    self.assertEqual(rendered["hooks"]["other"], "retained")
        self.assertNotIn("hooks", yaml.safe_load(render_config({}, "{}", replace(self.settings, preload=True))))
        with self.assertRaisesRegex(ManagerError, "list of model IDs"):
            render_config(registry, "hooks:\n  on_startup:\n    preload: all", self.settings)

    def test_header_model_collision_refused(self):
        with self.assertRaises(ManagerError):
            render_config({}, "models: {}", self.settings)

    def test_header_flow_mapping_document_end_and_invalid_yaml(self):
        for header in ("{globalTTL: 0}", "globalTTL: 0\n...\n"):
            result = yaml.safe_load(render_config({}, header, self.settings))
            self.assertEqual(result["globalTTL"], 0)
            self.assertEqual(result["models"], {})
        for header in ("[]", "macros: [", "hooks: []"):
            with self.subTest(header=header), self.assertRaises(ManagerError):
                render_config({}, header, self.settings)

    def test_context_never_exceeds_training(self):
        for trained in (1, 128, 1024, 6000, 8192, 131072):
            for budget in (None, 1, 1000, 8000, 80000):
                with self.subTest(trained=trained, budget=budget):
                    metadata = {"general.architecture": "llama", "llama.context_length": trained, "llama.block_count": 40, "llama.attention.head_count_kv": 8}
                    ctx = suggest_ctx(metadata, 8 * 2**30, budget)
                    self.assertGreater(ctx, 0)
                    self.assertLessEqual(ctx, trained)
        self.assertEqual(suggest_ctx({}, 0), 4096)
        oversized = {"general.architecture": "llama", "llama.context_length": 131072,
                     "llama.block_count": 40, "llama.attention.head_count_kv": 8}
        self.assertEqual(suggest_ctx(oversized, 16 * 2**30, 8192), 4096)

    def test_render_context_bounds_preserve_split_slots(self):
        result = yaml.safe_load(render_config({"unified": entry(ctx=8000), "split": entry(ctx=4096, parallel=2, kv_unified=False)}, "{}", self.settings))
        for name, context in (("unified", "2048"), ("split", "4096")):
            args = shlex.split(result["models"][name]["cmd"])
            self.assertEqual(args[args.index("-c") + 1], context)

    def test_shards_validate_entire_group_and_use_first(self):
        files = [f"sub/model-Q4_K_M-{i:05}-of-00003.gguf" for i in (1, 2, 3)]
        self.assertEqual(shard_group(files[2], files), files)
        self.assertEqual(collapse_shards(files), files[:1])
        for bad in (files[1:], files[:2], files + files[:1], files + ["sub/model-Q4_K_M-00004-of-00004.gguf"]):
            with self.subTest(bad=bad), self.assertRaises(ManagerError):
                shard_group(bad[0], bad)

    def test_resolve_filters_drafts_and_preserves_target_mtp(self):
        files = ["Model-MTP-Q4_K_M.gguf", "mtp-draft-Q4_K_M.gguf", "mmproj-F16.gguf", "MTP/head-Q4_K_M.gguf"]
        self.assertEqual(pick_model_file(files), files[0])
        plan = resolve_source("owner/repo:Q4_K_M", files)
        self.assertEqual(plan["repo"], "owner/repo")
        self.assertEqual(plan["mmproj_file"], "mmproj-F16.gguf")
        self.assertIsNone(resolve_source("owner/repo", files, no_mmproj=True)["mmproj_file"])
        for source in ("bad", "hf://owner/repo/missing.gguf"):
            with self.subTest(source=source), self.assertRaises(ManagerError):
                resolve_source(source, files)
        with self.assertRaises(ManagerError):
            resolve_source("owner/repo", ["../escape.gguf"])


class MutationTests(IsolatedTest):
    def test_init_non_overwriting_and_no_other_effects(self):
        created = self.manager.init()
        self.assertEqual(len(created), 3)
        before = {p: Path(p).read_bytes() for p in created}
        self.assertEqual(ModelManager(replace(self.settings, server="different")).init(), [])
        self.assertEqual(before, {p: Path(p).read_bytes() for p in created})
        self.assertFalse(self.settings.output.exists())
        self.assertFalse(self.settings.client_path.exists())
        self.runner.assert_not_called()
        self.opener.assert_not_called()

    def test_atomic_write_and_cleanup(self):
        path = self.root / "new" / "file"
        atomic_write(path, "old")
        with patch("llama_swap_manager.manager.os.replace", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                atomic_write(path, "new")
        self.assertEqual(path.read_text(), "old")
        self.assertEqual(list(path.parent.iterdir()), [path])
        with self.assertRaises(FileExistsError):
            atomic_write(path, "overwrite", overwrite=False)

    def test_local_add_shard_combined_size_no_network_service_or_clients(self):
        self.manager.init()
        paths = [self.weights(f"target-Q4_K_M-{i:05}-of-00002.gguf", size=i * 100) for i in (1, 2)]
        name, model = self.manager.add(str(paths[1]), name="real-id", no_mmproj=True)
        self.assertEqual(name, "real-id")
        self.assertEqual(model["size_bytes"], 300)
        self.assertEqual(model["path"], str(paths[0]))
        self.assertEqual(model["ctx"], 1024)
        self.probe.assert_called_once_with(paths[0])
        self.assertIn(name, yaml.safe_load(self.settings.output.read_text())["models"])
        self.assertFalse(self.settings.client_path.exists())
        self.runner.assert_not_called()
        self.opener.assert_not_called()

    def test_incomplete_local_shard_rejected_before_probe(self):
        path = self.weights("target-00001-of-00002.gguf")
        with self.assertRaises(ManagerError):
            self.manager.add(str(path))
        self.probe.assert_not_called()
        self.assertFalse(self.settings.registry.exists())

    def test_hf_download_pins_every_shard_and_projector_to_sha_cache(self):
        self.manager.init()
        files = ["target-Q4_K_M-00001-of-00002.gguf", "target-Q4_K_M-00002-of-00002.gguf", "mmproj-F16.gguf"]
        sha = "a" * 40
        hub = Mock()
        hub.model_info.return_value = SimpleNamespace(sha=sha, siblings=[SimpleNamespace(rfilename=f) for f in files])
        hub.hf_hub_download.side_effect = lambda **kwargs: str(self.weights(kwargs["filename"], folder=self.root / "downloads"))
        self.manager.hub = hub
        _, model = self.manager.add("owner/repo:Q4_K_M", revision="v1")
        hub.model_info.assert_called_once_with("owner/repo", revision="v1")
        self.assertEqual(hub.hf_hub_download.call_count, 3)
        for call in hub.hf_hub_download.call_args_list:
            self.assertEqual(call.kwargs["revision"], sha)
            self.assertEqual(call.kwargs["cache_dir"], str(self.settings.hf_cache))
        self.assertEqual(model["sha"], sha)
        self.assertEqual(model["size_bytes"], 64)

    def test_hf_unpinned_revision_refused(self):
        self.manager.hub = Mock()
        self.manager.hub.model_info.return_value = SimpleNamespace(sha="main")
        with self.assertRaises(ManagerError):
            self.manager.plan("owner/repo")
        self.manager.hub.hf_hub_download.assert_not_called()

    def test_hf_errors_reported_and_missing_probe_fails_before_download(self):
        self.manager.hub = Mock()
        self.manager.hub.model_info.side_effect = OSError("offline")
        with self.assertRaisesRegex(ManagerError, "could not list"):
            self.manager.plan("owner/repo")
        manager = ModelManager(self.settings, hub=self.manager.hub)
        with patch("llama_swap_manager.manager.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(ManagerError, "nothing downloaded"):
                manager.add("owner/repo")
        self.manager.hub.hf_hub_download.assert_not_called()

    def test_duplicate_name_refused_and_invalid_header_does_not_write_registry(self):
        self.registry({"same": entry()})
        path = self.weights()
        with self.assertRaises(ManagerError):
            self.manager.add(str(path), name="same")
        self.assertEqual(list(self.manager.load_registry()), ["same"])
        atomic_write(self.settings.header, "models: {}")
        with self.assertRaises(ManagerError):
            self.manager.add(str(path), name="new")
        self.assertEqual(list(self.manager.load_registry()), ["same"])

    def test_remove_offline_does_not_unload_restart_or_sync(self):
        self.registry({"a": entry()})
        self.manager.remove("a")
        self.assertEqual(self.manager.load_registry(), {})
        self.runner.assert_not_called()
        self.opener.assert_not_called()
        self.assertFalse(self.settings.client_path.exists())

    def test_clients_explicit_preserve_credentials_other_providers(self):
        self.registry({"a": entry(ctx=4096, trained_ctx=2048, parallel=2, kv_unified=False), "b": entry(mmproj="vision.gguf")})
        atomic_write(self.settings.client_path, json.dumps({"providers": {"other": {"keep": True}, "llama-swap": {"baseUrl": "https://custom/v1", "apiKey": "secret"}}}))
        self.manager.ensure_clients()
        data = json.loads(self.settings.client_path.read_text())
        self.assertEqual(data["providers"]["other"], {"keep": True})
        block = data["providers"]["llama-swap"]
        self.assertEqual(block["apiKey"], "secret")
        self.assertEqual(block["baseUrl"], "https://custom/v1")
        self.assertEqual(block["models"][0]["contextWindow"], 2048)
        self.assertEqual(block["models"][1]["input"], ["text", "image"])

    def test_malformed_client_file_unchanged(self):
        atomic_write(self.settings.client_path, "not json")
        with self.assertRaises(ValueError):
            self.manager.ensure_clients()
        self.assertEqual(self.settings.client_path.read_text(), "not json")


class PurgeTests(IsolatedTest):
    def test_purge_always_refused_before_any_io(self):
        self.manager.load_registry = Mock(side_effect=AssertionError("registry read"))
        for restart in (False, True):
            with self.subTest(restart=restart), self.assertRaisesRegex(ManagerError, "purge is disabled.*shared HF cache"):
                self.manager.remove("even-unknown-model", purge=True, restart=restart, sync_clients=True)
        self.manager.load_registry.assert_not_called()
        self.runner.assert_not_called()
        self.opener.assert_not_called()
        self.assertFalse(self.settings.config_dir.exists())

    def test_header_and_other_registry_shared_cache_unchanged(self):
        path = self.cache_weights()
        self.registry({"a": entry(path)})
        atomic_write(self.settings.header, yaml.safe_dump({"macros": {"common": "-md " + str(path)}}))
        other = ModelManager(Settings(self.root / "other", hf_cache=self.settings.hf_cache))
        other.init()
        atomic_write(other.settings.registry, json.dumps({"different-consumer": entry(path)}))
        self.manager.render()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ManagerError, "exclusive ownership.*cannot be proven"):
            self.manager.remove("a", purge=True, restart=True, sync_clients=True)
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after, before)
        self.runner.assert_not_called()
        self.opener.assert_not_called()


class APITests(IsolatedTest):
    def response(self, value):
        self.opener.side_effect = None
        self.opener.return_value = io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())

    def test_malformed_and_error_api_fail(self):
        for body in (b"not json", b"", [], {"error": "failed"}, {"success": False}):
            with self.subTest(body=body):
                self.response(body)
                with self.assertRaises(ManagerError):
                    self.manager.api("/running")

    def test_malformed_running_and_models_fail(self):
        for body in ({}, {"running": {}}, {"running": [{}]}, {"running": [{"model": "a"}]}):
            self.response(body)
            with self.assertRaises(ManagerError):
                self.manager.running()
        self.manager.running = Mock(return_value=[])
        for body in ({}, {"data": [{}]}, {"data": "wrong"}):
            self.response(body)
            with self.assertRaises(ManagerError):
                self.manager.status()

    def test_use_requires_answer_and_sends_real_model_id(self):
        self.registry({"real-id": entry()})
        for body in ({}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]},
                     {"choices": [{"message": {"role": "assistant", "content": 123}}]},
                     {"choices": [{"message": {"role": "assistant", "content": None}}]}):
            self.response(body)
            with self.assertRaises(ManagerError):
                self.manager.use("real-id")
        self.response({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        self.manager.use("real-id")
        request = self.opener.call_args.args[0]
        self.assertEqual(json.loads(request.data)["model"], "real-id")

    def test_doctor_offline_no_network(self):
        self.manager.init()
        checks = self.manager.doctor()
        self.assertTrue(next(c for c in checks if c["check"] == "config")["ok"])
        self.opener.assert_not_called()
        self.runner.assert_not_called()

    def test_doctor_malformed_header_and_live_failure(self):
        self.manager.init()
        atomic_write(self.settings.header, "macros: [")
        self.opener.side_effect = urllib.error.URLError("offline")
        checks = self.manager.doctor(live=True)
        for key in ("config", "live"):
            self.assertFalse(next(c for c in checks if c["check"] == key)["ok"])

class ServiceTests(IsolatedTest):
    def setUp(self):
        super().setUp()
        self.binary = self.weights("llama-swap", folder=self.root / "bin with spaces")
        self.binary.chmod(0o755)
        self.settings = replace(self.settings, swap_binary=str(self.binary))
        self.manager = ModelManager(self.settings, probe=self.probe, opener=self.opener, runner=self.runner)
        self.unit = self.manager.install_service()
        self.argv = [str(self.binary), "-config", str(self.settings.output), "-listen", self.settings.listen]
        self.properties = {
            "Id": "llama-swap.service", "LoadState": "loaded", "FragmentPath": str(self.unit),
            "ExecStart": f"{{ path={self.binary} ; argv[]={' '.join(self.argv)} ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }}",
            "ActiveState": "inactive", "MainPID": "0", "ControlPID": "0",
            "NeedDaemonReload": "no", "DropInPaths": "", "Transient": "no", "Type": "simple",
        }
        self.actions = []
        self.action_result = SimpleNamespace(returncode=0, stdout="", stderr="")
        self.runner.side_effect = self.run_mock

    def run_mock(self, command, **kwargs):
        if command[2] == "show":
            self.assertEqual(command[:4], ["systemctl", "--user", "show", self.settings.unit])
            self.assertEqual(kwargs["timeout"], 10)
            return SimpleNamespace(returncode=0, stderr="", stdout="\n".join(f"{k}={v}" for k, v in self.properties.items()) + "\n")
        self.actions.append(command)
        return self.action_result

    @contextlib.contextmanager
    def live_process(self, *, argv=None, executable=None, error=None):
        self.properties.update(ActiveState="active", MainPID="4321")
        original_read, original_resolve = Path.read_bytes, Path.resolve
        def read(path):
            if path == Path("/proc/4321/cmdline"):
                if error:
                    raise error
                return b"\0".join(os.fsencode(a) for a in (self.argv if argv is None else argv)) + b"\0"
            return original_read(path)
        def resolve(path, strict=False):
            if path == Path("/proc/4321/exe"):
                return original_resolve(executable or self.binary, strict=strict)
            return original_resolve(path, strict=strict)
        with patch.object(Path, "read_bytes", read), patch.object(Path, "resolve", resolve):
            yield

    def test_install_nonoverwriting_and_does_not_touch_manager(self):
        self.assertIn('"' + str(self.binary) + '"', self.unit.read_text())
        self.assertIn(str(self.settings.output), self.unit.read_text())
        with self.assertRaises(FileExistsError):
            self.manager.install_service()
        self.runner.assert_not_called()

    def test_start_restart_stop_require_matching_loaded_unit(self):
        for action in ("start", "restart", "stop"):
            self.manager.service(action)
            self.assertEqual(self.actions[-1], ["systemctl", "--user", action, "llama-swap.service"])
        self.assertEqual(self.runner.call_count, 6)
        self.opener.assert_not_called()

    def test_matching_live_process_allows_actions(self):
        with self.live_process():
            for action in ("start", "restart", "stop"):
                self.manager.service(action)
        self.assertEqual(len(self.actions), 3)

    def test_missing_unit_is_refused_before_systemctl(self):
        self.unit.unlink()
        with self.assertRaisesRegex(ManagerError, "install the unit first"):
            self.manager.service("start")
        self.runner.assert_not_called()

    def test_unlinked_external_unit_refused_with_exact_link_instruction(self):
        self.properties.update(LoadState="not-found", FragmentPath="", ExecStart="")
        with self.assertRaisesRegex(ManagerError, "not discoverable") as raised:
            self.manager.service("start")
        self.assertIn(shlex.join(["systemctl", "--user", "link", str(self.unit)]), str(raised.exception))
        self.assertEqual(self.actions, [])

    def test_linked_unit_resolves_to_expected_source(self):
        linked = self.root / "user-unit-path" / self.settings.unit
        linked.parent.mkdir()
        linked.symlink_to(self.unit)
        self.properties["FragmentPath"] = str(linked)
        self.manager.service("start")
        self.assertEqual(len(self.actions), 1)

    def test_other_instance_same_unit_name_refused(self):
        other = self.root / "other.service"
        other.write_text(self.unit.read_text())
        self.properties["FragmentPath"] = str(other)
        for action in ("start", "restart", "stop"):
            with self.subTest(action=action), self.assertRaisesRegex(ManagerError, "different unit source"):
                self.manager.service(action)
        self.assertEqual(self.actions, [])

    def test_changed_source_refused_even_if_loaded_exec_matches(self):
        self.unit.write_text(self.unit.read_text().replace(str(self.settings.output), "/other/config.yaml"))
        with self.assertRaisesRegex(ManagerError, "not the unmodified unit"):
            self.manager.service("stop")
        self.runner.assert_not_called()

    def test_loaded_execstart_mismatch_refused_for_every_binding(self):
        original = self.properties["ExecStart"]
        for old, new in ((str(self.binary), "/other/swap"), (str(self.settings.output), "/other/config.yaml"), (self.settings.listen, "127.0.0.1:9999")):
            with self.subTest(old=old):
                self.properties["ExecStart"] = original.replace(old, new)
                with self.assertRaisesRegex(ManagerError, "loaded ExecStart"):
                    self.manager.service("restart")
        self.assertEqual(self.actions, [])

    def test_stale_dropin_transient_and_unstable_units_refused(self):
        for key, value in (("NeedDaemonReload", "yes"), ("DropInPaths", "/override.conf"),
                           ("Transient", "yes"), ("Type", "forking"), ("ActiveState", "activating"),
                           ("MainPID", "4321"), ("MainPID", "bad"), ("ControlPID", "32")):
            original = self.properties[key]
            with self.subTest(key=key), self.assertRaises(ManagerError):
                self.properties[key] = value
                self.manager.service("restart")
            self.properties[key] = original
        self.assertEqual(self.actions, [])

    def test_live_binding_drift_after_daemon_reload_refused(self):
        for index, value in ((0, "/other/binary"), (2, "/old-instance/config.yaml"), (4, "127.0.0.1:9999")):
            argv = self.argv.copy()
            argv[index] = value
            with self.live_process(argv=argv):
                for action in ("start", "restart", "stop"):
                    with self.subTest(index=index, action=action), self.assertRaisesRegex(ManagerError, "live process binding"):
                        self.manager.service(action)
        self.assertEqual(self.actions, [])

    def test_live_executable_and_unreadable_proc_refused(self):
        other = self.weights("other-executable")
        with self.live_process(executable=other), self.assertRaisesRegex(ManagerError, "live process binding"):
            self.manager.service("stop")
        with self.live_process(error=PermissionError("proc unavailable")), self.assertRaisesRegex(ManagerError, "cannot verify"):
            self.manager.service("restart")
        self.assertEqual(self.actions, [])

    def test_systemctl_failure_timeout_and_malformed_show_refused(self):
        for response in (SimpleNamespace(returncode=1, stderr="no user manager", stdout=""),
                         SimpleNamespace(returncode=0, stderr="", stdout="Id=llama-swap.service\n"),
                         SimpleNamespace(returncode=0, stderr="", stdout="Id=x\nId=y\n")):
            self.runner.side_effect = None
            self.runner.return_value = response
            with self.subTest(response=response), self.assertRaises(ManagerError):
                self.manager.service("start")
        self.runner.side_effect = subprocess.TimeoutExpired("show", 10)
        with self.assertRaisesRegex(ManagerError, "cannot verify"):
            self.manager.service("restart")
        self.assertEqual(self.actions, [])

    def test_action_errors_still_propagate_after_verification(self):
        self.action_result = SimpleNamespace(returncode=1, stdout="", stderr="stop failure")
        with self.assertRaisesRegex(ManagerError, "stop failed: stop failure"):
            self.manager.service("stop")

    def test_add_remove_preflight_before_mutation_or_network(self):
        self.registry({"a": entry()})
        before = self.settings.registry.read_bytes()
        self.properties["FragmentPath"] = "/some-other-instance.service"
        for operation in (lambda: self.manager.add("owner/repo", restart=True),
                          lambda: self.manager.remove("a", restart=True)):
            with self.assertRaises(ManagerError):
                operation()
        self.assertEqual(self.settings.registry.read_bytes(), before)
        self.assertFalse(self.settings.output.exists())
        self.probe.assert_not_called()
        self.opener.assert_not_called()
        self.assertEqual(self.actions, [])

    def test_automatic_api_operations_require_matching_endpoint(self):
        self.registry({"a": entry()})
        for url in ("http://127.0.0.1:9999", "http://remote:18080", "https://127.0.0.1:18080", "http://127.0.0.1:18080/proxy"):
            manager = ModelManager(replace(self.settings, swap_url=url), runner=self.runner, opener=self.opener)
            for operation in (lambda: manager.add("owner/repo", restart=True, load=True),
                              lambda: manager.remove("a", restart=True)):
                with self.subTest(url=url), self.assertRaisesRegex(ManagerError, "listen endpoint"):
                    operation()
        self.assertIn("a", self.manager.load_registry())
        self.opener.assert_not_called()
        self.assertEqual(self.actions, [])

    def test_remove_from_inactive_verified_service_skips_api(self):
        self.registry({"a": entry()})
        self.manager.remove("a", restart=True)
        self.assertEqual(self.manager.load_registry(), {})
        self.assertEqual(self.actions, [["systemctl", "--user", "restart", "llama-swap.service"]])
        self.opener.assert_not_called()

    def test_cli_no_load_skips_explicit_request_not_configured_preload(self):
        self.registry({})
        path = self.weights()
        self.manager.use = Mock(side_effect=AssertionError("unexpected warm-up"))
        self.manager.wait_ready = Mock(side_effect=AssertionError("unexpected API polling"))
        for preload, name in ((True, "first"), (False, "second")):
            self.manager.settings = replace(self.settings, preload=preload)
            with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ, {"HOME": str(self.root)}, clear=True):
                code = main(["--config-dir", str(self.settings.config_dir), "add", name, str(path), "--no-load"],
                            manager_factory=lambda _: self.manager)
            self.assertEqual(code, 0)
            rendered = yaml.safe_load(self.settings.output.read_text())
            if preload:
                self.assertEqual(rendered["hooks"]["on_startup"]["preload"], [name])
            else:
                self.assertNotIn("hooks", rendered)
        self.assertEqual(len(self.actions), 2)
        self.manager.use.assert_not_called()
        self.manager.wait_ready.assert_not_called()
        self.opener.assert_not_called()

    def test_unload_failure_or_still_running_leaves_registry(self):
        self.registry({"a": entry()})
        self.manager.running = Mock(return_value=[{"model": "a", "state": "ready"}])
        self.manager.api = Mock(side_effect=ManagerError("unload failed"))
        with self.live_process(), self.assertRaisesRegex(ManagerError, "unload failed"):
            self.manager.remove("a", restart=True)
        self.manager.api.side_effect = None
        self.manager.api.return_value = {}
        with self.live_process(), self.assertRaisesRegex(ManagerError, "unload did not stop"):
            self.manager.remove("a", restart=True)
        self.assertIn("a", self.manager.load_registry())
        self.assertEqual(self.actions, [])

    def test_remove_verified_active_service_unloads_real_id(self):
        self.registry({"a/b": entry()})
        self.manager.running = Mock(side_effect=[[{"model": "a/b", "state": "ready"}], []])
        self.manager.api = Mock(return_value={})
        with self.live_process():
            self.manager.remove("a/b", restart=True)
        self.manager.api.assert_called_once_with("/api/models/unload/a%2Fb", method="POST", allow_empty=True)
        self.assertEqual(self.manager.load_registry(), {})


class CLITests(IsolatedTest):
    @unittest.skipUnless(os.name == "posix", "checkout executable shim is POSIX-only")
    def test_checkout_shim_executable_help(self):
        shim = Path(__file__).resolve().parents[1] / "bin" / "llm"
        if not shim.exists():
            self.skipTest("checkout shim not available")
        result = subprocess.run([str(shim), "--help"], capture_output=True, text=True,
                                env={**os.environ, "HOME": str(self.root), "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config-dir", result.stdout)

    def invoke(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"HOME": str(self.root)}, clear=True), contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(["--config-dir", str(self.settings.config_dir), *args], manager_factory=lambda settings: ModelManager(settings, probe=self.probe, opener=self.opener, runner=self.runner))
        return code, output.getvalue(), error.getvalue()

    def test_mocked_cli_init_add_render_remove(self):
        self.assertEqual(self.invoke("init", "--server", "fake-server")[0], 0)
        model = self.weights()
        self.assertEqual(self.invoke("add", "tiny", str(model), "--no-restart")[0], 0)
        self.assertEqual(self.invoke("render")[0], 0)
        self.assertEqual(self.invoke("ls", "--offline")[0], 0)
        self.assertIn("tiny", self.manager.load_registry())
        self.assertEqual(self.invoke("rm", "tiny", "--no-restart")[0], 0)
        self.assertFalse((self.root / ".config").exists())
        self.assertFalse((self.root / ".pi").exists())
        self.runner.assert_not_called()
        self.opener.assert_not_called()

    def test_cli_purge_refused_and_no_load_help_precise(self):
        self.assertEqual(self.invoke("init", "--server", "fake-server")[0], 0)
        before = self.settings.registry.read_bytes()
        for args in (("rm", "unknown", "--purge"), ("rm", "unknown", "--purge", "--no-restart")):
            code, _, error = self.invoke(*args)
            self.assertEqual(code, 1)
            self.assertIn("purge is disabled", error)
        self.assertEqual(self.settings.registry.read_bytes(), before)
        self.runner.assert_not_called()
        self.opener.assert_not_called()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()) as output:
            main(["add", "--help"])
        self.assertIn("explicit warm-up", output.getvalue())
        self.assertIn("startup preload", " ".join(output.getvalue().split()))

    def test_install_cli_prints_external_link_without_running_it(self):
        self.manager.init()
        with patch("llama_swap_manager.manager.shutil.which", return_value="/opt/llama-swap"):
            code, output, error = self.invoke("install-service")
        self.assertEqual(code, 0, error)
        path = self.settings.service_dir / self.settings.unit
        self.assertIn(shlex.join(["systemctl", "--user", "link", str(path)]), output)
        self.assertIn("systemctl --user daemon-reload", output)
        self.runner.assert_not_called()

    def test_install_cli_standard_user_dir_does_not_suggest_link(self):
        standard = self.root / ".config" / "systemd" / "user"
        settings = replace(self.settings, service_dir=standard)
        manager = ModelManager(settings, runner=self.runner)
        with patch("llama_swap_manager.manager.shutil.which", return_value="/opt/llama-swap"), \
             patch.dict(os.environ, {"HOME": str(self.root)}, clear=True), \
             contextlib.redirect_stdout(output := io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--config-dir", str(self.settings.config_dir), "install-service"],
                        manager_factory=lambda _: manager)
        self.assertEqual(code, 0)
        self.assertNotIn("--user link", output.getvalue())
        self.assertIn("systemctl --user daemon-reload", output.getvalue())
        self.runner.assert_not_called()

    def test_cli_api_failures_nonzero(self):
        self.opener.side_effect = urllib.error.URLError("offline")
        for command in ("status", "ls"):
            code, _, error = self.invoke(command)
            self.assertEqual(code, 1)
            self.assertIn("API", error)

    def test_cli_waits_until_started_service_is_ready(self):
        manager = MagicMock()
        with patch.dict(os.environ, {"HOME": str(self.root)}, clear=True):
            for command in ("start", "restart", "stop"):
                manager.reset_mock()
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--config-dir", str(self.settings.config_dir), command],
                                manager_factory=lambda _settings: manager)
                self.assertEqual(code, 0)
                manager.service.assert_called_once_with(command)
                if command == "stop":
                    manager.wait_ready.assert_not_called()
                else:
                    manager.wait_ready.assert_called_once_with()

    def test_import_and_help_no_settings_io(self):
        bad_dir = self.root / "bad"
        bad_dir.mkdir()
        (bad_dir / "settings.json").write_text("INVALID")
        env = {**os.environ, "HOME": str(self.root), "LLM_CONFIG_DIR": str(bad_dir), "PYTHONDONTWRITEBYTECODE": "1"}
        code = "import os; from unittest.mock import patch; import pathlib;\nwith patch('pathlib.Path.read_text', side_effect=AssertionError('read')), patch('subprocess.run', side_effect=AssertionError('process')), patch('urllib.request.urlopen', side_effect=AssertionError('network')):\n import llama_swap_manager\n from llama_swap_manager.cli import main\n main(['--help'])\n"
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-m", "llama_swap_manager", "--help"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config-dir", result.stdout)
        self.assertFalse((self.root / ".config").exists())

    @unittest.skipUnless(importlib.util.find_spec("gguf"), "optional gguf extra not installed; mocked smoke still runs")
    def test_real_synthetic_gguf_subprocess_smoke(self):
        from gguf import GGUFWriter
        import numpy as np

        path = self.root / "tiny-Q4_K_M.gguf"
        writer = GGUFWriter(path, "llama")
        writer.add_name("Synthetic tiny model")
        writer.add_context_length(1024)
        writer.add_block_count(1)
        writer.add_head_count(1)
        writer.add_head_count_kv(1)
        writer.add_embedding_length(4)
        writer.add_tensor("token_embd.weight", np.zeros((4, 4), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self.assertEqual(probe_gguf(path)["llama.context_length"], 1024)
        env = {**os.environ, "HOME": str(self.root), "XDG_CONFIG_HOME": str(self.root / "xdg"), "PYTHONDONTWRITEBYTECODE": "1"}
        for command in (["init", "--server", "/nonexistent/llama-server"], ["add", "tiny", str(path), "--no-restart", "--no-mmproj"], ["render"], ["ls", "--offline"]):
            result = subprocess.run([sys.executable, "-m", "llama_swap_manager", "--config-dir", str(self.settings.config_dir), *command], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.manager.load_registry()["tiny"]["trained_ctx"], 1024)
        self.assertFalse((self.root / "xdg").exists())


if __name__ == "__main__":
    unittest.main()
