"""bench/lib/promote.py: turning a measured variant into production."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "lib"))

from promote import PromoteError, load_measured  # noqa: E402
from stack import Variant  # noqa: E402

Q8V = Variant("q8V", entries={"m": {"extra_args": ["--cache-type-v q8_0"]}})


def row(model="m", variant=Q8V, status="OK", digest="d" * 64, cmd="/opt/e/llama-server -m m.gguf", **extra):
    return {"model": model, "variant": variant.name, "depth": "4096", "status": status,
            "patch": json.dumps(variant.to_dict(), sort_keys=True), "rendered_cmd": cmd,
            "production": "active", "base_digest": digest, **extra}


class LoadMeasuredTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tsv(self, rows, name="sweep.tsv"):
        path = self.root / name
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def kv_json(self, provenance, model="m", records=1, name="kv.json"):
        path = self.root / name
        path.write_text(json.dumps({"tag": "t", "model": model, "provenance": provenance, "summary": {},
                                    "records": [{"depth": 4096, "rep": i + 1} for i in range(records)]}))
        return path

    def test_tsv_variant_rows_agree(self):
        path = self.tsv([row(variant=Variant()), row(), row(depth="65536")])
        measured = load_measured(path, "q8V")
        self.assertEqual((measured.model, measured.variant.entries, measured.base_digest, measured.rows),
                         ("m", Q8V.entries, "d" * 64, 2))
        self.assertEqual(measured.rendered_cmd, "/opt/e/llama-server -m m.gguf")

    def test_failed_rows_do_not_count(self):
        path = self.tsv([row(status="LOAD_FAIL"), row(status="REQUEST_FAIL")])
        with self.assertRaisesRegex(PromoteError, "no OK rows for variant q8V"):
            load_measured(path, "q8V")

    def test_unknown_variant_names_the_ones_present(self):
        path = self.tsv([row(variant=Variant()), row()])
        with self.assertRaisesRegex(PromoteError, r"no variant 'fast'.*production, q8V"):
            load_measured(path, "fast")

    def test_rows_that_disagree_are_refused(self):
        for field, value in (("base_digest", "e" * 64), ("rendered_cmd", "/other"),
                             ("patch", json.dumps(Variant("q8V", entries={"m": {"ctx": 1}}).to_dict()))):
            with self.subTest(field=field):
                path = self.tsv([row(), {**row(), field: value}])
                with self.assertRaisesRegex(PromoteError, f"disagree on {field}"):
                    load_measured(path, "q8V")

    def test_results_without_base_digest_are_refused(self):
        old = {k: v for k, v in row().items() if k != "base_digest"}
        for path in (self.tsv([old]), self.tsv([row(digest="")], name="empty.tsv")):
            with self.subTest(path=path.name), self.assertRaisesRegex(PromoteError, "no base digest.*re-run"):
                load_measured(path, "q8V")

    def test_two_models_need_model(self):
        path = self.tsv([row(model="m"), row(model="n")])
        with self.assertRaisesRegex(PromoteError, r"several models.*--model.*m, n"):
            load_measured(path, "q8V")
        self.assertEqual(load_measured(path, "q8V", model="n").model, "n")
        with self.assertRaisesRegex(PromoteError, "no OK rows for variant q8V of model x"):
            load_measured(path, "q8V", model="x")

    def test_kv_probe_json(self):
        provenance = {"mode": "stack", "variant": "q8V", "patch": json.dumps(Q8V.to_dict(), sort_keys=True),
                      "rendered_cmd": "/opt/e/llama-server", "production": "active", "base_digest": "d" * 64}
        measured = load_measured(self.kv_json(provenance, records=2), "q8V")
        self.assertEqual((measured.model, measured.rows, measured.variant.entries), ("m", 2, Q8V.entries))
        with self.assertRaisesRegex(PromoteError, "no variant 'other'.*q8V"):
            load_measured(self.kv_json(provenance), "other")
        with self.assertRaisesRegex(PromoteError, "no OK rows"):
            load_measured(self.kv_json(provenance, records=0), "q8V")

    def test_standalone_results_are_refused(self):
        path = self.kv_json({"mode": "standalone", "server": "http://127.0.0.1:8080"})
        with self.assertRaisesRegex(PromoteError, "standalone.*re-measure.*--variant"):
            load_measured(path, "production")


class ProductionCase(unittest.TestCase):
    """A temporary production config dir; measurements come from Stack's own config code."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prod = self.root / "prod"
        self.prod.mkdir()
        self.env = {"HOME": str(self.root), "PATH": "/usr/bin:/bin"}
        (self.prod / "config.header.yaml").write_text("macros:\n  common: --jinja\n")
        self.registry = {
            "m": {"path": "/models/m.gguf", "capability": "chat", "ctx": 4096, "engine": "e"},
            "n": {"path": "/models/n.gguf", "capability": "chat", "ctx": 8192, "engine": "e"},
            "solo": {"path": "/models/s.gguf", "capability": "chat", "ctx": 2048, "engine": "f"},
        }
        (self.prod / "registry.json").write_text(json.dumps(self.registry, indent=2, sort_keys=True) + "\n")
        self.settings_text = json.dumps({
            "service_manager": "systemd",
            "swap_binary": "llama-swap",
            "engines": {"e": {"server": "/opt/e/llama-server", "args": ["-ngl", "999"]},
                        "f": {"server": "/opt/f/llama-server", "args": ["-ngl", "999"]}},
        }, indent=4) + "\n"
        (self.prod / "settings.json").write_text(self.settings_text)

    def production(self):
        from llms import Settings
        return Settings.from_env(self.prod, environ=self.env)

    def measure(self, variant, model="m"):
        """What a benchmark run of variant would record, via Stack's own isolated render."""
        from promote import Measured
        from stack import Stack, base_digest
        stack = Stack(variant, production_dir=self.prod, environ=self.env, log=lambda *_: None)
        stack._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(stack._tmp.cleanup)
        stack._write_isolated_config()
        registry = json.loads((self.prod / "registry.json").read_text())
        return Measured(results=self.root / "r.tsv", model=model, variant=variant,
                        base_digest=base_digest(self.production(), registry, model, variant),
                        rendered_cmd=stack.rendered_command(model), rows=1)


class PlanTests(ProductionCase):
    def plan(self, measured, **kwargs):
        from promote import plan
        return plan(measured, self.production(), **kwargs)

    def test_entry_patch_is_planned(self):
        planned = self.plan(self.measure(Q8V))
        self.assertEqual(planned.registry["m"]["extra_args"], ["--cache-type-v q8_0"])
        self.assertEqual(planned.registry["n"], self.registry["n"])
        self.assertEqual(planned.affected, [])
        self.assertTrue(planned.changes)

    def test_production_variant_changes_nothing(self):
        planned = self.plan(self.measure(Variant()))
        self.assertFalse(planned.changes)

    def test_patch_reaching_another_entry_is_refused(self):
        variant = Variant("both", entries={"m": {"ctx": 2048}, "n": {"ctx": 2048}})
        with self.assertRaisesRegex(PromoteError, "patches registry entries other than m: n"):
            self.plan(self.measure(variant))

    def test_production_edited_after_the_run_is_refused(self):
        measured = self.measure(Q8V)
        self.registry["m"]["ctx"] = 8192
        (self.prod / "registry.json").write_text(json.dumps(self.registry))
        with self.assertRaisesRegex(PromoteError, "base digest differs"):
            self.plan(measured)

    def test_render_mismatch_is_refused(self):
        measured = self.measure(Q8V)
        measured.rendered_cmd += " --extra"
        with self.assertRaisesRegex(PromoteError, "would render a different command for m"):
            self.plan(measured)

    def test_shared_engine_needs_acceptance(self):
        variant = Variant("pin", engines={"e": {"env": {"CUDA_VISIBLE_DEVICES": "1"}}})
        measured = self.measure(variant)
        with self.assertRaisesRegex(PromoteError, r"also changes n.*--accept-shared-engine"):
            self.plan(measured)
        planned = self.plan(measured, accept_shared_engine=True)
        self.assertEqual(planned.affected, ["n"])
        self.assertEqual(planned.settings["engines"]["e"],
                         {"server": "/opt/e/llama-server", "args": ["-ngl", "999"],
                          "env": {"CUDA_VISIBLE_DEVICES": "1"}})
        self.assertEqual(planned.settings["swap_binary"], "llama-swap")

    def test_unshared_engine_patch_needs_no_acceptance(self):
        variant = Variant("f-ngl", engines={"f": {"args": ["-ngl", "40"]}})
        planned = self.plan(self.measure(variant, model="solo"))
        self.assertEqual((planned.affected, planned.settings["engines"]["f"]["args"]), ([], ["-ngl", "40"]))


class ApplyCase(ProductionCase):
    """Production with a real ModelManager whose service calls are mocked."""

    def setUp(self):
        super().setUp()
        from llms import ModelManager
        self.manager = ModelManager(self.production())
        self.manager.render()
        self.paths = [self.prod / "settings.json", self.prod / "registry.json", self.production().output]
        self.before = {p: p.read_bytes() for p in self.paths}
        for name, value in (("_verify_service", True), ("service", None), ("wait_ready", None), ("use", {})):
            patcher = unittest.mock.patch.object(self.manager, name, return_value=value)
            setattr(self, name.strip("_"), patcher.start())
            self.addCleanup(patcher.stop)

    def apply(self, variant, model="m", **kwargs):
        from promote import apply, plan
        planned = plan(self.measure(variant, model=model), self.production(), **kwargs)
        return apply(planned, self.production(), manager=self.manager)


class ApplyTests(ApplyCase):
    def test_success_writes_restarts_and_warms(self):
        variant = Variant("both", entries={"solo": {"ctx": 4096}}, engines={"f": {"args": ["-ngl", "40"]}})
        self.assertEqual(self.apply(variant, model="solo"), "ok")
        registry = json.loads((self.prod / "registry.json").read_text())
        settings = json.loads((self.prod / "settings.json").read_text())
        self.assertEqual(registry["solo"]["ctx"], 4096)
        self.assertEqual({k: v for k, v in registry.items() if k != "solo"},
                         {k: v for k, v in self.registry.items() if k != "solo"})
        self.assertEqual(settings["engines"]["f"]["args"], ["-ngl", "40"])
        self.assertEqual(settings["engines"]["e"], json.loads(self.settings_text)["engines"]["e"])
        self.assertEqual(settings["swap_binary"], "llama-swap")
        self.assertIn("-ngl 40", self.production().output.read_text())
        from llms import ModelManager
        written = self.production().output.read_text()
        ModelManager(self.production()).render()  # what `llms render` makes from the new files
        self.assertEqual(self.production().output.read_text(), written)
        self.assertEqual([c.args for c in self.service.call_args_list], [("restart",)])
        self.use.assert_called_once_with("solo")

    def test_failed_warmup_restores_files_and_restarts_again(self):
        from llms import ManagerError
        self.use.side_effect = ManagerError("malformed chat completion")
        with self.assertRaisesRegex(PromoteError, "warm-up.*malformed.*restored"):
            self.apply(Q8V)
        self.assertEqual({p: p.read_bytes() for p in self.paths}, self.before)
        self.assertEqual([c.args for c in self.service.call_args_list], [("restart",), ("restart",)])

    def test_failed_rollback_restart_is_reported_after_restoring(self):
        from llms import ManagerError
        self.use.side_effect = ManagerError("no answer")
        self.service.side_effect = [None, ManagerError("unit failed")]
        with self.assertRaisesRegex(PromoteError, "(?s)no answer.*restart after restoring also failed: unit failed"):
            self.apply(Q8V)
        self.assertEqual({p: p.read_bytes() for p in self.paths}, self.before)

    def test_entry_only_patch_leaves_settings_bytes_alone(self):
        settings = self.prod / "settings.json"
        settings.chmod(0o644)
        before = settings.read_bytes()
        self.assertEqual(self.apply(Q8V), "ok")
        self.assertEqual(settings.read_bytes(), before)  # hand formatting kept: not rewritten
        self.assertEqual(settings.stat().st_mode & 0o777, 0o644)

    def test_engine_only_patch_leaves_registry_bytes_alone(self):
        registry = self.prod / "registry.json"
        registry.write_text(json.dumps(self.registry, indent=4))  # not llms' own layout
        before = registry.read_bytes()
        self.before = {p: p.read_bytes() for p in self.paths}
        self.assertEqual(self.apply(Variant("f-ngl", engines={"f": {"args": ["-ngl", "40"]}}), model="solo"), "ok")
        self.assertEqual(registry.read_bytes(), before)

    def test_rollback_restores_bytes_and_mode_of_a_rewritten_settings_file(self):
        from llms import ManagerError
        settings = self.prod / "settings.json"
        settings.chmod(0o644)
        self.use.side_effect = ManagerError("no answer")
        with self.assertRaises(PromoteError):
            self.apply(Variant("f-ngl", engines={"f": {"args": ["-ngl", "40"]}}), model="solo")
        self.assertEqual({p: p.read_bytes() for p in self.paths}, self.before)
        self.assertEqual(settings.stat().st_mode & 0o777, 0o644)

    def test_inactive_service_is_left_stopped(self):
        self.verify_service.return_value = False
        self.assertEqual(self.apply(Q8V), "skipped")
        self.service.assert_not_called()
        self.use.assert_not_called()
        self.assertIn("--cache-type-v q8_0", self.production().output.read_text())

    def test_production_variant_changes_no_file(self):
        self.assertEqual(self.apply(Variant()), "unchanged")
        self.assertEqual({p: p.read_bytes() for p in self.paths}, self.before)
        self.service.assert_not_called()


class LogTests(ApplyCase):
    def results(self, variant, model="m"):
        measured = self.measure(variant, model=model)
        path = self.root / f"{variant.name}.tsv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(row()), delimiter="\t")
            writer.writeheader()
            writer.writerow(row(model=model, variant=variant, digest=measured.base_digest, cmd=measured.rendered_cmd))
        return path

    def promote(self, results, variant, **kwargs):
        from promote import promote
        return promote(results, variant, production=self.production(), manager=self.manager,
                       log_path=self.root / "promotions.tsv", **kwargs)

    def log_rows(self):
        with (self.root / "promotions.tsv").open(newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))

    def test_successes_append_rows_under_one_header(self):
        self.assertEqual(self.promote(self.results(Variant()), "production"), "unchanged")
        self.assertEqual(self.promote(self.results(Q8V), "q8V"), "ok")
        text = (self.root / "promotions.tsv").read_text()
        self.assertEqual(text.count("utc\thost\tmodel"), 1)
        rows = self.log_rows()
        self.assertEqual([(r["model"], r["variant"], r["warmup"]) for r in rows],
                         [("m", "production", "unchanged"), ("m", "q8V", "ok")])
        self.assertTrue(rows[1]["results"].endswith("q8V.tsv"))
        self.assertIn("--cache-type-v q8_0", rows[1]["rendered_cmd"])
        self.assertEqual(json.loads(rows[1]["patch"])["entries"], Q8V.entries)
        self.assertEqual(len(rows[1]["base_digest"]), 64)

    def test_failures_append_nothing(self):
        from llms import ManagerError
        results = self.results(Q8V)
        self.use.side_effect = ManagerError("no answer")
        with self.assertRaises(PromoteError):
            self.promote(results, "q8V")
        self.registry["m"]["ctx"] = 1
        (self.prod / "registry.json").write_text(json.dumps(self.registry))
        with self.assertRaisesRegex(PromoteError, "base digest differs"):
            self.promote(results, "q8V")
        self.assertFalse((self.root / "promotions.tsv").exists())


class CliTests(ProductionCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "promote.py"

    def run_cli(self, *args):
        import subprocess
        return subprocess.run([sys.executable, str(self.SCRIPT), *args], capture_output=True, text=True,
                              env={**self.env, "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"}, timeout=60)

    def test_help_lists_options(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("RESULTS", "--variant", "--model", "--config-dir", "--accept-shared-engine"):
            self.assertIn(option, result.stdout)

    def test_pre_digest_results_are_refused_without_changes(self):
        path = self.root / "old.tsv"
        old = {k: v for k, v in row().items() if k != "base_digest"}
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(old), delimiter="\t")
            writer.writeheader()
            writer.writerow(old)
        before = {p: p.read_bytes() for p in self.prod.iterdir()}
        result = self.run_cli(str(path), "--variant", "q8V", "--config-dir", str(self.prod))
        self.assertEqual(result.returncode, 1)
        self.assertIn("promote: ", result.stderr)
        self.assertIn("no base digest", result.stderr)
        self.assertEqual({p: p.read_bytes() for p in self.prod.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
