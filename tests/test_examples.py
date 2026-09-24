"""config/*.example.* must stay loadable: they are the schema documentation."""

import json
import shlex
from pathlib import Path
import tempfile
import unittest

import yaml

from llms import Settings, render_config

CONFIG = Path(__file__).resolve().parents[1] / "config"


def strip_comments(value):
    if isinstance(value, dict):
        return {k: strip_comments(v) for k, v in value.items() if k != "_comment"}
    if isinstance(value, list):
        return [strip_comments(v) for v in value]
    return value


class ExampleConfigTests(unittest.TestCase):
    def test_settings_and_registry_examples_render(self):
        settings_data = strip_comments(json.loads((CONFIG / "settings.example.json").read_text()))
        registry = strip_comments(json.loads((CONFIG / "registry.example.json").read_text()))
        header = (CONFIG / "config.header.example.yaml").read_text()
        with tempfile.TemporaryDirectory() as root:
            for manager in ("systemd", "launchd"):
                with self.subTest(manager=manager):
                    settings = Settings(Path(root), **{**settings_data, "service_manager": manager})
                    rendered = yaml.safe_load(render_config(registry, header, settings))
                    models = rendered["models"]
                    self.assertIn("example-mlx", models)
                    tokens = shlex.split(models["example-mlx"]["cmd"].replace("${PORT}", "1"))
                    self.assertEqual(tokens[:2], ["/Applications/oMLX.app/Contents/MacOS/omlx-cli", "serve"])
                    self.assertEqual(models["example-mlx"]["useModelName"], "example-mlx")
                    self.assertIn("--spec-type draft-mtp", models["example-engine"]["cmd"])
                    self.assertTrue(set(settings.engines) >= {"metal", "omlx"})
                    self.assertEqual(settings.engines["metal"]["mtp_args"][:2], ["--spec-type", "draft-mtp"])


if __name__ == "__main__":
    unittest.main()
