from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import yaml

from llms import Settings, default_model, render_config


class VideoCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        settings = Settings(Path(self.temp.name) / "manager", service_manager="systemd")
        self.settings = replace(settings, engines={"rad": {
            "kind": "custom", "server": "/bin/run-radiance", "args": ["/cfg.yaml", "/m.rad"],
            "check_endpoint": "/health"}})
        self.registry = {
            "chat-model": {"capability": "chat", "engine": "rad", "ctx": 262144},
            "video-model": {"capability": "video", "engine": "rad", "ctx": 1024},
            "embedder": {"capability": "embedding", "engine": "rad", "ctx": 512},
        }

    def test_video_models_are_rendered_beside_chat_models(self):
        models = yaml.safe_load(render_config(self.registry, "{}", self.settings))["models"]
        self.assertEqual(sorted(models), ["chat-model", "video-model"])
        self.assertTrue(models["video-model"]["cmd"].startswith("/bin/run-radiance"))
        self.assertEqual(models["video-model"]["metadata"]["capability"], "video")

    def test_a_video_model_is_never_the_default_chat_model(self):
        self.assertEqual(default_model({"a-video": self.registry["video-model"]}), None)
        self.assertEqual(default_model(self.registry), "chat-model")


if __name__ == "__main__":
    unittest.main()
