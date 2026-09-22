import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "build-vllm"


class BuildVllmTest(unittest.TestCase):
    def run_script(self, *args: str) -> str:
        env = os.environ.copy()
        env["HOME"] = "/tmp/vllm-test-home"
        return subprocess.check_output(
            [SCRIPT, *args], cwd=ROOT, env=env, text=True
        )

    def test_print_config_loads_pins(self) -> None:
        output = self.run_script("--print-config")
        self.assertIn("source=/tmp/vllm-test-home/dev/vllm", output)
        self.assertIn("arch=gfx1201", output)
        self.assertIn("openmpi=4.1.8", output)

    def test_command_line_overrides_upgradeable_pins(self) -> None:
        output = self.run_script(
            "--ref", "v1.2.3", "--arch", "gfx1100", "--jobs", "7", "--print-config"
        )
        self.assertIn("ref=v1.2.3", output)
        self.assertIn("arch=gfx1100", output)
        self.assertIn("jobs=7", output)


if __name__ == "__main__":
    unittest.main()
