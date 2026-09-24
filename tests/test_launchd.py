"""launchd backend and service-manager settings; hermetic on Linux and macOS."""

import contextlib
import json
from dataclasses import replace
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from llms import ManagerError, ModelManager, Settings
from llms.services import resolve_service_manager
from llms.services.launchd import parse_print

FIXTURES = Path(__file__).parent / "fixtures" / "launchctl"


def fixture(name, **values):
    text = (FIXTURES / f"print-{name}.txt").read_text()
    for key, value in values.items():
        text = text.replace(f"@{key}@", str(value))
    return text


class ServiceManagerSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_auto_resolution_by_platform(self):
        for platform, expected in (("darwin", "launchd"), ("linux", "systemd")):
            with self.subTest(platform=platform):
                self.assertEqual(resolve_service_manager("auto", platform), expected)
        with self.assertRaisesRegex(ValueError, "win32"):
            resolve_service_manager("auto", "win32")
        with self.assertRaisesRegex(ValueError, "auto, systemd, or launchd"):
            resolve_service_manager("upstart", "linux")
        for platform, expected in (("darwin", "launchd"), ("linux", "systemd")):
            with self.subTest(platform=platform), patch("llms.services.sys.platform", platform):
                settings = Settings(self.root / platform)
                self.assertEqual(settings.resolved_service_manager, expected)
                self.assertEqual(settings.service_dir.name, expected)
        with patch("llms.services.sys.platform", "sunos5"), self.assertRaisesRegex(ValueError, "sunos5"):
            Settings(self.root / "other")

    def test_from_env_default_service_dirs(self):
        env = {"HOME": str(self.root), "XDG_CONFIG_HOME": str(self.root / "xdg")}
        with patch("llms.services.sys.platform", "linux"):
            self.assertEqual(Settings.from_env(environ=env).service_dir, self.root / "xdg" / "systemd" / "user")
        with patch("llms.services.sys.platform", "darwin"):
            self.assertEqual(Settings.from_env(environ=env).service_dir, self.root / "xdg" / "llms" / "launchd")
            pinned = Settings.from_env(environ={**env, "LLMS_SERVICE_MANAGER": "systemd"})
            self.assertEqual(pinned.service_dir, self.root / "xdg" / "systemd" / "user")

    def test_legacy_and_logical_unit_names(self):
        for manager, unit, filename in (("systemd", "llama-swap.service", "llama-swap.service"),
                                        ("systemd", "llama-swap", "llama-swap.service"),
                                        ("launchd", "llama-swap.service", "llms.llama-swap.plist"),
                                        ("launchd", "llama-swap", "llms.llama-swap.plist")):
            with self.subTest(manager=manager, unit=unit):
                settings = Settings(self.root / "c", service_manager=manager, unit=unit)
                self.assertEqual(settings.definition_name(settings.unit), filename)
        for unit in ("../bad.service", "-x", "a/b", ".service", "sp ace", ""):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                Settings(self.root / "c", service_manager="launchd", unit=unit)

    def test_service_path_and_distinct_launchd_paths(self):
        with self.assertRaisesRegex(ValueError, "service_path"):
            Settings(self.root / "c", service_manager="launchd", service_path="relative/bin:/usr/bin")
        settings = Settings(self.root / "c", service_manager="launchd")
        plist = settings.service_dir / "llms.llama-swap.plist"
        with self.assertRaisesRegex(ValueError, "distinct"):
            replace(settings, client_path=plist)
        with self.assertRaisesRegex(ValueError, "duplicate companion"):
            Settings(self.root / "c", service_manager="launchd", companions={
                "a": {"command": ["x"], "url": "http://127.0.0.1:1", "unit": "same"},
                "b": {"command": ["x"], "url": "http://127.0.0.1:2", "unit": "same.service"}})


class ParsePrintTests(unittest.TestCase):
    def test_running_fixture(self):
        text = fixture("running", PLIST="/cfg/llms.x.plist", LOG="/log", LABEL="llms.x", UID=501,
                       PROGRAM="/bin/sleep", ARGUMENTS="\t\t/bin/sleep\n\t\ta b\n\t\t", PID=4321)
        scalars, blocks = parse_print(text)
        self.assertEqual(scalars["state"], "running")
        self.assertEqual(scalars["pid"], "4321")
        self.assertEqual(scalars["path"], "/cfg/llms.x.plist")
        self.assertEqual(blocks["arguments"], ["/bin/sleep", "a b", ""])
        # nested coalition "state = active" must not leak into the top level
        self.assertNotIn("active", scalars.values())

    def test_not_running_fixture_has_no_pid(self):
        text = fixture("notrunning", PLIST="/p", LOG="/l", LABEL="llms.x", UID=501,
                       PROGRAM="/bin/sh", ARGUMENTS="\t\t/bin/sh")
        scalars, _ = parse_print(text)
        self.assertEqual(scalars["state"], "not running")
        self.assertNotIn("pid", scalars)

    def test_malformed_variants_refused(self):
        good = fixture("notrunning", PLIST="/p", LOG="/l", LABEL="llms.x", UID=501,
                       PROGRAM="/bin/sh", ARGUMENTS="\t\t/bin/sh")
        variants = {
            "empty": "",
            "no header": good.split("\n", 1)[1],
            "no footer": good.rstrip().rstrip("}"),
            "duplicate": good.replace("\tstate = not running\n", "\tstate = not running\n\tstate = running\n"),
            "unterminated": good.replace("\t}\n\n\tstdout path", "\n\tstdout path", 1),
            "bad indent": good.replace("\tstate = not running", "  state = not running"),
            "no separator": good.replace("\tstate = not running", "\tstate not running"),
        }
        for name, text in variants.items():
            with self.subTest(name=name), self.assertRaisesRegex(ManagerError, "cannot verify"):
                parse_print(text)


class LaunchdBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        patcher = patch.dict(os.environ, {"HOME": str(self.home)})
        patcher.start()
        self.addCleanup(patcher.stop)
        port_probe = patch("llms.manager.socket.create_connection", side_effect=ConnectionRefusedError)
        port_probe.start()
        self.addCleanup(port_probe.stop)
        self.binary = self.root / "bin with spaces" / "llama-swap"
        self.binary.parent.mkdir()
        self.binary.write_text("#!/bin/sh\n")
        self.binary.chmod(0o755)
        self.server = self.root / "bin" / "pooling-server"
        self.server.parent.mkdir()
        self.server.write_text("#!/bin/sh\n")
        self.server.chmod(0o755)
        self.settings = Settings(self.root / "manager", service_manager="launchd", swap_binary=str(self.binary),
                                 companions={"embeddings": {"command": [str(self.server), "--port", "8011"],
                                                            "url": "http://127.0.0.1:8011"}})
        self.runner = Mock(side_effect=self.run_mock)
        self.opener = Mock(side_effect=AssertionError("unexpected network"))
        self.manager = ModelManager(self.settings, runner=self.runner, opener=self.opener)
        self.argv = [str(self.binary), "-config", str(self.settings.output), "-listen", self.settings.listen]
        self.jobs = {}      # label -> state dict for launchctl print
        self.actions = []
        self.action_result = SimpleNamespace(returncode=0, stdout="", stderr="")
        self.uid = os.getuid()

    # -- fake launchctl ------------------------------------------------------------
    def job(self, label, *, argv=None, state="not running", pid=None, path=None):
        path = path or self.settings.service_dir / f"{label}.plist"
        self.jobs[label] = dict(argv=argv or self.argv, state=state, pid=pid, path=path)

    def run_mock(self, command, **kwargs):
        if command[1] == "print":
            self.assertEqual(kwargs.get("timeout"), 10)
            target = command[2]
            label = target.rsplit("/", 1)[-1]
            if target == f"gui/{self.uid}":
                return SimpleNamespace(returncode=0, stdout="gui/501 = {\n}\n", stderr="")
            if label not in self.jobs:
                return SimpleNamespace(returncode=113, stdout=fixture("notfound", LABEL=label, UID=self.uid), stderr="")
            job = self.jobs[label]
            name = "running" if job["state"] == "running" else "notrunning"
            text = fixture(name, PLIST=job["path"], LOG="/log", LABEL=label, UID=self.uid,
                           PROGRAM=job["argv"][0], ARGUMENTS="\n".join("\t\t" + a for a in job["argv"]),
                           PID=job["pid"])
            if job["state"] not in ("running", "not running"):
                text = text.replace("\tstate = not running", f"\tstate = {job['state']}")
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        self.actions.append(command)
        return self.action_result

    @contextlib.contextmanager
    def live(self, argv=None, executable=None, error=None):
        self.job("llms.llama-swap", state="running", pid="4321")
        def read_argv(pid):
            if error:
                raise error
            return list(argv or self.argv)
        with patch("llms._proc.argv", read_argv), \
             patch("llms._proc.exe", lambda pid: Path(executable or self.binary).resolve()):
            yield

    # -- 4.2 plist generation ------------------------------------------------------
    def test_plist_golden(self):
        path = self.manager.install_service()
        self.assertEqual(path, self.settings.service_dir / "llms.llama-swap.plist")
        job = plistlib.loads(path.read_bytes())
        self.assertEqual(job, {
            "Label": "llms.llama-swap",
            "ProgramArguments": self.argv,
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(self.home / "Library" / "Logs" / "llms" / "llama-swap.log"),
            "StandardErrorPath": str(self.home / "Library" / "Logs" / "llms" / "llama-swap.log"),
            "EnvironmentVariables": {"PATH": f"{self.home}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
            "ProcessType": "Interactive",
        })
        [companion] = self.manager.install_companions()
        self.assertEqual(companion.name, "llms.embeddings.plist")
        self.assertEqual(plistlib.loads(companion.read_bytes())["ProgramArguments"], [str(self.server), "--port", "8011"])
        self.assertEqual(plistlib.loads(companion.read_bytes())["StandardOutPath"],
                         str(self.home / "Library" / "Logs" / "llms" / "embeddings.log"))
        # deterministic: regenerating yields identical bytes
        self.assertEqual(self.manager._service_definition()[1], path.read_text())

    @unittest.skipUnless(shutil.which("plutil"), "plutil is macOS-only")
    def test_plutil_lint(self):
        path = self.manager.install_service()
        result = subprocess.run(["plutil", "-lint", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    # -- 4.3 install -------------------------------------------------------------
    def test_install_writes_without_launchctl_and_never_overwrites(self):
        path = self.manager.install_service()
        self.assertTrue(path.is_file())
        with self.assertRaises(FileExistsError):
            self.manager.install_service()
        self.manager.install_companions()
        with self.assertRaises(FileExistsError):
            self.manager.install_companions()
        self.runner.assert_not_called()
        hint = "\n".join(self.manager.backend.activation_hint([path]))
        self.assertIn(f"launchctl bootstrap gui/{self.uid} {path}", hint)
        self.assertIn("Library/LaunchAgents", hint)

    # -- 4.5 verification --------------------------------------------------------
    def test_not_loaded_is_inactive_and_start_bootstraps(self):
        path = self.manager.install_service()
        self.assertFalse(self.manager._verify_service())
        self.manager.service("start")
        self.assertEqual(self.actions, [["launchctl", "bootstrap", f"gui/{self.uid}", str(path)]])
        self.assertTrue((self.home / "Library" / "Logs" / "llms").is_dir())

    def test_modified_definition_refused_before_launchctl(self):
        path = self.manager.install_service()
        path.write_text(path.read_text().replace("Interactive", "Background"))
        with self.assertRaisesRegex(ManagerError, "not the unmodified definition generated by llms"):
            self.manager.service("start")
        self.runner.assert_not_called()

    def test_missing_definition_refused(self):
        with self.assertRaisesRegex(ManagerError, "install it first"):
            self.manager.service("start")
        self.runner.assert_not_called()

    def test_other_source_path_refused(self):
        self.manager.install_service()
        other = self.root / "elsewhere" / "llms.llama-swap.plist"
        other.parent.mkdir()
        other.write_text(self.manager.backend.definition_path("llama-swap.service").read_text())
        self.job("llms.llama-swap", path=other)
        with self.assertRaisesRegex(ManagerError, "different definition source"):
            self.manager.service("restart")
        self.assertEqual(self.actions, [])

    def test_launch_agents_copy_accepted_only_if_identical(self):
        path = self.manager.install_service()
        agents = self.home / "Library" / "LaunchAgents" / path.name
        agents.parent.mkdir(parents=True)
        agents.write_text(path.read_text())
        self.job("llms.llama-swap", path=agents)
        self.manager.service("stop")
        self.assertEqual(self.actions, [["launchctl", "bootout", f"gui/{self.uid}/llms.llama-swap"]])
        agents.write_text(path.read_text().replace("Interactive", "Standard"))
        with self.assertRaisesRegex(ManagerError, "different definition source"):
            self.manager.service("stop")

    def test_loaded_arguments_mismatch_refused(self):
        self.manager.install_service()
        for index, value in ((0, "/other/swap"), (2, "/other/config.yaml"), (4, "127.0.0.1:9999")):
            argv = self.argv.copy()
            argv[index] = value
            with self.subTest(index=index):
                self.job("llms.llama-swap", argv=argv)
                with self.assertRaisesRegex(ManagerError, "loaded program/arguments"):
                    self.manager.service("restart")
        self.assertEqual(self.actions, [])

    def test_unknown_or_inconsistent_state_refused(self):
        self.manager.install_service()
        for state, pid in (("spawn scheduled", None), ("running", None), ("running", "0"), ("not running", "12")):
            with self.subTest(state=state, pid=pid):
                self.job("llms.llama-swap", state=state, pid=pid)
                if state == "not running":
                    # a pid line on a not-running job cannot come from the fixture; inject it
                    original = self.run_mock
                    def with_pid(command, **kwargs):
                        result = original(command, **kwargs)
                        if command[1] == "print" and command[2].endswith("llms.llama-swap"):
                            result.stdout = result.stdout.replace("\tlast exit code", "\tpid = 12\n\tlast exit code")
                        return result
                    self.runner.side_effect = with_pid
                with self.assertRaises(ManagerError):
                    self.manager.service("restart")
                self.runner.side_effect = self.run_mock
        self.assertEqual(self.actions, [])

    def test_live_argv_and_exe_drift_refused(self):
        self.manager.install_service()
        drifted = self.argv.copy()
        drifted[2] = "/old/config.yaml"
        with self.live(argv=drifted), self.assertRaisesRegex(ManagerError, "live process binding"):
            self.manager.service("stop")
        other = self.root / "other-exe"
        other.write_text("")
        with self.live(executable=other), self.assertRaisesRegex(ManagerError, "live process binding"):
            self.manager.service("stop")
        with self.live(error=PermissionError("denied")), self.assertRaisesRegex(ManagerError, "cannot verify"):
            self.manager.service("stop")
        self.assertEqual(self.actions, [])

    def test_launchctl_errors_and_timeouts_refuse(self):
        self.manager.install_service()
        self.runner.side_effect = lambda command, **kw: SimpleNamespace(returncode=5, stdout="", stderr="Input/output error")
        with self.assertRaisesRegex(ManagerError, "cannot inspect launchd job"):
            self.manager.service("start")
        self.runner.side_effect = subprocess.TimeoutExpired("launchctl", 10)
        with self.assertRaisesRegex(ManagerError, "cannot verify"):
            self.manager.service("start")

    # -- 4.6 lifecycle ------------------------------------------------------------
    def test_lifecycle_argv_when_running(self):
        self.manager.install_service()
        target = f"gui/{self.uid}/llms.llama-swap"
        with self.live():
            for action, expected in (("start", ["launchctl", "kickstart", target]),
                                     ("restart", ["launchctl", "kickstart", "-k", target]),
                                     ("stop", ["launchctl", "bootout", target])):
                with self.subTest(action=action):
                    self.actions.clear()
                    self.manager.service(action)
                    self.assertEqual(self.actions, [expected])

    def test_lifecycle_when_unloaded(self):
        path = self.manager.install_service()
        self.manager.service("stop")
        self.assertEqual(self.actions, [])
        self.manager.service("restart")
        self.assertEqual(self.actions, [["launchctl", "bootstrap", f"gui/{self.uid}", str(path)]])

    def test_action_failure_propagates(self):
        self.manager.install_service()
        self.action_result = SimpleNamespace(returncode=5, stdout="", stderr="Bootstrap failed: 5")
        with self.assertRaisesRegex(ManagerError, "start failed: Bootstrap failed: 5"):
            self.manager.service("start")

    def test_companions_verify_then_act(self):
        [path] = self.manager.install_companions()
        self.manager.companion_service("start")
        self.assertEqual(self.actions, [["launchctl", "bootstrap", f"gui/{self.uid}", str(path)]])
        path.write_text("tampered")
        self.actions.clear()
        with self.assertRaisesRegex(ManagerError, "not the unmodified definition"):
            self.manager.companion_service("stop")
        self.assertEqual(self.actions, [])
        path.unlink()
        self.manager.install_companions()
        self.job("llms.embeddings", argv=[str(self.server), "--port", "8011"], state="running", pid="99")
        self.opener.side_effect = None
        self.opener.return_value = MagicMock()
        self.opener.return_value.__enter__.return_value.status = 200
        self.assertEqual(self.manager.companion_status(), [{
            "name": "embeddings", "unit": "llms.embeddings", "url": "http://127.0.0.1:8011",
            "active": True, "healthy": True, "error": None}])

    # -- 5.3 doctor ----------------------------------------------------------------
    def doctor_checks(self, settings, *, domain_rc=0, registry=None):
        def run(command, **kwargs):
            return SimpleNamespace(returncode=domain_rc, stdout="", stderr="")
        manager = ModelManager(settings, runner=Mock(side_effect=run), opener=self.opener)
        manager.init()
        if registry is not None:
            settings.registry.write_text(json.dumps(registry))
        return {c["check"]: c for c in manager.doctor()}

    def test_doctor_reports_manager_domain_and_launchd_path(self):
        tools = self.root / "tools"
        tools.mkdir()
        for name in ("launchctl", "llama-server"):
            (tools / name).write_text("#!/bin/sh\n")
            (tools / name).chmod(0o755)
        settings = replace(self.settings, launchctl=str(tools / "launchctl"), server="llama-server",
                           service_path=f"{tools}:/usr/bin",
                           engines={"missing": {"server": "not-a-real-server-binary"},
                                    "absolute": {"server": "/opt/x/llama-server"}})
        # an entry with no engine renders through the top-level server, so it is checked
        checks = self.doctor_checks(settings, registry={"m": {"capability": "chat", "path": "/m.gguf", "ctx": 8}})
        self.assertTrue(checks["service_manager"]["ok"])
        self.assertIn("launchd via", checks["service_manager"]["detail"])
        self.assertTrue(checks["launchd-domain"]["ok"])
        self.assertTrue(checks["service-path:server"]["ok"])
        failing = checks["service-path:engine:missing"]
        self.assertFalse(failing["ok"])
        self.assertIn(f"{tools}:/usr/bin", failing["detail"])
        self.assertNotIn("service-path:engine:absolute", checks)
        self.assertNotIn("service-path:server", self.doctor_checks(settings, registry={}))
        missing_domain = self.doctor_checks(settings, domain_rc=113)["launchd-domain"]
        self.assertFalse(missing_domain["ok"])
        self.assertIn("console", missing_domain["detail"])

    def test_doctor_systemd_has_no_launchd_checks(self):
        checks = self.doctor_checks(replace(self.settings, service_manager="systemd"))
        self.assertIn("systemd via systemctl", checks["service_manager"]["detail"])
        self.assertFalse(any(k.startswith(("launchd", "service-path")) for k in checks))


class LaunchdCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config"
        self.runner = Mock(side_effect=AssertionError("unexpected launchctl"))

    def invoke(self, *args, manager="launchd"):
        import io
        from llms.cli import main

        env = {"HOME": str(self.root), "LLMS_SERVICE_MANAGER": manager, "LLMS_SWAP_BINARY": "/bin/sh"}
        output, error = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(error):
            code = main(["--config-dir", str(self.config), *args],
                        manager_factory=lambda settings: ModelManager(settings, runner=self.runner))
        return code, output.getvalue(), error.getvalue()

    def test_install_service_prints_launchd_steps_without_running_them(self):
        code, output, error = self.invoke("install-service")
        self.assertEqual(code, 0, error)
        plist = self.config / "launchd" / "llms.llama-swap.plist"
        self.assertTrue(plist.is_file())
        self.assertIn(f"launchctl bootstrap gui/{os.getuid()} {plist}", output)
        self.assertIn(f"cp {plist} {self.root}/Library/LaunchAgents/", output)
        self.assertNotIn("systemctl", output)
        self.runner.assert_not_called()

    def test_logs_tails_the_log_file(self):
        log = self.root / "Library" / "Logs" / "llms" / "llama-swap.log"
        code, _, error = self.invoke("logs")
        self.assertEqual(code, 1)
        self.assertIn("no log file yet", error)
        log.parent.mkdir(parents=True)
        log.write_text("one\ntwo\n")
        with patch("llms.cli.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            for args, expected in ((("logs",), ["tail", "-n", "200", str(log)]),
                                   (("logs", "-f", "-n", "5"), ["tail", "-n", "5", "-F", str(log)])):
                with self.subTest(args=args):
                    self.assertEqual(self.invoke(*args)[0], 0)
                    self.assertEqual(run.call_args.args[0], expected)

    def test_logs_follow_really_streams(self):
        import signal
        import threading
        import time

        log = self.root / "Library" / "Logs" / "llms" / "llama-swap.log"
        log.parent.mkdir(parents=True)
        log.write_text("first\n")
        env = {**os.environ, "HOME": str(self.root), "LLMS_SERVICE_MANAGER": "launchd", "LLMS_SWAP_BINARY": "/bin/sh"}
        # Own session, so killing the group also reaps tail(1), which holds the pipe.
        child = subprocess.Popen([sys.executable, "-m", "llms", "--config-dir", str(self.config), "logs", "-f"],
                                 env=env, stdout=subprocess.PIPE, text=True, start_new_session=True)
        def kill_group():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        self.addCleanup(child.stdout.close)
        self.addCleanup(child.wait)
        self.addCleanup(kill_group)
        timer = threading.Timer(10, kill_group)
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertEqual(child.stdout.readline().strip(), "first")
        time.sleep(0.3)
        with log.open("a") as stream:
            stream.write("appended\n")
        self.assertEqual(child.stdout.readline().strip(), "appended")

    def test_logs_companion_and_systemd_journal(self):
        with patch("llms.cli.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            code, _, error = self.invoke("logs", "--companion", "nope", manager="systemd")
            self.assertEqual(code, 1)
            self.assertIn("unknown companion", error)
            self.assertEqual(self.invoke("logs", "-f", manager="systemd")[0], 0)
            self.assertEqual(run.call_args.args[0],
                             ["journalctl", "--user", "-u", "llama-swap.service", "-n", "200", "--no-pager", "-f"])


if __name__ == "__main__":
    unittest.main()
