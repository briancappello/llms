"""launchd per-user agent backend (macOS).

Lifecycle (design D2): the plist sets KeepAlive = {SuccessfulExit: false},
which implies RunAtLoad, so loading a job starts it:

    start    launchctl bootstrap gui/<uid> <plist>     (kickstart if loaded)
    stop     launchctl bootout   gui/<uid>/<label>     (never respawns)
    restart  launchctl kickstart -k gui/<uid>/<label>  (bootstrap if unloaded)

Plists live in the config dir, not ~/Library/LaunchAgents, so installing never
starts anything at login; copying the plist there is the explicit "enable".

State comes from `launchctl print`, which Apple documents as unstable. It is
parsed strictly and any surprise refuses the operation (design D4).
"""

import os
from pathlib import Path
import plistlib
import shlex
import subprocess

from ..errors import ManagerError
from .base import ServiceBackend
from .systemd import verify_live_process

NOT_FOUND_EXIT = 113  # launchctl print: "Could not find service"


def parse_print(text):
    """Top-level scalar keys and list blocks of a `launchctl print` job dump.

    Only one-tab-indented `key = value` lines and `key = {` blocks are read;
    nested blocks (coalitions, environments) are skipped. Duplicate top-level
    keys, unbalanced braces, or a missing header raise ManagerError.
    """
    lines = text.splitlines()
    if not lines or not lines[0].endswith(" = {") or lines[-1] != "}":
        raise ManagerError("cannot verify launchd job: malformed launchctl print output")
    scalars, blocks = {}, {}
    body = lines[1:-1]
    i = 0
    while i < len(body):
        line = body[i]
        i += 1
        if not line.strip():
            continue
        if not line.startswith("\t") or line.startswith("\t\t"):
            raise ManagerError("cannot verify launchd job: unexpected indentation in launchctl print output")
        key, separator, value = line[1:].partition(" = ")
        if not separator or not key:
            raise ManagerError("cannot verify launchd job: malformed launchctl print line")
        if key in scalars or key in blocks:
            raise ManagerError(f"cannot verify launchd job: duplicate key {key!r} in launchctl print output")
        if value == "{":
            items = []
            while True:
                if i >= len(body):
                    raise ManagerError("cannot verify launchd job: unterminated block in launchctl print output")
                inner = body[i]
                i += 1
                if inner == "\t}":
                    break
                if not inner.startswith("\t\t"):
                    raise ManagerError("cannot verify launchd job: malformed block in launchctl print output")
                items.append(inner[2:])
            blocks[key] = items
        else:
            scalars[key] = value
    return scalars, blocks


class LaunchdBackend(ServiceBackend):
    name = "launchd"

    # -- naming -------------------------------------------------------------------
    def unit_name(self, logical):
        from . import launchd_label

        return launchd_label(logical)

    def definition_path(self, logical):
        return self.settings.service_dir / f"{self.unit_name(logical)}.plist"

    def domain(self):
        return f"gui/{os.getuid()}"

    def target(self, logical):
        return f"{self.domain()}/{self.unit_name(logical)}"

    def home(self):
        return Path(os.environ.get("HOME", str(Path.home())))

    def launch_agents_path(self, logical):
        return self.home() / "Library" / "LaunchAgents" / f"{self.unit_name(logical)}.plist"

    def log_path(self, logical):
        name = self.unit_name(logical).removeprefix("llms.")
        return self.home() / "Library" / "Logs" / "llms" / f"{name}.log"

    def service_path(self):
        return ":".join(str(Path(part).expanduser()) for part in self.settings.service_path.split(":"))

    # -- definitions --------------------------------------------------------------
    def render(self, logical, description, args):
        for value in args:
            self._require_single_line(value)
        log = str(self.log_path(logical))
        job = {
            "Label": self.unit_name(logical),
            "ProgramArguments": [str(a) for a in args],
            # Restart only after an abnormal exit; bootout and clean exits stay down.
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": log,
            "StandardErrorPath": log,
            "EnvironmentVariables": {"PATH": self.service_path()},
            # Background agents are throttled (CPU, I/O, timers); inference is not background work.
            "ProcessType": "Interactive",
        }
        return plistlib.dumps(job, sort_keys=True).decode("utf-8")

    # -- state --------------------------------------------------------------------
    def _print(self, logical):
        result = self.runner([self.settings.launchctl, "print", self.target(logical)],
                             capture_output=True, text=True, timeout=10)
        if result.returncode == NOT_FOUND_EXIT or "Could not find service" in (result.stderr or "") + (result.stdout or ""):
            return None
        if result.returncode:
            raise ManagerError(f"cannot inspect launchd job: {(result.stderr or result.stdout).strip()}")
        return parse_print(result.stdout)

    def verify(self, logical, args, text, *, verify_process=True):
        path = self.definition_path(logical)
        hint = f"install it first with `llms install-service` (expected {path})"
        try:
            if path.read_text() != text:
                raise ManagerError(f"service refused: {path} is not the unmodified definition generated by llms")
            loaded = self._print(logical)
            if loaded is None:
                return False
            scalars, blocks = loaded
            for key in ("path", "state", "program"):
                if key not in scalars:
                    raise ManagerError(f"cannot verify launchd job: launchctl print lacks {key!r}")
            if "arguments" not in blocks:
                raise ManagerError("cannot verify launchd job: launchctl print lacks 'arguments'")
            source = Path(scalars["path"])
            agents = self.launch_agents_path(logical)
            if not source.is_absolute():
                raise ManagerError("service refused: loaded job has no absolute definition path")
            if self._same_file(source, path):
                pass
            elif (source.exists() and agents.exists() and self._same_file(source, agents)
                    and agents.read_text() == text):
                pass
            else:
                raise ManagerError("service refused: loaded job path belongs to a different definition source")
            if scalars["program"] != str(args[0]) or blocks["arguments"] != [str(a) for a in args]:
                raise ManagerError("service refused: loaded program/arguments do not match generated command")
            state, pid = scalars["state"], scalars.get("pid")
            if state == "running":
                if pid is None or not pid.isdecimal() or int(pid) <= 0:
                    raise ManagerError("service refused: running job has no verifiable main process")
                if verify_process:
                    verify_live_process(int(pid), args)
                return True
            if state == "not running" and pid is None:
                return False
            raise ManagerError(f"service refused: unstable or unknown launchd state {state!r}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ManagerError(f"cannot verify installed/live service binding: {exc}; {hint}") from exc

    def act(self, action, logicals, *, prefix=""):
        for logical in logicals:
            loaded = self._print(logical) is not None
            if action == "stop":
                if not loaded:
                    continue
                command = [self.settings.launchctl, "bootout", self.target(logical)]
            elif not loaded:
                self.log_path(logical).parent.mkdir(parents=True, exist_ok=True)
                command = [self.settings.launchctl, "bootstrap", self.domain(), str(self.definition_path(logical))]
            elif action == "restart":
                command = [self.settings.launchctl, "kickstart", "-k", self.target(logical)]
            else:
                command = [self.settings.launchctl, "kickstart", self.target(logical)]
            result = self.runner(command, capture_output=True, text=True)
            if result.returncode:
                raise ManagerError(f"{prefix}{action} failed: {(result.stderr or result.stdout).strip()}")

    # -- operator guidance --------------------------------------------------------
    def activation_hint(self, paths):
        agents = self.home() / "Library" / "LaunchAgents"
        lines = ["Nothing was loaded. Start it with `llms start` (or `llms companions-start`), which runs:"]
        lines += [shlex.join([self.settings.launchctl, "bootstrap", self.domain(), str(p)]) for p in paths]
        lines.append("To also start at login, copy the definition into ~/Library/LaunchAgents:")
        lines += [shlex.join(["cp", str(p), str(agents) + "/"]) for p in paths]
        return lines

    def logs_command(self, logical, *, follow=False, lines=200):
        return ["tail", "-n", str(lines), *(["-F"] if follow else []), str(self.log_path(logical))]

    def domain_available(self):
        result = self.runner([self.settings.launchctl, "print", self.domain()],
                             capture_output=True, text=True, timeout=10)
        return result.returncode == 0
