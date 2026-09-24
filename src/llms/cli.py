"""Argument parsing and presentation for the reusable manager."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from .manager import ManagerError, ModelManager
from .settings import Settings


def parser():
    ap = argparse.ArgumentParser(prog="llms", description="Manage GGUF models and a llama-swap registry.")
    ap.add_argument("--config-dir", help="settings/registry/header directory; default output is isolated here")
    sub = ap.add_subparsers(dest="command")
    init = sub.add_parser("init", help="create missing settings, empty registry, and header; never overwrite")
    init.add_argument("--server", required=True, help="llama-server executable path or command")
    doctor = sub.add_parser("doctor", help="check dependencies and configuration without starting services")
    doctor.add_argument("--live", action="store_true", help="also check running/models API responses")
    add = sub.add_parser("add", help="download or adopt a local GGUF, probe, register, and render")
    add.add_argument("source")
    add.add_argument("target", nargs="?", help="with two positionals: NAME SOURCE")
    add.add_argument("--name")
    add.add_argument("--quant")
    add.add_argument("--file")
    add.add_argument("--revision", default="main", help="HF revision to resolve and pin before download")
    add.add_argument("--no-mmproj", action="store_true")
    add.add_argument("--no-restart", action="store_true", help="offline register/render only; also disables pre-warm")
    add.add_argument("--no-load", action="store_true", help="skip the explicit warm-up request; configured startup preload hooks still apply")
    add.add_argument("--sync-clients", action="store_true", help="explicitly update configured pi models file")
    add.add_argument("--dry-run", action="store_true", help="resolve plan only; may query HF but never downloads/writes")
    add.add_argument("-y", "--yes", action="store_true", help="accepted for scripts; selection is always deterministic")
    inventory = sub.add_parser("ls", help="list registered models and running state")
    inventory.add_argument("--offline", action="store_true", help="registry only; do not query API")
    inventory.add_argument("--unregistered", action="store_true",
                           help="list complete cached GGUF/MLX models no registry entry references (offline)")
    use = sub.add_parser("use", aliases=["load"], help="pre-warm using the real registry model ID")
    use.add_argument("name")
    use.add_argument("--sync-clients", action="store_true",
                     help="after a valid warm-up, update pi's models and make NAME its default model")
    unload = sub.add_parser("unload", help="unload one model, or all running models; llama-swap stays up")
    unload.add_argument("name", nargs="?")
    remove = sub.add_parser("rm", aliases=["remove"], help="remove registry entry and render")
    remove.add_argument("name")
    remove.add_argument("--purge", action="store_true", help="refused in v0.1: exclusive ownership of the shared HF cache cannot be proven")
    remove.add_argument("--no-restart", action="store_true", help="offline edit; running server remains unchanged")
    remove.add_argument("--sync-clients", action="store_true")
    logs = sub.add_parser("logs", help="show the service's recent output (journal on systemd, log file on launchd)")
    logs.add_argument("-f", "--follow", action="store_true", help="keep streaming new output")
    logs.add_argument("-n", "--lines", type=int, default=200, help="lines of history to show first")
    logs.add_argument("--companion", metavar="NAME", help="a configured companion instead of llama-swap")
    for name, help_text in {
        "render": "regenerate llama-swap YAML without restarting",
        "status": "check loaded and advertised models",
        "ensure-clients": "explicitly update pi provider; does not install opencode plugins",
        "install-service": "write a non-overwriting service definition; never load, enable, or start it",
        "install-companions": "write non-overwriting companion service definitions",
        "companions-status": "check companion units and health endpoints",
        "companions-start": "start configured companion model services",
        "companions-stop": "stop configured companion model services",
        "companions-restart": "restart configured companion model services",
        "start": "start the configured user service",
        "stop": "stop the configured user service",
        "restart": "restart the configured user service",
    }.items():
        sub.add_parser(name, help=help_text)
    return ap


def main(argv=None, *, manager_factory=ModelManager):
    """Return an exit code; injectable manager factory for embedding and tests."""
    ap = parser()
    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_help()
        return 0
    try:
        settings = Settings.from_env(args.config_dir)
        if args.command == "init":
            server = str(Path(args.server).expanduser().absolute()) if "/" in args.server or args.server.startswith("~") else args.server
            settings = replace(settings, server=server)
        manager = manager_factory(settings)
        if args.command == "init":
            print(json.dumps({"created": manager.init()}, indent=2))
        elif args.command == "doctor":
            checks = manager.doctor(live=args.live)
            print(json.dumps(checks, indent=2))
            return int(any(not c["ok"] and c["required"] for c in checks))
        elif args.command == "add":
            source, name = args.source, args.name
            if args.target is not None:
                if name and name != args.source:
                    raise ManagerError("registry name given twice")
                source, name = args.target, args.source
            options = {"quant": args.quant, "file": args.file, "no_mmproj": args.no_mmproj, "revision": args.revision}
            if args.dry_run:
                print(json.dumps(manager.plan(source, **options), indent=2))
            else:
                name, entry = manager.add(source, name=name, restart=not args.no_restart,
                                          load=not (args.no_restart or args.no_load), sync_clients=args.sync_clients, **options)
                print(json.dumps({"name": name, "entry": entry}, indent=2))
                if entry.get("embedded_sampling"):
                    values = ", ".join(f"{k}={v}" for k, v in entry["embedded_sampling"].items())
                    print(f"note: this GGUF embeds sampling defaults that llama.cpp applies unless "
                          f"overridden: {values}", file=sys.stderr)
        elif args.command == "ls" and args.unregistered:
            print(json.dumps({"hf_cache": str(settings.hf_cache), "unregistered": manager.unregistered()}, indent=2))
        elif args.command == "ls":
            registry = manager.load_registry()
            running = None if args.offline else manager.running()
            print(json.dumps({"registry": registry, "running": running}, indent=2))
        elif args.command in ("use", "load"):
            if args.sync_clients:
                manager.use_and_sync(args.name)
                print(f"ready: {args.name}; pi default model set in {manager.pi_settings_path}")
            else:
                manager.use(args.name)
                print(f"ready: {args.name}")
        elif args.command == "unload":
            manager.unload(args.name)
            print(f"unloaded: {args.name or 'all running models'}")
        elif args.command in ("rm", "remove"):
            manager.remove(args.name, purge=args.purge, restart=not args.no_restart, sync_clients=args.sync_clients)
            print(f"removed: {args.name}")
        elif args.command == "render":
            print(f"wrote {manager.render()}")
        elif args.command == "status":
            print(json.dumps(manager.status(), indent=2))
        elif args.command == "ensure-clients":
            print(f"wrote {manager.ensure_clients()}")
        elif args.command == "install-service":
            path = manager.install_service()
            print(f"wrote {path}; no service-manager changes made")
            for line in manager.backend.activation_hint([path]):
                print(line)
        elif args.command == "install-companions":
            paths = manager.install_companions()
            print(json.dumps({"written": [str(path) for path in paths]}, indent=2))
            for line in manager.backend.activation_hint(paths):
                print(line)
        elif args.command == "logs":
            unit = settings.unit
            if args.companion:
                if args.companion not in settings.companions:
                    raise ManagerError(f"unknown companion: {args.companion}")
                unit = settings.companions[args.companion].get("unit", f"llms-{args.companion}.service")
            command = manager.backend.logs_command(unit, follow=args.follow, lines=args.lines)
            log = manager.backend.log_path(unit)
            if log is not None and not log.exists():
                raise ManagerError(f"no log file yet at {log}; start the service first")
            try:
                return subprocess.run(command).returncode
            except KeyboardInterrupt:
                return 130
        elif args.command == "companions-status":
            print(json.dumps(manager.companion_status(), indent=2))
        elif args.command.startswith("companions-"):
            action = args.command.removeprefix("companions-")
            manager.companion_service(action)
            print(f"companions-{action}: {', '.join(sorted(settings.companions))}")
        else:
            manager.service(args.command)
            if args.command in ("start", "restart"):
                manager.wait_ready()
            print(f"{args.command}: {manager.backend.unit_name(settings.unit)}")
        return 0
    except (ManagerError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
