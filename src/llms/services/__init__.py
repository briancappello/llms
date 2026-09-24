"""Per-platform user-service backends behind one interface."""

import sys

from ..errors import ManagerError
from .base import ServiceBackend
from .launchd import LaunchdBackend
from .systemd import SystemdBackend

BACKENDS = {"systemd": SystemdBackend, "launchd": LaunchdBackend}


def resolve_service_manager(value, platform=None):
    """Map a service_manager setting to a concrete backend name."""
    platform = sys.platform if platform is None else platform
    if value == "auto":
        if platform == "darwin":
            return "launchd"
        if platform.startswith("linux"):
            return "systemd"
        raise ValueError(f"no supported service manager on platform {platform!r}; set service_manager explicitly")
    if value not in ("systemd", "launchd"):
        raise ValueError(f"service_manager must be auto, systemd, or launchd, not {value!r}")
    return value


def systemd_unit(logical):
    return logical if logical.endswith(".service") else f"{logical}.service"


def launchd_label(logical):
    """llama-swap -> llms.llama-swap; llms-embeddings(.service) -> llms.embeddings."""
    return "llms." + logical.removesuffix(".service").removeprefix("llms-")


def definition_filename(manager, logical):
    return systemd_unit(logical) if manager == "systemd" else launchd_label(logical) + ".plist"


def backend_for(settings, runner):
    try:
        return BACKENDS[settings.resolved_service_manager](settings, runner)
    except KeyError:
        raise ManagerError(f"service manager {settings.service_manager!r} is not available") from None


__all__ = ["LaunchdBackend", "ServiceBackend", "SystemdBackend", "backend_for", "definition_filename",
           "launchd_label", "resolve_service_manager", "systemd_unit"]
