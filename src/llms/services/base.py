"""Interface shared by the per-platform user-service backends."""

from pathlib import Path

from ..errors import ManagerError


class ServiceBackend:
    """One host service manager (systemd user units, launchd agents).

    A backend renders definitions, verifies that what is loaded and running is
    exactly what it rendered, and performs lifecycle actions. It never decides
    *what* to run: ModelManager supplies the argument vector.
    """

    name = "abstract"

    def __init__(self, settings, runner):
        self.settings = settings
        self.runner = runner

    # -- naming ---------------------------------------------------------------
    def unit_name(self, logical):
        """Native identifier for a logical service name (unit or label)."""
        raise NotImplementedError

    def definition_path(self, logical):
        """Where install writes, and verification expects, the definition."""
        raise NotImplementedError

    # -- definitions ------------------------------------------------------------
    def render(self, logical, description, args):
        """Deterministic definition text for this service."""
        raise NotImplementedError

    # -- state ------------------------------------------------------------------
    def verify(self, logical, args, text, *, verify_process=True):
        """Return True if active, False if inactive; ManagerError otherwise."""
        raise NotImplementedError

    def act(self, action, logicals, *, prefix=""):
        """Run start/stop/restart for already-verified services."""
        raise NotImplementedError

    # -- operator guidance ------------------------------------------------------
    def activation_hint(self, paths):
        """Lines telling the operator how to make installed definitions live."""
        raise NotImplementedError

    def logs_command(self, logical, *, follow=False, lines=200):
        """argv that prints (or follows) the service's output."""
        raise NotImplementedError

    def log_path(self, logical):
        """File the service writes to, if the backend uses files."""
        return None

    @staticmethod
    def _require_single_line(value, what="service arguments"):
        if any(c in str(value) for c in "\x00\r\n"):
            raise ManagerError(f"{what} cannot contain control characters")
        return str(value)

    @staticmethod
    def _same_file(a, b):
        return Path(a).resolve(strict=True) == Path(b).resolve(strict=True)
