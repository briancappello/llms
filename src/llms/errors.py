"""Exception types shared by the manager and the service backends."""


class ManagerError(RuntimeError):
    """An operation failed without a successful service/configuration outcome."""
