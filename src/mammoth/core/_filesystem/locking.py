"""Descriptor locks with stable OS-error semantics, implemented by portalocker."""

import errno

import portalocker


def lock_exclusive(descriptor: int, *, blocking: bool = False) -> None:
    """Acquire ownership, optionally waiting, retaining the caller's descriptor."""
    flags = portalocker.LOCK_EX
    if not blocking:
        flags |= portalocker.LOCK_NB
    try:
        portalocker.lock(descriptor, flags)
    except portalocker.exceptions.AlreadyLocked as error:
        raise BlockingIOError(errno.EAGAIN, "file is already locked") from error
    except portalocker.exceptions.LockException as error:
        # Preserve errno-based recovery in existing lease/work-store callers.
        if isinstance(error.__cause__, OSError):
            raise error.__cause__ from None
        raise OSError(errno.EIO, str(error)) from error


def unlock(descriptor: int) -> None:
    """Release a descriptor lock without closing its file."""
    portalocker.unlock(descriptor)
