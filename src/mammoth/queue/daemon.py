"""Compose and run one queue service process under an exclusive state lock."""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
from types import FrameType
from typing import Union

from mammoth.core._filesystem.locking import lock_exclusive
from mammoth.queue.api import QueueApiServer
from mammoth.queue.executor import ActlExecutor, Executor
from mammoth.queue.model import QueueError
from mammoth.queue.sampling import HostSampler, Sampler
from mammoth.queue.service import (
    QueueService,
    ServiceConfig,
    check_prerequisites,
    prepare_private_directory,
)
from mammoth.queue.store import JobStore

logger = logging.getLogger(__name__)


class ServiceAlreadyRunningError(QueueError):
    """Raised when another service instance owns this state directory."""


def hold_state_lock(state_dir: Path) -> int:
    """Take the exclusive service lock and return its descriptor, held for life."""
    descriptor = os.open(state_dir / "service.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        lock_exclusive(descriptor)
    except BlockingIOError as error:
        os.close(descriptor)
        raise ServiceAlreadyRunningError(
            f"Another queue service owns {state_dir}; this instance launches nothing"
        ) from error
    return descriptor


def run_service(
    state_dir: Path,
    socket_path: Path,
    config: Union[ServiceConfig, None] = None,
    *,
    sampler: Union[Sampler, None] = None,
    executor: Union[Executor, None] = None,
) -> None:
    """Run until SIGTERM or SIGINT; running jobs keep running after it returns."""
    if sampler is None or executor is None:
        check_prerequisites(("actl", "nvidia-smi", "systemctl"))
    prepare_private_directory(state_dir)
    prepare_private_directory(socket_path.parent)
    lock = hold_state_lock(state_dir)
    store = JobStore(state_dir / "state.db")
    service = QueueService(
        store,
        sampler if sampler is not None else HostSampler(),
        executor if executor is not None else ActlExecutor(),
        config,
    )

    def stop(_signal: int, _frame: Union[FrameType, None]) -> None:
        service.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server = QueueApiServer(socket_path, service)
    server.serve_in_thread()
    try:
        logger.info("Queue service listening on %s", socket_path)
        service.run()
    finally:
        server.close()
        store.close()
        os.close(lock)
