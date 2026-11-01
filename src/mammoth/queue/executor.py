"""Durable job execution through ``actl``, confined to this one module.

Each job runs in its own transient unit, so stopping or upgrading the queue
service never stops a job. The stable execution name lets a restarted service
find a job whose ``tx:`` ID it never recorded.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Union

from mammoth.compat import DATACLASS_SLOTS
from mammoth.queue.model import QueueError

_COMMAND_TIMEOUT_SECONDS = 180.0
_LOG_LINES = 400
_SUBMITTED = re.compile(r"^submitted (tx:[0-9a-f]{32}):", re.MULTILINE)
_LISTED = re.compile(r"^(tx:[0-9a-f]{32})\s+(\S+)\s+\S+\s+cleanup=", re.MULTILINE)
TERMINAL_EXECUTION_STATUSES = frozenset(
    {"ok", "failed", "timeout", "skipped", "error", "cancelled", "abandoned"}
)


class ExecutorError(QueueError):
    """Raised when ``actl`` fails or its answer cannot be trusted."""


@dataclass(frozen=True, **DATACLASS_SLOTS)
class ExecutionRecord:
    """One execution's status as ACTL reports it."""

    tx_id: str
    status: str
    exit_code: Union[int, None] = None
    started_at: Union[float, None] = None
    finished_at: Union[float, None] = None

    @property
    def is_terminal(self) -> bool:
        """Return whether this execution has finished."""
        return self.status in TERMINAL_EXECUTION_STATUSES


class Executor(Protocol):
    """The four durable-execution operations, plus lookup and log reading."""

    def start(
        self,
        name: str,
        argv: Sequence[str],
        cwd: str,
        env: Mapping[str, str],
        env_files: Sequence[str],
        timeout_seconds: int,
    ) -> str:
        """Start one execution and return its ``tx:`` ID."""
        ...

    def status(self, tx_id: str) -> Union[ExecutionRecord, None]:
        """Return one execution's record, or ``None`` when it was confirmed gone."""
        ...

    def find(self, name: str) -> Union[str, None]:
        """Return the ``tx:`` ID with this name, or ``None`` when confirmed absent."""
        ...

    def cancel(self, tx_id: str) -> None:
        """Stop one execution, retaining its record."""
        ...

    def collect(self, tx_id: str) -> None:
        """Reconcile one execution, marking it abandoned if its registration is gone."""
        ...

    def stderr(self, tx_id: str) -> str:
        """Return the tail of one execution's stderr."""
        ...


class ActlExecutor:
    """Implement :class:`Executor` with the ``actl`` command line."""

    def __init__(self, command: str = "actl") -> None:
        self._command = command

    def start(
        self,
        name: str,
        argv: Sequence[str],
        cwd: str,
        env: Mapping[str, str],
        env_files: Sequence[str],
        timeout_seconds: int,
    ) -> str:
        """Start one execution and return its ``tx:`` ID."""
        arguments = [
            "submit",
            "--transient",
            name,
            "--cwd",
            cwd,
            "--timeout",
            f"{timeout_seconds}s",
        ]
        for key, value in env.items():
            arguments += ["--env", f"{key}={value}"]
        for path in env_files:
            arguments += ["--env-file", path]
        stdout = self._run([*arguments, "--", *argv])
        match = _SUBMITTED.search(stdout)
        if match is None:
            raise ExecutorError(f"Unexpected actl submit output: {stdout!r}")
        return match.group(1)

    def status(self, tx_id: str) -> Union[ExecutionRecord, None]:
        """Return one execution's record, or ``None`` when it was confirmed gone."""
        try:
            stdout = self._run(["status", tx_id])
        except ExecutorError:
            # A failed read is only proof of absence if the full listing agrees.
            if any(listed == tx_id for listed, _ in self._listed()):
                raise
            return None
        try:
            record = json.loads(stdout)
            return ExecutionRecord(
                tx_id=tx_id,
                status=str(record["status"]),
                exit_code=record.get("exit_code"),
                started_at=_timestamp(record.get("started_at")),
                finished_at=_timestamp(record.get("finished_at")),
            )
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutorError(f"Unexpected actl status output: {error!r}") from error

    def find(self, name: str) -> Union[str, None]:
        """Return the ``tx:`` ID with this name, or ``None`` when confirmed absent."""
        for tx_id, listed_name in self._listed():
            if listed_name == name:
                return tx_id
        return None

    def cancel(self, tx_id: str) -> None:
        """Stop one execution, retaining its record."""
        self._run(["cancel", tx_id])

    def collect(self, tx_id: str) -> None:
        """Reconcile one execution, marking it abandoned if its registration is gone."""
        self._run(["reconcile", tx_id])

    def stderr(self, tx_id: str) -> str:
        """Return the tail of one execution's stderr."""
        return self._run(["logs", tx_id, "--stderr", "--lines", str(_LOG_LINES)])

    def _listed(self) -> list[tuple[str, str]]:
        return [(match[0], match[1]) for match in _LISTED.findall(self._run(["list"]))]

    def _run(self, arguments: Sequence[str]) -> str:
        try:
            result = subprocess.run(
                [self._command, *arguments],
                capture_output=True,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            # Never format the error itself: a timeout's text holds argv and its env values.
            raise ExecutorError(f"actl {arguments[0]} failed: {type(error).__name__}") from None
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ExecutorError(f"actl {arguments[0]} failed: {detail}")
        return result.stdout


def _timestamp(value: Union[str, None]) -> Union[float, None]:
    return datetime.fromisoformat(value).timestamp() if value else None
