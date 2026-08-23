"""Spool-directory job model, submission, listing, cancellation, and journal.

A job is an opaque, caller-final argv plus the run names it expects to
produce, a device requirement, and submission metadata. Mammoth never parses
or interprets a job's argv; see ``docs/ARCHITECTURE.md`` for the full queue
component boundary and on-disk spool layout this module implements.

Submitted jobs live as individual JSON files under
``QueueLayout(entry).pending_dir``, named ``<sequence>-<job-id>.json`` so
lexicographic directory order is FIFO submission order regardless of clock
resolution. :mod:`mammoth.queue.serve` claims a pending job by renaming its
file into one device lane's ``claimed_lane_dir``; that rename is the single
atomic exclusive-claim operation two contending lanes race on. Terminal
outcomes are appended to one shared, append-only ``journal.jsonl`` completion
journal that multiple concurrent lane processes may write to; each record is
written with one raw ``os.write()`` call under ``O_APPEND`` so concurrent
appends from independent processes never interleave on local filesystems,
and each record is self-contained (no cross-record hash chain), so no
cross-process coordination is required to append one.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from mammoth.core.artifacts import artifact_open_flags, atomic_write_json
from mammoth.core.identity import validate_device_spec, validate_run_name
from mammoth.core.layout import QueueLayout

QUEUE_SCHEMA_VERSION = 1
QUEUE_JOURNAL_SCHEMA_VERSION = 1

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ImmutableJsonValue = (
    None
    | bool
    | int
    | float
    | str
    | tuple["ImmutableJsonValue", ...]
    | Mapping[str, "ImmutableJsonValue"]
)
JobStatus = Literal["completed", "failed", "interrupted"]

_JOB_ID_MAX_LENGTH = 128
_MAX_METADATA_DEPTH = 32


class QueueError(RuntimeError):
    """Base error for spool submission, listing, cancellation, and the journal."""


class JobNotFoundError(QueueError):
    """Raised when a caller-supplied job ID names no queued or terminal job."""


class JobAlreadyClaimedError(QueueError):
    """Raised when ``cancel_job`` targets a job that a lane already claimed."""


@dataclass(frozen=True, slots=True)
class Job:
    """One immutable spooled job: opaque argv, device requirement, and metadata."""

    schema_version: int
    job_id: str
    sequence: int
    argv: tuple[str, ...]
    expected_run_names: tuple[str, ...]
    device: str
    submitted_at: str
    cwd: str | None
    metadata: Mapping[str, ImmutableJsonValue]

    def __post_init__(self) -> None:
        if self.schema_version != QUEUE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported queue job schema version: "
                f"{self.schema_version!r}; expected {QUEUE_SCHEMA_VERSION}."
            )
        _validate_job_id(self.job_id)
        _validate_sequence(self.sequence)
        argv = _validate_argv(self.argv)
        object.__setattr__(self, "argv", argv)
        run_names = _validate_run_names(self.expected_run_names)
        object.__setattr__(self, "expected_run_names", run_names)
        validate_device_spec(self.device)
        _parse_utc_time(self.submitted_at)
        if self.cwd is not None and (not isinstance(self.cwd, str) or not self.cwd):
            raise ValueError("Job cwd must be a non-empty string or None.")
        object.__setattr__(self, "metadata", _freeze_json(_validate_json_compatible(self.metadata)))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the JSON payload written to this job's pending spool file."""
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "sequence": self.sequence,
            "argv": list(self.argv),
            "expected_run_names": list(self.expected_run_names),
            "device": self.device,
            "submitted_at": self.submitted_at,
            "cwd": self.cwd,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Job:
        """Validate and reconstruct one job read back from its spool file."""
        if not isinstance(payload, Mapping):
            raise ValueError("queue job must be a JSON object.")
        argv = payload.get("argv")
        if not isinstance(argv, list):
            raise ValueError("queue job argv must be a list of strings.")
        run_names = payload.get("expected_run_names")
        if not isinstance(run_names, list):
            raise ValueError("queue job expected_run_names must be a list of strings.")
        metadata = payload.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, Mapping):
            raise ValueError("queue job metadata must be an object.")
        cwd = payload.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError("queue job cwd must be a string or null.")
        return cls(
            schema_version=_required_int(payload, "schema_version"),
            job_id=_required_string(payload, "job_id"),
            sequence=_required_int(payload, "sequence"),
            argv=tuple(argv),
            expected_run_names=tuple(run_names),
            device=_required_string(payload, "device"),
            submitted_at=_required_string(payload, "submitted_at"),
            cwd=cwd,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """One immutable terminal record appended to the completion journal."""

    schema_version: int
    job_id: str
    sequence: int
    device: str
    status: JobStatus
    time: str
    expected_run_names: tuple[str, ...]
    return_code: int | None = None
    signal: int | None = None
    duration_seconds: float | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != QUEUE_JOURNAL_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported queue journal schema version: "
                f"{self.schema_version!r}; expected {QUEUE_JOURNAL_SCHEMA_VERSION}."
            )
        _validate_job_id(self.job_id)
        _validate_sequence(self.sequence)
        validate_device_spec(self.device)
        if self.status not in ("completed", "failed", "interrupted"):
            raise ValueError(f"Unknown job outcome status: {self.status!r}.")
        _parse_utc_time(self.time)
        run_names = _validate_run_names(self.expected_run_names)
        object.__setattr__(self, "expected_run_names", run_names)
        if self.return_code is not None and (
            not isinstance(self.return_code, int) or isinstance(self.return_code, bool)
        ):
            raise ValueError("Job outcome return_code must be an integer or None.")
        if self.signal is not None and (
            not isinstance(self.signal, int) or isinstance(self.signal, bool)
        ):
            raise ValueError("Job outcome signal must be an integer or None.")
        if self.duration_seconds is not None and (
            not isinstance(self.duration_seconds, int | float)
            or isinstance(self.duration_seconds, bool)
            or self.duration_seconds < 0
        ):
            raise ValueError("Job outcome duration_seconds must be a non-negative number or None.")
        if self.message is not None and (not isinstance(self.message, str) or not self.message):
            raise ValueError("Job outcome message must be a non-empty string or None.")

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the flattened JSON object appended to the completion journal."""
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "sequence": self.sequence,
            "device": self.device,
            "status": self.status,
            "time": self.time,
            "expected_run_names": list(self.expected_run_names),
        }
        optional: dict[str, JsonValue] = {
            "return_code": self.return_code,
            "signal": self.signal,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload

    def to_json_line(self) -> bytes:
        """Serialize one complete UTF-8 JSONL record for the completion journal."""
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JobOutcome:
        """Validate one JSON object read back from the completion journal."""
        if not isinstance(payload, Mapping):
            raise ValueError("queue journal record must be a JSON object.")
        run_names = payload.get("expected_run_names")
        if not isinstance(run_names, list):
            raise ValueError("queue journal record expected_run_names must be a list of strings.")
        return cls(
            schema_version=_required_int(payload, "schema_version"),
            job_id=_required_string(payload, "job_id"),
            sequence=_required_int(payload, "sequence"),
            device=_required_string(payload, "device"),
            status=cast(JobStatus, _required_string(payload, "status")),
            time=_required_string(payload, "time"),
            expected_run_names=tuple(run_names),
            return_code=_optional_int(payload, "return_code"),
            signal=_optional_int(payload, "signal"),
            duration_seconds=_optional_number(payload, "duration_seconds"),
            message=_optional_string(payload, "message"),
        )


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    """Passively readable pending, claimed, and terminal queue state.

    ``malformed_job_files`` lists the paths of any pending or claimed spool
    file that failed to parse; those jobs are omitted from ``pending`` /
    ``claimed`` rather than aborting the whole snapshot, but their paths are
    still surfaced here so one corrupt file never silently hides the rest of
    a healthy spool. ``journal_error`` is set instead of raising when the
    completion journal itself fails to parse (for example a corrupted
    non-trailing line); in that case ``completed`` / ``failed`` /
    ``interrupted`` are all empty because Mammoth cannot safely determine
    them, and callers should treat that as fail-closed, not as "no terminal
    jobs yet".
    """

    pending: tuple[Job, ...]
    claimed: tuple[Job, ...]
    completed: tuple[JobOutcome, ...]
    failed: tuple[JobOutcome, ...]
    interrupted: tuple[JobOutcome, ...]
    malformed_job_files: tuple[str, ...] = ()
    journal_error: str | None = None


def submit_job(
    entry: Path,
    *,
    argv: Sequence[str],
    expected_run_names: Sequence[str],
    device: str,
    cwd: str | os.PathLike[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Job:
    """Durably enqueue one opaque job under ``entry``'s pending spool.

    The job file is published with :func:`mammoth.core.artifacts.atomic_write_json`
    after a monotonic submission sequence is allocated under a short-held
    exclusive lease, so concurrent submitters from independent processes never
    race for the same FIFO position.
    """
    layout = QueueLayout(entry).prepare()
    job = Job(
        schema_version=QUEUE_SCHEMA_VERSION,
        job_id=_generate_job_id(),
        sequence=_allocate_sequence(layout),
        argv=tuple(argv),
        expected_run_names=tuple(expected_run_names),
        device=device,
        submitted_at=_utc_now_iso(),
        cwd=None if cwd is None else os.fspath(cwd),
        metadata=metadata or {},
    )
    path = layout.pending_dir / _job_filename(job.sequence, job.job_id)
    atomic_write_json(path, job.to_dict())
    return job


def list_jobs(entry: Path) -> QueueSnapshot:
    """Return a passive snapshot of pending, claimed, and terminal queue state.

    Tolerates one malformed job file the way :func:`mammoth.queue.serve._claim_next_job`
    and :func:`mammoth.queue.serve.reconcile_interrupted_jobs` already do: a
    bad file is skipped rather than hiding every healthy job in the spool,
    but its path is still reported through
    :attr:`QueueSnapshot.malformed_job_files` so it is never silently lost.
    A malformed completion journal is reported through
    :attr:`QueueSnapshot.journal_error` instead, since a partially trusted
    journal cannot safely populate ``completed`` / ``failed`` / ``interrupted``.
    """
    layout = QueueLayout(entry)
    malformed: list[str] = []

    def _load_all(paths: Sequence[Path]) -> list[Job]:
        jobs: list[Job] = []
        for path in paths:
            try:
                jobs.append(_load_job(path))
            except QueueError:
                malformed.append(str(path))
        return jobs

    pending = tuple(sorted(_load_all(_glob_json(layout.pending_dir)), key=lambda job: job.sequence))
    claimed: list[Job] = []
    if layout.claimed_dir.is_dir():
        for lane_dir in sorted(layout.claimed_dir.iterdir()):
            if not lane_dir.is_dir():
                continue
            claimed.extend(_load_all(_glob_json(lane_dir)))
    claimed_sorted = tuple(sorted(claimed, key=lambda job: job.sequence))

    journal_error: str | None = None
    outcomes: tuple[JobOutcome, ...] = ()
    try:
        outcomes = read_job_journal(layout.journal_path)
    except QueueError as error:
        journal_error = str(error)

    return QueueSnapshot(
        pending=pending,
        claimed=claimed_sorted,
        completed=tuple(outcome for outcome in outcomes if outcome.status == "completed"),
        failed=tuple(outcome for outcome in outcomes if outcome.status == "failed"),
        interrupted=tuple(outcome for outcome in outcomes if outcome.status == "interrupted"),
        malformed_job_files=tuple(malformed),
        journal_error=journal_error,
    )


def cancel_job(entry: Path, job_id: str) -> Job:
    """Remove one unclaimed job; refuse a job already claimed or terminal.

    Raises :class:`JobAlreadyClaimedError` when the job has already been
    claimed by a lane or already reached a terminal journal outcome, and
    :class:`JobNotFoundError` when no job with this ID is known at all. This
    is a cooperating-local-writer protocol, matching the rest of Mammoth's
    filesystem contracts: a job claimed in the instant between this call's
    pending-directory scan and its removal attempt is reported as already
    claimed rather than silently cancelled or silently left in place.
    """
    _validate_job_id(job_id)
    layout = QueueLayout(entry)
    pending_match = _find_job_file(layout.pending_dir, job_id)
    if pending_match is not None:
        job = _load_job(pending_match)
        try:
            pending_match.unlink()
        except FileNotFoundError:
            pending_match = None
        else:
            return job

    claimed_match = _find_claimed_job_file(layout.claimed_dir, job_id)
    if claimed_match is not None:
        raise JobAlreadyClaimedError(
            f"Job {job_id!r} has already been claimed by a runner and cannot be cancelled."
        )

    outcomes_by_id = {outcome.job_id: outcome for outcome in read_job_journal(layout.journal_path)}
    if job_id in outcomes_by_id:
        raise JobAlreadyClaimedError(
            f"Job {job_id!r} already reached a terminal state "
            f"({outcomes_by_id[job_id].status!r}) and cannot be cancelled."
        )
    raise JobNotFoundError(f"No queued job found with ID {job_id!r}.")


def append_job_outcome(entry: Path, outcome: JobOutcome) -> None:
    """Durably append one terminal job outcome to the shared completion journal.

    Writes with a single raw ``os.write()`` call under ``O_APPEND`` so
    concurrent appends from independent lane processes never interleave on a
    local filesystem, then fsyncs the record before returning.
    """
    path = QueueLayout(entry).journal_path
    path.parent.mkdir(parents=True, exist_ok=True)
    line = outcome.to_json_line()
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError(f"short queue journal append: wrote {written!r} of {len(line)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_job_journal(path: Path) -> tuple[JobOutcome, ...]:
    """Return every complete record from the completion journal, oldest first.

    An incomplete trailing line (a crash mid-append) is silently ignored,
    matching :func:`mammoth.core.events.read_execution_events`'s tolerance
    for a torn final append.
    """
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return ()
    complete_lines = raw.split(b"\n")[:-1]
    outcomes: list[JobOutcome] = []
    for line_number, line in enumerate(complete_lines, start=1):
        if not line:
            continue
        try:
            payload = json.loads(line.decode("utf-8"))
            outcomes.append(JobOutcome.from_dict(payload))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise QueueError(
                f"Malformed queue journal record in {path} at line {line_number}: {error}"
            ) from error
    return tuple(outcomes)


def journal_job_ids(path: Path) -> frozenset[str]:
    """Return the set of job IDs that already reached a terminal outcome."""
    return frozenset(outcome.job_id for outcome in read_job_journal(path))


def load_job(path: Path) -> Job:
    """Read and validate one job spool file."""
    return _load_job(path)


def job_filename(job: Job) -> str:
    """Return the deterministic spool filename for one job."""
    return _job_filename(job.sequence, job.job_id)


def _job_filename(sequence: int, job_id: str) -> str:
    return f"{sequence:020d}-{job_id}.json"


def _glob_json(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _load_job(path: Path) -> Job:
    """Read and validate one job spool file, rejecting a final-component symlink.

    Uses :func:`mammoth.core.artifacts.artifact_open_flags` (``O_NOFOLLOW``)
    like every write path in this module already does through
    :func:`mammoth.core.artifacts.atomic_write_json`, since a job file's
    ``argv`` is what a lane will later execute.
    """
    try:
        descriptor = os.open(path, artifact_open_flags())
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return Job.from_dict(payload)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise QueueError(f"Malformed queue job file {path}: {error}") from error


def _find_job_file(directory: Path, job_id: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"*-{job_id}.json"))
    return matches[0] if matches else None


def _find_claimed_job_file(claimed_dir: Path, job_id: str) -> Path | None:
    if not claimed_dir.is_dir():
        return None
    matches = sorted(claimed_dir.glob(f"*/*-{job_id}.json"))
    return matches[0] if matches else None


def _allocate_sequence(layout: QueueLayout) -> int:
    """Allocate the next strictly monotonic submission sequence under a lease.

    Mirrors :func:`mammoth.core.execution.claim_logical_run_lease`'s
    O_NOFOLLOW-guarded open, but holds a blocking exclusive lock for the
    brief read-increment-write rather than failing fast: unlike a device
    lease, this counter is held for a bounded instant by every submitter and
    is expected to contend.
    """
    lock_path = layout.sequence_lock_path
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            raw = os.read(descriptor, 32).strip()
            try:
                current = int(raw) if raw else 0
            except ValueError as error:
                raise QueueError(f"Corrupt queue sequence counter: {lock_path}") from error
            if current < 0:
                raise QueueError(f"Corrupt queue sequence counter: {lock_path}")
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, str(current + 1).encode("ascii"))
            os.fsync(descriptor)
            return current
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _generate_job_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def _validate_sequence(sequence: int) -> int:
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError(f"Job sequence must be a non-negative integer, got {sequence!r}.")
    return sequence


def _validate_job_id(job_id: str) -> str:
    if (
        not isinstance(job_id, str)
        or not job_id
        or len(job_id) > _JOB_ID_MAX_LENGTH
        or job_id in {".", ".."}
        or any(character in job_id for character in "*?[]/\\\x00")
    ):
        raise ValueError(f"Invalid queue job ID: {job_id!r}.")
    return job_id


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise ValueError("Job argv must be a sequence of arguments, not a scalar.")
    values = tuple(argv)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Job argv must contain at least one non-empty string.")
    return values


def _validate_run_names(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError("expected_run_names must be a sequence of strings, not a scalar.")
    values = tuple(names)
    if not values:
        raise ValueError("A job must declare at least one expected run name.")
    for name in values:
        validate_run_name(name)
    return values


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc_time(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"timestamp must be a non-empty string, got {value!r}.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid UTC timestamp: {value!r}.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"Timestamp must use UTC: {value!r}.")
    return parsed


def _validate_json_compatible(value: Any, *, depth: int = 0) -> JsonValue:
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("Job metadata nesting is too deep.")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # noqa: PLR0124 (NaN check)
            raise ValueError("Job metadata must contain finite numbers.")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        detached: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Job metadata keys must be strings.")
            detached[key] = _validate_json_compatible(item, depth=depth + 1)
        return detached
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_validate_json_compatible(item, depth=depth + 1) for item in value]
    raise ValueError(
        f"Job metadata must contain JSON-compatible values, got {type(value).__name__}."
    )


def _freeze_json(value: Any) -> ImmutableJsonValue:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return cast(None | bool | int | float | str, value)


def _thaw_json(value: ImmutableJsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"queue field {key!r} must be a non-empty string.")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"queue field {key!r} must be a non-empty string or null.")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"queue field {key!r} must be an integer.")
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"queue field {key!r} must be an integer or null.")
    return value


def _optional_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"queue field {key!r} must be a number or null.")
    return float(value)
