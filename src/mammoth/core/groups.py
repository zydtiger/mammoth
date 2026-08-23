"""Entry-level workflow-group manifests and event streams for Mammoth artifacts.

A group is an optional record between an :class:`~mammoth.core.layout.RunLayout`
entry and its member runs. It is produced only by :mod:`mammoth.workflow` and
consumed passively: nothing in this module or elsewhere requires a group to
exist, and an entry without a ``.mammoth/groups/`` subtree remains fully valid.
The immutable :class:`GroupManifest` mirrors :mod:`mammoth.core.execution`'s
``ExecutionMetadata`` (one atomically published record, generated-ID collision
retry), while :class:`GroupEvent` and :class:`GroupEventWriter` reuse the
schema-v1 JSONL conventions from :mod:`mammoth.core.events` at group scope:
member run and step lifecycle transitions instead of one execution's phases.
"""

from __future__ import annotations

import json
import logging
import math
import os
import stat
import threading
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Literal, cast

from mammoth.core.artifacts import atomic_write_json
from mammoth.core.events import ImmutableJsonValue, JsonValue, utc_event_time
from mammoth.core.execution import sanitize_metadata_fields
from mammoth.core.identity import validate_group_id, validate_run_name

GROUP_SCHEMA_VERSION = 1
GROUP_EVENT_SCHEMA_VERSION = 1
GROUPS_RELATIVE_DIR = Path(".mammoth") / "groups"
GROUP_MANIFEST_FILENAME = "manifest.json"
GROUP_EVENT_STREAM_FILENAME = "events.jsonl"
_GENERATED_ID_ATTEMPTS = 32
_GROUP_EVENT_APPEND_GUARD_BYTES = 4096

logger = logging.getLogger(__name__)

GroupOrder = Literal["run-major", "step-major"]
GroupEventName = Literal[
    "group_started",
    "group_completed",
    "group_failed",
    "group_interrupted",
    "run_started",
    "run_completed",
    "run_failed",
    "run_blocked",
    "run_interrupted",
    "step_started",
    "step_completed",
    "step_failed",
    "step_interrupted",
]

_GROUP_EVENT_NAMES: frozenset[str] = frozenset(
    {
        "group_started",
        "group_completed",
        "group_failed",
        "group_interrupted",
        "run_started",
        "run_completed",
        "run_failed",
        "run_blocked",
        "run_interrupted",
        "step_started",
        "step_completed",
        "step_failed",
        "step_interrupted",
    }
)
_GROUP_SCOPED_EVENTS = frozenset(
    {"group_started", "group_completed", "group_failed", "group_interrupted"}
)
_RUN_SCOPED_EVENTS = frozenset(
    {"run_started", "run_completed", "run_failed", "run_blocked", "run_interrupted"}
)
_STEP_SCOPED_EVENTS = frozenset(
    {"step_started", "step_completed", "step_failed", "step_interrupted"}
)
_TERMINAL_GROUP_EVENTS = frozenset({"group_completed", "group_failed", "group_interrupted"})
_TERMINAL_RUN_EVENTS = frozenset({"run_completed", "run_failed", "run_blocked", "run_interrupted"})
_TERMINAL_STEP_EVENTS = frozenset({"step_completed", "step_failed", "step_interrupted"})
_INTERRUPTED_EVENTS = frozenset({"group_interrupted", "run_interrupted", "step_interrupted"})


@dataclass(frozen=True, slots=True)
class GroupMember:
    """One workflow-declared run name and its ordered planned step names."""

    run_name: str
    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_run_name(self.run_name)
        if isinstance(self.steps, str) or not isinstance(self.steps, Sequence):
            raise ValueError(f"Group member {self.run_name!r} steps must be a sequence")
        steps = tuple(self.steps)
        if not steps or any(not isinstance(step, str) or not step for step in steps):
            raise ValueError(f"Group member {self.run_name!r} steps must contain non-empty strings")
        if len(steps) != len(set(steps)):
            raise ValueError(f"Group member {self.run_name!r} step names must be unique")
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the JSON object written inside one manifest's ``members`` list."""
        return {"run_name": self.run_name, "steps": list(self.steps)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GroupMember:
        """Validate and reconstruct one member record read from a manifest."""
        if not isinstance(payload, Mapping):
            raise ValueError("group member must be a JSON object.")
        run_name = payload.get("run_name")
        if not isinstance(run_name, str) or not run_name:
            raise ValueError("group member run_name must be a non-empty string.")
        steps = payload.get("steps")
        if not isinstance(steps, list) or any(
            not isinstance(step, str) or not step for step in steps
        ):
            raise ValueError("group member steps must be a list of non-empty strings.")
        return cls(run_name=run_name, steps=tuple(steps))


@dataclass(frozen=True)
class GroupManifest:
    """Immutable provenance published once for one workflow group invocation."""

    schema_version: int
    group_id: str
    created_at: str
    order: GroupOrder
    members: tuple[GroupMember, ...]
    metadata: Mapping[str, ImmutableJsonValue]

    def __post_init__(self) -> None:
        if self.schema_version != GROUP_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported group manifest schema version: "
                f"{self.schema_version!r}; expected {GROUP_SCHEMA_VERSION}."
            )
        validate_group_id(self.group_id)
        _parse_utc_time(self.created_at)
        if self.order not in {"run-major", "step-major"}:
            raise ValueError(
                f"group manifest order must be 'run-major' or 'step-major', got {self.order!r}."
            )
        members = tuple(self.members)
        if not members or any(not isinstance(member, GroupMember) for member in members):
            raise ValueError("group manifest members must contain at least one GroupMember.")
        names = [member.run_name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("group manifest member run names must be unique.")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "metadata", freeze_group_metadata(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the JSON payload written by :func:`publish_group_manifest`."""
        return {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "created_at": self.created_at,
            "order": self.order,
            "members": [member.to_dict() for member in self.members],
            "metadata": thaw_group_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GroupManifest:
        """Validate and reconstruct a manifest read from disk."""
        if not isinstance(payload, Mapping):
            raise ValueError("group manifest must be a JSON object.")
        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ValueError("group manifest schema_version must be an integer.")
        group_id = payload.get("group_id")
        if not isinstance(group_id, str):
            raise ValueError("group manifest group_id must be a string.")
        created_at = payload.get("created_at")
        if not isinstance(created_at, str):
            raise ValueError("group manifest created_at must be a string.")
        order = payload.get("order")
        if order not in {"run-major", "step-major"}:
            raise ValueError(f"Invalid group manifest order: {order!r}.")
        raw_members = payload.get("members")
        if not isinstance(raw_members, list):
            raise ValueError("group manifest members must be a list.")
        metadata = payload.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, Mapping):
            raise ValueError("group manifest metadata must be an object.")
        return cls(
            schema_version=schema_version,
            group_id=group_id,
            created_at=created_at,
            order=cast(GroupOrder, order),
            members=tuple(GroupMember.from_dict(member) for member in raw_members),
            metadata=metadata,
        )


def generate_group_id() -> str:
    """Generate a timestamped, filesystem-safe group ID with UUID4 collision resistance."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def groups_dir_for(entry: Path) -> Path:
    """Return the stable entry-level container for every published group."""
    return Path(entry) / GROUPS_RELATIVE_DIR


def publish_group_manifest(
    entry: Path,
    *,
    order: GroupOrder,
    members: Sequence[GroupMember],
    metadata: Mapping[str, Any] | None = None,
    group_id: str | None = None,
    created_at: str | None = None,
) -> GroupManifest:
    """Create and atomically publish one immutable group manifest.

    Mirrors :func:`mammoth.core.execution.create_execution_context`'s
    generated-ID collision retry: a caller-supplied ``group_id`` must be
    unique, while an omitted ID is retried against fresh candidates. Manifest
    bytes are published with :func:`mammoth.core.artifacts.atomic_write_json`.
    """
    entry_path = Path(entry)
    if order not in {"run-major", "step-major"}:
        raise ValueError(f"group order must be 'run-major' or 'step-major', got {order!r}.")
    member_tuple = tuple(members)
    if not member_tuple or any(not isinstance(member, GroupMember) for member in member_tuple):
        raise ValueError("group manifest requires at least one GroupMember.")
    resolved_metadata: Mapping[str, Any] = {} if metadata is None else metadata
    if created_at is not None and not isinstance(created_at, str):
        raise ValueError(f"created_at must be a UTC string or None, got {created_at!r}.")
    created_at_value = created_at or _utc_now_iso()
    _parse_utc_time(created_at_value)

    candidate_ids: tuple[str, ...]
    if group_id is not None:
        candidate_ids = (validate_group_id(group_id),)
    else:
        candidate_ids = tuple(
            validate_group_id(generate_group_id()) for _ in range(_GENERATED_ID_ATTEMPTS)
        )

    groups_root = groups_dir_for(entry_path)
    groups_root.mkdir(parents=True, exist_ok=True)

    collision: FileExistsError | None = None
    for candidate_id in candidate_ids:
        group_dir = groups_root / candidate_id
        try:
            group_dir.mkdir()
        except FileExistsError as error:
            collision = error
            if group_id is not None:
                raise
            continue

        manifest = GroupManifest(
            schema_version=GROUP_SCHEMA_VERSION,
            group_id=candidate_id,
            created_at=created_at_value,
            order=order,
            members=member_tuple,
            metadata=resolved_metadata,
        )
        manifest_path = group_dir / GROUP_MANIFEST_FILENAME
        try:
            atomic_write_json(manifest_path, manifest.to_dict())
        except BaseException:
            with suppress(OSError):
                group_dir.rmdir()
            raise
        return manifest

    raise FileExistsError(
        f"Could not allocate a unique group ID after {_GENERATED_ID_ATTEMPTS} attempts."
    ) from collision


def load_group_manifest(manifest_path: Path) -> GroupManifest:
    """Read and validate one previously published group manifest."""
    with Path(manifest_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Group manifest must contain a JSON object: {manifest_path}.")
    return GroupManifest.from_dict(payload)


def freeze_group_metadata(metadata: Mapping[str, Any]) -> Mapping[str, ImmutableJsonValue]:
    """Validate JSON compatibility and return an immutable recursive copy.

    Caller-supplied group metadata is opaque: Mammoth validates only that it is
    JSON-compatible and never redacts, reorders, or otherwise interprets it, so
    a caller reading it back through :meth:`GroupManifest.to_dict` or
    :func:`thaw_group_metadata` observes the exact same JSON value it supplied.
    Any ``Mapping``/``Sequence`` container is accepted at the boundary (not
    only a plain ``dict``/``list``), matching how other Mammoth metadata
    fields detach caller collections such as ``UserDict`` or ``UserList``.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("group metadata must be a mapping.")
    validated = _validate_json_compatible(metadata)
    assert isinstance(validated, dict)
    return cast(Mapping[str, ImmutableJsonValue], _freeze_json(validated))


def thaw_group_metadata(metadata: Mapping[str, ImmutableJsonValue]) -> dict[str, Any]:
    """Return a detached JSON-compatible copy of frozen group metadata."""
    return {key: _thaw_json(value) for key, value in metadata.items()}


@dataclass(frozen=True)
class GroupEvent:
    """One validated schema-v1-style record on a group's append-only stream.

    Reuses the JSONL conventions from :mod:`mammoth.core.events` (monotonic
    ``sequence``, UTC ``time``, flush-per-record durability) at group scope:
    each record names either the group itself, one member run, or one step of
    one member run, instead of one execution's own phases.
    """

    schema_version: int
    sequence: int
    time: str
    group_id: str
    event: GroupEventName
    run_name: str | None = None
    step_name: str | None = None
    signal: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise ValueError("group event schema_version must be an integer.")
        if self.schema_version != GROUP_EVENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported group event schema version: "
                f"{self.schema_version!r}; expected {GROUP_EVENT_SCHEMA_VERSION}."
            )
        invalid_sequence = (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 1
        )
        if invalid_sequence:
            raise ValueError(
                f"group event sequence must be a positive integer, got {self.sequence!r}."
            )
        _parse_utc_time(self.time)
        validate_group_id(self.group_id)
        if self.event not in _GROUP_EVENT_NAMES:
            raise ValueError(f"Unknown group event: {self.event!r}.")
        if self.event in _GROUP_SCOPED_EVENTS:
            if self.run_name is not None or self.step_name is not None:
                raise ValueError(f"{self.event} must omit run_name and step_name.")
        elif self.event in _RUN_SCOPED_EVENTS:
            if not isinstance(self.run_name, str) or not self.run_name:
                raise ValueError(f"{self.event} requires a non-empty run_name.")
            if self.step_name is not None:
                raise ValueError(f"{self.event} must omit step_name.")
        elif self.event in _STEP_SCOPED_EVENTS:
            if not isinstance(self.run_name, str) or not self.run_name:
                raise ValueError(f"{self.event} requires a non-empty run_name.")
            if not isinstance(self.step_name, str) or not self.step_name:
                raise ValueError(f"{self.event} requires a non-empty step_name.")
        else:  # pragma: no cover - unreachable once every _GROUP_EVENT_NAMES entry is scoped
            raise AssertionError(f"group event {self.event!r} is not assigned a scope.")
        if self.signal is not None:
            if not isinstance(self.signal, int) or isinstance(self.signal, bool):
                raise ValueError("group event signal must be an integer.")
            if self.event not in _INTERRUPTED_EVENTS:
                raise ValueError("signal is valid only for *_interrupted group events.")
        if self.message is not None:
            sanitized = sanitize_metadata_fields({"message": self.message})["message"]
            object.__setattr__(self, "message", cast(str, sanitized))

    @property
    def is_terminal(self) -> bool:
        """Return whether this event closes a group, run, or step scope."""
        return self.event in (_TERMINAL_GROUP_EVENTS | _TERMINAL_RUN_EVENTS | _TERMINAL_STEP_EVENTS)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the flattened JSON object written to the append-only stream."""
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "time": self.time,
            "group_id": self.group_id,
            "event": self.event,
        }
        optional: dict[str, JsonValue] = {
            "run_name": self.run_name,
            "step_name": self.step_name,
            "signal": self.signal,
            "message": self.message,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload

    def to_json_line(self) -> bytes:
        """Serialize one complete UTF-8 JSONL record for :class:`GroupEventWriter`."""
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
    def from_dict(cls, payload: Mapping[str, Any]) -> GroupEvent:
        """Validate a JSON object read back from one group event stream."""
        if not isinstance(payload, Mapping):
            raise ValueError("group event must be a JSON object.")
        return cls(
            schema_version=_required_int(payload, "schema_version"),
            sequence=_required_int(payload, "sequence"),
            time=_required_string(payload, "time"),
            group_id=_required_string(payload, "group_id"),
            event=cast(GroupEventName, _required_string(payload, "event")),
            run_name=_optional_string(payload, "run_name"),
            step_name=_optional_string(payload, "step_name"),
            signal=_optional_int(payload, "signal"),
            message=_optional_string(payload, "message"),
        )


class GroupEventWriter:
    """Own one append-only group event stream for one workflow invocation.

    Each group directory is freshly and exclusively created by
    :func:`publish_group_manifest`, so unlike
    :class:`mammoth.core.events.ExecutionEventWriter` this writer never reopens
    an existing populated stream: it exclusively creates ``events.jsonl`` and
    owns it for the lifetime of one :class:`mammoth.workflow.Workflow` run.
    A write or open failure disables the writer instead of raising, matching
    the JSONL durability contract that a writer failure never terminates the
    workload it observes.
    """

    def __init__(
        self,
        path: Path,
        *,
        group_id: str,
        utc_clock: Any = None,
        failure_logger: logging.Logger = logger,
    ) -> None:
        self.path = Path(path)
        self.group_id = validate_group_id(group_id)
        if self.path.name != GROUP_EVENT_STREAM_FILENAME:
            raise ValueError(
                f"group event writer path must end with {GROUP_EVENT_STREAM_FILENAME!r}, "
                f"got {self.path}."
            )
        self._utc_clock = utc_clock or utc_event_time
        self._failure_logger = failure_logger
        self._sequence = 0
        self._closed = False
        self._failure_logged = False
        self._lock = threading.RLock()
        self._stream: BinaryIO | None = None
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        opened_stream: BinaryIO | None = None
        try:
            descriptor = os.open(self.path, flags, 0o600)
            opened_stream = cast(BinaryIO, os.fdopen(descriptor, "wb", buffering=0))
            self._stream = opened_stream
        except Exception as error:
            if opened_stream is not None:
                with suppress(Exception):
                    opened_stream.close()
            self._disable_after_io_failure("open", error)

    @property
    def enabled(self) -> bool:
        """Return whether this writer may still append records."""
        with self._lock:
            return self._stream is not None and not self._closed and not self._failure_logged

    @property
    def sequence(self) -> int:
        """Return the last sequence successfully flushed by this writer."""
        with self._lock:
            return self._sequence

    def emit(self, event: GroupEventName, **fields: Any) -> GroupEvent | None:
        """Validate and immediately flush one group lifecycle event."""
        with self._lock:
            if not self.enabled:
                return None
            candidate = GroupEvent(
                schema_version=GROUP_EVENT_SCHEMA_VERSION,
                sequence=self._sequence + 1,
                time=self._utc_clock(),
                group_id=self.group_id,
                event=event,
                **fields,
            )
            return self._write_event(candidate)

    def close(self) -> None:
        """Close the stream without propagating I/O failures."""
        with self._lock:
            if self._closed:
                return
            stream = self._stream
            self._stream = None
            self._closed = True
            if stream is None:
                return
            try:
                stream.close()
            except Exception as error:
                self._disable_after_io_failure("close", error)

    def __enter__(self) -> GroupEventWriter:
        """Return this writer for context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the stream without masking a computation error."""
        self.close()

    def _write_event(self, event: GroupEvent) -> GroupEvent | None:
        stream = self._stream
        if stream is None:
            return None
        record = event.to_json_line()
        try:
            written = stream.write(record)
            if written != len(record):
                raise OSError(f"short group-event append: wrote {written!r} of {len(record)} bytes")
            stream.flush()
        except Exception as error:
            self._disable_after_io_failure("write", error)
            return None
        self._sequence = event.sequence
        return event

    def _disable_after_io_failure(self, operation: str, error: Exception) -> None:
        if self._failure_logged:
            return
        self._failure_logged = True
        stream = self._stream
        self._stream = None
        if stream is not None:
            with suppress(Exception):
                stream.close()
        with suppress(Exception):
            self._failure_logger.error(
                "Group event writer disabled after %s failure for %s: %s",
                operation,
                self.path,
                error,
                exc_info=True,
            )


def read_group_events(path: Path) -> list[GroupEvent]:
    """Return every complete record from one group event stream, oldest first.

    Any incomplete trailing line (a crash mid-append) is silently ignored,
    matching :func:`mammoth.core.events.read_execution_events`'s tolerance for
    a torn final append.
    """
    events: list[GroupEvent] = []
    next_sequence = 1
    stream_group_id: str | None = None
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return []
    # A trailing "\n" yields an empty final element; an incomplete torn append
    # yields a non-empty partial final element. Either way, dropping the last
    # split element discards exactly the incomplete tail, never a complete line.
    complete_lines = raw.split(b"\n")[:-1]
    for line_number, line in enumerate(complete_lines, start=1):
        if not line:
            continue
        try:
            payload = json.loads(line.decode("utf-8"))
            event = GroupEvent.from_dict(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"Malformed group event in {path} at line {line_number}: {error}"
            ) from error
        if event.sequence != next_sequence:
            raise ValueError(
                f"group event sequence {event.sequence} does not match expected "
                f"{next_sequence} in {path} at line {line_number}."
            )
        if stream_group_id is None:
            stream_group_id = event.group_id
        elif event.group_id != stream_group_id:
            raise ValueError(f"group event identity differs from earlier records in {path}.")
        next_sequence += 1
        events.append(event)
    return events


class GroupEventReadError(ValueError):
    """A complete JSONL group event record failed with precise stream context."""

    def __init__(
        self,
        path: Path,
        line_number: int,
        detail: str,
        *,
        valid_events: tuple[GroupEvent, ...] = (),
    ) -> None:
        self.path = path
        self.line_number = line_number
        self.detail = detail
        self.valid_events = valid_events
        super().__init__(f"Malformed group event in {path} at line {line_number}: {detail}")


class GroupEventTailReader:
    """Poll one append-only group event stream while retaining an incomplete final line.

    Mirrors :class:`mammoth.core.events.ExecutionEventTailReader`'s incremental
    contract at group scope: each :meth:`poll` returns only newly appended
    complete records, an incomplete trailing line is retained for the next
    poll, and a changed or truncated file identity fails loudly instead of
    silently reinterpreting history. This is the passive folding source a
    fleet or group monitor uses instead of re-reading the whole stream with
    :func:`read_group_events` on every refresh.
    """

    def __init__(self, path: Path, *, allow_missing: bool = True) -> None:
        self.path = Path(path)
        self.allow_missing = allow_missing
        self._file_identity: tuple[int, int] | None = None
        self._offset = 0
        self._append_guard = b""
        self._buffer = b""
        self._line_number = 0
        self._next_sequence = 1
        self._group_id: str | None = None
        self._error: GroupEventReadError | None = None

    @property
    def line_number(self) -> int:
        """Return the number of complete records consumed, including a failed line."""
        return self._line_number

    @property
    def offset(self) -> int:
        """Return the byte offset already read from the active file identity."""
        return self._offset

    def poll(self) -> list[GroupEvent]:
        """Return records appended since the previous poll without blocking."""
        if self._error is not None:
            raise self._error
        try:
            appended = self._read_appended_bytes()
        except FileNotFoundError:
            if self.allow_missing and self._file_identity is None:
                return []
            raise
        if not appended:
            return []

        raw = self._buffer + appended
        parts = raw.split(b"\n")
        complete_lines, self._buffer = parts[:-1], parts[-1]
        events: list[GroupEvent] = []
        for line in complete_lines:
            self._line_number += 1
            try:
                payload = json.loads(line.decode("utf-8"))
                event = GroupEvent.from_dict(payload)
                self._validate_stream_record(event)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._fail(self._line_number, str(error), valid_events=tuple(events))
            events.append(event)
        return events

    def _read_appended_bytes(self) -> bytes:
        flags = os.O_RDONLY | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        try:
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode):
                raise OSError(f"Group event stream must be a regular file: {self.path}")
            identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
            if self._file_identity is not None and identity != self._file_identity:
                self._fail(self._line_number + 1, "active stream file identity changed")
            if descriptor_stat.st_size < self._offset:
                self._fail(
                    self._line_number + 1,
                    f"active stream was truncated from offset {self._offset}",
                )
            guard_offset = self._offset - len(self._append_guard)
            if (
                self._append_guard
                and os.pread(descriptor, len(self._append_guard), guard_offset)
                != self._append_guard
            ):
                self._fail(self._line_number + 1, "consumed stream prefix changed")
            os.lseek(descriptor, self._offset, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = descriptor_stat.st_size - self._offset
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    self._fail(self._line_number + 1, "active stream changed while polling")
                chunks.append(chunk)
                remaining -= len(chunk)
            appended = b"".join(chunks)
        except BaseException:
            with suppress(Exception):
                os.close(descriptor)
            raise
        os.close(descriptor)
        self._file_identity = identity
        self._offset += len(appended)
        self._append_guard = (self._append_guard + appended)[-_GROUP_EVENT_APPEND_GUARD_BYTES:]
        return appended

    def _validate_stream_record(self, event: GroupEvent) -> None:
        if event.sequence != self._next_sequence:
            raise ValueError(
                f"group event sequence {event.sequence} does not match expected "
                f"{self._next_sequence}"
            )
        if self._group_id is None:
            self._group_id = event.group_id
        elif event.group_id != self._group_id:
            raise ValueError("group event identity differs from earlier records in the stream")
        self._next_sequence += 1

    def _fail(
        self,
        line_number: int,
        detail: str,
        *,
        valid_events: tuple[GroupEvent, ...] = (),
    ) -> None:
        error = GroupEventReadError(self.path, line_number, detail, valid_events=valid_events)
        self._error = error
        raise error


def _validate_json_compatible(value: Any) -> JsonValue:
    """Recursively validate and detach one JSON-compatible caller value."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("group metadata must contain finite numbers.")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        detached: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("group metadata keys must be strings.")
            detached[key] = _validate_json_compatible(item)
        return detached
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_validate_json_compatible(item) for item in value]
    raise ValueError(
        f"group metadata must contain JSON-compatible values, got {type(value).__name__}."
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


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"group event field {key!r} must be a non-empty string.")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"group event field {key!r} must be a non-empty string or null.")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"group event field {key!r} must be an integer.")
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"group event field {key!r} must be an integer or null.")
    return value
