"""Immutable execution-attempt identities and metadata for Mammoth run artifacts.

Direct CLI commands and the experiment runner create or join contexts from this
module before framework-specific work begins. CLI logging uses the returned
rank-log paths, while later dashboard/event layers may consume the same atomic
``execution.json`` without introducing a dependency on PyTorch or PrettyTerm.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from mammoth.core.identity import validate_execution_id, validate_run_name

EXECUTION_SCHEMA_VERSION = 1
EXECUTION_ID_ENV = "MAMMOTH_EXECUTION_ID"
RUN_NAME_ENV = "MAMMOTH_RUN_NAME"
INVOCATION_KIND_ENV = "MAMMOTH_INVOCATION_KIND"
PHASE_ENV = "MAMMOTH_PHASE"
EXECUTION_METADATA_FILENAME = "execution.json"
EXECUTIONS_RELATIVE_DIR = Path("logs") / "executions"
LOGICAL_RUN_LEASE_FILENAME = ".logical-run.lock"

ExecutionMode = Literal["single", "distributed"]

_SENSITIVE_NAMES = (
    "token",
    "password",
    "passwd",
    "secret",
    "credential",
    "apikey",
    "accesskey",
    "privatekey",
    "authkey",
    "cookie",
    "session",
    "signature",
    "bearer",
    "jwt",
)
_SENSITIVE_EXACT_NAMES = frozenset({"auth", "authorization", "header", "key", "sig", "ticket"})
_STRUCTURED_SENSITIVE_NAMES = frozenset(
    {
        "token",
        "tokens",
        "apitoken",
        "accesstoken",
        "refreshtoken",
        "password",
        "passwords",
        "passwd",
        "passwds",
        "secret",
        "secrets",
        "clientsecret",
        "credential",
        "credentials",
        "apikey",
        "accesskey",
        "privatekey",
        "authkey",
        "key",
        "auth",
        "authorization",
        "cookie",
        "session",
        "sessionid",
        "signature",
        "bearer",
        "jwt",
        "sig",
        "ticket",
    }
)
_STRUCTURED_SENSITIVE_SUFFIXES = frozenset(
    {
        "token",
        "tokens",
        "password",
        "passwords",
        "passwd",
        "passwds",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "auth",
        "authorization",
        "cookie",
        "session",
        "signature",
        "bearer",
        "jwt",
        "sig",
        "ticket",
    }
)
_STRUCTURED_SENSITIVE_PATH_SEGMENTS = frozenset(
    {
        "token",
        "apitoken",
        "accesstoken",
        "refreshtoken",
        "password",
        "passwd",
        "secret",
        "clientsecret",
        "credential",
        "credentials",
        "apikey",
        "accesskey",
        "privatekey",
        "authkey",
        "signature",
    }
)
_STRUCTURED_SENSITIVE_PATH_SUFFIXES = frozenset(
    {
        "token",
        "tokens",
        "password",
        "passwords",
        "passwd",
        "passwds",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "cookie",
        "session",
        "signature",
        "bearer",
        "jwt",
        "sig",
        "ticket",
    }
)
_STRUCTURED_VALUE_SUFFIXES = frozenset({"data", "string", "value", "values"})
_REDACTED = "<redacted>"
_GENERATED_ID_ATTEMPTS = 32


@dataclass
class LogicalRunLease:
    """Hold exclusive producer ownership of one logical run until terminal state."""

    path: Path
    _descriptor: int
    _closed: bool = False

    def close(self) -> None:
        """Release producer ownership while retaining the harmless lock inode."""
        if self._closed:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> LogicalRunLease:
        """Return this lease for process-lifetime context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the logical-run lease after producer termination."""
        self.close()


def claim_logical_run_lease(run_dir: Path) -> LogicalRunLease:
    """Claim the advisory lease before publishing a new execution attempt."""
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / LOGICAL_RUN_LEASE_FILENAME
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise RuntimeError(f"Logical-run lease must be a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another execution is already active for logical run {run_dir.name!r}: {path}"
            ) from error
    except Exception:
        os.close(descriptor)
        raise
    return LogicalRunLease(path=path, _descriptor=descriptor)


@dataclass(frozen=True)
class ExecutionMetadata:
    """Immutable provenance published once for one Mammoth execution attempt."""

    schema_version: int
    run_name: str
    execution_id: str
    created_at: str
    invocation_kind: str
    intended_phases: tuple[str, ...]
    world_size: int
    execution_mode: ExecutionMode
    command: tuple[str, ...]
    config_reference: str
    previous_execution_id: str | None = None
    resume_checkpoint: str | None = None
    parent_execution_id: str | None = None
    starting_epoch: int | None = None
    starting_global_step: int | None = None
    runtime: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.runtime is not None:
            sanitized = sanitize_metadata_fields(self.runtime)
            object.__setattr__(self, "runtime", _freeze_metadata_mapping(sanitized))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload written by :func:`create_execution_context`."""
        payload = {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "execution_id": self.execution_id,
            "created_at": self.created_at,
            "invocation_kind": self.invocation_kind,
            "intended_phases": list(self.intended_phases),
            "world_size": self.world_size,
            "execution_mode": self.execution_mode,
            "command": list(self.command),
            "config_reference": self.config_reference,
            "previous_execution_id": self.previous_execution_id,
            "resume_checkpoint": self.resume_checkpoint,
            "parent_execution_id": self.parent_execution_id,
            "starting_epoch": self.starting_epoch,
            "starting_global_step": self.starting_global_step,
        }
        if self.runtime is not None:
            payload["runtime"] = _thaw_metadata_value(self.runtime)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionMetadata:
        """Validate and reconstruct metadata read by :func:`join_execution_context`."""
        schema_version = payload.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != EXECUTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported execution metadata schema version: "
                f"{schema_version!r}; expected {EXECUTION_SCHEMA_VERSION}."
            )

        run_name = _required_string(payload, "run_name")
        execution_id = validate_execution_id(_required_string(payload, "execution_id"))
        created_at = _required_string(payload, "created_at")
        _parse_created_at(created_at)
        invocation_kind = _required_string(payload, "invocation_kind")
        intended_phases = _string_tuple(payload, "intended_phases", require_nonempty=True)
        command = _string_tuple(payload, "command", require_nonempty=True)
        world_size = _required_nonnegative_int(payload, "world_size", minimum=1)
        mode_value = payload.get("execution_mode")
        if mode_value not in {"single", "distributed"}:
            raise ValueError(f"Invalid execution_mode in execution metadata: {mode_value!r}.")
        if (mode_value == "single") != (world_size == 1):
            raise ValueError(
                "Execution metadata mode/world-size mismatch: "
                f"execution_mode={mode_value!r}, world_size={world_size}."
            )

        runtime_payload = payload.get("runtime")
        if runtime_payload is None:
            runtime = None
        elif isinstance(runtime_payload, Mapping):
            runtime = sanitize_metadata_fields(runtime_payload)
        else:
            raise ValueError("Execution metadata runtime must be an object when present.")

        return cls(
            schema_version=schema_version,
            run_name=run_name,
            execution_id=execution_id,
            created_at=created_at,
            invocation_kind=invocation_kind,
            intended_phases=intended_phases,
            world_size=world_size,
            execution_mode=mode_value,
            command=command,
            config_reference=_required_string(payload, "config_reference", allow_empty=True),
            previous_execution_id=_optional_execution_id(payload, "previous_execution_id"),
            resume_checkpoint=_optional_string(payload, "resume_checkpoint"),
            parent_execution_id=_optional_execution_id(payload, "parent_execution_id"),
            starting_epoch=_optional_nonnegative_int(payload, "starting_epoch"),
            starting_global_step=_optional_nonnegative_int(payload, "starting_global_step"),
            runtime=runtime,
        )


@dataclass(frozen=True)
class ExecutionContext:
    """Resolved immutable metadata and artifact paths shared by all ranks."""

    run_dir: Path
    execution_dir: Path
    metadata_path: Path
    metadata: ExecutionMetadata

    def rank_log_path(self, rank: int, *, world_size: int | None = None) -> Path:
        """Return the process-exclusive log path for one runtime rank.

        ``world_size`` is used only by workflow children whose launcher topology
        differs from the single-process runner that owns the execution.
        """
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(f"rank must be an integer, got {rank!r}.")
        effective_world_size = self.metadata.world_size if world_size is None else world_size
        if (
            not isinstance(effective_world_size, int)
            or isinstance(effective_world_size, bool)
            or effective_world_size < 1
        ):
            raise ValueError(f"world_size must be a positive integer, got {world_size!r}.")
        if rank < 0 or rank >= effective_world_size:
            raise ValueError(f"rank={rank} is outside execution world_size={effective_world_size}.")
        return self.execution_dir / f"rank-{rank}.log"


def generate_execution_id() -> str:
    """Generate a timestamped, filesystem-safe ID with UUID4 collision resistance."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def execution_id_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    aliases: Sequence[str] = (),
) -> str | None:
    """Resolve the canonical execution ID and explicit caller-owned aliases.

    Callers may supply aliases only for their own compatibility policy. Aliases
    must be distinct, non-empty POSIX environment-variable names other than
    :data:`EXECUTION_ID_ENV`; malformed declarations fail deterministically.
    Every populated configured name must carry the same valid execution ID.
    """
    source = os.environ if environ is None else environ
    alias_names = normalize_execution_id_environment_aliases(aliases)
    configured_names = (EXECUTION_ID_ENV, *alias_names)
    values = tuple(
        (name, validate_execution_id(value))
        for name in configured_names
        if (value := source.get(name)) is not None
    )
    if not values:
        return None
    if len({value for _, value in values}) != 1:
        names = ", ".join(name for name, _ in values)
        raise ValueError(f"Configured execution ID environment variables disagree: {names}.")
    return values[0][1]


def normalize_execution_id_environment_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze caller-owned execution-ID alias declarations."""
    if not isinstance(aliases, Sequence) or isinstance(aliases, str):
        raise ValueError("aliases must be a sequence of environment variable names, not a string.")
    alias_names = tuple(aliases)
    for alias in alias_names:
        if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias):
            raise ValueError(f"Invalid execution ID environment alias: {alias!r}.")
        if alias == EXECUTION_ID_ENV:
            raise ValueError(f"aliases must not repeat canonical name {EXECUTION_ID_ENV}.")
    if len(set(alias_names)) != len(alias_names):
        raise ValueError("aliases must not contain duplicate environment variable names.")
    return alias_names


def sanitize_reference(reference: str | os.PathLike[str]) -> str:
    """Remove URL credentials, queries, and fragments from a metadata reference."""
    value = _path_string(reference, name="reference")
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _REDACTED if _looks_credential_bearing(value) else value
    if not parsed.scheme and not parsed.netloc:
        return _REDACTED if _contains_sensitive_text(value) else value
    if not parsed.netloc and parsed.scheme != "file":
        return _REDACTED if _looks_credential_bearing(value) else value

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"{_REDACTED}@{netloc.rsplit('@', maxsplit=1)[1]}"
    path = parsed.path
    path_segments = [segment for segment in unquote(path).split("/") if segment]
    if any(_is_sensitive_name(segment) for segment in path_segments):
        path = f"/{_REDACTED}"
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


def sanitize_command(command: Sequence[str | os.PathLike[str]]) -> tuple[str, ...]:
    """Redact credential-like option values and sanitize URL-like arguments."""
    arguments = _validated_command_arguments(command)
    sanitized: list[str] = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            sanitized.append(_REDACTED)
            redact_next = False
            continue

        name, separator, value = argument.partition("=")
        is_named_argument = argument.startswith("-") or (
            bool(separator) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None
        )
        if is_named_argument and _is_sensitive_name(name):
            if separator:
                sanitized.append(f"{name}={_REDACTED}")
            else:
                sanitized.append(argument)
                redact_next = True
            continue
        if is_named_argument:
            if separator:
                sanitized.append(f"{name}={sanitize_reference(value)}")
            else:
                sanitized.append(argument)
            continue
        sanitized.append(sanitize_reference(argument))
    return tuple(sanitized)


def sanitize_metadata_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and redact structured metadata fields used by event records.

    ``mammoth.core.execution_events`` uses this helper for optional v1 extension
    fields so nested credentials and command arguments follow the same rules as
    immutable ``execution.json`` metadata.
    """
    if not isinstance(fields, Mapping):
        raise ValueError("metadata fields must be a mapping.")
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata field names must be non-empty strings.")
        sanitized[key] = _sanitize_metadata_value(key, value)
    return sanitized


def executions_dir_for(run_dir: Path) -> Path:
    """Return the stable execution-attempt container under a logical run."""
    return run_dir / EXECUTIONS_RELATIVE_DIR


def create_execution_context(
    run_dir: Path,
    *,
    run_name: str,
    invocation_kind: str,
    intended_phases: Sequence[str],
    world_size: int,
    execution_mode: ExecutionMode,
    command: Sequence[str | os.PathLike[str]],
    config_reference: str | os.PathLike[str] = "",
    execution_id: str | None = None,
    previous_execution_id: str | None = None,
    resume_checkpoint: str | os.PathLike[str] | None = None,
    parent_execution_id: str | None = None,
    starting_epoch: int | None = None,
    starting_global_step: int | None = None,
    runtime: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> ExecutionContext:
    """Create and atomically publish one immutable execution attempt.

    Direct commands call this on their primary rank, then distribute the ID and
    make peers join with :func:`join_execution_context`. The experiment runner
    creates one context per selected logical experiment and provides
    ``MAMMOTH_EXECUTION_ID`` to every phase subprocess.
    """
    validate_run_name(run_name)
    if not isinstance(invocation_kind, str) or not invocation_kind:
        raise ValueError("invocation_kind must be a non-empty string.")
    if isinstance(intended_phases, (str, bytes, os.PathLike)):
        raise ValueError("intended_phases must be a sequence of phase strings, not a scalar.")
    try:
        phases = tuple(intended_phases)
    except TypeError as error:
        raise ValueError("intended_phases must be a sequence of phase strings.") from error
    if not phases or any(not isinstance(phase, str) or not phase for phase in phases):
        raise ValueError("intended_phases must contain at least one non-empty string.")
    sanitized_command = sanitize_command(command)
    sanitized_config_reference = (
        "" if config_reference == "" else sanitize_reference(config_reference)
    )
    sanitized_resume_checkpoint = (
        sanitize_reference(resume_checkpoint) if resume_checkpoint is not None else None
    )
    sanitized_runtime = sanitize_metadata_fields(runtime) if runtime is not None else None
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}.")
    if execution_mode not in {"single", "distributed"}:
        raise ValueError(
            f"execution_mode must be 'single' or 'distributed', got {execution_mode!r}."
        )
    if (execution_mode == "single") != (world_size == 1):
        raise ValueError(
            "execution_mode and world_size disagree: "
            f"execution_mode={execution_mode!r}, world_size={world_size}."
        )
    _validate_starting_coordinate("starting_epoch", starting_epoch)
    _validate_starting_coordinate("starting_global_step", starting_global_step)

    if created_at is not None and not isinstance(created_at, str):
        raise ValueError(f"created_at must be a UTC string or None, got {created_at!r}.")
    created_at_value = created_at or _utc_now_iso()
    _parse_created_at(created_at_value)
    if previous_execution_id is not None:
        validate_execution_id(previous_execution_id)
    if parent_execution_id is not None:
        validate_execution_id(parent_execution_id)
    candidate_ids: tuple[str, ...]
    if execution_id is not None:
        candidate_ids = (validate_execution_id(execution_id),)
    else:
        candidate_ids = tuple(
            validate_execution_id(generate_execution_id()) for _ in range(_GENERATED_ID_ATTEMPTS)
        )
    executions_dir = executions_dir_for(run_dir)
    executions_dir.mkdir(parents=True, exist_ok=True)

    collision: FileExistsError | None = None
    for candidate_id in candidate_ids:
        execution_dir = executions_dir / candidate_id
        try:
            execution_dir.mkdir()
        except FileExistsError as error:
            collision = error
            if execution_id is not None:
                raise
            continue

        metadata = ExecutionMetadata(
            schema_version=EXECUTION_SCHEMA_VERSION,
            run_name=run_name,
            execution_id=candidate_id,
            created_at=created_at_value,
            invocation_kind=invocation_kind,
            intended_phases=phases,
            world_size=world_size,
            execution_mode=execution_mode,
            command=sanitized_command,
            config_reference=sanitized_config_reference,
            previous_execution_id=previous_execution_id,
            resume_checkpoint=sanitized_resume_checkpoint,
            parent_execution_id=parent_execution_id,
            starting_epoch=starting_epoch,
            starting_global_step=starting_global_step,
            runtime=sanitized_runtime,
        )
        metadata_path = execution_dir / EXECUTION_METADATA_FILENAME
        try:
            _publish_metadata(metadata_path, metadata)
        except BaseException:
            with suppress(OSError):
                execution_dir.rmdir()
            raise
        return ExecutionContext(
            run_dir=run_dir,
            execution_dir=execution_dir,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    raise FileExistsError(
        f"Could not allocate a unique execution ID after {_GENERATED_ID_ATTEMPTS} attempts."
    ) from collision


def join_execution_context(
    run_dir: Path,
    execution_id: str,
    *,
    expected_run_name: str | None = None,
) -> ExecutionContext:
    """Join an explicitly pre-established execution without modifying it."""
    safe_id = validate_execution_id(execution_id)
    execution_dir = executions_dir_for(run_dir) / safe_id
    metadata_path = execution_dir / EXECUTION_METADATA_FILENAME
    metadata = _load_metadata(metadata_path)
    if metadata.execution_id != safe_id:
        raise ValueError(
            f"Execution metadata ID {metadata.execution_id!r} does not match directory {safe_id!r}."
        )
    if expected_run_name is not None and metadata.run_name != expected_run_name:
        raise ValueError(
            f"Execution {safe_id!r} belongs to run {metadata.run_name!r}, "
            f"not {expected_run_name!r}."
        )
    return ExecutionContext(
        run_dir=run_dir,
        execution_dir=execution_dir,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def latest_execution_id(run_dir: Path) -> str | None:
    """Return the newest valid immutable execution under ``run_dir`` when known."""
    records = _valid_execution_records(run_dir)
    if not records:
        return None
    latest = max(
        records,
        key=lambda item: (_parse_created_at(item.created_at), item.execution_id),
    )
    return latest.execution_id


def parent_execution_id_for_checkpoint(run_dir: Path, checkpoint_path: Path) -> str | None:
    """Find the latest training attempt established before a checkpoint write."""
    try:
        checkpoint_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime, UTC)
        resolved_checkpoint = checkpoint_path.resolve(strict=True)
        resolved_checkpoint_dir = (run_dir / "checkpoints").resolve(strict=True)
    except OSError:
        return None
    if not resolved_checkpoint.is_relative_to(resolved_checkpoint_dir):
        return None

    candidates = [
        metadata
        for metadata in _valid_execution_records(run_dir)
        if "train" in metadata.intended_phases
        and _parse_created_at(metadata.created_at) <= checkpoint_time
    ]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda item: (_parse_created_at(item.created_at), item.execution_id),
    )
    return latest.execution_id


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_created_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid UTC execution timestamp: {value!r}.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"Execution timestamp must use UTC: {value!r}.")
    return parsed


def _is_sensitive_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower().lstrip("-"))
    return normalized in _SENSITIVE_EXACT_NAMES or any(
        marker in normalized for marker in _SENSITIVE_NAMES
    )


def _contains_sensitive_text(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    return (
        "bearer" in normalized
        or "authorization" in normalized
        or any(marker in normalized for marker in _SENSITIVE_NAMES)
    )


def _looks_credential_bearing(value: str) -> bool:
    return _contains_sensitive_text(value) or (
        ("://" in value or value.startswith("//")) and "@" in value
    )


def _sanitize_metadata_value(field_name: str, value: Any) -> Any:
    name_parts = _structured_name_parts(field_name)
    if _structured_name_is_sensitive(field_name):
        return _REDACTED
    is_command_container = bool(set(name_parts).intersection({"command", "commands", "argv"}))
    if (
        is_command_container
        and isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, os.PathLike))
        and bool(value)
        and all(isinstance(item, str | os.PathLike) for item in value)
    ):
        return list(sanitize_command(value))
    if is_command_container:
        return _REDACTED
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metadata field {field_name!r} must contain finite numbers.")
        return value
    if isinstance(value, str):
        return _sanitize_structured_string(value)
    if isinstance(value, Mapping):
        return sanitize_metadata_fields(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, os.PathLike)):
        if _contains_sensitive_command_argument(value):
            if all(isinstance(item, str | os.PathLike) for item in value):
                return list(sanitize_command(value))
            return _REDACTED
        return [_sanitize_metadata_value(field_name, item) for item in value]
    raise ValueError(f"metadata field {field_name!r} must contain JSON-compatible values.")


def _sanitize_structured_string(value: str) -> str:
    if not value:
        return value
    if re.search(r"(?i)\bauthorization\s*:\s*\S+\s+\S+", value) is not None:
        return _REDACTED
    bearer_match = re.match(r"(?i)^bearer\s+\S+$", value)
    if bearer_match is not None:
        return _REDACTED
    for option in re.finditer(r"(?i)(?<!\w)(--?[a-z][a-z0-9_-]*)(?:=|\s+)\S+", value):
        if _is_sensitive_name(option.group(1)):
            return _REDACTED
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _REDACTED if _looks_credential_bearing(value) else value
    path_segments = [segment for segment in unquote(parsed.path).split("/") if segment]
    has_sensitive_path = any(
        _structured_path_segment_is_sensitive(segment) for segment in path_segments
    )
    if not parsed.netloc:
        for json_field in re.finditer(r'(?i)"([a-z][a-z0-9_-]*)"\s*:\s*"[^"]*"', value):
            if _structured_name_is_sensitive(json_field.group(1)):
                return _REDACTED
        for assignment in re.finditer(
            r"""(?ix)(?<!\w)["']?([a-z][a-z0-9_-]*)["']?\s*=\s*\S+""",
            value,
        ):
            if _structured_name_is_sensitive(assignment.group(1)):
                return _REDACTED
        for label in re.finditer(
            r"""(?ix)(?<!\w)["']?([a-z][a-z0-9_-]*)["']?\s*:\s*\S+""",
            value,
        ):
            if _structured_name_is_sensitive(label.group(1)):
                return _REDACTED
        if "://" in value and "@" in value:
            return _REDACTED
        if has_sensitive_path:
            return _REDACTED
        return value

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"{_REDACTED}@{netloc.rsplit('@', maxsplit=1)[1]}"
    path = parsed.path
    if has_sensitive_path:
        path = f"/{_REDACTED}"
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    query = parsed.query
    if any(_structured_query_key_is_sensitive(key) for key, _ in query_items):
        query = urlencode(
            [
                (key, _REDACTED if _structured_query_key_is_sensitive(key) else item)
                for key, item in query_items
            ],
            doseq=True,
            safe="<>",
        )
    fragment_items = parse_qsl(parsed.fragment, keep_blank_values=True)
    fragment = parsed.fragment
    if any(_structured_query_key_is_sensitive(key) for key, _ in fragment_items):
        fragment = urlencode(
            [
                (key, _REDACTED if _structured_query_key_is_sensitive(key) else item)
                for key, item in fragment_items
            ],
            doseq=True,
            safe="<>",
        )
    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


def _structured_query_key_is_sensitive(key: str) -> bool:
    return _structured_name_is_sensitive(key)


def _structured_path_segment_is_sensitive(segment: str) -> bool:
    parts = _structured_name_parts(segment)
    if not parts:
        return False
    normalized = "".join(parts)
    return (
        normalized in _STRUCTURED_SENSITIVE_PATH_SEGMENTS
        or parts[-1] in _STRUCTURED_SENSITIVE_PATH_SUFFIXES
        or (
            parts[-1] == "key"
            and bool(set(parts[:-1]).intersection({"api", "access", "private", "auth", "secret"}))
        )
    )


def _structured_name_is_sensitive(name: str) -> bool:
    parts = _structured_name_parts(name)
    if not parts:
        return False
    normalized = "".join(parts)
    return (
        normalized in _STRUCTURED_SENSITIVE_NAMES
        or parts[-1] in _STRUCTURED_SENSITIVE_SUFFIXES
        or (
            len(parts) >= 2
            and parts[-1] in _STRUCTURED_VALUE_SUFFIXES
            and parts[-2] in _STRUCTURED_SENSITIVE_SUFFIXES
        )
        or (
            parts[-1] == "key"
            and bool(set(parts[:-1]).intersection({"api", "access", "private", "auth", "secret"}))
        )
        or (
            len(parts) >= 3
            and parts[-2:] == ("key", "id")
            and bool(set(parts[:-2]).intersection({"api", "access", "private", "auth", "secret"}))
        )
    )


def _freeze_metadata_mapping(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_metadata_value(value) for key, value in fields.items()})


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_metadata_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_metadata_value(item) for item in value)
    return value


def _thaw_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata_value(item) for item in value]
    return value


def _structured_name_parts(name: str) -> tuple[str, ...]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return tuple(part for part in re.split(r"[^a-z0-9]+", separated.lower()) if part)


def _contains_sensitive_command_argument(value: Sequence[Any]) -> bool:
    if not value:
        return False
    for item in value:
        if not isinstance(item, str | os.PathLike):
            continue
        argument = os.fspath(item)
        name, separator, _ = argument.partition("=")
        if (argument.startswith("-") or separator) and _is_sensitive_name(name):
            return True
    return False


def _path_string(value: str | os.PathLike[str], *, name: str) -> str:
    try:
        path_value = os.fspath(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a non-empty string or path-like value.") from error
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{name} must be a non-empty string or path-like value.")
    return path_value


def _validated_command_arguments(
    command: Sequence[str | os.PathLike[str]],
) -> tuple[str, ...]:
    if isinstance(command, (str, bytes, os.PathLike)):
        raise ValueError("command must be a sequence of arguments, not a scalar.")
    try:
        raw_arguments = tuple(command)
    except TypeError as error:
        raise ValueError("command must be a sequence of arguments.") from error
    if not raw_arguments:
        raise ValueError("command must contain at least one argument.")
    return tuple(_path_string(argument, name="command argument") for argument in raw_arguments)


def _required_string(payload: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"Execution metadata field {key!r} must be a string.")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Execution metadata field {key!r} must be a string or null.")
    return value


def _string_tuple(
    payload: Mapping[str, Any],
    key: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Execution metadata field {key!r} must be a list of strings.")
    if require_nonempty and (not value or any(not item for item in value)):
        raise ValueError(f"Execution metadata field {key!r} must contain non-empty strings.")
    return tuple(value)


def _required_nonnegative_int(payload: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"Execution metadata field {key!r} must be an integer >= {minimum}.")
    return value


def _optional_nonnegative_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"Execution metadata field {key!r} must be a non-negative integer or null."
        )
    return value


def _optional_execution_id(payload: Mapping[str, Any], key: str) -> str | None:
    value = _optional_string(payload, key)
    return validate_execution_id(value) if value is not None else None


def _validate_starting_coordinate(name: str, value: int | None) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None, got {value!r}.")


def _publish_metadata(metadata_path: Path, metadata: ExecutionMetadata) -> None:
    temporary_path = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(metadata.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, metadata_path)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        with suppress(OSError):
            temporary_path.unlink()
        raise


def _load_metadata(metadata_path: Path) -> ExecutionMetadata:
    with metadata_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Execution metadata must contain a JSON object: {metadata_path}.")
    return ExecutionMetadata.from_dict(payload)


def _valid_execution_records(run_dir: Path) -> list[ExecutionMetadata]:
    executions_dir = executions_dir_for(run_dir)
    if not executions_dir.is_dir():
        return []
    records: list[ExecutionMetadata] = []
    for child in executions_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            metadata = _load_metadata(child / EXECUTION_METADATA_FILENAME)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if metadata.execution_id == child.name:
            records.append(metadata)
    return records
