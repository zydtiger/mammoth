"""Durable, recoverable publication transactions for local artifact groups.

This module builds on :mod:`mammoth.core.artifacts` exact-byte file receipts to
publish caller-prepared files and directory trees across several stable paths.
It owns local transaction topology checks, advisory leases, journals, sealing,
and restart recovery; callers retain artifact creation and semantic validation.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mammoth.core.artifacts import ArtifactReceipt, inspect_artifact

type ArtifactKind = Literal["file", "directory"]
type PublicationMode = Literal["create_only", "replace"]
type RecoveryPolicy = Literal["roll_forward", "rollback_before_commit"]
type ArtifactValidator = Callable[[Path], object]

_JOURNAL_VERSION = 2
_TRANSACTION_DIRECTORY_NAME = ".mammoth-transactions"
_JOURNAL_DIRECTORY_NAME = "journals"
_TRANSACTION_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_ARTIFACT_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")
_JOURNAL_STATES = frozenset({"prepared", "committed"})
_ARTIFACT_STATES = frozenset(
    {"pending", "backup_moving", "backup_moved", "publishing", "published"}
)
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_ACTIVE_TRANSACTION_ROOTS: ContextVar[tuple[tuple[Path, int], ...] | None] = ContextVar(
    "mammoth_active_transaction_roots", default=None
)


class ArtifactTransactionError(RuntimeError):
    """Base error for multi-artifact publication and recovery failures."""


class ArtifactTransactionValidationError(ArtifactTransactionError):
    """Raised when a plan or local filesystem topology is unsafe."""


class ArtifactTransactionConflictError(ArtifactTransactionError):
    """Raised when another publisher holds an overlapping target lease."""


class ArtifactTransactionRecoveryRequired(ArtifactTransactionError):
    """Raised when an existing journal must be recovered explicitly."""


class ArtifactTransactionRecoveryError(ArtifactTransactionError):
    """Raised when recovery cannot authenticate an object before changing it."""


@dataclass(frozen=True, slots=True)
class TransactionArtifact:
    """One caller-prepared staged object and its stable publication target.

    ``stage`` must be the transaction-specific sibling path returned by
    :func:`transaction_stage_path`.  ``validator`` receives the sealed stage
    and every visible final generation, but Mammoth never interprets its data.
    """

    key: str
    stage: Path
    target: Path
    kind: ArtifactKind
    validator: ArtifactValidator | None = None


@dataclass(frozen=True, slots=True)
class ArtifactTransactionPlan:
    """Immutable caller contract for one local publication transaction.

    ``lease_root`` is the pre-existing coordinator root that owns the journal.
    ``lease_roots`` optionally declares the local filesystem boundaries that
    contain artifacts. Omitting it preserves the original one-root contract.
    ``create_only`` plans roll forward after interruption; ``replace`` plans
    restore their authenticated prior generation before the commit point.
    """

    transaction_id: str
    lease_root: Path
    artifacts: tuple[TransactionArtifact, ...]
    mode: PublicationMode
    recovery_policy: RecoveryPolicy
    lease_roots: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class TransactionObjectIdentity:
    """Transaction-local identity used to authenticate one file or tree."""

    kind: ArtifactKind
    device: int
    inode: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactTransactionResult:
    """Outcome of publication or recovery without hiding retained evidence."""

    transaction_id: str
    committed_targets: tuple[Path, ...]
    restored_targets: tuple[Path, ...]
    cleanup_complete: bool
    preserved_evidence: tuple[Path, ...] = ()


@dataclass(slots=True)
class _TransactionLease:
    """One held advisory target lease managed by a transaction operation."""

    path: Path
    descriptor: int

    def close(self) -> None:
        """Release this lock while preserving its harmless lease inode."""
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)


@dataclass(slots=True)
class _JournalHandle:
    """Current exact-byte receipt that authorizes one journal replacement."""

    path: Path
    receipt: ArtifactReceipt


def transaction_stage_path(plan: ArtifactTransactionPlan, key: str) -> Path:
    """Return the only safe caller-owned staging path for one plan artifact."""
    artifact = transaction_artifact_for_key(plan, key)
    return artifact.target.parent / f".mammoth-txn-{plan.transaction_id}-{key}.stage"


def transaction_journal_path(plan: ArtifactTransactionPlan) -> Path:
    """Return the durable journal location confined below ``plan.lease_root``."""
    return (
        Path(plan.lease_root)
        / _TRANSACTION_DIRECTORY_NAME
        / _JOURNAL_DIRECTORY_NAME
        / f"{plan.transaction_id}.json"
    )


def transaction_retired_journal_path(plan: ArtifactTransactionPlan) -> Path:
    """Return the deterministic post-commit journal location used during deletion."""
    journal = transaction_journal_path(plan)
    return journal.with_name(f"{journal.name}.retired")


def transaction_journal_swap_path(path: Path) -> Path:
    """Return the deterministic predecessor name used by journal compare-and-swap."""
    return path.with_name(f".{path.name}.swap")


def transaction_retired_object_path(
    plan: ArtifactTransactionPlan, artifact: TransactionArtifact, role: str
) -> Path:
    """Return a private deterministic cleanup location for one committed remnant."""
    if role not in {"stage", "backup", "rollback-stage", "rollback-backup", "rollback-target"}:
        raise ValueError(f"transaction cleanup role is unsupported: {role}")
    return (
        transaction_artifact_lease_root(plan, artifact)
        / _TRANSACTION_DIRECTORY_NAME
        / "retired"
        / plan.transaction_id
        / f"{artifact.key}-{role}"
    )


def locate_transaction_journal(plan: ArtifactTransactionPlan) -> Path:
    """Find the one recoverable journal name, including deterministic retirement."""
    journal = transaction_journal_path(plan)
    retired = transaction_retired_journal_path(plan)
    journal_exists = object_exists(journal)
    retired_exists = object_exists(retired)
    if journal_exists and retired_exists:
        raise ArtifactTransactionRecoveryError(
            "both active and retired transaction journals are visible; "
            f"preserving evidence: {journal}"
        )
    if journal_exists:
        return journal
    if retired_exists:
        return retired
    raise FileNotFoundError(f"transaction journal is missing: {journal}")


def seal_artifact_transaction(
    plan: ArtifactTransactionPlan,
) -> tuple[TransactionObjectIdentity, ...]:
    """Validate and durably seal every caller-prepared staged artifact.

    :func:`publish_artifact_transaction` invokes this automatically.  Exposing
    it lets callers fail early after writing their stages, but a later publish
    seals and re-authenticates again before it creates a journal.
    """
    validated = validate_artifact_transaction_plan(plan)
    return tuple(
        inspect_transaction_object(artifact.stage, artifact.kind, synchronize=True)
        for artifact in validated.artifacts
    )


def publish_artifact_transaction(plan: ArtifactTransactionPlan) -> ArtifactTransactionResult:
    """Seal, journal, publish, validate, and clean up one artifact transaction.

    An existing journal is intentionally not recovered implicitly: callers must
    invoke :func:`recover_artifact_transaction` with their expected plan so an
    interrupted operation cannot be mistaken for a new invocation.
    """
    normalized = validate_artifact_transaction_plan(plan, allow_missing_stages=True)
    with claim_artifact_transaction_leases(normalized):
        journal_path = transaction_journal_path(normalized)
        if object_exists(journal_path) or object_exists(
            transaction_retired_journal_path(normalized)
        ):
            raise ArtifactTransactionRecoveryRequired(
                f"transaction journal already exists; recover explicitly: {journal_path}"
            )
        validated = validate_artifact_transaction_plan(normalized)
        records = create_transaction_records(validated)
        journal = create_journal_payload(validated, records)
        journal_handle = create_transaction_journal(journal_path, journal)
        try:
            publish_journaled_transaction(validated, journal_handle, journal)
            return cleanup_committed_transaction(validated, journal_handle, journal)
        except BaseException:
            raise


def recover_artifact_transaction(plan: ArtifactTransactionPlan) -> ArtifactTransactionResult:
    """Idempotently recover one recorded transaction against its expected plan."""
    validated = validate_artifact_transaction_plan(plan, allow_missing_stages=True)
    with claim_artifact_transaction_leases(validated):
        journal_path = locate_transaction_journal(validated)
        journal, journal_handle = read_transaction_journal(journal_path)
        validate_journal_matches_plan(journal, validated)
        cleanup_transaction_journal_swap(validated)
        if journal["state"] == "committed":
            validate_committed_generation(validated, journal)
            return cleanup_committed_transaction(validated, journal_handle, journal)
        if validated.recovery_policy == "roll_forward":
            publish_journaled_transaction(validated, journal_handle, journal)
            return cleanup_committed_transaction(validated, journal_handle, journal)
        return rollback_journaled_transaction(validated, journal_handle, journal)


def validate_artifact_transaction_plan(
    plan: ArtifactTransactionPlan, *, allow_missing_stages: bool = False
) -> ArtifactTransactionPlan:
    """Reject unsafe topology before a transaction creates state or mutates targets."""
    if not isinstance(plan, ArtifactTransactionPlan):
        raise TypeError("plan must be an ArtifactTransactionPlan")
    if not isinstance(plan.transaction_id, str) or not _TRANSACTION_ID_PATTERN.fullmatch(
        plan.transaction_id
    ):
        raise ArtifactTransactionValidationError(
            "transaction_id must contain lowercase letters, digits, and hyphens"
        )
    if plan.mode not in {"create_only", "replace"}:
        raise ArtifactTransactionValidationError(
            "transaction mode must be 'create_only' or 'replace'"
        )
    if plan.recovery_policy not in {"roll_forward", "rollback_before_commit"}:
        raise ArtifactTransactionValidationError("transaction recovery policy is invalid")
    if (plan.mode == "create_only" and plan.recovery_policy != "roll_forward") or (
        plan.mode == "replace" and plan.recovery_policy != "rollback_before_commit"
    ):
        raise ArtifactTransactionValidationError(
            "create_only requires roll_forward and replace requires rollback_before_commit"
        )
    lease_root = require_safe_directory(Path(plan.lease_root), "lease_root")
    lease_roots = normalize_transaction_lease_roots(plan, lease_root)
    if not plan.artifacts or len(plan.artifacts) < 2:
        raise ArtifactTransactionValidationError(
            "a transaction must contain at least two artifacts"
        )
    keys: set[str] = set()
    normalized: list[TransactionArtifact] = []
    for artifact in plan.artifacts:
        if not isinstance(artifact, TransactionArtifact):
            raise TypeError("transaction artifacts must be TransactionArtifact values")
        if not isinstance(artifact.key, str) or not _ARTIFACT_KEY_PATTERN.fullmatch(artifact.key):
            raise ArtifactTransactionValidationError("transaction artifact key is unsafe")
        if artifact.key in keys:
            raise ArtifactTransactionValidationError("transaction artifact keys must be unique")
        keys.add(artifact.key)
        if artifact.kind not in {"file", "directory"}:
            raise ArtifactTransactionValidationError("transaction artifact kind is invalid")
        if artifact.validator is not None and not callable(artifact.validator):
            raise TypeError("transaction artifact validator must be callable or None")
        target = normalize_transaction_path(artifact.target, "target")
        stage = normalize_transaction_path(artifact.stage, "stage")
        target_root = transaction_root_for_path(target, lease_roots, "target")
        stage_root = transaction_root_for_path(stage, lease_roots, "stage")
        if target_root != stage_root:
            raise ArtifactTransactionValidationError(
                "transaction target and stage must share one declared lease_root"
            )
        if target == target_root or target.is_relative_to(
            target_root / _TRANSACTION_DIRECTORY_NAME
        ):
            raise ArtifactTransactionValidationError("transaction target overlaps Mammoth metadata")
        if target.parent != stage.parent:
            raise ArtifactTransactionValidationError(
                "transaction stage must be a sibling of its target"
            )
        expected_stage = target.parent / f".mammoth-txn-{plan.transaction_id}-{artifact.key}.stage"
        if stage != expected_stage:
            raise ArtifactTransactionValidationError(
                f"transaction stage must be the reserved sibling path: {expected_stage}"
            )
        if target == stage:
            raise ArtifactTransactionValidationError("transaction stage and target must differ")
        validate_existing_object(stage, artifact.kind, "stage", allow_missing=allow_missing_stages)
        validate_existing_object(target, artifact.kind, "target", allow_missing=True)
        if target.parent.stat().st_dev != target_root.stat().st_dev:
            raise ArtifactTransactionValidationError(
                "transaction target parent must share its lease_root's filesystem"
            )
        if object_exists(stage) and stage.stat().st_dev != target_root.stat().st_dev:
            raise ArtifactTransactionValidationError(
                "transaction stage must share its lease_root's filesystem"
            )
        normalized.append(
            TransactionArtifact(
                key=artifact.key,
                stage=stage,
                target=target,
                kind=artifact.kind,
                validator=artifact.validator,
            )
        )
    normalized_plan = ArtifactTransactionPlan(
        transaction_id=plan.transaction_id,
        lease_root=lease_root,
        artifacts=tuple(normalized),
        mode=plan.mode,
        recovery_policy=plan.recovery_policy,
        lease_roots=lease_roots,
    )
    for index, first in enumerate(normalized):
        for second in normalized[index + 1 :]:
            validate_artifact_topology_pair(normalized_plan, first, second)
    return normalized_plan


def validate_artifact_topology_pair(
    plan: ArtifactTransactionPlan,
    first: TransactionArtifact,
    second: TransactionArtifact,
) -> None:
    """Reject every cross-artifact target, stage, or backup overlap before journaling."""
    first_paths = (
        (first.target, first.kind),
        (first.stage, first.kind),
        (transaction_backup_path(plan, first), first.kind),
    )
    second_paths = (
        (second.target, second.kind),
        (second.stage, second.kind),
        (transaction_backup_path(plan, second), second.kind),
    )
    for first_path, first_kind in first_paths:
        for second_path, second_kind in second_paths:
            if transaction_paths_overlap(first_path, first_kind, second_path, second_kind):
                raise ArtifactTransactionValidationError(
                    "transaction artifact target, stage, and backup paths must not overlap"
                )


def transaction_paths_overlap(
    first_path: Path,
    first_kind: ArtifactKind,
    second_path: Path,
    second_kind: ArtifactKind,
) -> bool:
    """Return whether two declared file-or-directory artifact objects overlap."""
    return (
        first_path == second_path
        or (first_kind == "directory" and second_path.is_relative_to(first_path))
        or (second_kind == "directory" and first_path.is_relative_to(second_path))
    )


def transaction_artifact_for_key(plan: ArtifactTransactionPlan, key: str) -> TransactionArtifact:
    """Find one artifact by stable key without accepting ambiguous lookup."""
    for artifact in plan.artifacts:
        if artifact.key == key:
            return artifact
    raise KeyError(f"transaction plan has no artifact key {key!r}")


def require_safe_directory(path: Path, label: str) -> Path:
    """Return one existing non-symlink directory with no symlinked ancestors."""
    absolute = Path(os.path.abspath(path))
    require_no_symlink_components(absolute)
    try:
        file_stat = os.lstat(absolute)
    except FileNotFoundError:
        raise ArtifactTransactionValidationError(
            f"{label} must already exist: {absolute}"
        ) from None
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISDIR(file_stat.st_mode):
        raise ArtifactTransactionValidationError(f"{label} must be a real directory: {absolute}")
    return absolute


def normalize_transaction_lease_roots(
    plan: ArtifactTransactionPlan, coordinator_root: Path
) -> tuple[Path, ...]:
    """Validate caller-selected artifact roots without inferring mount boundaries."""
    raw_roots = plan.lease_roots or (coordinator_root,)
    roots = tuple(require_safe_directory(Path(root), "lease_root") for root in raw_roots)
    if len(set(roots)) != len(roots):
        raise ArtifactTransactionValidationError("transaction lease_roots must be unique")
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ArtifactTransactionValidationError(
                    "transaction lease_roots must not overlap or nest"
                )
    return tuple(sorted(roots, key=str))


def transaction_root_for_path(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    """Find the one declared root that confines one artifact path."""
    matches = tuple(root for root in roots if path.is_relative_to(root))
    if len(matches) != 1:
        raise ArtifactTransactionValidationError(
            "transaction "
            f"{label} must be beneath lease_root or exactly one declared lease root: {path}"
        )
    return matches[0]


def transaction_artifact_lease_root(
    plan: ArtifactTransactionPlan, artifact: TransactionArtifact
) -> Path:
    """Return the declared local root that owns one artifact's durable mutations."""
    roots = plan.lease_roots or (Path(plan.lease_root),)
    return transaction_root_for_path(Path(artifact.target), roots, "target")


def normalize_transaction_path(path: Path, label: str) -> Path:
    """Normalize one no-follow artifact path before assigning its declared root."""
    if not isinstance(path, Path):
        raise TypeError(f"transaction {label} must be a pathlib.Path")
    absolute = Path(os.path.abspath(path))
    require_no_symlink_components(absolute.parent)
    if absolute.exists() or absolute.is_symlink():
        final_stat = os.lstat(absolute)
        if stat.S_ISLNK(final_stat.st_mode):
            raise ArtifactTransactionValidationError(f"transaction {label} must not be a symlink")
    return absolute


def require_no_symlink_components(path: Path) -> None:
    """Reject every existing symlink component instead of resolving through it."""
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    current = Path(parts[0])
    for component in parts[1:]:
        current /= component
        if not current.exists() and not current.is_symlink():
            break
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(current_stat.st_mode):
            raise ArtifactTransactionValidationError(
                f"transaction path contains a symlink: {current}"
            )


def validate_existing_object(
    path: Path,
    kind: ArtifactKind,
    label: str,
    *,
    allow_missing: bool = False,
) -> None:
    """Require an existing regular file or safe directory when one is present."""
    try:
        object_stat = os.lstat(path)
    except FileNotFoundError:
        if allow_missing:
            return
        raise ArtifactTransactionValidationError(
            f"transaction {label} is missing: {path}"
        ) from None
    if stat.S_ISLNK(object_stat.st_mode):
        raise ArtifactTransactionValidationError(
            f"transaction {label} must not be a symlink: {path}"
        )
    expected = stat.S_ISREG if kind == "file" else stat.S_ISDIR
    if not expected(object_stat.st_mode):
        raise ArtifactTransactionValidationError(
            f"transaction {label} kind does not match declared {kind}: {path}"
        )


def claim_artifact_transaction_leases(plan: ArtifactTransactionPlan) -> _TransactionLeases:
    """Acquire every target and ancestor lease in deterministic order."""
    validated = validate_artifact_transaction_plan(plan, allow_missing_stages=True)
    leases: list[_TransactionLease] = []
    root_descriptors: list[tuple[Path, int]] = []
    try:
        roots = tuple(dict.fromkeys((validated.lease_root, *validated.lease_roots)))
        for root in roots:
            root_descriptors.append((root, open_absolute_directory_without_symlinks(root)))
        for protected_path, _root in transaction_protected_paths(validated):
            digest = hashlib.sha256(str(protected_path).encode("utf-8")).hexdigest()
            lease_path = protected_path.parent / f".mammoth-txn-lease-{digest}.lock"
            descriptor = open_transaction_lease(lease_path)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                os.close(descriptor)
                raise ArtifactTransactionConflictError(
                    f"another transaction holds an overlapping target lease: {protected_path}"
                ) from error
            leases.append(_TransactionLease(path=lease_path, descriptor=descriptor))
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        for _root, descriptor in reversed(root_descriptors):
            os.close(descriptor)
        raise
    return _TransactionLeases(leases, root_descriptors)


@dataclass(slots=True)
class _TransactionLeases:
    """Context manager that owns all leases acquired for one operation."""

    leases: list[_TransactionLease]
    root_descriptors: list[tuple[Path, int]]
    context_token: Token[tuple[tuple[Path, int], ...] | None] | None = None

    def __enter__(self) -> _TransactionLeases:
        """Keep all target locks until publication or recovery has finished."""
        self.context_token = _ACTIVE_TRANSACTION_ROOTS.set(tuple(self.root_descriptors))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release every lease in reverse order after the operation boundary."""
        if self.context_token is not None:
            _ACTIVE_TRANSACTION_ROOTS.reset(self.context_token)
            self.context_token = None
        for lease in reversed(self.leases):
            lease.close()
        for _root, descriptor in reversed(self.root_descriptors):
            os.close(descriptor)


def transaction_protected_paths(plan: ArtifactTransactionPlan) -> tuple[tuple[Path, Path], ...]:
    """Return ordered target and ancestor identities so overlap shares a lease."""
    protected: set[tuple[Path, Path]] = set()
    for artifact in plan.artifacts:
        root = transaction_artifact_lease_root(plan, artifact)
        current = artifact.target
        while True:
            protected.add((current, root))
            if current == root:
                break
            current = current.parent
    return tuple(sorted(protected, key=lambda item: (str(item[1]), str(item[0]))))


def ensure_transaction_metadata_directory(lease_root: Path) -> Path:
    """Create and authenticate Mammoth's metadata directory below one lease root."""
    metadata_directory = lease_root / _TRANSACTION_DIRECTORY_NAME
    ensure_confined_directory(metadata_directory, lease_root)
    return metadata_directory


def ensure_transaction_journal_directory(lease_root: Path) -> Path:
    """Create and sync the journal directory before any journal file can appear."""
    metadata_directory = ensure_transaction_metadata_directory(lease_root)
    journal_directory = metadata_directory / _JOURNAL_DIRECTORY_NAME
    ensure_confined_directory(journal_directory, lease_root)
    return journal_directory


def ensure_transaction_retired_directory(
    plan: ArtifactTransactionPlan, artifact: TransactionArtifact
) -> Path:
    """Create the private deterministic holding directory for one transaction cleanup."""
    root = transaction_artifact_lease_root(plan, artifact)
    metadata_directory = ensure_transaction_metadata_directory(root)
    retired_directory = metadata_directory / "retired"
    transaction_directory = retired_directory / plan.transaction_id
    ensure_confined_directory(retired_directory, root)
    ensure_confined_directory(transaction_directory, root)
    return transaction_directory


def ensure_confined_directory(path: Path, lease_root: Path) -> None:
    """Create a metadata directory through a root-anchored no-follow parent FD."""
    created = False
    try:
        descriptor = open_confined_directory(path, lease_root=lease_root)
    except FileNotFoundError:
        try:
            parent_descriptor, name = open_confined_parent(path, lease_root=lease_root)
        except OSError as error:
            raise ArtifactTransactionValidationError(
                f"transaction metadata path is unsafe: {path}"
            ) from error
        try:
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
                created = True
            except FileExistsError:
                pass
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        try:
            descriptor = open_confined_directory(path, lease_root=lease_root)
        except OSError as error:
            raise ArtifactTransactionValidationError(
                f"transaction metadata path is unsafe: {path}"
            ) from error
    except OSError as error:
        raise ArtifactTransactionValidationError(
            f"transaction metadata path is unsafe: {path}"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ArtifactTransactionValidationError(f"transaction metadata path is unsafe: {path}")
    finally:
        os.close(descriptor)
    if created:
        sync_directory_strict(path.parent, lease_root=lease_root)


def open_transaction_lease(path: Path) -> int:
    """Open one regular no-follow advisory lock file without truncating it."""
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
    if not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("artifact transaction leases require os.O_NOFOLLOW")
    flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactTransactionValidationError(f"transaction lease is not regular: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def create_transaction_records(plan: ArtifactTransactionPlan) -> list[dict[str, Any]]:
    """Seal stages and capture the authenticated prior generation before journaling."""
    records: list[dict[str, Any]] = []
    for artifact in plan.artifacts:
        root = transaction_artifact_lease_root(plan, artifact)
        stage_identity = inspect_transaction_object(artifact.stage, artifact.kind, synchronize=True)
        run_artifact_validator(artifact, artifact.stage, "staged")
        sync_directory_strict(artifact.stage.parent, lease_root=root)
        original_identity = (
            inspect_transaction_object(artifact.target, artifact.kind, synchronize=False)
            if artifact.target.exists()
            else None
        )
        if plan.mode == "create_only" and original_identity is not None:
            raise FileExistsError(
                f"create-only transaction target already exists: {artifact.target}"
            )
        records.append(
            {
                "key": artifact.key,
                "stage": str(artifact.stage),
                "target": str(artifact.target),
                "backup": str(transaction_backup_path(plan, artifact)),
                "kind": artifact.kind,
                "stage_identity": identity_to_json(stage_identity),
                "original_identity": (
                    identity_to_json(original_identity) if original_identity is not None else None
                ),
                "status": "pending",
            }
        )
    return records


def transaction_backup_path(plan: ArtifactTransactionPlan, artifact: TransactionArtifact) -> Path:
    """Return a reserved same-directory backup path for one replacement target."""
    return artifact.target.parent / f".mammoth-txn-{plan.transaction_id}-{artifact.key}.backup"


def inspect_transaction_object(
    path: Path,
    kind: ArtifactKind,
    *,
    synchronize: bool,
) -> TransactionObjectIdentity:
    """Authenticate and optionally seal one regular file or ordinary directory tree."""
    validate_existing_object(path, kind, "object")
    object_stat = os.lstat(path)
    if kind == "file":
        receipt = inspect_artifact(path)
        if synchronize:
            sync_file_strict(path)
        return identity_from_receipt(receipt, object_stat)
    digest = hashlib.sha256()
    inspect_directory_tree(path, path, digest, synchronize=synchronize)
    return TransactionObjectIdentity(
        kind="directory",
        device=object_stat.st_dev,
        inode=object_stat.st_ino,
        size_bytes=0,
        sha256=digest.hexdigest(),
    )


def inspect_directory_tree(
    root: Path,
    directory: Path,
    digest: Any,
    *,
    synchronize: bool,
) -> None:
    """Walk one tree without symlinks or special files and build its local identity."""
    relative = directory.relative_to(root)
    digest.update(b"D\0")
    digest.update(str(relative).encode("utf-8"))
    digest.update(b"\0")
    entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    for entry in entries:
        entry_stat = os.lstat(entry)
        relative_entry = entry.relative_to(root)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ArtifactTransactionValidationError(
                f"transaction tree contains a symlink: {entry}"
            )
        if stat.S_ISDIR(entry_stat.st_mode):
            inspect_directory_tree(root, entry, digest, synchronize=synchronize)
            continue
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ArtifactTransactionValidationError(
                f"transaction tree contains a special file: {entry}"
            )
        receipt = inspect_artifact(entry)
        digest.update(b"F\0")
        digest.update(str(relative_entry).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(receipt.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(receipt.sha256.encode("ascii"))
        digest.update(b"\0")
        if synchronize:
            sync_file_strict(entry)
    if synchronize:
        sync_directory_strict(directory)


def identity_from_receipt(
    receipt: ArtifactReceipt, object_stat: os.stat_result
) -> TransactionObjectIdentity:
    """Reuse Mammoth's regular-file receipt as a transaction object identity."""
    return TransactionObjectIdentity(
        kind="file",
        device=object_stat.st_dev,
        inode=object_stat.st_ino,
        size_bytes=receipt.size_bytes,
        sha256=receipt.sha256,
    )


def run_artifact_validator(artifact: TransactionArtifact, path: Path, phase: str) -> None:
    """Run caller-owned semantic validation without assigning artifact meaning."""
    if artifact.validator is None:
        return
    try:
        artifact.validator(path)
    except BaseException as error:
        raise ArtifactTransactionValidationError(
            f"transaction validator failed for {artifact.key} during {phase}: {path}"
        ) from error


def sync_file_strict(path: Path) -> None:
    """Synchronize one regular file and fail on unsupported durability semantics."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("artifact transactions require os.O_NOFOLLOW")
    descriptor = os.open(path, flags | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactTransactionValidationError(f"transaction file is not regular: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_directory_strict(path: Path, *, lease_root: Path | None = None) -> None:
    """Synchronize a parent directory after every recovery-relevant name mutation."""
    descriptor = open_confined_directory(path, lease_root=lease_root)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ArtifactTransactionValidationError(f"transaction directory is unsafe: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def open_confined_directory(path: Path, *, lease_root: Path | None = None) -> int:
    """Open a directory without allowing a later ancestor symlink to redirect it."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("artifact transactions require os.O_NOFOLLOW")
    if lease_root is None:
        return os.open(path, flags | os.O_NOFOLLOW)
    root = Path(lease_root)
    try:
        relative = Path(path).relative_to(root)
    except ValueError as error:
        raise ArtifactTransactionValidationError(
            f"transaction path escapes lease_root: {path}"
        ) from error
    active_roots = _ACTIVE_TRANSACTION_ROOTS.get()
    active_root = None
    if active_roots is not None:
        active_root = next(
            (
                (active_path, descriptor)
                for active_path, descriptor in active_roots
                if active_path == root
            ),
            None,
        )
    if active_root is not None:
        current_descriptor = open_absolute_directory_without_symlinks(root)
        try:
            active_stat = os.fstat(active_root[1])
            current_stat = os.fstat(current_descriptor)
            if (active_stat.st_dev, active_stat.st_ino) != (
                current_stat.st_dev,
                current_stat.st_ino,
            ):
                raise ArtifactTransactionRecoveryError(
                    f"transaction lease_root changed during publication: {root}"
                )
        finally:
            os.close(current_descriptor)
        descriptor = os.dup(active_root[1])
    else:
        descriptor = open_absolute_directory_without_symlinks(root)
    try:
        for component in relative.parts:
            next_descriptor = os.open(component, flags | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_absolute_directory_without_symlinks(path: Path) -> int:
    """Anchor an absolute directory FD by opening every ancestor with no-follow."""
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("artifact transactions require os.O_NOFOLLOW")
    descriptor = os.open(absolute.anchor, flags | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_confined_parent(path: Path, *, lease_root: Path | None = None) -> tuple[int, str]:
    """Open one final path's parent through an anchored no-follow directory walk."""
    return open_confined_directory(path.parent, lease_root=lease_root), path.name


def identity_to_json(identity: TransactionObjectIdentity) -> dict[str, object]:
    """Serialize one bounded transaction-local identity into the strict journal."""
    return {
        "kind": identity.kind,
        "device": identity.device,
        "inode": identity.inode,
        "size_bytes": identity.size_bytes,
        "sha256": identity.sha256,
    }


def identity_from_json(value: object) -> TransactionObjectIdentity:
    """Decode and strictly validate one journal object identity."""
    mapping = require_exact_mapping(
        value, {"kind", "device", "inode", "size_bytes", "sha256"}, "identity"
    )
    raw_kind = require_string(mapping["kind"], "identity kind")
    if raw_kind == "file":
        kind: ArtifactKind = "file"
    elif raw_kind == "directory":
        kind = "directory"
    else:
        raise ArtifactTransactionRecoveryError("journal identity kind is invalid")
    device = require_nonnegative_integer(mapping["device"], "identity device")
    inode = require_nonnegative_integer(mapping["inode"], "identity inode")
    size_bytes = require_nonnegative_integer(mapping["size_bytes"], "identity size_bytes")
    sha256 = require_sha256(mapping["sha256"], "identity sha256")
    return TransactionObjectIdentity(kind, device, inode, size_bytes, sha256)


def create_journal_payload(
    plan: ArtifactTransactionPlan, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the durable schema-v2 record before any target path changes."""
    return {
        "version": _JOURNAL_VERSION,
        "transaction_id": plan.transaction_id,
        "lease_root": str(plan.lease_root),
        "lease_roots": [str(root) for root in plan.lease_roots],
        "mode": plan.mode,
        "recovery_policy": plan.recovery_policy,
        "state": "prepared",
        "artifacts": records,
    }


def create_transaction_journal(path: Path, payload: dict[str, Any]) -> _JournalHandle:
    """Create, write, and sync a no-clobber journal before target mutation."""
    journal_directory = path.parent
    metadata_directory = journal_directory.parent
    if metadata_directory.name != _TRANSACTION_DIRECTORY_NAME:
        raise ArtifactTransactionValidationError(f"transaction journal path is unsafe: {path}")
    lease_root = metadata_directory.parent
    expected_directory = ensure_transaction_journal_directory(lease_root)
    if journal_directory != expected_directory:
        raise ArtifactTransactionValidationError(f"transaction journal path is unsafe: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("artifact transactions require os.O_NOFOLLOW")
    try:
        parent_descriptor, name = open_confined_parent(path, lease_root=lease_root)
    except OSError as error:
        raise ArtifactTransactionValidationError(
            f"transaction journal path is unsafe: {path}"
        ) from error
    try:
        descriptor = os.open(
            name,
            flags | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            write_json_descriptor(descriptor, payload)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    sync_directory_strict(journal_directory, lease_root=lease_root)
    return journal_handle_for_payload(path, payload)


def update_transaction_journal(
    handle: _JournalHandle, payload: dict[str, Any], *, lease_root: Path
) -> None:
    """Atomically replace an authenticated journal and synchronize its parent."""
    path = handle.path
    temporary = transaction_journal_swap_path(path)
    if object_exists(temporary):
        raise ArtifactTransactionRecoveryRequired(
            f"transaction journal update remnant requires recovery: {temporary}"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("artifact transactions require os.O_NOFOLLOW")
    try:
        parent_descriptor, _ = open_confined_parent(path, lease_root=lease_root)
    except OSError as error:
        raise ArtifactTransactionRecoveryError(
            f"transaction journal path is unsafe: {path}"
        ) from error
    try:
        journal_stat = os.lstat(path.name, dir_fd=parent_descriptor)
        if stat.S_ISLNK(journal_stat.st_mode) or not stat.S_ISREG(journal_stat.st_mode):
            raise ArtifactTransactionRecoveryError(f"transaction journal is unsafe: {path}")
        descriptor = os.open(
            temporary.name,
            flags | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            write_json_descriptor(descriptor, payload)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary.name, dir_fd=parent_descriptor)
            raise
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    rename_exchange(path, temporary, lease_root=lease_root)
    sync_directory_strict(path.parent, lease_root=lease_root)
    if not receipt_matches_path(handle.receipt, temporary):
        rename_exchange(path, temporary, lease_root=lease_root)
        sync_directory_strict(path.parent, lease_root=lease_root)
        raise ArtifactTransactionRecoveryError(
            f"transaction journal changed before replacement; preserving evidence: {path}"
        )
    remove_file_at_path(temporary, lease_root=lease_root)
    sync_directory_strict(path.parent, lease_root=lease_root)
    handle.receipt = journal_handle_for_payload(path, payload).receipt


def write_json_descriptor(descriptor: int, payload: dict[str, Any]) -> None:
    """Write deterministic journal bytes through one descriptor and synchronize them."""
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    encoded = f"{serialized}\n".encode()
    offset = 0
    while offset < len(encoded):
        offset += os.write(descriptor, encoded[offset:])
    os.fsync(descriptor)


def journal_handle_for_payload(path: Path, payload: dict[str, Any]) -> _JournalHandle:
    """Bind a just-created or updated journal path to its expected exact bytes."""
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    expected = f"{serialized}\n".encode()
    receipt = inspect_artifact(path)
    if (receipt.size_bytes, receipt.sha256) != (
        len(expected),
        hashlib.sha256(expected).hexdigest(),
    ):
        raise ArtifactTransactionRecoveryError(
            f"transaction journal changed before it could be authenticated: {path}"
        )
    return _JournalHandle(path=path, receipt=receipt)


def receipt_matches_path(receipt: ArtifactReceipt, path: Path) -> bool:
    """Check a renamed journal predecessor without trusting its new pathname."""
    try:
        observed = inspect_artifact(path)
    except (FileNotFoundError, ValueError):
        return False
    return observed.size_bytes == receipt.size_bytes and observed.sha256 == receipt.sha256


def read_transaction_journal(path: Path) -> tuple[dict[str, Any], _JournalHandle]:
    """Read a strict supported journal without accepting symlink substitution."""
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(f"transaction journal is missing: {path}")
    journal_stat = os.lstat(path)
    if stat.S_ISLNK(journal_stat.st_mode) or not stat.S_ISREG(journal_stat.st_mode):
        raise ArtifactTransactionRecoveryError(f"transaction journal is unsafe: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("artifact transactions require os.O_NOFOLLOW")
    descriptor = os.open(path, flags | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactTransactionRecoveryError(f"transaction journal is unsafe: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    serialized = b"".join(chunks)
    try:
        decoded = json.loads(serialized.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactTransactionRecoveryError(
            f"transaction journal is malformed: {path}"
        ) from error
    validate_journal_schema(decoded)
    journal = require_mapping(decoded, "journal")
    receipt = inspect_artifact(path)
    if (receipt.size_bytes, receipt.sha256) != (
        len(serialized),
        hashlib.sha256(serialized).hexdigest(),
    ):
        raise ArtifactTransactionRecoveryError(
            f"transaction journal changed while being read: {path}"
        )
    return journal, _JournalHandle(path=path, receipt=receipt)


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object keys instead of silently accepting ambiguity."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate journal key: {key}")
        result[key] = value
    return result


def validate_journal_schema(value: object) -> None:
    """Require the complete versioned journal shape before recovery reads it."""
    journal = require_mapping(value, "journal")
    version = journal.get("version")
    required = {
        "version",
        "transaction_id",
        "lease_root",
        "mode",
        "recovery_policy",
        "state",
        "artifacts",
    }
    if version == 2:
        required.add("lease_roots")
    elif version != 1:
        raise ArtifactTransactionRecoveryError("transaction journal version is unsupported")
    if set(journal) != required:
        raise ArtifactTransactionRecoveryError("journal fields are unsupported")
    transaction_id = require_string(journal["transaction_id"], "journal transaction_id")
    if not _TRANSACTION_ID_PATTERN.fullmatch(transaction_id):
        raise ArtifactTransactionRecoveryError("journal transaction_id is invalid")
    require_string(journal["lease_root"], "journal lease_root")
    if version == 2:
        roots = journal["lease_roots"]
        if (
            not isinstance(roots, list)
            or not roots
            or not all(isinstance(root, str) and root for root in roots)
            or len(set(roots)) != len(roots)
        ):
            raise ArtifactTransactionRecoveryError("journal lease_roots are invalid")
    if journal["mode"] not in {"create_only", "replace"}:
        raise ArtifactTransactionRecoveryError("journal mode is invalid")
    if journal["recovery_policy"] not in {"roll_forward", "rollback_before_commit"}:
        raise ArtifactTransactionRecoveryError("journal recovery policy is invalid")
    if journal["state"] not in _JOURNAL_STATES:
        raise ArtifactTransactionRecoveryError("journal state is invalid")
    artifacts_value = journal["artifacts"]
    if not isinstance(artifacts_value, list) or not artifacts_value:
        raise ArtifactTransactionRecoveryError("journal artifacts are invalid")
    keys: set[str] = set()
    for record_value in artifacts_value:
        record = require_exact_mapping(
            record_value,
            {
                "key",
                "stage",
                "target",
                "backup",
                "kind",
                "stage_identity",
                "original_identity",
                "status",
            },
            "journal artifact",
        )
        key = require_string(record["key"], "journal artifact key")
        if not _ARTIFACT_KEY_PATTERN.fullmatch(key) or key in keys:
            raise ArtifactTransactionRecoveryError("journal artifact keys are invalid")
        keys.add(key)
        for path_field in ("stage", "target", "backup"):
            require_string(record[path_field], f"journal artifact {path_field}")
        if record["kind"] not in {"file", "directory"}:
            raise ArtifactTransactionRecoveryError("journal artifact kind is invalid")
        kind = record["kind"]
        if identity_from_json(record["stage_identity"]).kind != kind:
            raise ArtifactTransactionRecoveryError(
                "journal stage identity kind does not match its artifact kind"
            )
        if (
            record["original_identity"] is not None
            and identity_from_json(record["original_identity"]).kind != kind
        ):
            raise ArtifactTransactionRecoveryError(
                "journal original identity kind does not match its artifact kind"
            )
        if record["status"] not in _ARTIFACT_STATES:
            raise ArtifactTransactionRecoveryError("journal artifact state is invalid")


def require_mapping(value: object, label: str) -> dict[str, Any]:
    """Require one JSON object with string keys for strict journal decoding."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArtifactTransactionRecoveryError(f"{label} must be an object")
    return value


def require_exact_mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    """Reject unknown or missing fields instead of accepting journal drift."""
    mapping = require_mapping(value, label)
    if set(mapping) != keys:
        raise ArtifactTransactionRecoveryError(f"{label} fields are unsupported")
    return mapping


def require_string(value: object, label: str) -> str:
    """Require one nonempty JSON string field."""
    if not isinstance(value, str) or not value:
        raise ArtifactTransactionRecoveryError(f"{label} must be a nonempty string")
    return value


def require_nonnegative_integer(value: object, label: str) -> int:
    """Require one non-boolean nonnegative JSON integer field."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactTransactionRecoveryError(f"{label} must be a nonnegative integer")
    return value


def require_sha256(value: object, label: str) -> str:
    """Require a lowercase hexadecimal SHA-256 digest in a journal identity."""
    digest = require_string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactTransactionRecoveryError(f"{label} must be lowercase SHA-256")
    return digest


def validate_journal_matches_plan(journal: dict[str, Any], plan: ArtifactTransactionPlan) -> None:
    """Bind an untrusted journal to the caller's complete expected topology."""
    if (
        journal["transaction_id"] != plan.transaction_id
        or journal["lease_root"] != str(plan.lease_root)
        or journal["mode"] != plan.mode
        or journal["recovery_policy"] != plan.recovery_policy
    ):
        raise ArtifactTransactionRecoveryError(
            "journal does not match the expected transaction plan"
        )
    if journal["version"] == 1:
        if plan.lease_roots != (plan.lease_root,):
            raise ArtifactTransactionRecoveryError(
                "schema-v1 journal cannot recover a multi-root transaction"
            )
    elif journal["lease_roots"] != [str(root) for root in plan.lease_roots]:
        raise ArtifactTransactionRecoveryError(
            "journal does not match the expected transaction roots"
        )
    records = journal["artifacts"]
    if not isinstance(records, list) or len(records) != len(plan.artifacts):
        raise ArtifactTransactionRecoveryError("journal artifact topology does not match plan")
    by_key = {require_string(record["key"], "journal artifact key"): record for record in records}
    for artifact in plan.artifacts:
        record = by_key.get(artifact.key)
        if record is None or (
            record["stage"] != str(artifact.stage)
            or record["target"] != str(artifact.target)
            or record["backup"] != str(transaction_backup_path(plan, artifact))
            or record["kind"] != artifact.kind
        ):
            raise ArtifactTransactionRecoveryError("journal artifact topology does not match plan")


def cleanup_transaction_journal_swap(plan: ArtifactTransactionPlan) -> None:
    """Reclaim only a deterministic prior journal that binds to the same plan."""
    swap = transaction_journal_swap_path(transaction_journal_path(plan))
    if not object_exists(swap):
        return
    try:
        prior_journal, _ = read_transaction_journal(swap)
        validate_journal_matches_plan(prior_journal, plan)
    except (ArtifactTransactionRecoveryError, FileNotFoundError) as error:
        raise ArtifactTransactionRecoveryError(
            f"transaction journal update remnant is preserved: {swap}"
        ) from error
    remove_file_at_path(swap, lease_root=plan.lease_root)
    sync_directory_strict(swap.parent, lease_root=plan.lease_root)


def publish_journaled_transaction(
    plan: ArtifactTransactionPlan, journal_handle: _JournalHandle, journal: dict[str, Any]
) -> None:
    """Complete a journaled publication in order and durably mark its commit point."""
    records = journal_records(journal)
    for artifact, record in zip(plan.artifacts, records, strict=True):
        publish_journal_record(plan, journal_handle, journal, artifact, record)
    validate_committed_generation(plan, journal, allow_uncommitted=True)
    journal["state"] = "committed"
    update_transaction_journal(journal_handle, journal, lease_root=plan.lease_root)


def journal_records(journal: dict[str, Any]) -> list[dict[str, Any]]:
    """Return schema-validated artifact records in caller-established plan order."""
    records = journal["artifacts"]
    if not isinstance(records, list):
        raise ArtifactTransactionRecoveryError("journal artifact records are invalid")
    return [require_mapping(record, "journal artifact") for record in records]


def publish_journal_record(
    plan: ArtifactTransactionPlan,
    journal_handle: _JournalHandle,
    journal: dict[str, Any],
    artifact: TransactionArtifact,
    record: dict[str, Any],
) -> None:
    """Publish or authenticate exactly one target while preserving its prior generation."""
    root = transaction_artifact_lease_root(plan, artifact)
    stage_identity = identity_from_json(record["stage_identity"])
    original_identity = (
        identity_from_json(record["original_identity"])
        if record["original_identity"] is not None
        else None
    )
    backup = transaction_backup_path(plan, artifact)
    target_exists = object_exists(artifact.target)
    stage_exists = object_exists(artifact.stage)
    if target_exists and object_matches(artifact.target, stage_identity):
        if stage_exists:
            raise ArtifactTransactionRecoveryError(
                f"both stage and target remain visible for {artifact.key}; preserving evidence"
            )
        run_artifact_validator(artifact, artifact.target, "published")
        record["status"] = "published"
        update_transaction_journal(journal_handle, journal, lease_root=plan.lease_root)
        return
    if target_exists and not stage_exists:
        raise ArtifactTransactionRecoveryError(
            f"target identity is substituted and stage is unavailable: {artifact.target}"
        )
    if not stage_exists:
        raise ArtifactTransactionRecoveryError(
            f"stage is missing before target publication: {artifact.stage}"
        )
    require_matching_object(artifact.stage, stage_identity, "stage")
    if plan.mode == "create_only":
        if target_exists:
            raise ArtifactTransactionRecoveryError(
                f"create-only target is occupied by another object: {artifact.target}"
            )
    else:
        move_original_to_backup(
            plan,
            journal_handle,
            journal,
            artifact,
            record,
            original_identity,
            backup,
            target_exists,
        )
    record["status"] = "publishing"
    update_transaction_journal(journal_handle, journal, lease_root=plan.lease_root)
    rename_without_overwrite(
        artifact.stage, artifact.target, lease_root=root
    )
    sync_directory_strict(artifact.target.parent, lease_root=root)
    require_matching_object(artifact.target, stage_identity, "published target")
    run_artifact_validator(artifact, artifact.target, "published")
    record["status"] = "published"
    update_transaction_journal(journal_handle, journal, lease_root=plan.lease_root)


def move_original_to_backup(
    plan: ArtifactTransactionPlan,
    journal_handle: _JournalHandle,
    journal: dict[str, Any],
    artifact: TransactionArtifact,
    record: dict[str, Any],
    original_identity: TransactionObjectIdentity | None,
    backup: Path,
    target_exists: bool,
) -> None:
    """Durably retain the authenticated original before replacement publication."""
    root = transaction_artifact_lease_root(plan, artifact)
    if original_identity is None:
        if target_exists:
            raise ArtifactTransactionRecoveryError(
                f"replacement target appeared after journal creation: {artifact.target}"
            )
        return
    if object_exists(backup):
        require_matching_object(backup, original_identity, "replacement backup")
        if target_exists:
            raise ArtifactTransactionRecoveryError(
                f"both original target and backup are visible: {artifact.target}"
            )
        record["status"] = "backup_moved"
        update_transaction_journal(journal_handle, journal, lease_root=plan.lease_root)
        return
    if not target_exists:
        raise ArtifactTransactionRecoveryError(
            f"replacement original is missing before backup: {artifact.target}"
        )
    require_matching_object(artifact.target, original_identity, "replacement original")
    record["status"] = "backup_moving"
    update_transaction_journal(journal_handle, journal, lease_root=plan.lease_root)
    rename_without_overwrite(artifact.target, backup, lease_root=root)
    sync_directory_strict(artifact.target.parent, lease_root=root)
    require_matching_object(backup, original_identity, "replacement backup")
    record["status"] = "backup_moved"
    update_transaction_journal(journal_handle, journal, lease_root=plan.lease_root)


def validate_committed_generation(
    plan: ArtifactTransactionPlan, journal: dict[str, Any], *, allow_uncommitted: bool = False
) -> None:
    """Authenticate every visible target before and after the durable commit point."""
    if not allow_uncommitted and journal["state"] != "committed":
        raise ArtifactTransactionRecoveryError("journal has not reached its commit point")
    for artifact, record in zip(plan.artifacts, journal_records(journal), strict=True):
        identity = identity_from_json(record["stage_identity"])
        require_matching_object(artifact.target, identity, "committed target")
        run_artifact_validator(artifact, artifact.target, "committed")


def cleanup_committed_transaction(
    plan: ArtifactTransactionPlan, journal_handle: _JournalHandle, journal: dict[str, Any]
) -> ArtifactTransactionResult:
    """Remove only authenticated remnants after validating the committed generation."""
    validate_committed_generation(plan, journal)
    for artifact, record in zip(plan.artifacts, journal_records(journal), strict=True):
        stage_identity = identity_from_json(record["stage_identity"])
        original_identity = (
            identity_from_json(record["original_identity"])
            if record["original_identity"] is not None
            else None
        )
        if object_exists(artifact.stage) or object_exists(
            transaction_retired_object_path(plan, artifact, "stage")
        ):
            retire_and_remove_transaction_object(
                plan, artifact, "stage", artifact.stage, stage_identity
            )
        backup = transaction_backup_path(plan, artifact)
        if object_exists(backup):
            if original_identity is None:
                raise ArtifactTransactionRecoveryError(
                    f"unexpected replacement backup is preserved: {backup}"
                )
            retire_and_remove_transaction_object(
                plan, artifact, "backup", backup, original_identity
            )
        elif original_identity is not None and object_exists(
            transaction_retired_object_path(plan, artifact, "backup")
        ):
            retire_and_remove_transaction_object(
                plan, artifact, "backup", backup, original_identity
            )
    delete_transaction_journal(plan, journal_handle)
    return ArtifactTransactionResult(
        transaction_id=plan.transaction_id,
        committed_targets=tuple(artifact.target for artifact in plan.artifacts),
        restored_targets=(),
        cleanup_complete=True,
    )


def rollback_journaled_transaction(
    plan: ArtifactTransactionPlan, journal_handle: _JournalHandle, journal: dict[str, Any]
) -> ArtifactTransactionResult:
    """Restore every authenticated replacement original before the commit point."""
    if journal["state"] == "committed":
        raise ArtifactTransactionRecoveryError(
            "committed transactions must clean up, not roll back"
        )
    restored: list[Path] = []
    for artifact, record in zip(plan.artifacts, journal_records(journal), strict=True):
        root = transaction_artifact_lease_root(plan, artifact)
        stage_identity = identity_from_json(record["stage_identity"])
        original_identity = (
            identity_from_json(record["original_identity"])
            if record["original_identity"] is not None
            else None
        )
        backup = transaction_backup_path(plan, artifact)
        target_exists = object_exists(artifact.target)
        if target_exists:
            if object_matches(artifact.target, stage_identity):
                retire_and_remove_transaction_object(
                    plan, artifact, "rollback-target", artifact.target, stage_identity
                )
            elif original_identity is not None and object_matches(
                artifact.target, original_identity
            ):
                pass
            else:
                raise ArtifactTransactionRecoveryError(
                    f"replacement target identity does not permit rollback: {artifact.target}"
                )
        if original_identity is not None:
            if object_exists(backup):
                require_matching_object(backup, original_identity, "rollback backup")
                if object_exists(artifact.target):
                    require_matching_object(artifact.target, original_identity, "rollback target")
                    retire_and_remove_transaction_object(
                        plan, artifact, "rollback-backup", backup, original_identity
                    )
                else:
                    rename_without_overwrite(
                        backup, artifact.target, lease_root=root
                    )
                    sync_directory_strict(artifact.target.parent, lease_root=root)
                    require_matching_object(artifact.target, original_identity, "restored target")
                    restored.append(artifact.target)
            elif not object_exists(artifact.target):
                raise ArtifactTransactionRecoveryError(
                    f"replacement original is missing and cannot be restored: {artifact.target}"
                )
            else:
                require_matching_object(artifact.target, original_identity, "restored target")
        elif object_exists(backup):
            raise ArtifactTransactionRecoveryError(f"unexpected backup is preserved: {backup}")
        if object_exists(artifact.stage) or object_exists(
            transaction_retired_object_path(plan, artifact, "rollback-stage")
        ):
            retire_and_remove_transaction_object(
                plan, artifact, "rollback-stage", artifact.stage, stage_identity
            )
        if object_exists(transaction_retired_object_path(plan, artifact, "rollback-target")):
            retire_and_remove_transaction_object(
                plan, artifact, "rollback-target", artifact.target, stage_identity
            )
        if original_identity is not None and object_exists(
            transaction_retired_object_path(plan, artifact, "rollback-backup")
        ):
            retire_and_remove_transaction_object(
                plan, artifact, "rollback-backup", backup, original_identity
            )
    delete_transaction_journal(plan, journal_handle)
    return ArtifactTransactionResult(
        transaction_id=plan.transaction_id,
        committed_targets=(),
        restored_targets=tuple(restored),
        cleanup_complete=True,
    )


def object_exists(path: Path) -> bool:
    """Distinguish absent paths from unsafe visible objects without following links."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def object_matches(path: Path, identity: TransactionObjectIdentity) -> bool:
    """Return whether a visible object still has one exact transaction identity."""
    if not object_exists(path):
        return False
    try:
        observed = inspect_transaction_object(path, identity.kind, synchronize=False)
    except (ArtifactTransactionValidationError, FileNotFoundError):
        return False
    return observed == identity


def require_matching_object(path: Path, identity: TransactionObjectIdentity, label: str) -> None:
    """Fail closed when a visible object no longer belongs to this transaction."""
    if not object_matches(path, identity):
        raise ArtifactTransactionRecoveryError(
            f"{label} is missing, unsafe, or identity-mismatched; preserving evidence: {path}"
        )


def rename_without_overwrite(
    source: Path, destination: Path, *, lease_root: Path | None = None
) -> None:
    """Atomically rename without overwriting a destination that races into view."""
    if object_exists(destination):
        raise ArtifactTransactionRecoveryError(
            f"refusing to overwrite transaction path: {destination}"
        )
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise NotImplementedError("artifact transactions require Linux renameat2(RENAME_NOREPLACE)")
    try:
        source_descriptor, source_name = open_confined_parent(source, lease_root=lease_root)
        destination_descriptor, destination_name = open_confined_parent(
            destination, lease_root=lease_root
        )
    except OSError as error:
        raise ArtifactTransactionRecoveryError(
            f"transaction rename path is unsafe from {source} to {destination}"
        ) from error
    try:
        result = renameat2(
            source_descriptor,
            os.fsencode(source_name),
            destination_descriptor,
            os.fsencode(destination_name),
            _RENAME_NOREPLACE,
        )
    except OSError as error:
        raise ArtifactTransactionRecoveryError(
            f"transaction rename failed from {source} to {destination}"
        ) from error
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ArtifactTransactionRecoveryError(
            f"refusing to overwrite transaction path: {destination}"
        )
    raise ArtifactTransactionRecoveryError(
        f"transaction rename failed from {source} to {destination}: {os.strerror(error_number)}"
    )


def rename_exchange(first: Path, second: Path, *, lease_root: Path | None = None) -> None:
    """Atomically exchange two same-filesystem names for safe cleanup quarantine."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise NotImplementedError("artifact transactions require Linux renameat2(RENAME_EXCHANGE)")
    try:
        first_descriptor, first_name = open_confined_parent(first, lease_root=lease_root)
        second_descriptor, second_name = open_confined_parent(second, lease_root=lease_root)
    except OSError as error:
        raise ArtifactTransactionRecoveryError(
            f"transaction exchange path is unsafe from {first} to {second}"
        ) from error
    try:
        result = renameat2(
            first_descriptor,
            os.fsencode(first_name),
            second_descriptor,
            os.fsencode(second_name),
            _RENAME_EXCHANGE,
        )
    finally:
        os.close(second_descriptor)
        os.close(first_descriptor)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise ArtifactTransactionRecoveryError(
        f"transaction exchange failed from {first} to {second}: {os.strerror(error_number)}"
    )


def retire_and_remove_transaction_object(
    plan: ArtifactTransactionPlan,
    artifact: TransactionArtifact,
    role: str,
    path: Path,
    identity: TransactionObjectIdentity,
) -> None:
    """Durably retire one known remnant before physically reclaiming its storage.

    The deterministic retired name is part of the recovery protocol: a crash
    after the first rename leaves a discoverable object that a later recovery
    can remove without requiring the original caller-visible name to reappear.
    """
    root = transaction_artifact_lease_root(plan, artifact)
    ensure_transaction_retired_directory(plan, artifact)
    retired = transaction_retired_object_path(plan, artifact, role)
    path_exists = object_exists(path)
    retired_exists = object_exists(retired)
    if path_exists and retired_exists:
        raise ArtifactTransactionRecoveryError(
            f"both active and retired cleanup objects are visible; preserving evidence: {path}"
        )
    if path_exists:
        require_matching_object(path, identity, f"{role} cleanup object")
        rename_without_overwrite(path, retired, lease_root=root)
        sync_directory_strict(path.parent, lease_root=root)
        sync_directory_strict(retired.parent, lease_root=root)
        if not object_matches(retired, identity):
            if not object_exists(path):
                rename_without_overwrite(retired, path, lease_root=root)
                sync_directory_strict(path.parent, lease_root=root)
                sync_directory_strict(retired.parent, lease_root=root)
            raise ArtifactTransactionRecoveryError(
                f"{role} cleanup object changed during retirement; preserving evidence: {path}"
            )
        retired_exists = True
    if not retired_exists:
        return
    if identity.kind == "file":
        require_matching_object(retired, identity, f"retired {role} cleanup object")
        remove_file_at_path(retired, lease_root=root)
        sync_directory_strict(retired.parent, lease_root=root)
        return
    remove_retired_directory_tree(retired, identity, lease_root=root)
    sync_directory_strict(retired.parent, lease_root=root)


def remove_file_at_path(path: Path, *, lease_root: Path) -> None:
    """Unlink a verified private transaction file through its anchored parent FD."""
    parent_descriptor, name = open_confined_parent(path, lease_root=lease_root)
    try:
        os.unlink(name, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)


def remove_retired_directory_tree(
    path: Path, identity: TransactionObjectIdentity, *, lease_root: Path
) -> None:
    """Reclaim only a complete authenticated retired directory tree."""
    require_matching_object(path, identity, "retired directory cleanup object")
    remove_authenticated_directory_tree(path, lease_root=lease_root)


def remove_authenticated_directory_tree(path: Path, *, lease_root: Path | None = None) -> None:
    """Recursively remove an ordinary tree through no-follow directory descriptors."""
    descriptor = open_confined_directory(path, lease_root=lease_root)
    try:
        remove_authenticated_directory_descriptor(descriptor, path)
    finally:
        os.close(descriptor)
    parent_descriptor, name = open_confined_parent(path, lease_root=lease_root)
    try:
        os.rmdir(name, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)


def remove_authenticated_directory_descriptor(descriptor: int, path: Path) -> None:
    """Remove one opened tree's remaining ordinary children using descriptor-relative names."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    for name in sorted(os.listdir(descriptor)):
        entry_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        entry_path = path / name
        if stat.S_ISLNK(entry_stat.st_mode) or not (
            stat.S_ISDIR(entry_stat.st_mode) or stat.S_ISREG(entry_stat.st_mode)
        ):
            raise ArtifactTransactionRecoveryError(
                f"cleanup tree changed after authentication; preserving evidence: {entry_path}"
            )
        if stat.S_ISDIR(entry_stat.st_mode):
            child_descriptor = os.open(name, flags, dir_fd=descriptor)
            try:
                remove_authenticated_directory_descriptor(child_descriptor, entry_path)
            finally:
                os.close(child_descriptor)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def delete_transaction_journal(plan: ArtifactTransactionPlan, handle: _JournalHandle) -> None:
    """Retire then remove the committed journal without losing its recovery name."""
    path = handle.path
    if not receipt_matches_path(handle.receipt, path):
        raise ArtifactTransactionRecoveryError(
            f"transaction journal changed before cleanup; preserving evidence: {path}"
        )
    retired = transaction_retired_journal_path(plan)
    if path == transaction_journal_path(plan):
        if object_exists(retired):
            raise ArtifactTransactionRecoveryError(
                f"retired transaction journal already exists; preserving evidence: {retired}"
            )
        rename_without_overwrite(path, retired, lease_root=plan.lease_root)
        sync_directory_strict(path.parent, lease_root=plan.lease_root)
        handle.path = retired
        path = retired
    elif path != retired:
        raise ArtifactTransactionRecoveryError(f"transaction journal path is unsafe: {path}")
    if not receipt_matches_path(handle.receipt, path):
        raise ArtifactTransactionRecoveryError(
            f"retired transaction journal changed before cleanup; preserving evidence: {path}"
        )
    remove_file_at_path(path, lease_root=plan.lease_root)
    sync_directory_strict(path.parent, lease_root=plan.lease_root)
