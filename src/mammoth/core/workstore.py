"""Recoverable chunked work stores for framework-neutral long-running producers.

A consumer with a long-running chunked workload (rendering tiles, exporting
shards, or any partitionable unit of work) uses this module to claim
exclusive ownership of one store path, durably record which opaque
consumer-defined chunk IDs have completed, and safely classify prior state
before deciding whether to resume. Mammoth owns store leasing, an
append-only hash-chained completion journal, durable creation, and fail-closed
classification of prior state. It never interprets the consumer's identity
payload, chunk IDs, or chunk contents: the identity payload only selects one
store path's ownership, and a chunk marker is an opaque caller-supplied
string recorded verbatim for the consumer's own later use.
``WorkStoreInspection.completed_chunks`` and ``WorkStoreSession.completed_chunks``
read those markers back, keyed by chunk ID, from journal state that already
passed hash-chain verification, so a consumer can re-verify a durable chunk
payload against its own marker at resume time.

This module shares the same scope as :mod:`mammoth.core.transactions`: local
POSIX filesystems, no network filesystems, no cross-host coordination, and no
hostile-writer model. A crashed or killed cooperating writer is recoverable;
a writer that deliberately bypasses this module's lease is not.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Literal, cast

from mammoth.core.execution import sanitize_metadata_fields

WORK_STORE_SCHEMA_VERSION = 1
WORK_STORE_FORMAT = "mammoth-work-store-jsonl-v1"
WORK_STORE_METADATA_NAME = "mammoth-work-store.json"
WORK_STORE_JOURNAL_NAME = "mammoth-work-store-completed.jsonl"
MAX_CHUNK_ID_LENGTH = 512
MAX_CHUNK_MARKER_LENGTH = 4096

# Identity discrimination uses scrypt, not a plain hash: a plain hash of a
# low-entropy identity payload (a short session ID, for example) is
# invertible offline by dictionary attack from the persisted metadata file
# alone. Salting per store and using scrypt's tunable, deliberately
# memory-hard cost raises that attack's cost; it does not eliminate it for a
# sufficiently low-entropy payload (see docs/ARCHITECTURE.md).
IDENTITY_KDF_NAME = "scrypt"
IDENTITY_KDF_SALT_BYTES = 16
IDENTITY_KDF_N = 2**14
IDENTITY_KDF_R = 8
IDENTITY_KDF_P = 1
IDENTITY_KDF_DKLEN = 32
_IDENTITY_KDF_MAXMEM = 64 * 1024 * 1024

type WorkStoreStatus = Literal[
    "absent",
    "resumable",
    "legacy",
    "incompatible",
    "relocated",
    "damaged",
    "concurrently_owned",
]


class WorkStoreError(RuntimeError):
    """Base error for recoverable work-store leasing, journaling, and recovery."""


class WorkStoreValidationError(WorkStoreError):
    """Raised when caller-supplied inputs or session usage are invalid."""


class WorkStoreStatusError(WorkStoreError):
    """Base error carrying the failed classification status of one store path."""

    status: WorkStoreStatus

    def __init__(self, message: str, *, store_path: Path) -> None:
        """Record the store path this classification failure concerns."""
        super().__init__(message)
        self.store_path = store_path


class WorkStoreConflictError(WorkStoreStatusError):
    """Raised when another process already owns the exclusive store lease."""

    status: WorkStoreStatus = "concurrently_owned"


class WorkStoreDamagedError(WorkStoreStatusError):
    """Raised when store metadata or its completion journal fails integrity checks."""

    status: WorkStoreStatus = "damaged"


class WorkStoreIncompatibleError(WorkStoreStatusError):
    """Raised when an existing store path cannot be safely adopted for this identity.

    ``status`` is ``"legacy"`` when the path holds pre-existing content
    without Mammoth's own metadata, ``"incompatible"`` when Mammoth's
    metadata is well-formed but was published for a different consumer
    identity, or ``"relocated"`` when Mammoth's metadata matches this
    identity but was durably recorded for a different canonical store path
    (the store directory was copied or moved since it was created).
    """

    def __init__(self, message: str, *, store_path: Path, status: WorkStoreStatus) -> None:
        """Record the store path and specific incompatibility classification."""
        super().__init__(message, store_path=store_path)
        self.status = status


@dataclass(frozen=True, slots=True)
class WorkStoreInspection:
    """Non-mutating classification of one work-store path for a given identity.

    ``completed_chunks`` maps each committed chunk ID to its opaque marker
    string, sourced only from a journal that passed hash-chain verification;
    every other classification leaves it empty. It is declared last and
    keyword-only so existing four-positional construction
    (``store_path``, ``status``, ``completed_chunk_ids``, ``detail``) keeps
    binding exactly as it did before this field existed.
    """

    store_path: Path
    status: WorkStoreStatus
    completed_chunk_ids: tuple[str, ...] = ()
    detail: str = ""
    completed_chunks: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), kw_only=True
    )

    def __hash__(self) -> int:
        """Hash over the hashable fields, preserving hashability despite the marker mapping.

        ``completed_chunks`` is a ``Mapping`` (backed by a ``MappingProxyType``
        over a ``dict``) and therefore unhashable, so the dataclass-generated
        ``__hash__`` would make every instance unhashable. This stays
        consistent with the generated ``__eq__``: two instances equal under
        ``__eq__`` have equal ``completed_chunk_ids`` and therefore equal
        hashes here.
        """
        return hash((self.store_path, self.status, self.completed_chunk_ids, self.detail))


@dataclass(slots=True)
class WorkStoreLease:
    """Hold the exclusive advisory lease for one work-store path."""

    path: Path
    _descriptor: int
    _closed: bool = False

    def close(self) -> None:
        """Release this lease exactly once."""
        if self._closed:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> WorkStoreLease:
        """Return this lease for context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release the lease on context exit."""
        self.close()


def claim_work_store_lease(store_path: Path) -> WorkStoreLease:
    """Claim the exclusive advisory lease for one work-store path without waiting.

    Failing closed here turns a concurrent second owner into an explicit
    :class:`WorkStoreConflictError` instead of racing writers.
    """
    path = Path(store_path)
    lease_path = _lease_path(path)
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lease_path, flags, 0o600)
    except OSError as exc:
        # For example a symlink at the lease path defeats O_NOFOLLOW
        # (ELOOP), or the parent cannot be created or traversed: none of
        # these are safely acquirable, so fail closed instead of letting a
        # raw OSError escape this module's WorkStoreError contract.
        raise WorkStoreDamagedError(
            f"Work-store lease path is not safely acquirable: {lease_path}.",
            store_path=path,
        ) from exc
    try:
        lease_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lease_stat.st_mode):
            raise WorkStoreDamagedError(
                f"Work-store lease path is not a safe regular file: {lease_path}.",
                store_path=path,
            )
        # Authenticate the already-opened descriptor's metadata, not a fresh
        # path re-stat, so a replacement between open() and this check
        # cannot slip an unsafe lease file past ownership validation.
        _require_owned(lease_stat, path=lease_path, store_path=path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkStoreConflictError(
                f"Work store {path} is owned by another process.",
                store_path=path,
            ) from error
        except OSError as error:
            raise WorkStoreDamagedError(
                f"Work-store lease could not be locked: {lease_path}.",
                store_path=path,
            ) from error
    except BaseException:
        os.close(descriptor)
        raise
    return WorkStoreLease(path=lease_path, _descriptor=descriptor)


def inspect_work_store(
    store_path: Path,
    identity_payload: Mapping[str, Any],
) -> WorkStoreInspection:
    """Classify existing store state under a short-held exclusive lease.

    Returns a status for every outcome instead of raising, so a caller can
    make an informed choice before claiming a session. It never creates,
    modifies, or removes the store.
    """
    path = Path(store_path)
    _sanitized, raw_payload_bytes = _sanitize_identity(identity_payload)
    try:
        lease = claim_work_store_lease(path)
    except WorkStoreConflictError as error:
        return WorkStoreInspection(store_path=path, status=error.status, detail=str(error))
    try:
        try:
            loaded = _load_existing_store(path, raw_payload_bytes=raw_payload_bytes)
        except WorkStoreStatusError as error:
            return WorkStoreInspection(store_path=path, status=error.status, detail=str(error))
        if loaded is None:
            return WorkStoreInspection(
                store_path=path,
                status="absent",
                detail="No work store exists at this path yet.",
            )
        return WorkStoreInspection(
            store_path=path,
            status="resumable",
            completed_chunk_ids=tuple(sorted(loaded.completed)),
            completed_chunks=MappingProxyType(dict(loaded.completed)),
            detail=f"Resumable work store with {len(loaded.completed)} committed chunk(s).",
        )
    finally:
        lease.close()


class WorkStoreSession:
    """Own one exclusively leased recoverable work store and its completion journal.

    Construct through :meth:`open_or_create`, which claims the store's lease
    and either resumes an authenticated matching store or durably creates a
    new one. Any other prior state (legacy, incompatible, relocated,
    damaged, or concurrently owned) fails closed with a
    :class:`WorkStoreStatusError` instead of silently adopting it. See
    ``docs/ARCHITECTURE.md`` for this module's thread- and
    process-confinement contract.
    """

    def __init__(
        self,
        *,
        identity_payload: Mapping[str, Any],
        identity_digest: str,
        store_path: Path,
        lease: WorkStoreLease,
        completed: Mapping[str, str],
        next_sequence: int,
        previous_sha256: str,
        chain_seed: str,
        recovered: bool,
    ) -> None:
        """Assemble a session from already-claimed ownership and loaded state."""
        self.identity_payload = identity_payload
        self.identity_digest = identity_digest
        self.store_path = store_path
        self.recovered = recovered
        self._lease = lease
        self._completed = dict(completed)
        self._next_sequence = next_sequence
        self._previous_sha256 = previous_sha256
        self._chain_seed = chain_seed
        self._closed = False
        self._poisoned = False

    @property
    def completed_chunk_ids(self) -> tuple[str, ...]:
        """Return committed chunk IDs in stable sorted order."""
        return tuple(sorted(self._completed))

    @property
    def completed_chunks(self) -> Mapping[str, str]:
        """Return a read-only snapshot mapping committed chunk IDs to their markers."""
        return MappingProxyType(dict(self._completed))

    @classmethod
    def open_or_create(
        cls,
        store_path: Path,
        identity_payload: Mapping[str, Any],
    ) -> WorkStoreSession:
        """Claim exclusive ownership, resume a matching store, or create a new one."""
        path = Path(store_path)
        sanitized, raw_payload_bytes = _sanitize_identity(identity_payload)
        lease = claim_work_store_lease(path)
        created = False
        try:
            loaded = _load_existing_store(path, raw_payload_bytes=raw_payload_bytes)
            if loaded is not None:
                _truncate_journal_tail(path, valid_bytes=loaded.valid_journal_bytes)
                return cls(
                    identity_payload=_freeze(sanitized),
                    identity_digest=loaded.identity_digest,
                    store_path=path,
                    lease=lease,
                    completed=loaded.completed,
                    next_sequence=loaded.next_sequence,
                    previous_sha256=loaded.previous_sha256,
                    chain_seed=loaded.chain_seed,
                    recovered=True,
                )
            created = True
            identity_digest, chain_seed = _create_store(
                path,
                sanitized=sanitized,
                raw_payload_bytes=raw_payload_bytes,
            )
            return cls(
                identity_payload=_freeze(sanitized),
                identity_digest=identity_digest,
                store_path=path,
                lease=lease,
                completed={},
                next_sequence=0,
                previous_sha256=chain_seed,
                chain_seed=chain_seed,
                recovered=False,
            )
        except BaseException:
            if created:
                # Prefer the quarantine-rename-then-remove path used by
                # every other deletion. If that cannot even be attempted
                # (for example the rename itself fails), best-effort fall
                # back to removing the live path directly rather than
                # leaving it certainly blocked for a retry; either path is
                # best-effort, so a remnant can still remain, and the next
                # inspection or open classifies whatever it finds (absent
                # if fully removed, damaged if only partial Mammoth debris
                # such as a bare journal file survives).
                with suppress(OSError):
                    _retire_and_remove_store(path)
                if path.exists():
                    with suppress(OSError):
                        shutil.rmtree(path)
            lease.close()
            raise

    def commit(self, chunk_markers: Mapping[str, str]) -> None:
        """Durably append one completion record covering one or more chunk IDs.

        The completion journal is fsynced before this call returns, and the
        store directory is fsynced afterward so a newly created journal
        entry's directory entry is durable too. A crash can therefore never
        observe completion without a durable journal record, and never lose
        one that was already durable.

        After *any* exception from this method, the session is unusable:
        every further ``commit()`` or ``verify()`` raises
        :class:`WorkStoreDamagedError`, and the caller must ``close()`` it
        and, if it still wants to continue the work, reopen the store fresh
        and trust its ``completed_chunk_ids`` rather than any in-memory
        state. Two failure shapes reach this poisoned state, and they leave
        different durable outcomes:

        - A failure while writing, flushing, or fsyncing the record itself
          (including a short write) means the record is not confirmed
          durable. Mammoth best-effort truncates the just-written bytes back
          off the journal file, even if the truncation itself fails, so a
          later reopen never adopts a record this call did not confirm.
        - A failure fsyncing the store directory afterward happens only
          after the record's own fsync already succeeded: the commit *is*
          durable, and nothing is truncated. A reopen after this failure may
          legitimately show the chunk as already completed.
        """
        if self._closed:
            raise WorkStoreValidationError("Work-store session is closed.")
        self._require_not_poisoned()
        normalized = _validate_chunk_markers(chunk_markers)
        ids = tuple(chunk_id for chunk_id, _marker in normalized)
        if any(chunk_id in self._completed for chunk_id in ids):
            raise WorkStoreValidationError("commit() chunk IDs must not already be completed.")
        payload = _journal_record_payload(self._next_sequence, self._previous_sha256, normalized)
        record_sha256 = _sha256_payload(payload)
        record = {**payload, "sha256": record_sha256}
        journal_path = self.store_path / WORK_STORE_JOURNAL_NAME
        if os.path.lexists(journal_path):
            _require_regular_file(journal_path, store_path=self.store_path, description="journal")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(journal_path, flags, 0o600)
        except OSError as exc:
            raise WorkStoreDamagedError(
                f"Work-store completion journal is not safely writable: {journal_path}.",
                store_path=self.store_path,
            ) from exc
        with os.fdopen(descriptor, "ab") as handle:
            journal_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(journal_stat.st_mode):
                raise WorkStoreDamagedError(
                    f"Work-store completion journal is not a regular file: {journal_path}.",
                    store_path=self.store_path,
                )
            _require_owned(journal_stat, path=journal_path, store_path=self.store_path)
            pre_write_size = journal_stat.st_size
            line = _canonical_json(record) + b"\n"
            try:
                written = handle.write(line)
                if written != len(line):
                    raise OSError(f"short journal append: wrote {written!r} of {len(line)} bytes")
                handle.flush()
                os.fsync(handle.fileno())
            except OSError as append_error:
                # write()/flush()/fsync() can each fail after some or all of
                # the record's bytes are already visible to a later reader
                # of this file (a short write, a deferred ENOSPC/EIO
                # surfacing only at flush, or an outright fsync failure): a
                # record that is not confirmed durable must never be
                # adoptable, so best-effort truncate it back off and poison
                # this session even if the truncate itself fails.
                self._poisoned = True
                with suppress(OSError):
                    os.ftruncate(handle.fileno(), pre_write_size)
                raise WorkStoreDamagedError(
                    f"Work-store completion journal append could not be made durable; "
                    f"this session is now poisoned: {journal_path}.",
                    store_path=self.store_path,
                ) from append_error
        # The record is durable as soon as its own fsync succeeds; update the
        # cached state now so it stays consistent with disk even if the
        # directory fsync below (only load-bearing for a brand-new journal
        # file's directory entry) subsequently fails.
        self._completed.update(normalized)
        self._next_sequence += 1
        self._previous_sha256 = record_sha256
        try:
            _fsync_directory(self.store_path)
        except OSError as directory_fsync_error:
            # The record's own fsync already succeeded, so the cached state
            # updated above is correct and durable; only the (rare, and only
            # load-bearing for a brand-new journal file's) directory-entry
            # fsync failed. There is nothing unsynced to roll back, so this
            # does not truncate. Poison the session anyway: it just observed
            # a filesystem failure, and the caller must reopen and trust the
            # durable on-disk state (completed_chunk_ids there may
            # legitimately include this chunk) rather than keep using it.
            self._poisoned = True
            raise WorkStoreDamagedError(
                f"Work-store directory fsync failed after a durable commit; this "
                f"session is now poisoned and must be reopened: {self.store_path}.",
                store_path=self.store_path,
            ) from directory_fsync_error

    def verify(self) -> None:
        """Re-validate the durable journal against this session's cached state.

        Raises :class:`WorkStoreDamagedError` if the on-disk journal no
        longer reproduces the exact completed-chunk state this session has
        observed, including a broken hash chain or a truncated record.
        """
        if self._closed:
            raise WorkStoreValidationError("Work-store session is closed.")
        self._require_not_poisoned()
        completed, next_sequence, previous_sha256, _valid_bytes = _load_journal(
            self.store_path,
            chain_seed=self._chain_seed,
        )
        if (
            completed != self._completed
            or next_sequence != self._next_sequence
            or previous_sha256 != self._previous_sha256
        ):
            raise WorkStoreDamagedError(
                f"Work-store completion journal no longer matches session state: "
                f"{self.store_path}.",
                store_path=self.store_path,
            )

    def _require_not_poisoned(self) -> None:
        """Reject further commits or verification after an unsynced-append failure."""
        if self._poisoned:
            raise WorkStoreDamagedError(
                "Work-store session is poisoned after a prior durability failure "
                "and must not be reused for commit() or verify().",
                store_path=self.store_path,
            )

    def close(self) -> None:
        """Release exclusive store ownership while preserving the store for resume."""
        if self._closed:
            return
        self._closed = True
        self._lease.close()

    def remove_after_publication(self) -> None:
        """Delete this validated, caller-published store, then release ownership.

        Callers must have already confirmed final publication succeeded;
        Mammoth does not infer publication from committed chunk markers.
        """
        if self._closed:
            return
        try:
            if self.store_path.exists():
                _retire_and_remove_store(self.store_path)
        finally:
            self.close()

    def __enter__(self) -> WorkStoreSession:
        """Return this session for context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Release ownership without deleting the store on context exit."""
        self.close()


def cleanup_work_store(
    store_path: Path,
    identity_payload: Mapping[str, Any],
    *,
    validate_publication: Callable[[], object],
) -> None:
    """Revalidate a store under a fresh exclusive lease, then delete it.

    Use this from a process or invocation that did not hold the committing
    session, after independently confirming that final publication
    succeeded. Any status other than ``resumable`` or ``absent`` fails
    closed instead of deleting ambiguous or foreign state.
    """
    path = Path(store_path)
    _sanitized, raw_payload_bytes = _sanitize_identity(identity_payload)
    lease = claim_work_store_lease(path)
    try:
        loaded = _load_existing_store(path, raw_payload_bytes=raw_payload_bytes)
        if loaded is None:
            return
        validate_publication()
        _retire_and_remove_store(path)
    finally:
        lease.close()


@dataclass(frozen=True, slots=True)
class _LoadedStore:
    """One authenticated matching store and its completion-journal state."""

    completed: dict[str, str]
    next_sequence: int
    previous_sha256: str
    valid_journal_bytes: int
    chain_seed: str
    identity_digest: str


@dataclass(frozen=True, slots=True)
class _StoredMetadata:
    """Trustworthy fields parsed from one store's metadata file."""

    created_at: str
    identity_digest: str
    identity_kdf: _IdentityKdfParams
    store_path: str


def _lease_path(store_path: Path) -> Path:
    """Return the deterministic sibling lease path for one store path."""
    return store_path.parent / f".{store_path.name}.mammoth-work-store.lock"


def _retired_store_path(store_path: Path) -> Path:
    """Return the deterministic sibling quarantine path for one store path."""
    return store_path.parent / f".{store_path.name}.mammoth-work-store.retired"


def _retire_and_remove_store(store_path: Path) -> None:
    """Quarantine a store via a durable rename before removing it.

    A crash between the rename and the final removal never leaves a
    half-deleted store at the live path: the live path is simply absent
    (classified ``absent``, never a false-empty ``resumable`` store), and
    the retired remnant is reclaimed deterministically by the next call
    that retires a store at the same path.
    """
    retired_path = _retired_store_path(store_path)
    if os.path.lexists(retired_path):
        # A prior cleanup was interrupted after quarantining but before the
        # final removal; finish it before reusing the deterministic name.
        shutil.rmtree(retired_path, ignore_errors=True)
    os.rename(store_path, retired_path)
    _fsync_directory(store_path.parent)
    shutil.rmtree(retired_path)


def _canonical_json(value: object) -> bytes:
    """Serialize store records deterministically for integrity hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_payload(value: object) -> str:
    """Return a canonical SHA-256 digest for one JSON-compatible payload."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _journal_chain_seed(*, identity_digest: str, store_path: str) -> str:
    """Return the domain-separated chain root binding a journal to one store.

    Seeding the first completion record's ``previous_sha256`` with a digest
    over the format identifier, identity digest, and metadata's own recorded
    canonical store path means the chain only reproduces for the exact store
    it was written for. A journal grafted from a different store's directory
    fails this seed check on its very first record, and so does a journal
    whose store's metadata had its recorded ``store_path`` field edited
    without regenerating the chain that field seeds: both classify as
    ``damaged`` rather than silently validating or degrading only to
    ``relocated``.
    """
    payload = {
        "format": WORK_STORE_FORMAT,
        "identity_digest": identity_digest,
        "store_path": store_path,
    }
    return _sha256_payload(payload)


def _is_hex_string(value: object, *, length: int | None = None) -> bool:
    """Return whether a value is a lowercase hex string, optionally of one length."""
    return (
        isinstance(value, str)
        and bool(value)
        and (length is None or len(value) == length)
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_int(value: object, expected: int) -> bool:
    """Return whether a value is exactly one specific plain (non-bool) integer."""
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


@dataclass(frozen=True, slots=True)
class _IdentityKdfParams:
    """One store's scrypt salt and cost parameters for its identity digest."""

    salt: bytes
    n: int
    r: int
    p: int
    dklen: int


def _generate_identity_kdf() -> _IdentityKdfParams:
    """Generate a fresh per-store random salt with this module's fixed KDF cost."""
    return _IdentityKdfParams(
        salt=os.urandom(IDENTITY_KDF_SALT_BYTES),
        n=IDENTITY_KDF_N,
        r=IDENTITY_KDF_R,
        p=IDENTITY_KDF_P,
        dklen=IDENTITY_KDF_DKLEN,
    )


def _identity_digest(raw_payload_bytes: bytes, *, kdf: _IdentityKdfParams) -> str:
    """Return the salted, memory-hard scrypt discrimination digest for one raw payload.

    See the ``IDENTITY_KDF_*`` module constants for why this is scrypt and
    not a plain hash, and ``docs/ARCHITECTURE.md`` for the residual risk
    against a sufficiently low-entropy payload.
    """
    derived = hashlib.scrypt(
        raw_payload_bytes,
        salt=kdf.salt,
        n=kdf.n,
        r=kdf.r,
        p=kdf.p,
        dklen=kdf.dklen,
        maxmem=_IDENTITY_KDF_MAXMEM,
    )
    return derived.hex()


def _utc_now_iso() -> str:
    """Return the current UTC time in the metadata timestamp format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sanitize_identity(identity_payload: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    """Validate one opaque consumer identity payload; canonicalize its raw, unredacted form.

    The identity digest (see :func:`_identity_digest`) is computed over the
    caller's raw payload, not the redacted copy this function also returns
    for persistence and display. Hashing the sanitized copy would collide
    two genuinely different identities whenever they differ only in a field
    whose *name* happens to match ``sanitize_metadata_fields``'s
    sensitive-name heuristics (for example two different ``session_id``
    values both becoming ``"<redacted>"``); hashing the raw payload cannot
    collide that way. Only the sanitized copy is ever persisted or exposed
    publicly.
    """
    if not isinstance(identity_payload, Mapping):
        raise WorkStoreValidationError("identity_payload must be a mapping.")
    try:
        sanitized = sanitize_metadata_fields(identity_payload)
    except ValueError as error:
        raise WorkStoreValidationError(f"identity_payload is invalid: {error}") from error
    # sanitize_metadata_fields already proved identity_payload is
    # JSON-compatible (same structure, only some leaf values redacted), so
    # this canonicalization step only needs to normalize container types
    # for hashing, not re-validate; but it still has to cover every type
    # sanitize_metadata_fields itself accepts (os.PathLike values inside a
    # command-named field, and arbitrary Sequence implementations, not just
    # list/tuple), or a value that passed sanitization could still crash
    # json.dumps here.
    try:
        raw_payload_bytes = _canonical_json(_canonicalize_json_value(identity_payload))
    except TypeError as error:
        raise WorkStoreValidationError(f"identity_payload is invalid: {error}") from error
    return sanitized, raw_payload_bytes


def _canonicalize_json_value(value: Any) -> Any:
    """Recursively normalize an already-validated value to dict/list/scalars.

    Only safe to call after :func:`sanitize_metadata_fields` (or equivalent)
    has already validated ``value`` is JSON-compatible: this mirrors that
    function's accepted domain, including ``os.PathLike`` values (which it
    accepts inside command-named containers) and arbitrary ``Sequence``
    implementations (not just ``list``/``tuple``), without redacting
    anything, so the raw payload can still be hashed for identity
    discrimination. A residual value neither function's domain covers still
    reaches ``json.dumps`` unchanged, so callers wrap that in their own
    ``TypeError`` guard.
    """
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {key: _canonicalize_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonicalize_json_value(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    """Deeply freeze a JSON-compatible value for immutable public exposure."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_chunk_markers(chunk_markers: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Validate one caller-supplied chunk-ID-to-marker completion batch."""
    if not isinstance(chunk_markers, Mapping) or not chunk_markers:
        raise WorkStoreValidationError(
            "commit() requires a non-empty mapping of chunk IDs to markers."
        )
    normalized: list[tuple[str, str]] = []
    for chunk_id, marker in chunk_markers.items():
        if not isinstance(chunk_id, str) or not chunk_id or len(chunk_id) > MAX_CHUNK_ID_LENGTH:
            raise WorkStoreValidationError(f"Invalid work-store chunk ID: {chunk_id!r}.")
        if not isinstance(marker, str) or not marker or len(marker) > MAX_CHUNK_MARKER_LENGTH:
            raise WorkStoreValidationError(f"Invalid work-store chunk marker for {chunk_id!r}.")
        normalized.append((chunk_id, marker))
    ids = tuple(chunk_id for chunk_id, _marker in normalized)
    if len(set(ids)) != len(ids):
        raise WorkStoreValidationError("commit() chunk IDs must be unique within one call.")
    return tuple(normalized)


def _journal_record_payload(
    sequence: int,
    previous_sha256: str,
    chunks: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Return the hashed portion of one completion-journal record."""
    return {
        "sequence": sequence,
        "previous_sha256": previous_sha256,
        "chunks": [{"chunk_id": chunk_id, "marker": marker} for chunk_id, marker in chunks],
    }


def _require_owned(metadata: os.stat_result, *, path: Path, store_path: Path) -> None:
    """Require current-process ownership and reject group/world-writable state."""
    if metadata.st_uid != os.geteuid():
        raise WorkStoreDamagedError(
            f"Work-store object is not owned by this process user: {path}.",
            store_path=store_path,
        )
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise WorkStoreDamagedError(
            f"Work-store object has unsafe writable permissions: {path}.",
            store_path=store_path,
        )


def _require_regular_file(path: Path, *, store_path: Path, description: str) -> os.stat_result:
    """Return lstat metadata for a non-symlink regular store file."""
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise WorkStoreDamagedError(
            f"Work-store {description} is unavailable: {path}.",
            store_path=store_path,
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise WorkStoreDamagedError(
            f"Work-store {description} is not a safe regular file: {path}.",
            store_path=store_path,
        )
    _require_owned(metadata, path=path, store_path=store_path)
    return metadata


def _open_regular_binary(path: Path, *, store_path: Path, description: str) -> BinaryIO:
    """Open one existing regular store file for reading without following symlinks."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WorkStoreDamagedError(
            f"Work-store {description} is not safely accessible: {path}.",
            store_path=store_path,
        ) from exc
    handle = cast(BinaryIO, os.fdopen(descriptor, "rb"))
    metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        handle.close()
        raise WorkStoreDamagedError(
            f"Work-store {description} is not a regular file: {path}.",
            store_path=store_path,
        )
    _require_owned(metadata, path=path, store_path=store_path)
    return handle


def _fsync_directory(path: Path) -> None:
    """Fsync one directory's entries after a create, rename, or append."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_metadata(metadata_path: Path, *, store_path: Path) -> _StoredMetadata:
    """Load and authenticate one store's durable metadata record."""
    try:
        with _open_regular_binary(
            metadata_path,
            store_path=store_path,
            description="metadata",
        ) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkStoreDamagedError(
            f"Work-store metadata is unreadable: {metadata_path}.",
            store_path=store_path,
        ) from exc
    if not isinstance(payload, Mapping):
        raise WorkStoreDamagedError(
            f"Work-store metadata root must be an object: {metadata_path}.",
            store_path=store_path,
        )
    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != WORK_STORE_SCHEMA_VERSION
    ):
        raise WorkStoreDamagedError(
            f"Work-store metadata has an unsupported schema version: {metadata_path}.",
            store_path=store_path,
        )
    if payload.get("format") != WORK_STORE_FORMAT:
        raise WorkStoreDamagedError(
            f"Work-store metadata format is invalid: {metadata_path}.",
            store_path=store_path,
        )
    identity_payload = payload.get("identity_payload")
    if not isinstance(identity_payload, Mapping):
        raise WorkStoreDamagedError(
            f"Work-store metadata identity payload is invalid: {metadata_path}.",
            store_path=store_path,
        )
    # No self-check against a recomputed digest here: identity_digest is
    # computed over the caller's raw payload (see _sanitize_identity), and
    # only the redacted identity_payload is ever persisted, so the two can
    # never be expected to hash to each other. Authentication instead
    # happens by comparing the caller's freshly recomputed raw-payload
    # digest against this stored identity_digest in _load_existing_store.
    kdf_payload = payload.get("identity_kdf")
    if not isinstance(kdf_payload, Mapping) or kdf_payload.get("name") != IDENTITY_KDF_NAME:
        raise WorkStoreDamagedError(
            f"Work-store metadata identity KDF is invalid or unsupported: {metadata_path}.",
            store_path=store_path,
        )
    salt_hex = kdf_payload.get("salt")
    kdf_n = kdf_payload.get("n")
    kdf_r = kdf_payload.get("r")
    kdf_p = kdf_payload.get("p")
    kdf_dklen = kdf_payload.get("dklen")
    # The format is unreleased and Mammoth is the only writer, so this
    # requires exactly the cost parameters this module itself generates
    # rather than accepting an open-ended range. hashlib.scrypt's cost
    # parameters are attacker-adjacent (an inflated n can exhaust process
    # memory) and an odd-length or non-hex salt crashes bytes.fromhex, so a
    # single corrupted field must classify damaged here, never reach the
    # KDF as a raw crash.
    if (
        not _is_exact_int(kdf_n, IDENTITY_KDF_N)
        or not _is_exact_int(kdf_r, IDENTITY_KDF_R)
        or not _is_exact_int(kdf_p, IDENTITY_KDF_P)
        or not _is_exact_int(kdf_dklen, IDENTITY_KDF_DKLEN)
        or not _is_hex_string(salt_hex, length=IDENTITY_KDF_SALT_BYTES * 2)
    ):
        raise WorkStoreDamagedError(
            f"Work-store metadata identity KDF parameters are invalid: {metadata_path}.",
            store_path=store_path,
        )
    identity_digest = payload.get("identity_digest")
    if not _is_hex_string(identity_digest, length=IDENTITY_KDF_DKLEN * 2):
        raise WorkStoreDamagedError(
            f"Work-store metadata identity digest is invalid: {metadata_path}.",
            store_path=store_path,
        )
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise WorkStoreDamagedError(
            f"Work-store metadata is missing a creation timestamp: {metadata_path}.",
            store_path=store_path,
        )
    recorded_store_path = payload.get("store_path")
    if not isinstance(recorded_store_path, str) or not recorded_store_path:
        raise WorkStoreDamagedError(
            f"Work-store metadata is missing its recorded store path: {metadata_path}.",
            store_path=store_path,
        )
    return _StoredMetadata(
        created_at=created_at,
        identity_digest=cast(str, identity_digest),
        # Already validated equal to these exact module constants above;
        # using the constants directly (not the parsed values) keeps their
        # type as plain int for callers without a redundant cast.
        identity_kdf=_IdentityKdfParams(
            salt=bytes.fromhex(cast(str, salt_hex)),
            n=IDENTITY_KDF_N,
            r=IDENTITY_KDF_R,
            p=IDENTITY_KDF_P,
            dklen=IDENTITY_KDF_DKLEN,
        ),
        store_path=recorded_store_path,
    )


def _load_journal(store_path: Path, *, chain_seed: str) -> tuple[dict[str, str], int, str, int]:
    """Validate the completion chain, ignoring only an unterminated tail.

    ``chain_seed`` (see :func:`_journal_chain_seed`) is the expected root for
    the first record's ``previous_sha256``; a journal grafted from a
    different store, or one whose seed no longer matches its store's
    metadata, fails on that very first record.

    An unterminated trailing line is the signature of a commit interrupted
    before its final newline; it is treated as recoverable, not damage,
    matching this module's crash-then-resume contract. A validly created
    store's journal file always exists (possibly empty), so a *missing*
    journal file is always damage, never mistakeable for a fresh store with
    no commits yet.
    """
    error_path = store_path
    journal_path = store_path / WORK_STORE_JOURNAL_NAME
    try:
        journal_path.lstat()
    except FileNotFoundError as exc:
        raise WorkStoreDamagedError(
            f"Work-store completion journal is missing: {journal_path}.",
            store_path=error_path,
        ) from exc
    with _open_regular_binary(
        journal_path,
        store_path=error_path,
        description="completion journal",
    ) as handle:
        journal = handle.read()
    completed: dict[str, str] = {}
    previous_sha256 = chain_seed
    expected_sequence = 0
    valid_bytes = 0
    for line_number, line in enumerate(journal.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n"):
            break
        valid_bytes += len(line)
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkStoreDamagedError(
                f"Work-store completion journal line {line_number} is damaged: {journal_path}.",
                store_path=error_path,
            ) from exc
        if not isinstance(record, Mapping):
            raise WorkStoreDamagedError(
                f"Work-store completion journal line {line_number} is invalid: {journal_path}.",
                store_path=error_path,
            )
        chunks = record.get("chunks")
        sequence = record.get("sequence")
        record_previous = record.get("previous_sha256")
        record_sha256 = record.get("sha256")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence != expected_sequence
            or record_previous != previous_sha256
            or not isinstance(chunks, list)
            or not chunks
        ):
            raise WorkStoreDamagedError(
                f"Work-store completion journal line {line_number} is inconsistent: "
                f"{journal_path}.",
                store_path=error_path,
            )
        normalized: list[tuple[str, str]] = []
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise WorkStoreDamagedError(
                    f"Work-store completion journal line {line_number} has invalid chunk "
                    f"data: {journal_path}.",
                    store_path=error_path,
                )
            chunk_id = chunk.get("chunk_id")
            marker = chunk.get("marker")
            if (
                not isinstance(chunk_id, str)
                or not chunk_id
                or not isinstance(marker, str)
                or not marker
            ):
                raise WorkStoreDamagedError(
                    f"Work-store completion journal line {line_number} has invalid chunk "
                    f"data: {journal_path}.",
                    store_path=error_path,
                )
            normalized.append((chunk_id, marker))
        ids = tuple(chunk_id for chunk_id, _marker in normalized)
        if len(set(ids)) != len(ids) or any(chunk_id in completed for chunk_id in ids):
            raise WorkStoreDamagedError(
                f"Work-store completion journal line {line_number} repeats a chunk ID: "
                f"{journal_path}.",
                store_path=error_path,
            )
        hashed_payload = _journal_record_payload(sequence, previous_sha256, normalized)
        expected_sha256 = _sha256_payload(hashed_payload)
        if record_sha256 != expected_sha256:
            raise WorkStoreDamagedError(
                f"Work-store completion journal line {line_number} failed integrity checks: "
                f"{journal_path}.",
                store_path=error_path,
            )
        completed.update(normalized)
        previous_sha256 = expected_sha256
        expected_sequence += 1
    return completed, expected_sequence, previous_sha256, valid_bytes


def _load_existing_store(store_path: Path, *, raw_payload_bytes: bytes) -> _LoadedStore | None:
    """Authenticate one matching store, or classify why it cannot be adopted.

    Returns ``None`` when nothing exists at ``store_path`` yet. Raises
    :class:`WorkStoreIncompatibleError` for legacy or identity-mismatched
    content, and :class:`WorkStoreDamagedError` for anything structurally
    broken, so a caller can never silently adopt ambiguous prior state.
    """
    try:
        top_metadata = store_path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(top_metadata.st_mode):
        raise WorkStoreDamagedError(
            f"Work-store path is not a safe directory: {store_path}.",
            store_path=store_path,
        )
    _require_owned(top_metadata, path=store_path, store_path=store_path)
    metadata_path = store_path / WORK_STORE_METADATA_NAME
    try:
        metadata_path.lstat()
    except FileNotFoundError as exc:
        journal_path = store_path / WORK_STORE_JOURNAL_NAME
        if os.path.lexists(journal_path):
            # Mammoth's own completion journal without Mammoth's own
            # metadata is not a foreign store: it is debris from an
            # interrupted creation or an incomplete manual edit, and must
            # not be misclassified as adoptable-or-preservable legacy
            # content.
            raise WorkStoreDamagedError(
                f"Work-store path has a completion journal but no metadata: {store_path}.",
                store_path=store_path,
            ) from exc
        raise WorkStoreIncompatibleError(
            f"Work-store path lacks Mammoth's metadata and is treated as legacy: {store_path}.",
            store_path=store_path,
            status="legacy",
        ) from exc
    stored = _read_metadata(metadata_path, store_path=store_path)
    # Recompute the digest with the STORED salt and KDF parameters (a fresh
    # random salt would never match); the constant-time comparison avoids a
    # timing side-channel on an already deliberately expensive digest.
    # _read_metadata already restricts the KDF parameters and salt shape to
    # exactly what this module generates, so scrypt should never itself
    # reject them; this is a belt-and-braces guard so a gap there still
    # classifies damaged instead of crashing every public entry point with
    # a raw ValueError.
    try:
        candidate_digest = _identity_digest(raw_payload_bytes, kdf=stored.identity_kdf)
    except ValueError as exc:
        raise WorkStoreDamagedError(
            f"Work-store metadata identity KDF parameters were rejected: {store_path}.",
            store_path=store_path,
        ) from exc
    if not hmac.compare_digest(candidate_digest, stored.identity_digest):
        raise WorkStoreIncompatibleError(
            f"Work-store identity does not match the current caller identity: {store_path}.",
            store_path=store_path,
            status="incompatible",
        )
    resolved_store_path = str(store_path.resolve())
    if stored.store_path != resolved_store_path:
        raise WorkStoreIncompatibleError(
            f"Work-store metadata was recorded for {stored.store_path!r}, not "
            f"{resolved_store_path!r}; a copied or moved store is not adopted: {store_path}.",
            store_path=store_path,
            status="relocated",
        )
    # Seeded from metadata's own recorded store path, not the freshly
    # resolved one above: an honest whole-directory copy (caught as
    # ``relocated`` already) still validates here, while a journal grafted
    # from elsewhere, or a metadata store_path edited without regenerating
    # its journal, both fail this seed check as damaged.
    chain_seed = _journal_chain_seed(
        identity_digest=stored.identity_digest,
        store_path=stored.store_path,
    )
    completed, next_sequence, previous_sha256, valid_bytes = _load_journal(
        store_path,
        chain_seed=chain_seed,
    )
    return _LoadedStore(
        completed=completed,
        next_sequence=next_sequence,
        previous_sha256=previous_sha256,
        valid_journal_bytes=valid_bytes,
        chain_seed=chain_seed,
        identity_digest=stored.identity_digest,
    )


def _truncate_journal_tail(store_path: Path, *, valid_bytes: int) -> None:
    """Discard an interrupted commit's incomplete trailing journal bytes.

    Resuming must remove any unterminated tail before further commits are
    appended, or a new append would land after undelimited garbage and turn
    one damaged line into an unrecoverable journal.
    """
    journal_path = store_path / WORK_STORE_JOURNAL_NAME
    try:
        current_size = journal_path.lstat().st_size
    except FileNotFoundError:
        return
    if current_size == valid_bytes:
        return
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(journal_path, flags)
    try:
        journal_stat = os.fstat(descriptor)
        if not stat.S_ISREG(journal_stat.st_mode):
            raise WorkStoreDamagedError(
                f"Work-store completion journal is not a regular file: {journal_path}.",
                store_path=store_path,
            )
        _require_owned(journal_stat, path=journal_path, store_path=store_path)
        os.ftruncate(descriptor, valid_bytes)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_store(
    store_path: Path,
    *,
    sanitized: dict[str, Any],
    raw_payload_bytes: bytes,
) -> tuple[str, str]:
    """Durably create a new store directory and publish its identity metadata.

    Returns ``(identity_digest, chain_seed)`` (see :func:`_identity_digest`
    and :func:`_journal_chain_seed`) so the caller can construct a brand-new
    session consistently with what was just durably recorded in metadata.
    """
    store_path.mkdir(mode=0o700, parents=True)
    os.chmod(store_path, 0o700, follow_symlinks=False)
    _require_owned(store_path.lstat(), path=store_path, store_path=store_path)
    # The new directory's own dentry in its parent is not durable until the
    # parent itself is fsynced, matching the transactions module's
    # mkdir-then-parent-fsync precedent.
    _fsync_directory(store_path.parent)
    resolved_store_path = str(store_path.resolve())
    kdf = _generate_identity_kdf()
    identity_digest = _identity_digest(raw_payload_bytes, kdf=kdf)
    _write_metadata(
        store_path,
        sanitized=sanitized,
        identity_digest=identity_digest,
        kdf=kdf,
        resolved_store_path=resolved_store_path,
    )
    _create_empty_journal(store_path)
    chain_seed = _journal_chain_seed(
        identity_digest=identity_digest,
        store_path=resolved_store_path,
    )
    return identity_digest, chain_seed


def _write_metadata(
    store_path: Path,
    *,
    sanitized: dict[str, Any],
    identity_digest: str,
    kdf: _IdentityKdfParams,
    resolved_store_path: str,
) -> None:
    """Atomically publish immutable identity metadata for a newly created store."""
    payload: dict[str, Any] = {
        "schema_version": WORK_STORE_SCHEMA_VERSION,
        "format": WORK_STORE_FORMAT,
        "created_at": _utc_now_iso(),
        "identity_digest": identity_digest,
        "identity_kdf": {
            "name": IDENTITY_KDF_NAME,
            "salt": kdf.salt.hex(),
            "n": kdf.n,
            "r": kdf.r,
            "p": kdf.p,
            "dklen": kdf.dklen,
        },
        "identity_payload": sanitized,
        # Binds this metadata to one canonical location and seeds the
        # journal chain (see _journal_chain_seed): a copied or moved store
        # directory is caught as relocated, and a journal grafted from
        # elsewhere or a store_path edited without regenerating the chain
        # it seeds both fail journal validation as damaged.
        "store_path": resolved_store_path,
    }
    target = store_path / WORK_STORE_METADATA_NAME
    temporary = store_path / f".{WORK_STORE_METADATA_NAME}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise
    _fsync_directory(store_path)


def _create_empty_journal(store_path: Path) -> None:
    """Durably create the empty completion journal at store-creation time.

    Guaranteeing the journal file always exists for a validly created store
    means its later absence is always damage, never mistakeable for a
    legitimate fresh store that has not committed anything yet.
    """
    journal_path = store_path / WORK_STORE_JOURNAL_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(journal_path, flags, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(store_path)
