"""Filesystem-safe logical-run and execution identity validation.

Layout, execution metadata, and event streams share these validators so a
caller-controlled identity can never escape its configured entry directory.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

RUN_NAME_MAX_LENGTH = 255
EXECUTION_ID_MAX_LENGTH = 128
GROUP_ID_MAX_LENGTH = 128
DERIVED_RUN_NAME_DIGEST_LENGTH = 8

_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXECUTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GROUP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DERIVED_STEM_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_DERIVED_STEM_MAX_LENGTH = 64


def validate_run_name(run_name: str) -> str:
    """Return a safe single-directory logical-run name."""
    if (
        not isinstance(run_name, str)
        or len(run_name) > RUN_NAME_MAX_LENGTH
        or run_name in {".", ".."}
        or _RUN_NAME_PATTERN.fullmatch(run_name) is None
    ):
        raise ValueError(
            "Run names must be 1-255 filesystem-safe ASCII characters using "
            "letters, digits, '.', '_', or '-', must begin with a letter or "
            "digit, and may not be '.' or '..'."
        )
    return run_name


def validate_execution_id(execution_id: str) -> str:
    """Return a safe execution-attempt ID."""
    if (
        not isinstance(execution_id, str)
        or len(execution_id) > EXECUTION_ID_MAX_LENGTH
        or execution_id in {".", ".."}
        or _EXECUTION_ID_PATTERN.fullmatch(execution_id) is None
    ):
        raise ValueError(
            "Execution IDs must be 1-128 filesystem-safe ASCII characters "
            "using letters, digits, '.', '_', or '-', must begin with a letter "
            "or digit, and may not be '.' or '..'."
        )
    return execution_id


def validate_group_id(group_id: str) -> str:
    """Return a safe workflow-group identifier.

    Group IDs share their filesystem-safe syntax with execution IDs so both
    identities resolve safely beneath their respective entry-scoped and
    run-scoped directories.
    """
    if (
        not isinstance(group_id, str)
        or len(group_id) > GROUP_ID_MAX_LENGTH
        or group_id in {".", ".."}
        or _GROUP_ID_PATTERN.fullmatch(group_id) is None
    ):
        raise ValueError(
            "Group IDs must be 1-128 filesystem-safe ASCII characters using "
            "letters, digits, '.', '_', or '-', must begin with a letter or "
            "digit, and may not be '.' or '..'."
        )
    return group_id


def derive_run_name(prefix: str, target_path: str | Path) -> str:
    """Derive a stable, collision-resistant, path-safe run name for one target.

    The result has the shape ``<prefix>-<target-stem>-<digest>``, where
    ``target-stem`` is the filesystem-safe final path component of
    ``target_path`` (unsafe characters collapsed to ``-``, trimmed of
    leading/trailing ``-``, ``.``, and ``_``, and truncated to 64 characters)
    and ``digest`` is the first :data:`DERIVED_RUN_NAME_DIGEST_LENGTH` (8) hex
    characters of the SHA-256 digest of the target's absolute path. When the
    sanitized stem is empty the name falls back to ``<prefix>-<digest>``. The
    returned name never embeds the absolute path itself, only its
    filesystem-safe basename and digest, so it does not leak directory
    structure.

    The digest is computed from ``os.path.abspath(target_path)`` without
    resolving symlinks, so deriving a name never touches the filesystem and
    does not require the target to exist. Two calls with the same ``prefix``
    and the same absolute target path always derive the same name, so
    re-running the same workflow against the same target reuses the same
    logical run. The 8 hex digit digest is always appended, so distinct
    absolute targets that merely share a basename never collide on that
    basename alone; two genuinely different absolute paths collide only if
    their SHA-256 digests happen to share their first 8 hex characters, an
    approximately 1-in-4-billion (16**8) chance per pair. The result always
    satisfies :func:`validate_run_name`; callers remain responsible for
    supplying a ``prefix`` that is itself a non-empty valid run-name prefix.
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string")
    resolved = Path(os.path.abspath(target_path))
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[
        :DERIVED_RUN_NAME_DIGEST_LENGTH
    ]
    stem = _DERIVED_STEM_UNSAFE_PATTERN.sub("-", resolved.name).strip("-._")
    stem = stem[:_DERIVED_STEM_MAX_LENGTH]
    name = f"{prefix}-{stem}-{digest}" if stem else f"{prefix}-{digest}"
    return validate_run_name(name)
