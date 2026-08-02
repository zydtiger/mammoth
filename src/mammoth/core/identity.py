"""Filesystem-safe logical-run and execution identity validation.

Layout, execution metadata, and event streams share these validators so a
caller-controlled identity can never escape its configured entry directory.
"""

from __future__ import annotations

import re

RUN_NAME_MAX_LENGTH = 255
EXECUTION_ID_MAX_LENGTH = 128

_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXECUTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
