"""Shared pytest configuration for the Mammoth test suite."""

from __future__ import annotations

import os
from pathlib import Path

# Typer forces terminal mode when it sees GITHUB_ACTIONS, FORCE_COLOR or
# PY_COLORS (see typer/rich_utils.py), which renders CLI errors inside a
# coloured, box-drawn Rich panel. That panel interleaves ANSI escapes and
# hard-wraps the message, so the plain-substring assertions in test_cli.py stop
# matching. Typer's own escape hatch turns it back off, and the constant it
# guards is evaluated at import time, so set it here rather than per test.
os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")


# Windows supports direct single-process execution and training. Keep unrelated
# POSIX supervision/transaction suites out of collection.
# The shared suites below still run on Linux; Windows-specific handle tests are
# additional coverage rather than substitutes for their behavioral assertions.
_WINDOWS_SUITES = {
    "test_backend.py",
    "test_events.py",
    "test_filesystem.py",
    "test_execution_session.py",
    "test_layout.py",
    "test_logging.py",
    "test_pipeline.py",
    "test_portable_training.py",
    "test_windows.py",
}
collect_ignore = (
    [
        path.name
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name not in _WINDOWS_SUITES
    ]
    if os.name == "nt"
    else []
)
