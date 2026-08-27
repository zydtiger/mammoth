"""Shared pytest configuration for the Mammoth test suite."""

from __future__ import annotations

import os

# Typer forces terminal mode when it sees GITHUB_ACTIONS, FORCE_COLOR or
# PY_COLORS (see typer/rich_utils.py), which renders CLI errors inside a
# coloured, box-drawn Rich panel. That panel interleaves ANSI escapes and
# hard-wraps the message, so the plain-substring assertions in test_cli.py stop
# matching. Typer's own escape hatch turns it back off, and the constant it
# guards is evaluated at import time, so set it here rather than per test.
os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")
