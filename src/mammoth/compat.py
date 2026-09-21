"""Bridge Python runtime differences used by core, workflow, and Torch modules."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from itertools import zip_longest
from typing import Any

DATACLASS_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


def add_exception_note(error: BaseException, note: str) -> None:
    """Keep cleanup diagnostics on exceptions, including Python before native notes."""
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:
        notes = getattr(error, "__notes__", None)
        if notes is None:
            error.__notes__ = [note]
        else:
            notes.append(note)


def zip_strict(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """Pair scheduler and metric values without silently dropping unequal tails."""
    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("zip() arguments have different lengths")
        yield values
