from __future__ import annotations

import json
from pathlib import Path

import pytest

from mammoth.core import atomic_publish, atomic_write_bytes, atomic_write_json, atomic_write_text


def test_atomic_writers_publish_complete_payloads(tmp_path: Path) -> None:
    binary = atomic_write_bytes(tmp_path / "nested" / "payload.bin", b"abc")
    text = atomic_write_text(tmp_path / "payload.txt", "hello")
    document = atomic_write_json(tmp_path / "payload.json", {"b": 2, "a": 1})

    assert binary.read_bytes() == b"abc"
    assert text.read_text() == "hello"
    assert json.loads(document.read_text()) == {"a": 1, "b": 2}
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_publish_keeps_previous_file_when_writer_fails(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.bin"
    target.write_bytes(b"old")

    def fail_writer(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise RuntimeError("serialization failed")

    with pytest.raises(RuntimeError, match="serialization failed"):
        atomic_publish(target, fail_writer)

    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_publish_requires_writer_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="did not create"):
        atomic_publish(tmp_path / "missing.bin", lambda _path: None)
