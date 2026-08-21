"""Exercise entry-level group manifests, event streams, and layout resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from mammoth.core import GroupLayout, validate_group_id
from mammoth.core.groups import (
    GroupEvent,
    GroupEventWriter,
    GroupManifest,
    GroupMember,
    load_group_manifest,
    publish_group_manifest,
    read_group_events,
)


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "../escape", "nested/id", "/absolute", "white space", "-prefix"],
)
def test_validate_group_id_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_group_id(value)


def test_group_layout_resolves_stable_paths(tmp_path: Path) -> None:
    layout = GroupLayout(tmp_path / "runs", "group-1").prepare()

    assert layout.groups_root == tmp_path / "runs" / ".mammoth" / "groups"
    assert layout.group_dir == layout.groups_root / "group-1"
    assert layout.manifest_path == layout.group_dir / "manifest.json"
    assert layout.events_path == layout.group_dir / "events.jsonl"
    assert layout.group_dir.is_dir()


def test_publish_group_manifest_is_atomic_and_round_trips_opaque_metadata(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "runs"
    metadata = {
        "campaign": "demo",
        "tags": ["a", "b"],
        "budget": 3.5,
        "enabled": True,
        "note": None,
    }

    manifest = publish_group_manifest(
        entry,
        order="step-major",
        members=[GroupMember("alpha", ("prepare", "train")), GroupMember("beta", ("prepare",))],
        metadata=metadata,
    )
    metadata["tags"].append("mutated-after-publish")

    layout = GroupLayout(entry, manifest.group_id)
    assert layout.manifest_path.is_file()
    reloaded = load_group_manifest(layout.manifest_path)

    assert reloaded.group_id == manifest.group_id
    assert reloaded.order == "step-major"
    assert [member.run_name for member in reloaded.members] == ["alpha", "beta"]
    assert reloaded.members[0].steps == ("prepare", "train")
    assert reloaded.to_dict()["metadata"] == {
        "campaign": "demo",
        "tags": ["a", "b"],
        "budget": 3.5,
        "enabled": True,
        "note": None,
    }


def test_publish_group_manifest_rejects_non_json_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        publish_group_manifest(
            tmp_path / "runs",
            order="run-major",
            members=[GroupMember("alpha", ("prepare",))],
            metadata={"handle": object()},
        )


def test_publish_group_manifest_rejects_duplicate_explicit_group_id(tmp_path: Path) -> None:
    entry = tmp_path / "runs"
    publish_group_manifest(
        entry,
        order="run-major",
        members=[GroupMember("alpha", ("prepare",))],
        group_id="fixed-id",
    )

    with pytest.raises(FileExistsError):
        publish_group_manifest(
            entry,
            order="run-major",
            members=[GroupMember("beta", ("prepare",))],
            group_id="fixed-id",
        )


def test_publish_group_manifest_requires_at_least_one_member(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        publish_group_manifest(tmp_path / "runs", order="run-major", members=())


def test_publish_group_manifest_rejects_duplicate_member_run_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        GroupManifest(
            schema_version=1,
            group_id="group-1",
            created_at="2026-01-01T00:00:00Z",
            order="run-major",
            members=(GroupMember("alpha", ("prepare",)), GroupMember("alpha", ("train",))),
            metadata={},
        )


def test_group_event_writer_emits_ordered_records_and_read_back_matches(tmp_path: Path) -> None:
    layout = GroupLayout(tmp_path / "runs", "group-1").prepare()
    writer = GroupEventWriter(layout.events_path, group_id="group-1")

    writer.emit("group_started")
    writer.emit("run_started", run_name="alpha")
    writer.emit("step_started", run_name="alpha", step_name="prepare")
    writer.emit("step_failed", run_name="alpha", step_name="prepare", message="boom")
    writer.emit("run_failed", run_name="alpha", message="boom")
    writer.emit("group_failed", message="boom")
    writer.close()

    events = read_group_events(layout.events_path)
    assert [event.event for event in events] == [
        "group_started",
        "run_started",
        "step_started",
        "step_failed",
        "run_failed",
        "group_failed",
    ]
    assert [event.sequence for event in events] == list(range(1, 7))
    assert events[3].message == "boom"
    assert events[3].run_name == "alpha"
    assert events[3].step_name == "prepare"


def test_group_event_writer_disables_after_a_write_failure(tmp_path: Path) -> None:
    layout = GroupLayout(tmp_path / "runs", "group-1").prepare()
    writer = GroupEventWriter(layout.events_path, group_id="group-1")

    class FailingStream:
        def write(self, data: bytes) -> int:
            raise OSError("disk full")

        def close(self) -> None:
            return None

    writer._stream = FailingStream()  # type: ignore[assignment]

    result = writer.emit("group_started")

    assert result is None
    assert not writer.enabled
    assert writer.emit("group_failed") is None


def test_group_event_writer_close_is_idempotent(tmp_path: Path) -> None:
    layout = GroupLayout(tmp_path / "runs", "group-1").prepare()
    writer = GroupEventWriter(layout.events_path, group_id="group-1")
    writer.emit("group_started")
    writer.close()
    writer.close()

    assert read_group_events(layout.events_path)[-1].event == "group_started"


def test_group_event_writer_rejects_a_mismatched_stream_filename(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="events.jsonl"):
        GroupEventWriter(tmp_path / "wrong-name.jsonl", group_id="group-1")


def test_read_group_events_ignores_an_incomplete_trailing_line(tmp_path: Path) -> None:
    layout = GroupLayout(tmp_path / "runs", "group-1").prepare()
    writer = GroupEventWriter(layout.events_path, group_id="group-1")
    writer.emit("group_started")
    writer.emit("group_completed")
    with layout.events_path.open("ab") as handle:
        handle.write(b'{"schema_version":1,"sequence":3,"group_id":"group-1","event":"run_started"')

    events = read_group_events(layout.events_path)

    assert [event.event for event in events] == ["group_started", "group_completed"]


def test_read_group_events_returns_empty_for_a_missing_stream(tmp_path: Path) -> None:
    assert read_group_events(tmp_path / "does-not-exist.jsonl") == []


@pytest.mark.parametrize(
    ("event", "kwargs", "message"),
    (
        ("group_started", {"run_name": "alpha"}, "must omit run_name"),
        ("run_started", {}, "requires a non-empty run_name"),
        ("run_started", {"run_name": "alpha", "step_name": "prepare"}, "must omit step_name"),
        ("step_started", {"run_name": "alpha"}, "requires a non-empty step_name"),
        (
            "run_completed",
            {"run_name": "alpha", "signal": 15},
            "valid only for",
        ),
    ),
)
def test_group_event_enforces_scope_applicability(
    event: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GroupEvent(
            schema_version=1,
            sequence=1,
            time="2026-01-01T00:00:00Z",
            group_id="group-1",
            event=event,  # type: ignore[arg-type]
            **kwargs,
        )
