"""Exercise live compressed events, independent seek-table decoding, and failure boundaries."""

from __future__ import annotations

import io
import json
import os
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import zstandard as zstd

import mammoth.core._event_stream as storage
from mammoth.core import RunLayout, create_execution_context
from mammoth.core.events import (
    ExecutionEventReadError,
    ExecutionEventTailReader,
    ExecutionEventWriter,
    read_execution_events,
)
from mammoth.core.groups import GroupEventTailReader, GroupEventWriter
from mammoth.core.layout import GroupLayout
from mammoth.monitor import ExecutionMonitor, FleetMonitor


def context(tmp_path: Path):
    layout = RunLayout(tmp_path, "compressed-run").prepare()
    return create_execution_context(
        layout.run_dir,
        run_name=layout.run_name,
        invocation_kind="test",
        intended_phases=("work",),
        world_size=1,
        execution_mode="single",
        command=("python", "worker.py"),
        execution_id="attempt",
    )


def decode(path: Path) -> bytes:
    """Use the standard codec independently of Mammoth's reader."""
    with zstd.ZstdDecompressor().stream_reader(
        io.BytesIO(path.read_bytes()), read_across_frames=True
    ) as reader:
        return reader.read()


def frames_from_standard_table(path: Path) -> list[tuple[int, int]]:
    raw = path.read_bytes()
    count, flags, magic = struct.unpack("<IBI", raw[-9:])
    assert flags == 0 and magic == 0x8F92EAB1
    table_start = len(raw) - 17 - 8 * count
    assert struct.unpack("<II", raw[table_start : table_start + 8]) == (0x184D2A5E, 9 + 8 * count)
    entries = list(struct.iter_unpack("<II", raw[table_start + 8 : -9]))
    offset = 0
    for compressed, uncompressed in entries:
        frame = raw[offset : offset + compressed]
        with zstd.ZstdDecompressor().stream_reader(io.BytesIO(frame)) as reader:
            assert len(reader.read()) == uncompressed
        offset += compressed
    assert offset == table_start
    return entries


def test_default_sink_is_live_seekable_and_reopens_without_rewriting(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(storage, "FRAME_BYTES", 1024)
    ctx = context(tmp_path)
    writer = ExecutionEventWriter.for_process(ctx, rank=0)
    assert writer.path.name == "rank-0.jsonl.zst"
    reader = ExecutionEventTailReader(writer.path)
    for index in range(20):
        writer.emit("heartbeat", phase="work", force=True, message=f"record-{index}")
        assert [event.sequence for event in reader.poll()] == [index + 1]
        assert reader.poll() == []
    writer.close()
    assert reader.poll() == []
    entries = frames_from_standard_table(writer.path)
    assert len(entries) > 1
    prefix = writer.path.read_bytes()
    assert len(decode(writer.path).splitlines()) == 20
    with ExecutionEventWriter.for_process(ctx, rank=0) as reopened:
        assert reopened.enabled
        reopened.emit("process_completed", phase="work")
        assert [event.sequence for event in reader.poll()] == [21]
    assert writer.path.read_bytes().startswith(prefix)
    assert any(uncompressed == 0 for _, uncompressed in frames_from_standard_table(writer.path))
    assert [event.sequence for event in read_execution_events(writer.path)] == list(range(1, 22))


@pytest.mark.parametrize("sealed", [False, True])
def test_cold_tail_bounds_data_reads_and_includes_latest_event(tmp_path: Path, monkeypatch, sealed):
    monkeypatch.setattr(storage, "FRAME_BYTES", 4096)
    ctx = context(tmp_path)
    writer = ExecutionEventWriter.for_process(ctx, rank=0)
    rng = random.Random(4)
    for _ in range(400):
        writer.emit("heartbeat", phase="work", force=True, message=rng.randbytes(256).hex())
    writer.emit("process_completed", phase="work")
    if sealed:
        writer.close()
    total = writer.path.stat().st_size
    bytes_read = 0
    original = storage.read_at

    def counting_read(descriptor, length, offset):
        nonlocal bytes_read
        data = original(descriptor, length, offset)
        bytes_read += len(data)
        return data

    monkeypatch.setattr(storage, "read_at", counting_read)
    reader = ExecutionEventTailReader(writer.path, tail_window_bytes=2048)
    events = reader.poll()
    assert events[-1].event == "process_completed" and events[-1].sequence == 401
    assert events[0].sequence > 1
    assert 0 < bytes_read < total // 2
    before = bytes_read
    assert reader.poll() == []
    assert bytes_read - before <= 4096
    writer.close()


@pytest.mark.parametrize(
    "cache", ["missing", "invalid", "deeply-nested", "wrong-identity", "wrong-entries", "stale"]
)
def test_live_index_is_optional_and_never_hides_unindexed_suffix(
    tmp_path: Path, monkeypatch, cache
):
    monkeypatch.setattr(storage, "FRAME_BYTES", 1024)
    ctx = context(tmp_path)
    writer = ExecutionEventWriter.for_process(ctx, rank=0)
    try:
        writer.emit("process_started", phase="work")
        index = Path(str(writer.path) + ".idx")
        initial = index.read_bytes()
        for _ in range(50):
            writer.emit("heartbeat", phase="work", force=True)
        writer.emit("process_completed", phase="work")
        if cache == "missing":
            index.unlink()
        elif cache == "invalid":
            index.write_bytes(b"not json")
        elif cache == "deeply-nested":
            index.write_text("[" * 10000 + "0" + "]" * 10000)
        elif cache == "wrong-identity":
            payload = json.loads(index.read_bytes())
            payload["identity"] = [-1, -1]
            index.write_text(json.dumps(payload))
        elif cache == "wrong-entries":
            payload = json.loads(index.read_bytes())
            payload["frames"] = [[writer.path.stat().st_size, 0]]
            index.write_text(json.dumps(payload))
        else:
            index.write_bytes(initial)
        data_before = writer.path.read_bytes()
        events = ExecutionEventTailReader(writer.path, tail_window_bytes=1024).poll()
        assert events[-1].event == "process_completed" and events[-1].sequence == 52
        assert writer.path.read_bytes() == data_before
    finally:
        writer.close()


def test_partial_compressed_writes_and_footer_are_followed_exactly_once(tmp_path: Path):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        writer.emit("process_started", phase="work")
        writer.emit("process_completed", phase="work")
    data = writer.path.read_bytes()
    replay = tmp_path / "rank-0.jsonl.zst"
    replay.touch()
    reader = ExecutionEventTailReader(replay)
    seen = []
    with replay.open("ab", buffering=0) as handle:
        for value in data:
            handle.write(bytes([value]))
            seen.extend(reader.poll())
    assert [event.sequence for event in seen] == [1, 2]
    assert reader.poll() == []


def test_unsealed_crash_log_is_readable_but_not_reopened(tmp_path: Path):
    ctx = context(tmp_path)
    writer = ExecutionEventWriter.for_process(ctx, rank=0)
    writer.emit("process_started", phase="work")
    raw = writer.path.read_bytes()
    writer.close()
    writer.path.write_bytes(raw)  # Model process exit before frame/footer finalization.
    assert [event.event for event in read_execution_events(writer.path)] == ["process_started"]
    with ExecutionEventWriter.for_process(ctx, rank=0) as reopened:
        assert not reopened.enabled
    assert writer.path.read_bytes() == raw


def test_corruption_reports_valid_prior_frame_and_disables_reader(tmp_path: Path):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        writer.emit("process_started", phase="work")
    with writer.path.open("ab") as handle:
        handle.write(b"invalid compressed frame")
    reader = ExecutionEventTailReader(writer.path)
    with pytest.raises(ExecutionEventReadError) as error:
        reader.poll()
    assert [event.event for event in error.value.valid_events] == ["process_started"]
    with pytest.raises(ExecutionEventReadError):
        reader.poll()
    snapshot = ExecutionMonitor(RunLayout(tmp_path, "compressed-run"), "attempt").poll()
    assert snapshot.status == "running" and len(snapshot.warnings) == 1


def test_duplicate_formats_warn_instead_of_folding_twice(tmp_path: Path):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        writer.emit("process_started", phase="work")
    writer.path.with_suffix("").write_bytes(decode(writer.path))
    snapshot = ExecutionMonitor(RunLayout(tmp_path, "compressed-run"), "attempt").poll()
    assert len(snapshot.warnings) == 1
    assert "Both event stream formats" in snapshot.warnings[0]


def test_index_failure_does_not_disable_successful_events(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(storage, "FRAME_BYTES", 512)

    def fail(*args, **kwargs):
        raise OSError("index unavailable")

    monkeypatch.setattr(storage, "atomic_write_json", fail)
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        for _ in range(10):
            assert writer.emit("heartbeat", phase="work", force=True) is not None
        assert writer.sequence == 10 and writer.enabled
    assert len(read_execution_events(writer.path)) == 10
    frames_from_standard_table(writer.path)


def test_group_and_fleet_consume_compressed_and_legacy_events(tmp_path: Path):
    group = GroupLayout(tmp_path, "group").prepare()
    with GroupEventWriter(group.events_path, group_id="group") as writer:
        reader = GroupEventTailReader(group.events_path)
        writer.emit("group_started")
        assert [event.event for event in reader.poll()] == ["group_started"]
        writer.emit("group_completed")
        assert [event.event for event in reader.poll()] == ["group_completed"]
    frames_from_standard_table(group.events_path)
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        writer.emit("process_started", phase="work")
        assert (
            FleetMonitor(tmp_path).poll(stale_after_seconds=10**9).loose_runs[0].status == "running"
        )
        writer.emit("process_completed", phase="work")
    assert FleetMonitor(tmp_path).poll().loose_runs[0].status == "completed"


@pytest.mark.parametrize("mutation", ["truncate", "replace", "prefix"])
def test_compressed_tail_detects_file_mutations(tmp_path: Path, mutation):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        writer.emit("heartbeat", phase="work", force=True)
    reader = ExecutionEventTailReader(writer.path)
    assert len(reader.poll()) == 1
    if mutation == "truncate":
        writer.path.write_bytes(b"")
    elif mutation == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(writer.path.read_bytes())
        os.replace(replacement, writer.path)
    else:
        data = writer.path.read_bytes()
        writer.path.write_bytes(b"X" + data[1:])
    with pytest.raises(ExecutionEventReadError):
        reader.poll()


def test_missing_after_first_compressed_poll_is_not_treated_as_uncreated(tmp_path: Path):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        writer.emit("heartbeat", phase="work", force=True)
    reader = ExecutionEventTailReader(writer.path)
    assert len(reader.poll()) == 1
    writer.path.unlink()
    with pytest.raises(FileNotFoundError):
        reader.poll()


@pytest.mark.parametrize("failure", ["short-write", "flush", "footer"])
def test_io_failures_disable_only_the_event_writer(tmp_path: Path, failure):
    ctx = context(tmp_path)
    writer = ExecutionEventWriter.for_process(ctx, rank=0)
    writer.emit("process_started", phase="work")
    compressed = writer._stream
    assert isinstance(compressed, storage.ZstdEventWriter)
    original = compressed.stream

    class FailingStream:
        def write(self, data):
            if failure == "short-write":
                return original.write(data[:1])
            if failure == "footer" and data.startswith(struct.pack("<I", 0x184D2A5E)):
                raise OSError("footer unavailable")
            return original.write(data)

        def flush(self):
            if failure == "flush":
                raise OSError("flush unavailable")
            original.flush()

        def fileno(self):
            return original.fileno()

        def close(self):
            original.close()

    compressed.stream = FailingStream()
    if failure != "footer":
        assert writer.emit("process_completed", phase="work") is None
        assert not writer.enabled
    writer.close()
    assert original.closed
    # The preceding complete event survives even if finalization cannot run.
    events = read_execution_events(writer.path)
    assert events[0].event == "process_started"


def test_bounded_compressed_reader_still_rejects_sequence_gaps(tmp_path: Path):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        for _ in range(30):
            writer.emit("heartbeat", phase="work", force=True)
    reader = ExecutionEventTailReader(writer.path, tail_window_bytes=1024)
    assert reader.poll()[-1].sequence == 30
    bad = read_execution_events(writer.path)[-1].to_dict()
    bad["sequence"] = 32
    with writer.path.open("ab") as handle:
        handle.write(zstd.ZstdCompressor(level=1).compress(json.dumps(bad).encode() + b"\n"))
    with pytest.raises(ExecutionEventReadError, match="expected 31"):
        reader.poll()


@pytest.mark.parametrize("kind", ["execution", "group"])
def test_old_producer_can_appear_after_reader_starts(tmp_path: Path, kind):
    ctx = context(tmp_path)
    if kind == "execution":
        path = ctx.execution_dir / "rank-0.jsonl.zst"
        reader = ExecutionEventTailReader(path)
        assert reader.poll() == []
        path.with_suffix("").touch()
        with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
            writer.emit("process_started", phase="work")
            assert [event.event for event in reader.poll()] == ["process_started"]
    else:
        path = tmp_path / "events.jsonl.zst"
        reader = GroupEventTailReader(path)
        assert reader.poll() == []
        with GroupEventWriter(path.with_suffix(""), group_id="group") as writer:
            writer.emit("group_started")
            assert [event.event for event in reader.poll()] == ["group_started"]


def test_skipping_an_empty_old_footer_does_not_skip_sequence_validation(tmp_path: Path):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        pass
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        writer.emit("process_started", phase="work")
    raw = json.loads(decode(writer.path))
    raw["sequence"] = 2
    # Keep a leading empty table, followed by a complete frame containing an
    # invalid first sequence and a valid standard table indexing both frames.
    empty_table = struct.pack("<IIIBI", 0x184D2A5E, 9, 0, 0, 0x8F92EAB1)
    record = json.dumps(raw).encode() + b"\n"
    frame = zstd.ZstdCompressor(level=1).compress(record)
    body = struct.pack("<IIIIIBI", len(empty_table), 0, len(frame), len(record), 2, 0, 0x8F92EAB1)
    writer.path.write_bytes(empty_table + frame + struct.pack("<II", 0x184D2A5E, len(body)) + body)
    with pytest.raises(ExecutionEventReadError, match="expected 1"):
        ExecutionEventTailReader(writer.path, tail_window_bytes=128 * 1024).poll()


def test_first_bounded_poll_retains_terminal_before_corrupt_suffix(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(storage, "FRAME_BYTES", 512)
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        for _ in range(3500):
            writer.emit("heartbeat", phase="work", force=True)
        writer.emit("process_completed", phase="work")
    with writer.path.open("ab") as handle:
        handle.write(b"invalid compressed frame")
    reader = ExecutionEventTailReader(writer.path, tail_window_bytes=1024)
    with pytest.raises(ExecutionEventReadError) as error:
        reader.poll()
    assert error.value.valid_events[0].sequence > 1
    assert error.value.valid_events[-1].event == "process_completed"
    snapshot = FleetMonitor(tmp_path).poll().loose_runs[0]
    assert snapshot.status == "completed" and len(snapshot.warnings) == 1
    assert "sequence" not in snapshot.warnings[0]


@pytest.mark.parametrize("kind", ["execution", "group"])
@pytest.mark.parametrize("present", [False, True])
def test_overlapping_polls_consume_compressed_events_once(
    tmp_path: Path, monkeypatch, kind, present
):
    ctx = context(tmp_path)
    if kind == "execution":
        writer = ExecutionEventWriter.for_process(ctx, rank=0)
        reader = ExecutionEventTailReader(writer.path)
        for _ in range(30):
            writer.emit("heartbeat", phase="work", force=True)
    else:
        writer = GroupEventWriter(tmp_path / "events.jsonl.zst", group_id="group")
        reader = GroupEventTailReader(writer.path)
        for _ in range(30):
            writer.emit("group_started")
    writer.close()
    if not present:
        writer.path.unlink()
    original_open = storage.open_event_file

    def slow_open(*args):
        try:
            return original_open(*args)
        finally:
            time.sleep(0.01)

    monkeypatch.setattr(storage, "open_event_file", slow_open)
    original = storage.read_at

    def slow_read(*args):
        data = original(*args)
        time.sleep(0.01)  # Model I/O releasing the GIL during overlapping refreshes.
        return data

    monkeypatch.setattr(storage, "read_at", slow_read)
    ready = Barrier(2)

    def poll():
        ready.wait(timeout=5)
        return reader.poll()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: poll(), range(2)))
    expected = list(range(1, 31)) if present else []
    assert sorted(event.sequence for result in results for event in result) == expected


@pytest.mark.parametrize("source", ["process", "runner"])
def test_ambiguous_formats_disable_writer_without_escaping_factory(tmp_path: Path, source):
    ctx = context(tmp_path)
    factory = (
        (lambda: ExecutionEventWriter.for_process(ctx, rank=0))
        if source == "process"
        else (lambda: ExecutionEventWriter.for_runner(ctx))
    )
    with factory() as writer:
        pass
    writer.path.with_suffix("").touch()
    before = writer.path.read_bytes()
    with factory() as reopened:
        assert not reopened.enabled
    assert writer.path.read_bytes() == before
    assert writer.path.with_suffix("").read_bytes() == b""


@pytest.mark.parametrize("cache", ["missing", "invalid", "valid"])
def test_corrupt_first_poll_keeps_the_same_window_as_clean_stream(tmp_path: Path, cache):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        for _ in range(1200):
            writer.emit("heartbeat", phase="work", force=True, message="x" * 256)
        writer.emit("process_completed", phase="work")
    expected = ExecutionEventTailReader(writer.path, tail_window_bytes=2048).poll()
    index = Path(str(writer.path) + ".idx")
    if cache == "missing":
        index.unlink()
    elif cache == "invalid":
        index.write_bytes(b"invalid cache")
    with writer.path.open("ab") as handle:
        handle.write(b"invalid compressed frame")
    with pytest.raises(ExecutionEventReadError) as error:
        ExecutionEventTailReader(writer.path, tail_window_bytes=2048).poll()
    assert list(error.value.valid_events) == expected


@pytest.mark.parametrize("cache", [False, True])
def test_damaged_seek_table_does_not_hide_valid_event_frames(tmp_path: Path, cache):
    ctx = context(tmp_path)
    with ExecutionEventWriter.for_process(ctx, rank=0) as writer:
        writer.emit("process_started", phase="work")
        writer.emit("process_completed", phase="work")
    raw = bytearray(writer.path.read_bytes())
    raw[-5] = 1  # Reserved seek-table flag; the skippable frame is still decodable.
    writer.path.write_bytes(raw)
    if not cache:
        Path(str(writer.path) + ".idx").unlink()
    assert read_execution_events(writer.path)[-1].event == "process_completed"
    events = ExecutionEventTailReader(writer.path, tail_window_bytes=2048).poll()
    assert events[-1].event == "process_completed"
    assert FleetMonitor(tmp_path).poll().loose_runs[0].status == "completed"
    with ExecutionEventWriter.for_process(ctx, rank=0) as reopened:
        assert not reopened.enabled
    assert writer.path.read_bytes() == raw


@pytest.mark.parametrize("existing_suffix", [".jsonl", ".jsonl.zst"])
def test_group_writer_cannot_create_a_second_format(tmp_path: Path, existing_suffix):
    existing = tmp_path / ("events" + existing_suffix)
    other = tmp_path / ("events.jsonl.zst" if existing_suffix == ".jsonl" else "events.jsonl")
    with GroupEventWriter(existing, group_id="group") as writer:
        writer.emit("group_started")
    original = existing.read_bytes()
    with GroupEventWriter(other, group_id="group") as duplicate:
        assert not duplicate.enabled
    assert not other.exists()
    assert existing.read_bytes() == original
    assert GroupEventTailReader(other).poll()[-1].event == "group_started"
