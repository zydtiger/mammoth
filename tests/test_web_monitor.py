"""Exercise the read-only web monitor service and HTTP routes."""

from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from mammoth.core import (
    GroupMember,
    RunLayout,
    create_execution_context,
    publish_group_manifest,
)
from mammoth.core.events import ExecutionEventWriter
from mammoth.core.telemetry import TelemetryRecorder, telemetry_stream_path
from mammoth.monitor.web import (
    MAX_DOCUMENT_ROWS,
    MAX_POINTS_LIMIT,
    MonitorService,
    _compact_spans,
    _Span,
    _stage_summary,
    create_queue_monitor_server,
    create_web_server,
    discover_entries,
    downsample,
    queue_entries,
)


def create_run(
    entry: Path,
    name: str = "run",
    *,
    execution_id: str = "attempt",
    created_at: str = "2026-01-01T00:00:00Z",
    parent_execution_id: str | None = None,
) -> tuple[RunLayout, object]:
    layout = RunLayout(entry, name).prepare()
    context = create_execution_context(
        layout.run_dir,
        run_name=name,
        invocation_kind="test",
        intended_phases=("train",),
        world_size=1,
        execution_mode="single",
        command=("train",),
        execution_id=execution_id,
        created_at=created_at,
        parent_execution_id=parent_execution_id,
    )
    return layout, context


def write_training_events(context: object, *, epochs: int = 3) -> None:
    with ExecutionEventWriter.for_process(context, rank=0, world_size=1) as writer:
        writer.emit("process_started", phase="train")
        writer.emit("stage_changed", stage="train", lane="main")
        writer.emit("stage_changed", stage="write", lane="background")
        for epoch in range(epochs):
            writer.emit(
                "progress",
                phase="train",
                task_id="epochs",
                completed=epoch + 1,
                total=epochs,
                final=True,
                coordinates={"epoch": epoch + 1},
                display_metrics={"train/loss": 1.0 / (epoch + 1), "val/error": 0.5},
            )
        writer.emit("stage_changed", stage="validate", lane="main")
        writer.emit("display", kind="table", name="Domains", data={"columns": ["a"], "rows": [[1]]})
        writer.emit("process_completed", phase="train", exit_code=0)


def write_telemetry(context: object, samples: list[dict[str, object]]) -> None:
    recorder = TelemetryRecorder(context.execution_dir, environment={"CUDA_VISIBLE_DEVICES": ""})
    recorder.interval_seconds = 3600.0
    recorder.start()
    for sample in samples:
        recorder._write(sample)
    recorder.stop()


def swept(service: MonitorService) -> MonitorService:
    service.sweep()
    return service


def snapshot_tree(path: Path) -> dict[str, tuple[int, int]]:
    return {
        str(child): (child.stat().st_size, child.stat().st_mtime_ns)
        for child in path.rglob("*")
        if child.is_file()
    }


@pytest.fixture
def server(tmp_path: Path) -> Iterator[object]:
    instance = create_web_server([tmp_path], port=0, interval_seconds=0.05)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.shutdown_monitor()
    thread.join(timeout=5)


def refresh(server: object) -> None:
    """Make the running server discover entries created after it started."""
    server.service._discovered_at = None
    server.service.sweep()


def get(server: object, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f"{server.url}{path}", timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_discovery_finds_nested_entries_without_entering_runs(tmp_path: Path) -> None:
    create_run(tmp_path, "top")
    create_run(tmp_path / "experiment", "R1")
    create_run(tmp_path / "a" / "b", "R2")
    # Entries inside a run's own artifacts and hidden directories stay invisible.
    create_run(tmp_path / "top" / "code", "inner")
    create_run(tmp_path / ".hidden", "secret")
    (tmp_path / "no-runs" / "deep").mkdir(parents=True)

    entries = discover_entries([tmp_path])

    assert set(entries.values()) == {tmp_path, tmp_path / "experiment", tmp_path / "a" / "b"}


def test_discovery_respects_the_depth_bound(tmp_path: Path) -> None:
    create_run(tmp_path / "a" / "b" / "c", "run")

    assert discover_entries([tmp_path], max_depth=2) == {}
    assert len(discover_entries([tmp_path], max_depth=3)) == 1


def test_discovery_keeps_same_named_roots_distinct(tmp_path: Path) -> None:
    create_run(tmp_path / "one" / "runs")
    create_run(tmp_path / "two" / "runs")

    entries = discover_entries([tmp_path / "one" / "runs", tmp_path / "two" / "runs"])

    assert len(entries) == 2


def test_discovery_includes_group_only_entries(tmp_path: Path) -> None:
    (tmp_path / ".mammoth" / "groups").mkdir(parents=True)

    assert list(discover_entries([tmp_path]).values()) == [tmp_path]


def test_fleet_hides_directories_without_executions(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    (tmp_path / "experiment-without-runs").mkdir()

    fleet = swept(MonitorService([tmp_path])).fleet()

    assert fleet["ready"] is True
    assert [run["run_name"] for run in fleet["entries"][0]["loose_runs"]] == ["run"]
    assert fleet["entries"][0]["loose_runs"][0]["status"] == "completed"


def test_group_route_returns_members(tmp_path: Path) -> None:
    create_run(tmp_path, "member")
    manifest = publish_group_manifest(
        tmp_path,
        order="run-major",
        members=[GroupMember("member", ("train",))],
    )
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    group = service.group(entry, manifest.group_id)

    assert group is not None
    assert group["entry"] == entry
    assert [member["run_name"] for member in group["members"]] == ["member"]


def test_new_entries_and_runs_appear_without_restart(tmp_path: Path) -> None:
    (tmp_path / "experiment").mkdir()
    service = swept(MonitorService([tmp_path]))
    assert service.fleet()["entries"] == []

    _layout, context = create_run(tmp_path / "experiment", "R1")
    write_training_events(context)
    service._discovered_at = None
    service.sweep()
    entry = service.fleet()["entries"][0]
    assert [run["run_name"] for run in entry["loose_runs"]] == ["R1"]

    _layout, context = create_run(tmp_path / "experiment", "R2")
    write_training_events(context)
    service.sweep()
    assert {run["run_name"] for run in service.fleet()["entries"][0]["loose_runs"]} == {"R1", "R2"}


def test_exact_execution_and_resume_lineage(tmp_path: Path) -> None:
    _layout, first = create_run(tmp_path, execution_id="first")
    write_training_events(first, epochs=1)
    _layout, second = create_run(
        tmp_path,
        execution_id="second",
        created_at="2026-01-02T00:00:00Z",
        parent_execution_id="first",
    )
    write_training_events(second, epochs=1)
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    newest = service.run(entry, "run", None)
    exact = service.run(entry, "run", "first")

    assert newest["execution_id"] == "second"
    assert [attempt["execution_id"] for attempt in newest["attempts"]] == ["first", "second"]
    assert exact["execution_id"] == "first"
    train = next(row for row in newest["stage_summary"] if row["stage"] == "train")
    assert train["count"] == 2


def test_run_document_includes_stages_tables_and_metrics(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    service = swept(MonitorService([tmp_path]))
    entry = next(iter(service.fleet()["entries"]))["id"]

    run = service.run(entry, "run", None)

    assert run is not None
    assert {metric["name"] for metric in run["metrics"]} == {"train/loss", "val/error"}
    assert run["display_items"][0]["data"] == {"columns": ["a"], "rows": [[1]]}
    assert {(row["lane"], row["stage"]) for row in run["stage_summary"]} == {
        ("main", "train"),
        ("main", "validate"),
        ("background", "write"),
    }
    main_share = sum(row["share"] for row in run["stage_summary"] if row["lane"] == "main")
    assert main_share == pytest.approx(1.0)
    json.dumps(run, allow_nan=False)


def test_stage_summary_attributes_telemetry_to_stages(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context, epochs=1)
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]
    stages = service.stages(entry, "run", None, since=0.0, max_points=100)
    main = [time for time, lane in zip(stages["t"], stages["lane"]) if lane == "main"]
    write_telemetry(
        context,
        [
            {"kind": "gpu", "time": main[0], "gpu": "0", "utilization": 80.0},
            {"kind": "gpu", "time": main[1] + 1, "gpu": "0", "utilization": 20.0},
            {"kind": "cpu", "time": main[0], "main_cores": 1.5, "worker_cores": 0.5},
            {"kind": "status", "time": main[0], "source": "gpu", "detail": "gone"},
        ],
    )
    service = swept(MonitorService([tmp_path]))

    run = service.run(entry, "run", None)

    train = next(row for row in run["stage_summary"] if row["stage"] == "train")
    assert train["gpu_mean"] == 80.0
    assert train["gpu_median"] == 80.0
    assert train["main_cores"] == 1.5
    assert train["worker_cores"] == 0.5
    assert "telemetry/gpu/0/utilization" in run["telemetry_series"]
    assert telemetry_stream_path(context.execution_dir).is_file()


def test_series_honors_cursor_and_point_limit(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_telemetry(
        context,
        [
            {"kind": "cpu", "time": float(index), "main_cores": float(index), "worker_cores": 0.0}
            for index in range(1, 1001)
        ],
    )
    service = swept(MonitorService([tmp_path], interval_seconds=0.0))
    entry = service.fleet()["entries"][0]["id"]
    name = "telemetry/cpu/main_cores"

    limited = service.series(entry, "run", None, [name], since=0.0, max_points=50)
    later = service.series(entry, "run", None, [name], since=990.0, max_points=50)
    nothing = service.series(entry, "run", None, [name], since=later["cursor"], max_points=50)

    assert len(limited["series"][name]["t"]) <= 50
    # Each bucket keeps its extremes, so the full range survives.
    assert limited["series"][name]["y"][0] == 1.0
    assert limited["series"][name]["y"][-1] == 1000.0
    assert later["series"][name]["t"] == [float(value) for value in range(991, 1001)]
    assert later["cursor"] == 1000.0
    assert nothing["series"][name]["t"] == []


def test_metric_series_use_epoch_axis_when_every_point_has_one(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    series = service.series(entry, "run", None, ["train/loss"], since=0.0, max_points=100)

    assert series["series"]["train/loss"]["axis"] == "epoch"
    assert series["series"]["train/loss"]["x"] == [1.0, 2.0, 3.0]


def test_unknown_series_names_are_ignored(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    series = service.series(entry, "run", None, ["nope"], since=0.0, max_points=100)

    assert series["series"] == {}


def test_series_after_resumes_each_series_from_its_own_cursor(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_telemetry(
        context,
        [
            {"kind": "cpu", "time": float(index), "main_cores": 1.0, "worker_cores": 2.0}
            for index in range(1, 3001)
        ],
    )
    service = swept(MonitorService([tmp_path], interval_seconds=0.0))
    entry = service.fleet()["entries"][0]["id"]
    busy, quiet = "telemetry/cpu/main_cores", "telemetry/cpu/worker_cores"

    response = service.series(
        entry, "run", None, [busy, quiet], since=0.0, max_points=1000, after=[2998.0, 100.0]
    )

    # A quiet series far behind must not make a live one resend or lose its tail.
    assert response["series"][busy]["t"] == [2999.0, 3000.0]
    assert len(response["series"][quiet]["t"]) <= 1000
    assert response["series"][quiet]["t"][0] > 100.0


def write_stage_pattern(
    context: object, transitions: list[tuple[str, str]], *, lane: str = "main"
) -> None:
    clock = ["2026-01-01T00:00:00Z"]
    with ExecutionEventWriter.for_process(
        context, rank=0, world_size=1, utc_clock=lambda: clock[0]
    ) as writer:
        for moment, stage in transitions:
            clock[0] = moment
            writer.emit("stage_changed", stage=stage, lane=lane)


def test_resume_gap_is_not_counted_as_stage_time(tmp_path: Path) -> None:
    _layout, parent = create_run(tmp_path, execution_id="parent")
    write_stage_pattern(
        parent, [("2026-01-01T00:00:10Z", "train"), ("2026-01-01T00:01:10Z", "validate")]
    )
    _layout, child = create_run(
        tmp_path,
        execution_id="child",
        created_at="2026-01-01T10:00:00Z",
        parent_execution_id="parent",
    )
    write_stage_pattern(
        child, [("2026-01-01T10:00:10Z", "train"), ("2026-01-01T10:01:10Z", "validate")]
    )
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    rows = {row["stage"]: row for row in service.run(entry, "run", None)["stage_summary"]}

    assert rows["train"]["mean_seconds"] == pytest.approx(60.0)
    assert rows["validate"]["mean_seconds"] == pytest.approx(0.0)
    assert rows["train"]["share"] == pytest.approx(1.0)


def test_stage_query_includes_the_span_active_at_the_window_start(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_stage_pattern(
        context, [("2026-01-01T00:00:10Z", "train"), ("2026-01-01T00:01:10Z", "validate")]
    )
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]
    train_start = service.stages(entry, "run", None, since=0.0, max_points=100)["t"][0]

    stages = service.stages(entry, "run", None, since=train_start + 30, max_points=100)

    assert stages["stage"] == ["train", "validate"]
    assert stages["end"][0] == train_start + 60


def test_stage_compaction_keeps_stage_shares_without_aliasing() -> None:
    spans = []
    for index in range(2000):
        spans.append(_Span(float(index * 10), float(index * 10 + 9), "compute", "main", "rank-0"))
        spans.append(_Span(float(index * 10 + 9), float(index * 10 + 10), "data", "main", "rank-0"))

    kept = _compact_spans(spans, 100, now=20_000.0)

    assert len(kept) <= 100
    total = sum(span.end - span.start for span in kept)
    data = sum(span.end - span.start for span in kept if span.stage == "data")
    assert data / total == pytest.approx(0.1, abs=0.02)
    assert kept == sorted(kept, key=lambda span: span.start)


def test_stage_compaction_of_an_even_pattern_shows_both_stages() -> None:
    spans = [
        _Span(float(index), float(index + 1), "data" if index % 2 else "compute", "main", "rank-0")
        for index in range(2000)
    ]

    kept = _compact_spans(spans, 1000, now=2000.0)

    assert {span.stage for span in kept} == {"compute", "data"}


def test_stage_compaction_gives_each_lane_a_budget_and_keeps_open_spans() -> None:
    main = [_Span(float(index), float(index + 1), "a", "main", "rank-0") for index in range(50)]
    main.append(_Span(50.0, None, "b", "main", "rank-0"))
    background = [_Span(1.0, 40.0, "write", "background", "rank-0")]

    kept = _compact_spans([*main, *background], 21, now=60.0)

    assert [span.stage for span in kept if span.lane == "background"] == ["write"]
    assert len(kept) <= 21
    assert next(span for span in reversed(kept) if span.lane == "main").end is None


def test_unreadable_directory_does_not_stop_discovery_or_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_run(tmp_path / "ok", "run")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert list(discover_entries([tmp_path]).values()) == [tmp_path / "ok"]
    finally:
        locked.chmod(0o755)

    calls = []
    service = MonitorService([tmp_path], interval_seconds=0.01)

    def fail_once() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        service._stop.set()

    monkeypatch.setattr(service, "sweep", fail_once)
    service._sweep_forever()
    assert len(calls) == 2


def test_transient_read_failure_is_not_reported_as_missing(
    tmp_path: Path, server: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    refresh(server)
    entry = json.loads(get(server, "api/fleet")[1])["entries"][0]["id"]

    def fail(self: object, execution: object = None) -> None:
        raise PermissionError("busy")

    monkeypatch.setattr("mammoth.monitor.web.RunMonitor.poll", fail)

    assert get(server, f"api/run?entry={entry}&run=run")[0] == 503


def test_loopback_server_refuses_a_foreign_host_header(server: object) -> None:
    request = urllib.request.Request(f"{server.url}api/fleet", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)
    assert caught.value.code == 403

    for allowed in ("localhost:1234", "127.0.0.1", "[::1]:8765"):
        request = urllib.request.Request(f"{server.url}api/fleet", headers={"Host": allowed})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200


def test_series_and_stages_reject_an_unknown_execution(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    assert service.series(entry, "run", "nope", ["train/loss"], since=0.0, max_points=9) is None
    assert service.stages(entry, "run", "nope", since=0.0, max_points=9) is None


def test_run_document_is_bounded_with_many_distinct_names(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    with ExecutionEventWriter.for_process(context, rank=0, world_size=1) as writer:
        for index in range(MAX_DOCUMENT_ROWS + 50):
            writer.emit("stage_changed", stage=f"stage-{index}", lane="main")
            writer.emit("display", kind="table", name=f"table-{index}", data={"rows": []})
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    run = service.run(entry, "run", None)

    assert len(run["stage_summary"]) == MAX_DOCUMENT_ROWS
    assert len(run["stages"]) == MAX_DOCUMENT_ROWS
    assert len(run["display_items"]) == MAX_DOCUMENT_ROWS
    assert run["display_items"][0]["name"] == f"table-{MAX_DOCUMENT_ROWS + 49}"


def test_stage_summary_is_reused_between_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    service = swept(MonitorService([tmp_path], interval_seconds=0.0))
    entry = service.fleet()["entries"][0]["id"]
    calls = []
    real = _stage_summary

    def counted(*args: object) -> object:
        calls.append(1)
        return real(*args)

    monkeypatch.setattr("mammoth.monitor.web._stage_summary", counted)

    service.run(entry, "run", None)
    service.run(entry, "run", None)
    assert len(calls) == 1

    monkeypatch.setattr("mammoth.monitor.web.SUMMARY_SECONDS", 0.0)
    service.run(entry, "run", None)
    assert len(calls) == 2


@pytest.mark.skipif(shutil.which("playwright") is None, reason="Requires the Playwright CLI")
@pytest.mark.parametrize("viewport", ["1280,800", "390,800"])
def test_browser_renders_run_and_fleet_pages(
    tmp_path: Path, server: object, viewport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry_dir = tmp_path / "entry"
    _layout, context = create_run(entry_dir)
    write_training_events(context)
    write_telemetry(context, [{"kind": "cpu", "time": 1.0, "main_cores": 1.0, "worker_cores": 0.5}])
    refresh(server)
    entry = json.loads(get(server, "api/fleet")[1])["entries"][0]["id"]
    monkeypatch.setenv("NO_PROXY", "*")
    pages = {
        f"{server.url}?{urllib.parse.urlencode({'entry': entry, 'run': 'run'})}": "#metrics h2",
        server.url: "#fleet table",
    }

    for url, selector in pages.items():
        result = subprocess.run(
            [
                "playwright",
                "screenshot",
                "--browser=chromium",
                f"--viewport-size={viewport}",
                f"--wait-for-selector={selector}",
                url,
                str(tmp_path / "page.png"),
            ],
            timeout=60,
            capture_output=True,
            text=True,
        )
        if "Executable doesn't exist" in result.stderr:
            pytest.skip("Requires the Playwright Chromium browser")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "page.png").stat().st_size > 0


def test_mixed_epoch_series_keeps_one_time_axis_across_refreshes(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    with ExecutionEventWriter.for_process(context, rank=0, world_size=1) as writer:
        for step in range(4):
            writer.emit(
                "progress",
                phase="train",
                task_id="steps",
                completed=step,
                final=True,
                coordinates={"epoch": step} if step < 2 else {"step": step},
                display_metrics={"loss": float(step)},
            )
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    full = service.series(entry, "run", None, ["loss"], since=0.0, max_points=9)["series"]["loss"]
    later = service.series(
        entry, "run", None, ["loss"], since=0.0, max_points=9, after=[full["t"][1]]
    )["series"]["loss"]
    done = service.series(
        entry, "run", None, ["loss"], since=0.0, max_points=9, after=[full["t"][-1]]
    )["series"]["loss"]

    assert (full["axis"], full["x"]) == ("time", None)
    assert (later["axis"], later["x"]) == ("time", None)
    assert done["t"] == []


def test_downsample_keeps_short_series_and_bounds_long_ones() -> None:
    times = [float(index) for index in range(100)]
    values = [float(index % 7) for index in range(100)]

    assert downsample(times[:5], values[:5], None, 10) == (times[:5], values[:5], None)
    kept_times, kept_values, kept_x = downsample(times, values, times, 20)
    assert len(kept_times) <= 20
    assert kept_times == kept_x
    assert max(kept_values) == 6.0
    assert kept_times == sorted(kept_times)


def test_views_are_lazy_and_released_after_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    service = swept(MonitorService([tmp_path]))
    assert service._views == {}
    entry = service.fleet()["entries"][0]["id"]

    service.run(entry, "run", None)
    assert len(service._views) == 1

    monkeypatch.setattr("mammoth.monitor.web.RUN_IDLE_SECONDS", 0.0)
    service.sweep()
    assert service._views == {}


def test_unknown_targets_return_none(tmp_path: Path) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    service = swept(MonitorService([tmp_path]))
    entry = service.fleet()["entries"][0]["id"]

    assert service.run("missing", "run", None) is None
    assert service.run(entry, "../run", None) is None
    assert service.run(entry, "missing", None) is None
    assert service.run(entry, "run", "not-an-execution") is None
    assert service.group(entry, "missing") is None


def test_server_binds_to_loopback_by_default(server: object) -> None:
    assert server.server_address[0] == "127.0.0.1"


def test_routes_serve_page_assets_and_documents(tmp_path: Path, server: object) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    refresh(server)

    status, body = get(server, "")
    assert status == 200
    assert b"Mammoth Monitor" in body
    for asset in ("app.js", "style.css", "uPlot.iife.min.js", "uPlot.min.css"):
        assert get(server, asset)[0] == 200

    status, body = get(server, "api/fleet")
    assert status == 200
    entry = json.loads(body)["entries"][0]["id"]
    assert get(server, f"api/run?entry={entry}&run=run")[0] == 200
    assert get(server, f"api/series?entry={entry}&run=run&name=train/loss")[0] == 200
    assert get(server, f"api/stages?entry={entry}&run=run")[0] == 200
    assert get(server, f"api/run?entry={entry}&run=missing")[0] == 404
    assert get(server, "api/run")[0] == 400
    assert get(server, "api/unknown")[0] == 404
    assert get(server, f"api/series?entry={entry}&run=run&name=a&name=b&after=1")[0] == 400
    assert get(server, f"api/series?entry={entry}&run=run&since=nan")[0] == 400


def test_static_routes_do_not_expose_other_files(server: object) -> None:
    assert get(server, "../web.py")[0] == 404
    assert get(server, "web_static/app.js")[0] == 404


def test_point_limit_is_clamped(tmp_path: Path, server: object) -> None:
    _layout, context = create_run(tmp_path)
    write_telemetry(
        context,
        [
            {"kind": "cpu", "time": float(index), "main_cores": 1.0, "worker_cores": 0.0}
            for index in range(1, MAX_POINTS_LIMIT + 500)
        ],
    )
    refresh(server)
    entry = json.loads(get(server, "api/fleet")[1])["entries"][0]["id"]

    _status, body = get(
        server,
        f"api/series?entry={entry}&run=run&name=telemetry/cpu/main_cores&max_points=999999",
    )

    assert len(json.loads(body)["series"]["telemetry/cpu/main_cores"]["t"]) <= MAX_POINTS_LIMIT


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
def test_server_has_no_state_changing_route(server: object, method: str) -> None:
    request = urllib.request.Request(f"{server.url}api/fleet", data=b"{}", method=method)

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)

    assert caught.value.code == 501


@pytest.mark.skipif(os.name == "nt", reason="File identity comparison is POSIX-specific")
def test_web_reads_write_nothing_under_the_execution(tmp_path: Path, server: object) -> None:
    _layout, context = create_run(tmp_path)
    write_training_events(context)
    write_telemetry(context, [{"kind": "cpu", "time": 1.0, "main_cores": 1.0, "worker_cores": 0.0}])
    before = snapshot_tree(tmp_path)
    refresh(server)
    entry = json.loads(get(server, "api/fleet")[1])["entries"][0]["id"]

    get(server, f"api/run?entry={entry}&run=run")
    get(server, f"api/series?entry={entry}&run=run&name=train/loss")
    get(server, f"api/stages?entry={entry}&run=run")

    assert snapshot_tree(tmp_path) == before


class _FakeQueue:
    """Record forwarded writes and return one job with a run link target."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self.calls: list[tuple[str, object]] = []
        self.failure: Exception | None = None

    def list_jobs(self) -> list[dict[str, object]]:
        return [{"job_id": "a" * 12, "status": "pending", "run_dir": self.run_dir}]

    def resources(self) -> dict[str, object]:
        return {"gpus": [], "ram": None}

    def run_dirs(self) -> list[str]:
        return [self.run_dir, f"{self.run_dir}-deleted"]

    def cancel(self, job_id: str) -> dict[str, object]:
        if self.failure:
            raise self.failure
        self.calls.append(("cancel", job_id))
        return {"job_id": job_id, "status": "cancelled"}

    def set_priority(self, job_id: str, priority: object) -> dict[str, object]:
        self.calls.append(("priority", priority))
        return {"job_id": job_id, "priority": priority}


@pytest.fixture
def queue_server(tmp_path: Path) -> Iterator[tuple[object, _FakeQueue]]:
    create_run(tmp_path, "run-a")
    fake = _FakeQueue(str(tmp_path / "run-a"))
    instance = create_queue_monitor_server(fake, port=0, interval_seconds=0.05)  # type: ignore[arg-type]
    instance.service.sweep()
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance, fake
    instance.shutdown()
    instance.shutdown_monitor()
    thread.join(timeout=5.0)


def _post(url: str, body: bytes = b"{}", **headers: str) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", headers.pop("content_type", "application/json"))
    for name, value in headers.items():
        request.add_header(name.replace("_", "-"), value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_queue_route_links_jobs_to_discovered_runs(queue_server) -> None:
    server, _ = queue_server
    with urllib.request.urlopen(f"{server.url}api/queue", timeout=10) as response:
        document = json.load(response)
    assert document["available"] is True
    assert document["jobs"][0]["link"]["run"] == "run-a"


def test_queue_route_reports_a_down_service(tmp_path: Path) -> None:
    from mammoth.queue import QueueClient

    service = MonitorService((), queue_client=QueueClient(tmp_path / "none.sock"))
    document = service.queue()
    assert document["available"] is False
    assert "not answering" in document["error"]


def test_queue_writes_forward_cancel_and_priority_only(queue_server) -> None:
    server, fake = queue_server
    job_id = "a" * 12
    assert _post(f"{server.url}api/queue/{job_id}/cancel")[0] == 200
    status, _ = _post(f"{server.url}api/queue/{job_id}/priority", b'{"priority": 3}')
    assert status == 200
    assert fake.calls == [("cancel", job_id), ("priority", 3)]
    assert _post(f"{server.url}api/queue/{job_id}/submit")[0] == 404
    assert _post(f"{server.url}api/jobs")[0] == 404
    assert _post(f"{server.url}api/queue")[0] == 404


def test_queue_writes_map_service_errors(queue_server) -> None:
    from mammoth.queue import JobStateError

    server, fake = queue_server
    fake.failure = JobStateError("already ended")
    status, document = _post(f"{server.url}api/queue/{'a' * 12}/cancel")
    assert (status, document["error"]) == (409, "already ended")


def test_queue_writes_refuse_foreign_origin_and_form_posts(queue_server) -> None:
    server, fake = queue_server
    url = f"{server.url}api/queue/{'a' * 12}/cancel"
    assert _post(url, Origin="http://evil.example")[0] == 403
    assert _post(url, content_type="text/plain")[0] == 403
    assert _post(url, content_type="application/x-www-form-urlencoded")[0] == 403
    assert _post(url, Host="evil.example")[0] == 403
    assert not fake.calls
    assert _post(url, Origin=server.url.rstrip("/"))[0] == 200


@pytest.mark.parametrize("path", ["cancel", "nowhere"])
def test_queue_write_leaves_a_reused_connection_usable(queue_server, path: str) -> None:
    server, _ = queue_server
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    try:
        connection.request(
            "POST",
            f"/api/queue/{'a' * 12}/{path}",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        connection.getresponse().read()
        connection.request("GET", "/api/queue")
        assert connection.getresponse().status == 200
    finally:
        connection.close()


@pytest.mark.parametrize("length", ["nope", "-1", "99999999"])
def test_queue_write_survives_a_bad_content_length(queue_server, length: str) -> None:
    server, fake = queue_server
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    try:
        connection.putrequest("POST", f"/api/queue/{'a' * 12}/priority")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", length)
        connection.endheaders()
        assert connection.getresponse().status == 400
    finally:
        connection.close()
    assert not fake.calls


def test_queue_writes_are_disabled_on_a_non_loopback_bind(tmp_path: Path) -> None:
    instance = create_queue_monitor_server(
        _FakeQueue(str(tmp_path / "run")),  # type: ignore[arg-type]
        host="0.0.0.0",
        port=0,
        interval_seconds=0.05,
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        port = instance.server_address[1]
        status, document = _post(f"http://127.0.0.1:{port}/api/queue/{'a' * 12}/cancel")
        assert status == 403
        assert "non-loopback" in document["error"]
    finally:
        instance.shutdown()
        instance.shutdown_monitor()
        thread.join(timeout=5.0)


def test_web_command_has_no_queue(server: object) -> None:
    assert json.loads(get(server, "api/mode")[1]) == {"queue": False, "writable": False}
    assert get(server, "api/queue")[0] == 404
    assert _post(f"{server.url}api/queue/{'a' * 12}/cancel")[0] == 404


def test_queue_entries_keep_started_runs_and_skip_deleted_ones(tmp_path: Path) -> None:
    create_run(tmp_path / "project", "started")
    (tmp_path / "project" / "not-started").mkdir()
    entries, names = queue_entries(
        [
            str(tmp_path / "project" / "started"),
            str(tmp_path / "project" / "not-started"),
            str(tmp_path / "gone" / "run"),
        ]
    )
    assert list(entries.values()) == [tmp_path / "project"]
    assert list(names.values()) == [{"started"}]


def test_serve_lists_only_queued_runs_from_any_project(tmp_path: Path) -> None:
    create_run(tmp_path / "one", "queued")
    create_run(tmp_path / "one", "sibling")
    create_run(tmp_path / "two", "other")

    class Client(_FakeQueue):
        def run_dirs(self) -> list[str]:
            return [str(tmp_path / "one" / "queued"), str(tmp_path / "two" / "other")]

    service = MonitorService((), queue_client=Client(""), interval_seconds=0.05)  # type: ignore[arg-type]
    service.sweep()
    runs = {run["run_name"] for entry in service.fleet()["entries"] for run in entry["loose_runs"]}
    assert runs == {"queued", "other"}


def test_serve_keeps_runs_readable_when_the_service_stops_answering(tmp_path: Path) -> None:
    from mammoth.queue import ServiceUnavailableError

    create_run(tmp_path, "queued")

    class Client(_FakeQueue):
        down = False

        def run_dirs(self) -> list[str]:
            if self.down:
                raise ServiceUnavailableError("down")
            return [str(tmp_path / "queued")]

    client = Client("")
    service = MonitorService((), queue_client=client, interval_seconds=0.05)  # type: ignore[arg-type]
    service.sweep()
    client.down = True
    service._discovered_at = None
    service.sweep()
    assert service.fleet()["entries"][0]["loose_runs"][0]["run_name"] == "queued"


def test_failed_discovery_retries_on_the_next_sweep(tmp_path: Path) -> None:
    from mammoth.queue import ServiceUnavailableError

    create_run(tmp_path, "queued")

    class Client(_FakeQueue):
        down = True

        def run_dirs(self) -> list[str]:
            if self.down:
                raise ServiceUnavailableError("starting")
            return [str(tmp_path / "queued")]

    client = Client("")
    service = MonitorService((), queue_client=client, interval_seconds=0.05)  # type: ignore[arg-type]
    service.sweep()
    assert service.fleet()["entries"] == []
    client.down = False
    service.sweep()
    assert service.fleet()["entries"][0]["loose_runs"][0]["run_name"] == "queued"


def test_deleted_run_stops_serving_its_cached_view(tmp_path: Path) -> None:
    create_run(tmp_path, "run-a")
    service = MonitorService([tmp_path], interval_seconds=0.05)
    service.sweep()
    entry_id = next(iter(service._entries))
    assert service.run(entry_id, "run-a", None) is not None
    shutil.rmtree(tmp_path / "run-a")
    assert service.run(entry_id, "run-a", None) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "0", ""])
def test_allow_queue_writes_refuses_any_wildcard_spelling(host: str) -> None:
    with pytest.raises(ValueError, match="specific"):
        create_queue_monitor_server(_FakeQueue(""), host=host, port=0, allow_queue_writes=True)  # type: ignore[arg-type]


def test_allow_queue_writes_checks_host() -> None:
    with pytest.raises(ValueError, match="specific"):
        create_queue_monitor_server(_FakeQueue(""), host="0.0.0.0", port=0, allow_queue_writes=True)  # type: ignore[arg-type]
    fake = _FakeQueue("")
    instance = create_queue_monitor_server(
        fake,  # type: ignore[arg-type]
        host="127.0.0.2",
        port=0,
        interval_seconds=0.05,
        allow_queue_writes=True,
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.2:{instance.server_address[1]}/api/queue/{'a' * 12}/cancel"
        assert _post(url, Host="rebind.example")[0] == 403
        assert _post(url, Host="rebind.example", Origin="http://rebind.example")[0] == 403
        assert _post(url)[0] == 200
        assert json.loads(get(instance, "api/mode")[1])["writable"] is True
        assert fake.calls == [("cancel", "a" * 12)]
    finally:
        instance.shutdown()
        instance.shutdown_monitor()
        thread.join(timeout=5.0)
