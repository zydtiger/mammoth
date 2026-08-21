"""Passive fleet and group folding above one logical run.

A fleet is every group and loose run beneath one entry. This module folds
that hierarchy from the same class of passive sources :mod:`mammoth.monitor.model`
already uses for one run: immutable group manifests
(:mod:`mammoth.core.groups`), the append-only group event stream (tailed
incrementally with :class:`~mammoth.core.groups.GroupEventTailReader`), and a
cheap :class:`~mammoth.monitor.model.ExecutionMonitor` tail of each member
run's newest execution. It never constructs a full
:class:`~mammoth.monitor.model.RunMonitor` (which reconstructs a run's entire
execution lineage); that stays reserved for the one run an operator drills
into.

An entry with no ``.mammoth/groups`` subtree, or a run outside any workflow,
remains fully valid: this module treats their absence as an empty fleet
section rather than an error, matching the monitor's passivity contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

from mammoth.core.groups import (
    GROUP_MANIFEST_FILENAME,
    GroupEvent,
    GroupEventName,
    GroupEventReadError,
    GroupEventTailReader,
    GroupManifest,
    GroupOrder,
    groups_dir_for,
    load_group_manifest,
)
from mammoth.core.identity import validate_group_id, validate_run_name
from mammoth.core.layout import GroupLayout, RunLayout
from mammoth.monitor.model import (
    ExecutionMonitor,
    MonitorSnapshot,
    ScopeStatus,
    select_execution,
)

MemberStatus = Literal[
    "pending",
    "running",
    "stale",
    "completed",
    "failed",
    "interrupted",
    "blocked",
]

_RUN_EVENT_STATUS: dict[GroupEventName, MemberStatus] = {
    "run_completed": "completed",
    "run_failed": "failed",
    "run_blocked": "blocked",
    "run_interrupted": "interrupted",
}
_STEP_EVENT_STATUS: dict[GroupEventName, ScopeStatus] = {
    "step_started": "running",
    "step_completed": "completed",
    "step_failed": "failed",
    "step_interrupted": "interrupted",
}
_TERMINAL_GROUP_EVENT_STATUS: dict[GroupEventName, str] = {
    "group_completed": "group_completed",
    "group_failed": "group_failed",
    "group_interrupted": "group_interrupted",
}
_FAILED_MEMBER_STATUSES = frozenset({"failed", "interrupted", "blocked"})
_ACTIVE_MEMBER_STATUSES = frozenset({"running", "stale"})
_PENDING_MEMBER_STATUSES = frozenset({"pending", "running", "stale"})


@dataclass(frozen=True, slots=True)
class MemberProgress:
    """One member run's current-task progress, folded from its execution tail."""

    completed: int
    total: int | None
    throughput: float | None
    eta_seconds: float | None


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    """One member's declared step name and its folded lifecycle status."""

    name: str
    status: ScopeStatus


@dataclass(frozen=True, slots=True)
class GroupMemberSnapshot:
    """One group member row: schedule position, step statuses, and run tail."""

    run_name: str
    steps: tuple[StepSnapshot, ...]
    status: MemberStatus
    active_step: str | None
    progress: MemberProgress | None
    updated_at: datetime | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GroupSnapshot:
    """Folded state for one manifest-backed group or one ``--match`` cohort.

    ``manifest`` and ``order`` are ``None`` only for an ad-hoc ``--match``
    cohort, which has no recorded schedule or group event stream; its members
    carry no step schedule and its status is derived from run tails alone.
    """

    group_id: str
    manifest: GroupManifest | None
    pattern: str | None
    order: GroupOrder | None
    members: tuple[GroupMemberSnapshot, ...]
    terminal_status: str | None
    created_at: datetime | None
    updated_at: datetime | None
    warnings: tuple[str, ...] = ()

    @property
    def total_count(self) -> int:
        """Return the number of declared members."""
        return len(self.members)

    @property
    def completed_count(self) -> int:
        """Return the number of members with a completed run status."""
        return sum(1 for member in self.members if member.status == "completed")

    @property
    def failed_count(self) -> int:
        """Return members that failed, were interrupted, or never activated.

        A ``blocked`` member never produced artifacts because an earlier
        group failure stopped dispatch before it activated (see
        ``docs/ARCHITECTURE.md``); it is counted here rather than in a fourth
        bucket because it is causally a failure outcome for the group.
        """
        return sum(1 for member in self.members if member.status in _FAILED_MEMBER_STATUSES)

    @property
    def active_member(self) -> str | None:
        """Return the most recently active running or stale member, if any."""
        active = [member for member in self.members if member.status in _ACTIVE_MEMBER_STATUSES]
        if not active:
            return None
        return max(
            active,
            key=lambda member: member.updated_at or datetime.min.replace(tzinfo=UTC),
        ).run_name

    @property
    def aggregate_eta_seconds(self) -> float | None:
        """Return one honest combined ETA, or ``None`` when it cannot be trusted.

        Only derived when every not-yet-terminal member currently reports
        both a total and a positive throughput; a single missing rate makes a
        synthetic aggregate dishonest, so the caller should fall back to
        showing each member's own progress instead of guessing.
        """
        pending = [member for member in self.members if member.status in _PENDING_MEMBER_STATUSES]
        if not pending:
            return None
        remaining = 0.0
        rate = 0.0
        for member in pending:
            progress = member.progress
            if (
                progress is None
                or progress.total is None
                or progress.throughput is None
                or progress.throughput <= 0
            ):
                return None
            remaining += max(0, progress.total - progress.completed)
            rate += progress.throughput
        if rate <= 0:
            return None
        return remaining / rate


@dataclass(frozen=True, slots=True)
class LooseRunSnapshot:
    """Folded state for one run not claimed by any group."""

    run_name: str
    status: MemberStatus
    progress: MemberProgress | None
    updated_at: datetime | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FleetSnapshot:
    """Complete entry-level roll-up: every group plus every loose run."""

    entry: Path
    groups: tuple[GroupSnapshot, ...]
    loose_runs: tuple[LooseRunSnapshot, ...]
    match_pattern: str | None
    warnings: tuple[str, ...] = ()


def discover_group_ids(entry: Path) -> list[str]:
    """Return every published group ID beneath one entry, oldest name first.

    A group directory missing its manifest (a publish that never completed)
    is skipped rather than reported as an error; :class:`FleetMonitor` treats
    an unreadable manifest the same way once it tries to load it.
    """
    groups_root = groups_dir_for(Path(entry))
    if not groups_root.is_dir():
        return []
    ids: list[str] = []
    for child in sorted(groups_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or not (child / GROUP_MANIFEST_FILENAME).is_file():
            continue
        try:
            ids.append(validate_group_id(child.name))
        except ValueError:
            continue
    return ids


def discover_run_names(entry: Path) -> list[str]:
    """Return every syntactically valid run directory name beneath one entry."""
    entry_path = Path(entry)
    if not entry_path.is_dir():
        return []
    names: list[str] = []
    for child in sorted(entry_path.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name == ".mammoth":
            continue
        try:
            names.append(validate_run_name(child.name))
        except ValueError:
            continue
    return names


class FleetMonitor:
    """Incrementally fold every group and loose run under one entry.

    Group manifests are immutable and cached after their first successful
    load. Group event streams are tailed incrementally, exactly like
    :class:`~mammoth.monitor.model.ExecutionMonitor` tails execution streams:
    a stream that fails once is never retried, and the warning is recorded
    once rather than on every subsequent poll.
    """

    def __init__(self, entry: Path, *, match: str | None = None) -> None:
        self.entry = Path(entry)
        self.match = match
        self._manifests: dict[str, GroupManifest] = {}
        self._invalid_manifests: set[str] = set()
        self._group_readers: dict[str, GroupEventTailReader] = {}
        self._group_events: dict[str, list[GroupEvent]] = {}
        self._failed_group_streams: set[str] = set()
        self._monitors: dict[str, ExecutionMonitor] = {}
        self._warnings: list[str] = []

    def poll(
        self,
        *,
        now: datetime | None = None,
        stale_after_seconds: float = 90.0,
    ) -> FleetSnapshot:
        """Discover groups and loose runs, tail each incrementally, and fold state."""
        observed_at = now or datetime.now(UTC)
        groups: list[GroupSnapshot] = []
        claimed: set[str] = set()
        for group_id in discover_group_ids(self.entry):
            manifest = self._load_manifest(group_id)
            if manifest is None:
                continue
            claimed.update(member.run_name for member in manifest.members)
            layout = GroupLayout(self.entry, group_id)
            events = self._poll_group_events(group_id, layout.events_path)
            groups.append(
                _fold_group(
                    self.entry,
                    manifest,
                    events,
                    monitors=self._monitors,
                    now=observed_at,
                    stale_after_seconds=stale_after_seconds,
                )
            )

        loose_names = [name for name in discover_run_names(self.entry) if name not in claimed]
        loose_runs: tuple[LooseRunSnapshot, ...] = ()
        if self.match is not None:
            matched = [name for name in loose_names if fnmatch(name, self.match)]
            if matched:
                groups.append(
                    _fold_adhoc_group(
                        self.entry,
                        self.match,
                        matched,
                        monitors=self._monitors,
                        now=observed_at,
                        stale_after_seconds=stale_after_seconds,
                    )
                )
        else:
            loose_runs = tuple(
                _fold_loose_run(
                    self.entry,
                    name,
                    monitors=self._monitors,
                    now=observed_at,
                    stale_after_seconds=stale_after_seconds,
                )
                for name in loose_names
            )

        return FleetSnapshot(
            entry=self.entry,
            groups=tuple(groups),
            loose_runs=loose_runs,
            match_pattern=self.match,
            warnings=tuple(self._warnings),
        )

    def _load_manifest(self, group_id: str) -> GroupManifest | None:
        if group_id in self._invalid_manifests:
            return None
        manifest = self._manifests.get(group_id)
        if manifest is not None:
            return manifest
        layout = GroupLayout(self.entry, group_id)
        try:
            manifest = load_group_manifest(layout.manifest_path)
        except (OSError, ValueError) as error:
            self._invalid_manifests.add(group_id)
            self._warnings.append(f"Ignored group {group_id!r}: {error}")
            return None
        self._manifests[group_id] = manifest
        return manifest

    def _poll_group_events(self, group_id: str, events_path: Path) -> list[GroupEvent]:
        accumulated = self._group_events.setdefault(group_id, [])
        if group_id in self._failed_group_streams:
            return accumulated
        reader = self._group_readers.setdefault(group_id, GroupEventTailReader(events_path))
        try:
            accumulated.extend(reader.poll())
        except GroupEventReadError as error:
            accumulated.extend(error.valid_events)
            self._warnings.append(str(error))
            self._failed_group_streams.add(group_id)
        except OSError as error:
            self._warnings.append(str(error))
            self._failed_group_streams.add(group_id)
        return accumulated


def _fold_group(
    entry: Path,
    manifest: GroupManifest,
    events: list[GroupEvent],
    *,
    monitors: dict[str, ExecutionMonitor],
    now: datetime,
    stale_after_seconds: float,
) -> GroupSnapshot:
    """Fold one manifest plus its group event stream into a full group row set."""
    terminal_status: str | None = None
    started_runs: set[str] = set()
    latest_run_event: dict[str, GroupEvent] = {}
    latest_step_event: dict[tuple[str, str], GroupEvent] = {}
    for event in events:
        if event.event in _TERMINAL_GROUP_EVENT_STATUS:
            terminal_status = _TERMINAL_GROUP_EVENT_STATUS[event.event]
        elif event.event == "run_started" and event.run_name is not None:
            started_runs.add(event.run_name)
        elif event.run_name is not None and event.step_name is not None:
            latest_step_event[(event.run_name, event.step_name)] = event
        elif event.run_name is not None and event.event in _RUN_EVENT_STATUS:
            latest_run_event[event.run_name] = event

    members: list[GroupMemberSnapshot] = []
    warnings: list[str] = []
    newest_heartbeat: datetime | None = None
    for member in manifest.members:
        steps = tuple(
            StepSnapshot(
                name=step_name,
                status=_step_status(latest_step_event, member.run_name, step_name),
            )
            for step_name in member.steps
        )
        tail = _poll_member_tail(entry, member.run_name, monitors)
        run_event = latest_run_event.get(member.run_name)
        if run_event is not None:
            status: MemberStatus = _RUN_EVENT_STATUS[run_event.event]
        elif tail is not None:
            status = _tail_status(tail, now, stale_after_seconds)
        elif member.run_name in started_runs:
            status = "running"
        else:
            status = "pending"
        active_step = next((step.name for step in steps if step.status == "running"), None)
        updated_at = tail.updated_at if tail is not None else None
        if updated_at is not None and (newest_heartbeat is None or updated_at > newest_heartbeat):
            newest_heartbeat = updated_at
        member_warnings = tuple(tail.warnings) if tail is not None else ()
        warnings.extend(member_warnings)
        members.append(
            GroupMemberSnapshot(
                run_name=member.run_name,
                steps=steps,
                status=status,
                active_step=active_step,
                progress=_member_progress(tail),
                updated_at=updated_at,
                warnings=member_warnings,
            )
        )

    return GroupSnapshot(
        group_id=manifest.group_id,
        manifest=manifest,
        pattern=None,
        order=manifest.order,
        members=tuple(members),
        terminal_status=terminal_status,
        created_at=_parse_manifest_time(manifest.created_at),
        updated_at=newest_heartbeat or _parse_manifest_time(manifest.created_at),
        warnings=tuple(warnings),
    )


def _fold_adhoc_group(
    entry: Path,
    pattern: str,
    run_names: list[str],
    *,
    monitors: dict[str, ExecutionMonitor],
    now: datetime,
    stale_after_seconds: float,
) -> GroupSnapshot:
    """Fold ``--match``-selected loose runs into one synthetic, manifest-free group."""
    members: list[GroupMemberSnapshot] = []
    warnings: list[str] = []
    newest_heartbeat: datetime | None = None
    for run_name in run_names:
        tail = _poll_member_tail(entry, run_name, monitors)
        status: MemberStatus = (
            "pending" if tail is None else _tail_status(tail, now, stale_after_seconds)
        )
        updated_at = tail.updated_at if tail is not None else None
        if updated_at is not None and (newest_heartbeat is None or updated_at > newest_heartbeat):
            newest_heartbeat = updated_at
        member_warnings = tuple(tail.warnings) if tail is not None else ()
        warnings.extend(member_warnings)
        members.append(
            GroupMemberSnapshot(
                run_name=run_name,
                steps=(),
                status=status,
                active_step=None,
                progress=_member_progress(tail),
                updated_at=updated_at,
                warnings=member_warnings,
            )
        )
    return GroupSnapshot(
        group_id=f"match:{pattern}",
        manifest=None,
        pattern=pattern,
        order=None,
        members=tuple(members),
        terminal_status=None,
        created_at=None,
        updated_at=newest_heartbeat,
        warnings=tuple(warnings),
    )


def _fold_loose_run(
    entry: Path,
    run_name: str,
    *,
    monitors: dict[str, ExecutionMonitor],
    now: datetime,
    stale_after_seconds: float,
) -> LooseRunSnapshot:
    """Fold one run not claimed by any group from a cheap execution tail alone."""
    tail = _poll_member_tail(entry, run_name, monitors)
    if tail is None:
        return LooseRunSnapshot(
            run_name=run_name,
            status="pending",
            progress=None,
            updated_at=None,
            warnings=(),
        )
    return LooseRunSnapshot(
        run_name=run_name,
        status=_tail_status(tail, now, stale_after_seconds),
        progress=_member_progress(tail),
        updated_at=tail.updated_at,
        warnings=tuple(tail.warnings),
    )


def _poll_member_tail(
    entry: Path,
    run_name: str,
    monitors: dict[str, ExecutionMonitor],
) -> MonitorSnapshot | None:
    """Return a cheap incremental tail of one run's newest execution, if any.

    Reuses the cached :class:`~mammoth.monitor.model.ExecutionMonitor` for
    that run when its newest execution is unchanged, and replaces it only
    when a new attempt has appeared; it never constructs a full
    :class:`~mammoth.monitor.model.RunMonitor` across the whole lineage.
    """
    layout = RunLayout(entry, run_name)
    try:
        newest = select_execution(layout)
    except (FileNotFoundError, OSError):
        monitors.pop(run_name, None)
        return None
    monitor = monitors.get(run_name)
    if monitor is None or monitor.context.metadata.execution_id != newest.metadata.execution_id:
        monitor = ExecutionMonitor(layout, newest.metadata.execution_id)
        monitors[run_name] = monitor
    return monitor.poll()


def _member_progress(tail: MonitorSnapshot | None) -> MemberProgress | None:
    """Project one member's current task into unitless progress fields."""
    if tail is None:
        return None
    task = tail.current_task
    if task is None:
        return None
    return MemberProgress(
        completed=task.completed,
        total=task.total,
        throughput=task.throughput,
        eta_seconds=task.eta_seconds,
    )


def _tail_status(
    tail: MonitorSnapshot,
    now: datetime,
    stale_after_seconds: float,
) -> MemberStatus:
    """Map one run tail's folded status to a member status, detecting staleness.

    Mirrors :mod:`mammoth.monitor.dashboard`'s single-run staleness check: a
    nominally running attempt whose only process producers have gone quiet
    past the observation horizon reports ``stale`` instead of ``running``.
    """
    if tail.status == "running" and any(
        producer.effective_status(now, stale_after_seconds) == "stale"
        for key, producer in tail.producers.items()
        if key.source == "process"
    ):
        return "stale"
    return tail.status


def _step_status(
    latest_step_event: dict[tuple[str, str], GroupEvent],
    run_name: str,
    step_name: str,
) -> ScopeStatus:
    event = latest_step_event.get((run_name, step_name))
    return _STEP_EVENT_STATUS[event.event] if event is not None else "pending"


def _parse_manifest_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
