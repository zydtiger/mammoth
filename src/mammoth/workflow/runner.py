"""Programmatic serial workflows with isolated execution attempts.

The public :class:`Workflow` model derives run layouts, execution phases, and
dispatch order from caller-owned Python values.  This module owns logical-run
leases, immutable execution metadata, lifecycle records, child supervision,
signal handling, and cleanup; :mod:`mammoth.workflow.launch` owns subprocess
groups and descendant reaping.
"""

from __future__ import annotations

import math
import os
import signal
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from mammoth.core import (
    ExecutionContext,
    LogicalRunLease,
    RunLayout,
    claim_logical_run_lease,
    create_execution_context,
    latest_execution_id,
    sanitize_command,
    sanitize_metadata_fields,
    sanitize_reference,
    validate_execution_id,
    validate_resume_checkpoint_sha256,
    validate_run_name,
)
from mammoth.core.events import ExecutionEventWriter
from mammoth.core.execution import (
    EXECUTION_ID_ENV,
    INVOCATION_KIND_ENV,
    PHASE_ENV,
    RUN_NAME_ENV,
)
from mammoth.logging import JsonlEventSink, RunObserver
from mammoth.workflow.launch import CommandPlan, ProcessResult, launch_process

StepOutcome = Literal["completed", "failed", "interrupted"]
RunOutcome = Literal["completed", "failed", "blocked", "interrupted"]
WorkflowOrder = Literal["run-major", "step-major"]

_MAMMOTH_ENVIRONMENT_PREFIX = "MAMMOTH_"
_FINALIZATION_STAGE_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class Step:
    """One named execution phase whose command is already the final argv."""

    name: str
    command: Sequence[str]
    cwd: Path | None = None
    timeout_seconds: float | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach mutable argv/environment inputs and validate launch policy."""
        _validate_phase_name(self.name)
        if isinstance(self.command, str) or not isinstance(self.command, Sequence):
            raise ValueError(f"Step {self.name!r} command must be a sequence of arguments")
        command = tuple(self.command)
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError(f"Step {self.name!r} command must contain non-empty strings")
        object.__setattr__(self, "command", command)
        if self.cwd is not None:
            try:
                object.__setattr__(self, "cwd", Path(self.cwd))
            except TypeError as error:
                raise ValueError(
                    f"Step {self.name!r} cwd must be path-like or None"
                ) from error
        timeout = self.timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(f"Step {self.name!r} timeout_seconds must be positive")
        if timeout is not None:
            object.__setattr__(self, "timeout_seconds", float(timeout))
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(_validate_environment(self.environment, scope=f"step {self.name!r}")),
        )


@dataclass(frozen=True, slots=True)
class Execution:
    """Caller-owned execution facts not derived from a :class:`Run` or workflow."""

    invocation_kind: str = "workflow"
    config_reference: str | Path = ""
    world_size: int = 1
    execution_mode: Literal["single", "distributed"] = "single"
    resume_checkpoint: str | Path | None = None
    resume_checkpoint_sha256: str | None = None
    parent_execution_id: str | None = None
    starting_epoch: int | None = None
    starting_global_step: int | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata now and retain a deeply immutable runtime copy."""
        if not isinstance(self.invocation_kind, str) or not self.invocation_kind:
            raise ValueError("Execution invocation_kind must be a non-empty string")
        if self.config_reference != "":
            sanitize_reference(self.config_reference)
        if (
            not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.world_size < 1
        ):
            raise ValueError("Execution world_size must be a positive integer")
        if self.execution_mode not in {"single", "distributed"}:
            raise ValueError("Execution mode must be 'single' or 'distributed'")
        if (self.execution_mode == "single") != (self.world_size == 1):
            raise ValueError("Execution mode and world_size disagree")
        _validate_resume_coordinate("starting_epoch", self.starting_epoch)
        _validate_resume_coordinate("starting_global_step", self.starting_global_step)
        _validate_resume_facts(self)
        if not isinstance(self.runtime, Mapping):
            raise ValueError("Execution runtime must be a mapping")
        object.__setattr__(
            self,
            "runtime",
            _freeze_runtime_mapping(sanitize_metadata_fields(self.runtime)),
        )


@dataclass(frozen=True, slots=True)
class Run:
    """One independently leased logical run and its declaration-ordered steps."""

    name: str
    steps: Sequence[Step]
    execution: Execution = field(default_factory=Execution)
    root: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    resolve_execution: Callable[[ExecutionResolutionContext], Execution] | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    before_first_step: Callable[[BeforeFirstStepContext], None] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Freeze run inputs while leaving cross-run validation to planning."""
        validate_run_name(self.name)
        if isinstance(self.steps, Step) or not isinstance(self.steps, Sequence):
            raise ValueError(f"Run {self.name!r} steps must be a sequence of Step values")
        steps = tuple(self.steps)
        if not steps or any(not isinstance(step, Step) for step in steps):
            raise ValueError(f"Run {self.name!r} must contain Step values")
        names = [step.name for step in steps]
        if len(names) != len(set(names)):
            raise ValueError(f"Run {self.name!r} step names must be unique")
        object.__setattr__(self, "steps", steps)
        if not isinstance(self.execution, Execution):
            raise ValueError(f"Run {self.name!r} execution must be an Execution")
        if self.root is not None:
            try:
                object.__setattr__(self, "root", Path(self.root))
            except TypeError as error:
                raise ValueError(f"Run {self.name!r} root must be path-like or None") from error
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(_validate_environment(self.environment, scope=f"run {self.name!r}")),
        )
        for name, hook in (
            ("resolve_execution", self.resolve_execution),
            ("before_first_step", self.before_first_step),
        ):
            if hook is not None and not callable(hook):
                raise ValueError(f"Run {self.name!r} {name} must be callable or None")

    def step(self, name: str) -> Step:
        """Return one exact step from this run."""
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"Run {self.name!r} has no step {name!r}")


@dataclass(frozen=True, slots=True)
class ExecutionResolutionContext:
    """Run-local facts available after layout preparation and lease acquisition."""

    run: Run
    layout: RunLayout
    previous_execution_id: str | None


@dataclass(frozen=True, slots=True)
class BeforeFirstStepContext:
    """Immutable attempt facts available immediately before the first phase."""

    run: Run
    layout: RunLayout
    spec: Execution
    execution: ExecutionContext


@dataclass(frozen=True, slots=True)
class StepResult:
    """One dispatched step's terminal process or launch outcome."""

    name: str
    outcome: StepOutcome
    process: ProcessResult | None = None
    reason: str | None = None
    signal: int | None = None


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """One step result retained in workflow dispatch order."""

    run_name: str
    step: StepResult


@dataclass(frozen=True, slots=True)
class RunResult:
    """One logical run attempt or one artifact-free blocked declaration."""

    run_name: str
    outcome: RunOutcome
    execution_id: str | None
    steps: tuple[StepResult, ...]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Structured terminal outcome for one :meth:`Workflow.run` invocation."""

    runs: tuple[RunResult, ...]
    dispatch: tuple[DispatchResult, ...]
    exit_code: int

    @property
    def successful(self) -> bool:
        """Return whether the entire workflow completed successfully."""
        return self.exit_code == 0

    def run(self, name: str) -> RunResult:
        """Return one result by stable logical run name."""
        for result in self.runs:
            if result.run_name == name:
                return result
        raise KeyError(f"Workflow result has no run {name!r}")

    def step(self, run_name: str, step_name: str) -> StepResult:
        """Return one dispatched result by run and step identity."""
        for result in self.dispatch:
            if result.run_name == run_name and result.step.name == step_name:
                return result.step
        raise KeyError(f"Workflow result has no dispatched step {run_name!r}/{step_name!r}")


@dataclass(frozen=True, slots=True)
class Workflow:
    """One validated programmatic workflow with two explicit serial orders."""

    runs: Sequence[Run]
    root: Path = Path("runs")
    order: WorkflowOrder = "run-major"
    step_order: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Detach top-level collections without touching the filesystem."""
        if isinstance(self.runs, Run) or not isinstance(self.runs, Sequence):
            raise ValueError("Workflow runs must be a sequence of Run values")
        runs = tuple(self.runs)
        if not runs or any(not isinstance(run, Run) for run in runs):
            raise ValueError("Workflow must contain Run values")
        object.__setattr__(self, "runs", runs)
        try:
            object.__setattr__(self, "root", Path(self.root))
        except TypeError as error:
            raise ValueError("Workflow root must be path-like") from error
        if self.order not in {"run-major", "step-major"}:
            raise ValueError("Workflow order must be 'run-major' or 'step-major'")
        if self.step_order is not None:
            if isinstance(self.step_order, str) or not isinstance(self.step_order, Sequence):
                raise ValueError("Workflow step_order must be a sequence of phase names")
            order = tuple(self.step_order)
            if any(not isinstance(name, str) or not name for name in order):
                raise ValueError("Workflow step_order must contain non-empty phase names")
            object.__setattr__(self, "step_order", order)

    def plan(self) -> tuple[CommandPlan, ...]:
        """Validate and return the complete dispatch plan without side effects."""
        return _plan_workflow(self)

    def run(
        self,
        *,
        base_environment: Mapping[str, str] | None = None,
    ) -> WorkflowResult:
        """Execute the validated plan with owned attempts, children, and cleanup."""
        return _run_workflow(self, base_environment=base_environment)


@dataclass(slots=True)
class _ActiveRun:
    """Executor-owned resources for one started logical run."""

    run: Run
    layout: RunLayout
    spec: Execution
    lease: LogicalRunLease
    context: ExecutionContext
    observer: RunObserver
    event_writer: ExecutionEventWriter
    before_first_step_done: bool = False
    failure_reason: str | None = None
    terminal_result: RunResult | None = None
    terminal_event_attempts: int = 0
    terminal_event_failed: bool = False
    observer_closed: bool = False
    observer_close_attempts: int = 0
    observer_close_failed: bool = False
    lease_closed: bool = False
    lease_close_attempts: int = 0
    lease_close_failed: bool = False


@dataclass(slots=True)
class _PendingStep:
    """Lifecycle state for a step interrupted between paired records."""

    active: _ActiveRun
    step: Step
    phase_started: bool = False
    task_started: bool = False
    result: StepResult | None = None
    registered: bool = False


@dataclass(frozen=True, slots=True)
class _Failure:
    """The first ordinary or interruption failure that stops dispatch."""

    reason: str
    exit_code: int
    signal: int | None = None
    precedence: int = 3
    workflow_interruption: bool = False


class _WorkflowInterrupted(BaseException):
    """Internal signal translation that preserves deterministic cleanup."""

    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number


@dataclass(slots=True)
class _InterruptionState:
    """Track signals deferred across owned resource transitions."""

    critical_depth: int = 0
    deferred_signal: int | None = None
    initial_guard_active: bool = False


_INTERRUPTION_STATE: ContextVar[_InterruptionState | None] = ContextVar(
    "mammoth_workflow_interruption_state",
    default=None,
)


def _plan_workflow(workflow: Workflow) -> tuple[CommandPlan, ...]:
    """Validate identities/order and derive layouts without filesystem access."""
    names = [run.name for run in workflow.runs]
    if len(names) != len(set(names)):
        raise ValueError("Workflow run names must be unique")
    dispatch = _dispatch_order(workflow)
    return tuple(
        CommandPlan(
            run_name=run.name,
            step_name=step.name,
            command=tuple(step.command),
            cwd=step.cwd,
            timeout_seconds=step.timeout_seconds,
            run_dir=_layout_for(workflow, run).run_dir,
        )
        for run, step in dispatch
    )


def _dispatch_order(workflow: Workflow) -> tuple[tuple[Run, Step], ...]:
    """Return the exact run-major or canonical-name step-major sequence."""
    if workflow.order == "run-major":
        if workflow.step_order is not None:
            raise ValueError("run-major workflows must omit step_order")
        return tuple((run, step) for run in workflow.runs for step in run.steps)

    if workflow.step_order is None or not workflow.step_order:
        raise ValueError("step-major workflows require a non-empty step_order")
    canonical = tuple(workflow.step_order)
    if len(canonical) != len(set(canonical)):
        raise ValueError("Workflow step_order must not contain duplicates")
    positions = {name: index for index, name in enumerate(canonical)}
    for run in workflow.runs:
        run_names = tuple(step.name for step in run.steps)
        missing = [name for name in run_names if name not in positions]
        if missing:
            raise ValueError(
                f"Run {run.name!r} has steps absent from step_order: {', '.join(missing)}"
            )
        run_positions = tuple(positions[name] for name in run_names)
        if run_positions != tuple(sorted(run_positions)):
            raise ValueError(
                f"Run {run.name!r} steps must be a subsequence of step_order"
            )
    by_phase = {run.name: {step.name: step for step in run.steps} for run in workflow.runs}
    return tuple(
        (run, by_phase[run.name][phase])
        for phase in canonical
        for run in workflow.runs
        if phase in by_phase[run.name]
    )


def _run_workflow(
    workflow: Workflow,
    *,
    base_environment: Mapping[str, str] | None,
) -> WorkflowResult:
    """Own serial dispatch and preserve the first failure through cleanup."""
    workflow.plan()
    environment = _base_environment(base_environment)
    invocation_command = tuple(sys.argv)
    dispatch = _dispatch_order(workflow)
    active: dict[str, _ActiveRun] = {}
    results_by_run: dict[str, list[StepResult]] = {run.name: [] for run in workflow.runs}
    dispatch_results: list[DispatchResult] = []
    first_failure: _Failure | None = None
    pending: _PendingStep | None = None
    transition_result: StepResult | None = None
    original_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    finalization_errors: list[tuple[str, BaseException]] = []
    run_results: tuple[RunResult, ...] = ()

    with _interruption_signals() as interruption_state:
        try:
            _release_initial_interrupt_guard(interruption_state)
            for run, step in dispatch:
                transition_result = None
                current = active.get(run.name)
                if current is None:
                    current = _start_run(
                        workflow,
                        run,
                        active,
                        invocation_command=invocation_command,
                    )
                hook_failure = _run_before_first_step(current)
                if hook_failure is not None:
                    first_failure = hook_failure
                    break
                pending = _PendingStep(current, step)
                result = _execute_step(current, step, environment, pending)
                with _blocked_interruption_signals():
                    _record_pending_result(pending, results_by_run, dispatch_results)
                    transition_result = result
                    pending = None
                if result.outcome != "completed":
                    first_failure = _failure_from_step(result)
                    break
        except _WorkflowInterrupted as error:
            first_failure = _failure_for_workflow_interruption(
                error.signal_number,
                pending=pending,
                transition_result=transition_result,
                results_by_run=results_by_run,
                dispatch_results=dispatch_results,
            )
            pending = None
        except KeyboardInterrupt:
            first_failure = _failure_for_workflow_interruption(
                signal.SIGINT,
                pending=pending,
                transition_result=transition_result,
                results_by_run=results_by_run,
                dispatch_results=dispatch_results,
            )
            pending = None
        except BaseException as error:
            original_error = error
        finally:
            while True:
                try:
                    run_results = _finalize_runs(
                        workflow,
                        active,
                        results_by_run,
                        failure=first_failure,
                        executor_failed=original_error is not None,
                        cleanup_errors=finalization_errors,
                    )
                except _WorkflowInterrupted as error:
                    if original_error is not None:
                        original_error.add_note(
                            "Workflow interrupted during resolver cleanup by signal "
                            f"{error.signal_number}"
                        )
                    else:
                        first_failure = _prefer_failure(
                            first_failure,
                            _signal_failure(error.signal_number),
                        )
                    continue
                except KeyboardInterrupt:
                    if original_error is not None:
                        original_error.add_note(
                            "Workflow interrupted during resolver cleanup by signal 2"
                        )
                    else:
                        first_failure = _prefer_failure(
                            first_failure,
                            _signal_failure(signal.SIGINT),
                        )
                    continue
                except BaseException as error:
                    cleanup_error = error
                break

    if original_error is not None:
        if cleanup_error is not None:
            original_error.add_note(
                "Workflow cleanup failure: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise original_error
    if cleanup_error is not None:
        raise cleanup_error
    return WorkflowResult(
        runs=run_results,
        dispatch=tuple(dispatch_results),
        exit_code=0 if first_failure is None else first_failure.exit_code,
    )


def _start_run(
    workflow: Workflow,
    run: Run,
    active: dict[str, _ActiveRun],
    *,
    invocation_command: tuple[str, ...],
) -> _ActiveRun:
    """Prepare layout/lease, resolve execution, and publish immutable metadata."""
    current: _ActiveRun | None = None
    lease: LogicalRunLease | None = None
    try:
        with _blocked_interruption_signals():
            layout = _layout_for(workflow, run).prepare()
            lease = claim_logical_run_lease(layout.run_dir)
            previous_execution_id = latest_execution_id(layout.run_dir)
        spec = run.execution
        if run.resolve_execution is not None:
            try:
                spec = run.resolve_execution(
                    ExecutionResolutionContext(
                        run=run,
                        layout=layout,
                        previous_execution_id=previous_execution_id,
                    )
                )
            except _WorkflowInterrupted:
                raise
            except KeyboardInterrupt as error:
                raise _WorkflowInterrupted(signal.SIGINT) from error
            if not isinstance(spec, Execution):
                raise ValueError(
                    f"Run {run.name!r} resolve_execution must return an Execution"
                )
        with _blocked_interruption_signals():
            context = create_execution_context(
                layout.run_dir,
                run_name=run.name,
                invocation_kind=spec.invocation_kind,
                intended_phases=tuple(step.name for step in run.steps),
                world_size=spec.world_size,
                execution_mode=spec.execution_mode,
                command=invocation_command,
                config_reference=spec.config_reference,
                previous_execution_id=previous_execution_id,
                resume_checkpoint=spec.resume_checkpoint,
                resume_checkpoint_sha256=spec.resume_checkpoint_sha256,
                parent_execution_id=spec.parent_execution_id,
                starting_epoch=spec.starting_epoch,
                starting_global_step=spec.starting_global_step,
                runtime=_thaw_runtime_mapping(spec.runtime),
            )
            event_writer = ExecutionEventWriter.for_runner(context)
            observer = RunObserver((JsonlEventSink(event_writer),))
            current = _ActiveRun(
                run,
                layout,
                spec,
                lease,
                context,
                observer,
                event_writer,
            )
            active[run.name] = current
            observer.emit("execution_started")
        return current
    except BaseException as original_error:
        if current is None and lease is not None:
            try:
                with _blocked_interruption_signals(deliver_deferred=False):
                    lease.close()
            except BaseException as cleanup_error:
                original_error.add_note(
                    "Workflow untransferred lease cleanup failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise


def _run_before_first_step(active: _ActiveRun) -> _Failure | None:
    """Call the run-local hook exactly once before any phase starts."""
    if active.before_first_step_done:
        return None
    active.before_first_step_done = True
    hook = active.run.before_first_step
    if hook is None:
        return None
    try:
        hook(
            BeforeFirstStepContext(
                run=active.run,
                layout=active.layout,
                spec=active.spec,
                execution=active.context,
            )
        )
    except _WorkflowInterrupted:
        raise
    except KeyboardInterrupt as error:
        raise _WorkflowInterrupted(signal.SIGINT) from error
    except BaseException as error:
        reason = f"before_first_step failed: {type(error).__name__}: {error}"
        active.failure_reason = reason
        return _Failure(reason=reason, exit_code=1)
    return None


def _execute_step(
    active: _ActiveRun,
    step: Step,
    base_environment: Mapping[str, str],
    pending: _PendingStep,
) -> StepResult:
    """Launch one final argv while emitting paired phase/task lifecycle records."""
    command = sanitize_command(step.command)
    with _blocked_interruption_signals():
        active.observer.emit("phase_started", phase=step.name)
        pending.phase_started = True
        active.observer.emit(
            "task_started",
            phase=step.name,
            task_id=step.name,
            command=command,
        )
        pending.task_started = True
    environment = _child_environment(active, step, base_environment)
    try:
        process = launch_process(
            tuple(step.command),
            cwd=step.cwd,
            environment=environment,
            timeout_seconds=step.timeout_seconds,
        )
    except _WorkflowInterrupted:
        raise
    except BaseException as error:
        reason = f"launch failed: {type(error).__name__}: {error}"
        result = StepResult(step.name, "failed", reason=reason)
    else:
        if process.interrupted and process.signal is not None:
            raise _WorkflowInterrupted(process.signal)
        result = _step_result(step, process)
    with _blocked_interruption_signals():
        _emit_step_terminal(active, step, result)
        pending.result = result
    return result


def _record_pending_result(
    pending: _PendingStep,
    results_by_run: Mapping[str, list[StepResult]],
    dispatch_results: list[DispatchResult],
) -> None:
    """Register one terminal step result exactly once in both public result views."""
    if pending.registered:
        return
    if pending.result is None:
        raise RuntimeError("Cannot register a step before its terminal result exists")
    results_by_run[pending.active.run.name].append(pending.result)
    dispatch_results.append(DispatchResult(pending.active.run.name, pending.result))
    pending.registered = True


def _failure_for_workflow_interruption(
    signal_number: int,
    *,
    pending: _PendingStep | None,
    transition_result: StepResult | None,
    results_by_run: Mapping[str, list[StepResult]],
    dispatch_results: list[DispatchResult],
) -> _Failure:
    """Preserve any recorded step result before applying workflow-signal precedence."""
    result = transition_result
    if pending is not None:
        if pending.result is None:
            result = _finish_pending_interruption(pending, signal_number)
        else:
            result = pending.result
        with _blocked_interruption_signals(deliver_deferred=False):
            _record_pending_result(pending, results_by_run, dispatch_results)
    step_failure = (
        _failure_from_step(result)
        if result is not None and result.outcome != "completed"
        else None
    )
    return _prefer_failure(step_failure, _signal_failure(signal_number))


def _signal_failure(signal_number: int) -> _Failure:
    """Return the structured failure for one workflow interruption signal."""
    return _Failure(
        reason=f"workflow interrupted by signal {signal_number}",
        exit_code=128 + signal_number,
        signal=signal_number,
        precedence=1,
        workflow_interruption=True,
    )


def _prefer_failure(*failures: _Failure | None) -> _Failure:
    """Choose timeout, signal, child-code, then generic failure stably."""
    present = tuple(failure for failure in failures if failure is not None)
    if not present:
        raise RuntimeError("At least one workflow failure is required")
    return min(
        present,
        key=lambda failure: (
            failure.precedence,
            not failure.workflow_interruption,
        ),
    )


def _step_result(step: Step, process: ProcessResult) -> StepResult:
    """Classify one process result without losing timeout or signal precedence."""
    if process.timed_out:
        return StepResult(
            step.name,
            "failed",
            process=process,
            reason=f"timed out after {step.timeout_seconds} seconds",
            signal=process.signal,
        )
    child_signal = process.signal
    if child_signal is None and process.return_code < 0:
        child_signal = -process.return_code
    if process.interrupted or child_signal is not None:
        return StepResult(
            step.name,
            "interrupted",
            process=process,
            reason=f"child interrupted by signal {child_signal or signal.SIGINT}",
            signal=child_signal or signal.SIGINT,
        )
    if process.return_code != 0:
        return StepResult(
            step.name,
            "failed",
            process=process,
            reason=f"child exited with code {process.return_code}",
        )
    return StepResult(step.name, "completed", process=process)


def _emit_step_terminal(active: _ActiveRun, step: Step, result: StepResult) -> None:
    """Close the task and phase scopes for one ordinary terminal outcome."""
    duration = result.process.duration_seconds if result.process is not None else None
    if result.outcome == "completed":
        active.observer.emit(
            "task_completed",
            phase=step.name,
            task_id=step.name,
            duration_seconds=duration,
            exit_code=0,
        )
        active.observer.emit(
            "phase_completed",
            phase=step.name,
            duration_seconds=duration,
            exit_code=0,
        )
        return
    fields: dict[str, Any] = {
        "phase": step.name,
        "duration_seconds": duration,
        "exit_code": _step_exit_code(result),
    }
    if result.reason is not None:
        fields["message"] = result.reason
    active.observer.emit("task_failed", task_id=step.name, **fields)
    active.observer.emit("phase_failed", **fields)


def _finish_pending_interruption(pending: _PendingStep, signal_number: int) -> StepResult:
    """Best-effort close phase/task scopes interrupted by the workflow signal."""
    result = StepResult(
        pending.step.name,
        "interrupted",
        reason=f"workflow interrupted by signal {signal_number}",
        signal=signal_number,
    )
    fields: dict[str, Any] = {
        "phase": pending.step.name,
        "message": result.reason,
        "exit_code": 128 + signal_number,
    }
    with _blocked_interruption_signals(deliver_deferred=False):
        if pending.task_started:
            pending.active.observer.emit(
                "task_failed",
                task_id=pending.step.name,
                **fields,
            )
        if pending.phase_started:
            pending.active.observer.emit("phase_failed", **fields)
    pending.result = result
    return result


def _failure_from_step(result: StepResult) -> _Failure:
    """Apply the required timeout, signal, child-code, then generic precedence."""
    if result.process is not None and result.process.timed_out:
        return _Failure(result.reason or "step timed out", exit_code=1, precedence=0)
    if result.signal is not None:
        return _Failure(
            result.reason or f"step interrupted by signal {result.signal}",
            exit_code=128 + result.signal,
            signal=result.signal,
            precedence=1,
        )
    if result.process is not None and result.process.return_code != 0:
        return _Failure(
            result.reason or "child failed",
            exit_code=result.process.return_code,
            precedence=2,
        )
    return _Failure(result.reason or "step failed", exit_code=1)


def _step_exit_code(result: StepResult) -> int:
    """Return the terminal lifecycle code for one step result."""
    return _failure_from_step(result).exit_code if result.outcome != "completed" else 0


def _finalize_runs(
    workflow: Workflow,
    active: Mapping[str, _ActiveRun],
    results_by_run: Mapping[str, list[StepResult]],
    *,
    failure: _Failure | None,
    executor_failed: bool,
    cleanup_errors: list[tuple[str, BaseException]],
) -> tuple[RunResult, ...]:
    """Emit execution terminals and release resources through resumable stages."""
    while True:
        finalized: dict[str, RunResult] = {}
        retry_pending = False
        for run in workflow.runs:
            current = active.get(run.name)
            if current is None:
                continue
            if current.terminal_result is None:
                step_results = tuple(results_by_run[run.name])
                outcome, reason = _run_outcome(
                    current,
                    step_results,
                    failure=failure,
                    executor_failed=executor_failed,
                )
                terminal_result = RunResult(
                    run_name=run.name,
                    outcome=outcome,
                    execution_id=current.context.metadata.execution_id,
                    steps=step_results,
                    reason=reason,
                )
                sequence_before = current.event_writer.sequence
                try:
                    _emit_execution_terminal(
                        current,
                        outcome=outcome,
                        reason=reason,
                        failure=failure,
                    )
                    current.terminal_result = terminal_result
                except _WorkflowInterrupted:
                    if current.event_writer.sequence > sequence_before:
                        current.terminal_result = terminal_result
                    raise
                except KeyboardInterrupt as error:
                    if current.event_writer.sequence > sequence_before:
                        current.terminal_result = terminal_result
                    raise _WorkflowInterrupted(signal.SIGINT) from error
                except BaseException as error:
                    _record_finalization_error(
                        cleanup_errors,
                        f"run {run.name!r} terminal event",
                        error,
                    )
                    if current.event_writer.sequence > sequence_before:
                        current.terminal_result = terminal_result
                    else:
                        current.terminal_event_attempts += 1
                        if (
                            current.terminal_event_attempts
                            < _FINALIZATION_STAGE_ATTEMPTS
                        ):
                            retry_pending = True
                        else:
                            current.terminal_event_failed = True
                            current.terminal_result = terminal_result
            if current.terminal_result is None:
                continue
            if not current.observer_closed and not current.observer_close_failed:
                stage_error: BaseException | None = None
                try:
                    with _blocked_interruption_signals():
                        try:
                            current.observer.close()
                        except BaseException as error:
                            stage_error = error
                            current.observer_close_attempts += 1
                            _record_finalization_error(
                                cleanup_errors,
                                f"run {run.name!r} observer",
                                error,
                            )
                        else:
                            current.observer_closed = True
                except _WorkflowInterrupted:
                    raise
                except KeyboardInterrupt as error:
                    raise _WorkflowInterrupted(signal.SIGINT) from error
                if stage_error is not None:
                    if current.observer_close_attempts < _FINALIZATION_STAGE_ATTEMPTS:
                        retry_pending = True
                    else:
                        current.observer_close_failed = True
            if not current.observer_closed and not current.observer_close_failed:
                continue
            if not current.lease_closed and not current.lease_close_failed:
                stage_error = None
                try:
                    with _blocked_interruption_signals():
                        try:
                            current.lease.close()
                        except BaseException as error:
                            stage_error = error
                            current.lease_close_attempts += 1
                            _record_finalization_error(
                                cleanup_errors,
                                f"run {run.name!r} lease",
                                error,
                            )
                        else:
                            current.lease_closed = True
                except _WorkflowInterrupted:
                    raise
                except KeyboardInterrupt as error:
                    raise _WorkflowInterrupted(signal.SIGINT) from error
                if stage_error is not None:
                    if current.lease_close_attempts < _FINALIZATION_STAGE_ATTEMPTS:
                        retry_pending = True
                    else:
                        current.lease_close_failed = True
            if not current.lease_closed and not current.lease_close_failed:
                continue
            finalized[run.name] = current.terminal_result
        if not retry_pending:
            break
    if cleanup_errors:
        first_label, first_error = cleanup_errors[0]
        first_error.add_note(f"Workflow cleanup stage: {first_label}")
        for label, cleanup_failure in cleanup_errors[1:]:
            first_error.add_note(
                f"Later workflow cleanup failure ({label}): "
                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
            )
        raise first_error
    return tuple(
        finalized.get(
            run.name,
            RunResult(
                run_name=run.name,
                outcome="blocked",
                execution_id=None,
                steps=(),
                reason=None if failure is None else failure.reason,
            ),
        )
        for run in workflow.runs
    )


def _emit_execution_terminal(
    current: _ActiveRun,
    *,
    outcome: RunOutcome,
    reason: str | None,
    failure: _Failure | None,
) -> None:
    """Emit one run terminal; the caller reconciles durable writes on interruption."""
    if outcome == "completed":
        current.observer.emit("execution_completed", exit_code=0)
        return
    if outcome == "interrupted":
        interrupted_signal = failure.signal if failure is not None else signal.SIGINT
        current.observer.emit(
            "execution_interrupted",
            signal=interrupted_signal,
            message=reason,
        )
        return
    current.observer.emit(
        "execution_failed",
        exit_code=1 if failure is None else failure.exit_code,
        message=reason,
    )


def _record_finalization_error(
    errors: list[tuple[str, BaseException]],
    label: str,
    error: BaseException,
) -> None:
    """Retain one cleanup failure across signal-driven finalization retries."""
    if any(existing_label == label and existing is error for existing_label, existing in errors):
        return
    errors.append((label, error))


def _run_outcome(
    active: _ActiveRun,
    results: tuple[StepResult, ...],
    *,
    failure: _Failure | None,
    executor_failed: bool,
) -> tuple[RunOutcome, str | None]:
    """Classify completed, direct failure, cross-run blocking, and interruption."""
    if failure is not None and failure.workflow_interruption:
        return "interrupted", failure.reason
    if active.failure_reason is not None:
        return "failed", active.failure_reason
    if any(result.outcome == "failed" for result in results):
        failed = next(result for result in results if result.outcome == "failed")
        return "failed", failed.reason
    if any(result.outcome == "interrupted" for result in results):
        interrupted = next(result for result in results if result.outcome == "interrupted")
        return "interrupted", interrupted.reason
    if len(results) == len(active.run.steps):
        return "completed", None
    if executor_failed:
        return "failed", "workflow aborted by execution-resolution failure"
    return "blocked", None if failure is None else failure.reason


def _layout_for(workflow: Workflow, run: Run) -> RunLayout:
    """Derive one run layout from its override or the workflow default root."""
    return RunLayout(workflow.root if run.root is None else run.root, run.name)


def _child_environment(
    active: _ActiveRun,
    step: Step,
    base: Mapping[str, str],
) -> dict[str, str]:
    """Compose caller fields and overwrite them with exactly four canonical values."""
    environment = {
        name: value
        for name, value in base.items()
        if not name.startswith(_MAMMOTH_ENVIRONMENT_PREFIX)
    }
    environment.update(active.run.environment)
    environment.update(step.environment)
    environment.update(
        {
            EXECUTION_ID_ENV: active.context.metadata.execution_id,
            RUN_NAME_ENV: active.run.name,
            INVOCATION_KIND_ENV: active.spec.invocation_kind,
            PHASE_ENV: step.name,
        }
    )
    return environment


def _base_environment(environment: Mapping[str, str] | None) -> Mapping[str, str]:
    """Copy the inherited environment without retaining caller-owned state."""
    values = os.environ if environment is None else environment
    return MappingProxyType(
        _validate_environment(values, scope="base environment", allow_mammoth=True)
    )


def _validate_environment(
    environment: Mapping[str, str],
    *,
    scope: str,
    allow_mammoth: bool = False,
) -> dict[str, str]:
    """Validate one explicit environment and reserve Mammoth child variables."""
    if not isinstance(environment, Mapping):
        raise ValueError(f"{scope} environment must be a mapping")
    validated: dict[str, str] = {}
    for name, value in environment.items():
        if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
            raise ValueError(f"{scope} environment has an invalid variable name")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"{scope} environment values must be strings")
        if not allow_mammoth and name.startswith(_MAMMOTH_ENVIRONMENT_PREFIX):
            raise ValueError(f"{scope} cannot define Mammoth-owned environment {name!r}")
        validated[name] = value
    return validated


def _validate_phase_name(name: str) -> None:
    """Accept opaque non-empty project phase names."""
    if not isinstance(name, str) or not name:
        raise ValueError("Step names must be non-empty phase strings")


def _validate_resume_coordinate(name: str, value: int | None) -> None:
    """Reject negative, boolean, and non-integral resume coordinates."""
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"Execution {name} must be a non-negative integer or None")


def _validate_resume_facts(execution: Execution) -> None:
    """Require absent resume metadata to be wholly null and validate resumed facts."""
    if execution.resume_checkpoint is None:
        populated = {
            "resume_checkpoint_sha256": execution.resume_checkpoint_sha256,
            "parent_execution_id": execution.parent_execution_id,
            "starting_epoch": execution.starting_epoch,
            "starting_global_step": execution.starting_global_step,
        }
        unexpected = sorted(name for name, value in populated.items() if value is not None)
        if unexpected:
            raise ValueError(
                "Execution resume facts require resume_checkpoint: " + ", ".join(unexpected)
            )
        return
    sanitize_reference(execution.resume_checkpoint)
    if execution.resume_checkpoint_sha256 is None:
        raise ValueError("Execution resume_checkpoint requires resume_checkpoint_sha256")
    validate_resume_checkpoint_sha256(execution.resume_checkpoint_sha256)
    if execution.starting_epoch is None:
        raise ValueError("Execution resume_checkpoint requires starting_epoch")
    if execution.parent_execution_id is not None:
        validate_execution_id(execution.parent_execution_id)


def _freeze_runtime_mapping(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze sanitized JSON-compatible runtime metadata."""
    return MappingProxyType(
        {name: _freeze_runtime_value(value) for name, value in runtime.items()}
    )


def _freeze_runtime_value(value: Any) -> Any:
    """Freeze one nested mapping or sequence value."""
    if isinstance(value, Mapping):
        return _freeze_runtime_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | os.PathLike):
        return tuple(_freeze_runtime_value(item) for item in value)
    return value


def _thaw_runtime_mapping(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached JSON-compatible runtime mapping for metadata publication."""
    return {name: _thaw_runtime_value(value) for name, value in runtime.items()}


def _thaw_runtime_value(value: Any) -> Any:
    """Thaw one frozen runtime metadata value."""
    if isinstance(value, Mapping):
        return {name: _thaw_runtime_value(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_runtime_value(item) for item in value]
    return value


@contextmanager
def _interruption_signals() -> Iterator[_InterruptionState | None]:
    """Translate process termination signals into executor-managed cleanup."""
    if not hasattr(signal, "SIGTERM"):
        yield None
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    state = _InterruptionState(critical_depth=1, initial_guard_active=True)
    token = _INTERRUPTION_STATE.set(state)
    previous: dict[signal.Signals, Any] = {}
    installed: list[signal.Signals] = []

    def interrupt(signal_number: int, _frame: object) -> None:
        if state.critical_depth:
            if state.deferred_signal is None:
                state.deferred_signal = signal_number
            return
        raise _WorkflowInterrupted(signal_number)

    try:
        try:
            for signal_number in signals:
                previous[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, interrupt)
                installed.append(signal_number)
        except ValueError:
            state.critical_depth = 0
            state.initial_guard_active = False
            yield None
        else:
            yield state
    finally:
        try:
            for signal_number in reversed(installed):
                signal.signal(signal_number, previous[signal_number])
        finally:
            _INTERRUPTION_STATE.reset(token)


def _release_initial_interrupt_guard(state: _InterruptionState | None) -> None:
    """Deliver an install-window signal after entering the managed executor."""
    if state is None or not state.initial_guard_active:
        return
    state.initial_guard_active = False
    state.critical_depth -= 1
    if state.critical_depth == 0 and state.deferred_signal is not None:
        signal_number = state.deferred_signal
        state.deferred_signal = None
        raise _WorkflowInterrupted(signal_number)


@contextmanager
def _blocked_interruption_signals(
    *,
    deliver_deferred: bool = True,
) -> Iterator[None]:
    """Defer SIGINT/SIGTERM across registration and cleanup boundaries."""
    state = _INTERRUPTION_STATE.get()
    if state is not None:
        state.critical_depth += 1
    try:
        yield
    finally:
        if state is not None:
            state.critical_depth -= 1
            if (
                deliver_deferred
                and state.critical_depth == 0
                and state.deferred_signal is not None
            ):
                signal_number = state.deferred_signal
                state.deferred_signal = None
                raise _WorkflowInterrupted(signal_number)
