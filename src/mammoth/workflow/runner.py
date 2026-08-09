"""Programmatic and schema-v1 workflow planning with isolated run attempts.

``mammoth.workflow.config`` parses Mammoth's optional schema-v1 YAML format.
This module owns the project-neutral compiled-plan API used by both that
compatibility adapter and callers with their own configuration languages.  It
does not interpret command arguments, phase names, or caller hook behavior.
``mammoth.workflow.launch`` supplies the bounded child-process supervision
used for each dispatched step.
"""

from __future__ import annotations

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
    validate_run_name,
)
from mammoth.core.events import EventName, ExecutionEventWriter
from mammoth.core.execution import (
    EXECUTION_ID_ENV,
    INVOCATION_KIND_ENV,
    PHASE_ENV,
    RUN_NAME_ENV,
)
from mammoth.logging import JsonlEventSink, RunObserver
from mammoth.workflow.config import StepConfig, WorkflowConfig, validate_step_graph
from mammoth.workflow.launch import CommandPlan, ProcessResult, command_for_step, launch_process

StepOutcome = Literal["completed", "failed", "skipped", "interrupted"]
RunOutcome = Literal["completed", "failed", "blocked", "interrupted", "dry-run"]
WorkflowFailurePolicy = Literal["stop", "continue"]


@dataclass(frozen=True, slots=True)
class ExecutionInputs:
    """Caller-owned immutable inputs for one runner-created execution context.

    ``ProgrammaticRun`` supplies the run name and selected phase names.  This
    value supplies the remaining generic metadata accepted by
    :func:`mammoth.core.create_execution_context`; all metadata is validated
    before the executor prepares a layout or claims a logical-run lease.
    """

    invocation_kind: str
    command: Sequence[str]
    config_reference: str | Path = ""
    world_size: int = 1
    execution_mode: Literal["single", "distributed"] = "single"
    previous_execution_id: str | None = None
    resume_checkpoint: str | Path | None = None
    parent_execution_id: str | None = None
    starting_epoch: int | None = None
    starting_global_step: int | None = None
    runtime: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Freeze structured runtime metadata without persisting environment state."""
        if isinstance(self.command, str) or not isinstance(self.command, Sequence):
            raise ValueError("execution command must be a sequence of arguments, not a string")
        command = tuple(self.command)
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("execution command must contain non-empty strings")
        object.__setattr__(self, "command", command)
        if self.runtime is not None:
            object.__setattr__(
                self,
                "runtime",
                _freeze_runtime_metadata(sanitize_metadata_fields(self.runtime)),
            )


@dataclass(frozen=True, slots=True)
class ProgrammaticRun:
    """One independently leased run in a caller-compiled workflow plan.

    The caller supplies generic steps with :class:`StepConfig`; the YAML
    adapter uses the same step value.  ``layout`` remains caller-owned so one
    plan may direct each logical run to a different artifact entry.
    """

    name: str
    layout: RunLayout
    steps: tuple[StepConfig, ...]
    execution: ExecutionInputs
    environment: Mapping[str, str] = field(default_factory=dict)
    resolve_previous_execution: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate run-local mechanics before the global plan is dispatched."""
        validate_run_name(self.name)
        if not isinstance(self.layout, RunLayout):
            raise ValueError("Programmatic run layout must be a RunLayout")
        if self.layout.run_name != self.name:
            raise ValueError(
                f"Programmatic run {self.name!r} does not match layout run name "
                f"{self.layout.run_name!r}"
            )
        if isinstance(self.steps, StepConfig) or not isinstance(self.steps, Sequence):
            raise ValueError("Programmatic run steps must be a sequence of StepConfig values")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError(f"Programmatic run {self.name!r} must contain at least one step")
        if any(not isinstance(step, StepConfig) for step in steps):
            raise ValueError("Programmatic run steps must be StepConfig values")
        if not isinstance(self.execution, ExecutionInputs):
            raise ValueError("Programmatic run execution must be ExecutionInputs")
        if not isinstance(self.resolve_previous_execution, bool):
            raise ValueError("Programmatic run resolve_previous_execution must be a boolean")
        object.__setattr__(self, "steps", steps)
        validate_step_graph(steps)
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(_validate_environment(self.environment)),
        )

    def step(self, name: str) -> StepConfig:
        """Return one named compiled step for dispatch validation and execution."""
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"Programmatic run {self.name!r} has no step {name!r}")


@dataclass(frozen=True, slots=True)
class DispatchEntry:
    """One caller-selected global ``(run, step)`` dispatch position."""

    run_name: str
    step_name: str

    def __post_init__(self) -> None:
        """Reject malformed identities before global plan validation."""
        validate_run_name(self.run_name)
        validate_run_name(self.step_name)


@dataclass(frozen=True, slots=True)
class PreDispatchContext:
    """Read-only caller context supplied immediately before a child launch.

    Hooks receive no observer, lease, or child supervisor, preserving Mammoth's
    ownership of lifecycle records and cleanup.  Callers may use the context
    for their own immediate pre-launch work, such as resolving project state.
    """

    dispatch: DispatchEntry
    run: ProgrammaticRun
    step: StepConfig
    execution: ExecutionContext


PreDispatchHook = Callable[[PreDispatchContext], None]


@dataclass(frozen=True, slots=True)
class ProgrammaticWorkflow:
    """Fully caller-compiled, serial workflow representation.

    ``dispatch`` is intentionally global rather than derived from the run
    declarations.  It can therefore interleave otherwise independent logical
    runs while preserving each run's own layout, execution metadata, event
    stream, and producer lease.
    """

    runs: tuple[ProgrammaticRun, ...]
    dispatch: tuple[DispatchEntry, ...]
    failure_policy: WorkflowFailurePolicy = "stop"
    pre_dispatch: PreDispatchHook | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Reject invalid top-level fields before a side-effect-free preflight."""
        if isinstance(self.runs, ProgrammaticRun) or not isinstance(self.runs, Sequence):
            raise ValueError(
                "Programmatic workflow runs must be a sequence of ProgrammaticRun values"
            )
        if isinstance(self.dispatch, DispatchEntry) or not isinstance(self.dispatch, Sequence):
            raise ValueError(
                "Programmatic workflow dispatch must be a sequence of DispatchEntry values"
            )
        runs = tuple(self.runs)
        dispatch = tuple(self.dispatch)
        if not runs:
            raise ValueError("Programmatic workflow must contain at least one run")
        if any(not isinstance(run, ProgrammaticRun) for run in runs):
            raise ValueError("Programmatic workflow runs must be ProgrammaticRun values")
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "dispatch", dispatch)
        if self.failure_policy not in {"stop", "continue"}:
            raise ValueError("Programmatic workflow failure_policy must be 'stop' or 'continue'")
        if self.pre_dispatch is not None and not callable(self.pre_dispatch):
            raise ValueError("Programmatic workflow pre_dispatch must be callable or None")


@dataclass(frozen=True, slots=True)
class StepResult:
    """One selected step's terminal workflow outcome and any deferred signal."""

    name: str
    outcome: StepOutcome
    process: ProcessResult | None = None
    reason: str | None = None
    signal: int | None = None


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """One terminal step result retained in global caller-selected order."""

    run_name: str
    step: StepResult


@dataclass(frozen=True, slots=True)
class RunResult:
    """One logical run attempt and its step outcomes."""

    run_name: str
    outcome: RunOutcome
    execution_id: str | None
    steps: tuple[StepResult, ...]


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Complete result for the selected runs in one workflow invocation."""

    runs: tuple[RunResult, ...]
    plans: tuple[CommandPlan, ...] = ()
    dispatch: tuple[DispatchResult, ...] = ()

    @property
    def successful(self) -> bool:
        """Return whether every selected run completed or was only planned."""
        return all(run.outcome in {"completed", "dry-run"} for run in self.runs)

    def run(self, name: str) -> RunResult:
        """Return one run result by its stable logical run name."""
        for result in self.runs:
            if result.run_name == name:
                return result
        raise KeyError(f"Workflow result has no run {name!r}")

    def step(self, run_name: str, step_name: str) -> StepResult:
        """Return one globally dispatched result by run and step identity."""
        for result in self.dispatch:
            if result.run_name == run_name and result.step.name == step_name:
                return result.step
        raise KeyError(f"Workflow result has no dispatched step {run_name!r}/{step_name!r}")


@dataclass(slots=True)
class _ActiveRun:
    """Executor-owned resources retained while interleaved steps are active."""

    run: ProgrammaticRun
    lease: LogicalRunLease
    context: ExecutionContext
    observer: RunObserver


@dataclass(slots=True)
class _PendingDispatch:
    """One dispatch whose terminal step result may precede ledger publication."""

    dispatch: DispatchEntry
    result: StepResult | None = None


class _WorkflowInterrupted(BaseException):
    """Internal signal translation that preserves deterministic cleanup."""

    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number


@dataclass(slots=True)
class _InterruptionState:
    """Track signals deferred while the executor mutates owned resources."""

    critical_depth: int = 0
    deferred_signal: int | None = None
    initial_guard_active: bool = False


_INTERRUPTION_STATE: ContextVar[_InterruptionState | None] = ContextVar(
    "mammoth_workflow_interruption_state",
    default=None,
)


def validate_programmatic_workflow(workflow: ProgrammaticWorkflow) -> None:
    """Validate a complete compiled plan without creating workflow artifacts.

    Callers and :func:`run_programmatic_workflow` use this exact preflight so
    unknown references, duplicate or omitted dispatches, invalid dependency
    order, layouts, environments, and execution metadata fail before leases or
    execution directories exist.
    """
    _validate_programmatic_workflow(workflow, validate_layout_paths=True)


def _validate_programmatic_workflow(
    workflow: ProgrammaticWorkflow,
    *,
    validate_layout_paths: bool,
) -> None:
    """Validate a compiled plan, optionally skipping synthetic planning layouts."""
    if not isinstance(workflow, ProgrammaticWorkflow):
        raise ValueError("workflow must be a ProgrammaticWorkflow")
    run_names = [run.name for run in workflow.runs]
    if len(run_names) != len(set(run_names)):
        raise ValueError("Programmatic workflow run names must be unique")
    by_name = {run.name: run for run in workflow.runs}
    if validate_layout_paths:
        _validate_programmatic_layouts(workflow.runs)
    expected = {(run.name, step.name) for run in workflow.runs for step in run.steps}
    if len(expected) != sum(len(run.steps) for run in workflow.runs):
        raise ValueError("Programmatic workflow step names must be unique within each run")

    seen: set[tuple[str, str]] = set()
    for entry in workflow.dispatch:
        if not isinstance(entry, DispatchEntry):
            raise ValueError("Programmatic workflow dispatch entries must be DispatchEntry values")
        run = by_name.get(entry.run_name)
        if run is None:
            raise ValueError(
                f"Programmatic workflow dispatch references unknown run {entry.run_name!r}"
            )
        try:
            step = run.step(entry.step_name)
        except KeyError as error:
            raise ValueError(
                f"Programmatic workflow dispatch references unknown step "
                f"{entry.run_name!r}/{entry.step_name!r}"
            ) from error
        key = (entry.run_name, entry.step_name)
        if key in seen:
            raise ValueError(
                f"Programmatic workflow dispatch duplicates {entry.run_name!r}/{entry.step_name!r}"
            )
        missing_dependencies = [
            dependency for dependency in step.needs if (entry.run_name, dependency) not in seen
        ]
        if missing_dependencies:
            raise ValueError(
                f"Programmatic workflow dispatch places {entry.run_name!r}/{entry.step_name!r} "
                f"before dependencies: {', '.join(missing_dependencies)}"
            )
        seen.add(key)
    omitted = sorted(expected.difference(seen))
    if omitted:
        rendered = ", ".join(f"{run_name}/{step_name}" for run_name, step_name in omitted)
        raise ValueError(f"Programmatic workflow dispatch omits selected steps: {rendered}")
    _validate_execution_inputs_for_runs(workflow.runs)


def plan_programmatic_workflow(workflow: ProgrammaticWorkflow) -> tuple[CommandPlan, ...]:
    """Return a fully resolved global dispatch plan without side effects."""
    validate_programmatic_workflow(workflow)
    return _command_plans(workflow)


def _command_plans(workflow: ProgrammaticWorkflow) -> tuple[CommandPlan, ...]:
    """Build command plans after the caller selected the applicable preflight."""
    by_name = {run.name: run for run in workflow.runs}
    return tuple(
        CommandPlan(
            run_name=entry.run_name,
            step_name=entry.step_name,
            command=command_for_step(by_name[entry.run_name].step(entry.step_name)),
            cwd=by_name[entry.run_name].step(entry.step_name).cwd,
            timeout_seconds=by_name[entry.run_name].step(entry.step_name).timeout_seconds,
        )
        for entry in workflow.dispatch
    )


def compile_workflow(
    workflow: WorkflowConfig,
    *,
    entry: Path,
    selected_runs: Sequence[str] | None = None,
    selected_steps: Sequence[str] | None = None,
    invocation_command: Sequence[str] | None = None,
    resolve_previous_executions: bool = False,
) -> ProgrammaticWorkflow:
    """Compile schema-v1 YAML values into the public programmatic plan model.

    Schema-v1 preserves its historical run-major dispatch and per-step failure
    behavior by using workflow-level ``continue`` policy.  Programmatic callers
    may instead provide any validated serial global order directly.
    """
    command = tuple(invocation_command or (sys.executable, "-m", "mammoth", "workflow", "run"))
    selected = workflow.select_runs(selected_runs)
    compiled_runs: list[ProgrammaticRun] = []
    dispatch: list[DispatchEntry] = []
    for run in selected:
        steps = run.ordered_steps(selected_steps)
        layout = RunLayout(Path(entry), run.name)
        compiled_runs.append(
            ProgrammaticRun(
                name=run.name,
                layout=layout,
                steps=steps,
                execution=ExecutionInputs(
                    invocation_kind="workflow",
                    command=command,
                    config_reference=workflow.source,
                ),
                environment=run.environment,
                resolve_previous_execution=resolve_previous_executions,
            )
        )
        dispatch.extend(DispatchEntry(run.name, step.name) for step in steps)
    return ProgrammaticWorkflow(
        runs=tuple(compiled_runs),
        dispatch=tuple(dispatch),
        failure_policy="continue",
    )


def plan_workflow(
    workflow: WorkflowConfig,
    *,
    selected_runs: Sequence[str] | None = None,
    selected_steps: Sequence[str] | None = None,
) -> tuple[CommandPlan, ...]:
    """Resolve schema-v1 commands through the shared programmatic preflight."""
    selected = workflow.select_runs(selected_runs)
    if not selected:
        return ()
    if selected_steps is not None and not tuple(selected_steps):
        return ()
    compiled = compile_workflow(
        workflow,
        entry=Path("."),
        selected_runs=selected_runs,
        selected_steps=selected_steps,
    )
    _validate_programmatic_workflow(compiled, validate_layout_paths=False)
    return _command_plans(compiled)


def run_programmatic_workflow(
    workflow: ProgrammaticWorkflow,
    *,
    dry_run: bool = False,
    base_environment: Mapping[str, str] | None = None,
) -> WorkflowResult:
    """Execute one validated caller-compiled plan in its explicit serial order.

    All validation and dry-run planning finish before layouts, leases, contexts,
    observers, or event streams are created.  Non-dry execution starts each
    run lazily at its first dispatch, then retains its independently leased
    attempt while later entries may target other runs.
    """
    plans = plan_programmatic_workflow(workflow)
    if dry_run:
        return WorkflowResult(
            runs=tuple(
                RunResult(run.name, "dry-run", execution_id=None, steps=()) for run in workflow.runs
            ),
            plans=plans,
        )

    environment = _validate_environment(
        os.environ if base_environment is None else base_environment
    )
    active: dict[str, _ActiveRun] = {}
    result_by_key: dict[tuple[str, str], StepResult] = {}
    dispatch_results: list[DispatchResult] = []
    stopped_runs: set[str] = set()
    workflow_failure: DispatchEntry | None = None
    interrupted_signal: int | None = None
    executor_failed = False
    pending: _PendingDispatch | None = None

    runs: tuple[RunResult, ...] = ()
    by_name = {run.name: run for run in workflow.runs}
    with _interruption_signals() as interruption_state:
        try:
            _release_initial_interrupt_guard(interruption_state)
            for entry in workflow.dispatch:
                current = active.get(entry.run_name)
                if current is None:
                    current = _start_run(by_name[entry.run_name], active)
                step = current.run.step(entry.step_name)
                pending = _PendingDispatch(entry)
                if workflow_failure is not None:
                    result = _skip_step(
                        current,
                        step,
                        "blocked after workflow failure "
                        f"{workflow_failure.run_name}/{workflow_failure.step_name}",
                        pending,
                    )
                elif entry.run_name in stopped_runs:
                    result = _skip_step(
                        current,
                        step,
                        "stopped after an earlier step failure",
                        pending,
                    )
                elif (
                    step.dependency_failure == "skip"
                    and any(
                        result_by_key[(entry.run_name, dependency)].outcome != "completed"
                        for dependency in step.needs
                    )
                ):
                    result = _skip_step(
                        current,
                        step,
                        "dependency did not complete successfully",
                        pending,
                    )
                else:
                    result = _execute_step(
                        current,
                        entry,
                        step,
                        workflow.pre_dispatch,
                        environment,
                        pending,
                    )
                with _blocked_interruption_signals():
                    _record_pending_result(pending, result_by_key, dispatch_results)
                    pending = None

                if result.signal is not None and result.outcome != "interrupted":
                    interrupted_signal = result.signal
                    break

                if result.outcome == "interrupted":
                    stopped_runs.add(entry.run_name)
                    if workflow.failure_policy == "stop":
                        workflow_failure = entry
                elif result.outcome == "failed":
                    if step.on_failure == "stop":
                        stopped_runs.add(entry.run_name)
                    if workflow.failure_policy == "stop":
                        workflow_failure = entry
        except _WorkflowInterrupted as error:
            with _blocked_interruption_signals(deliver_deferred=False):
                if pending is not None and pending.result is not None:
                    _record_pending_result(pending, result_by_key, dispatch_results)
                    pending = None
                interrupted_signal = error.signal_number
        except BaseException:
            executor_failed = True
            raise
        finally:
            with _blocked_interruption_signals(deliver_deferred=False):
                runs = _finalize_active_runs(
                    workflow,
                    active,
                    result_by_key,
                    interrupted_signal=interrupted_signal,
                    executor_failed=executor_failed,
                )
                runs = _complete_run_results(
                    workflow,
                    runs,
                    interrupted_signal=interrupted_signal,
                )

    return WorkflowResult(runs=runs, plans=plans, dispatch=tuple(dispatch_results))


def run_workflow(
    workflow: WorkflowConfig,
    *,
    entry: Path,
    selected_runs: Sequence[str] | None = None,
    selected_steps: Sequence[str] | None = None,
    dry_run: bool = False,
    invocation_command: Sequence[str] | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> WorkflowResult:
    """Run schema-v1 YAML through the same public programmatic executor."""
    selected = workflow.select_runs(selected_runs)
    if not selected:
        return WorkflowResult(runs=(), plans=())
    if dry_run and selected_steps is not None and not tuple(selected_steps):
        return WorkflowResult(
            runs=tuple(
                RunResult(run.name, "dry-run", execution_id=None, steps=()) for run in selected
            ),
            plans=(),
        )
    compiled = compile_workflow(
        workflow,
        entry=Path(entry),
        selected_runs=selected_runs,
        selected_steps=selected_steps,
        invocation_command=invocation_command,
        resolve_previous_executions=not dry_run,
    )
    return run_programmatic_workflow(
        compiled,
        dry_run=dry_run,
        base_environment=base_environment,
    )


def child_environment(
    base: Mapping[str, str],
    *,
    run_environment: Mapping[str, str],
    step_environment: Mapping[str, str],
    run_name: str,
    execution_id: str,
    invocation_kind: str,
    phase: str,
) -> dict[str, str]:
    """Build inherited child environment plus documented execution join hooks."""
    environment = dict(base)
    environment.update(run_environment)
    environment.update(step_environment)
    environment.update(
        {
            EXECUTION_ID_ENV: execution_id,
            RUN_NAME_ENV: run_name,
            INVOCATION_KIND_ENV: invocation_kind,
            PHASE_ENV: phase,
        }
    )
    return environment


def _start_run(run: ProgrammaticRun, active: dict[str, _ActiveRun]) -> _ActiveRun:
    """Prepare one independently owned layout, lease, context, and observer."""
    current: _ActiveRun | None = None
    lease: LogicalRunLease | None = None
    try:
        with _blocked_interruption_signals():
            layout = run.layout.prepare()
            lease = claim_logical_run_lease(layout.run_dir)
            inputs = run.execution
            previous_execution_id = inputs.previous_execution_id
            if run.resolve_previous_execution:
                previous_execution_id = latest_execution_id(layout.run_dir)
            context = create_execution_context(
                layout.run_dir,
                run_name=run.name,
                invocation_kind=inputs.invocation_kind,
                intended_phases=tuple(step.name for step in run.steps),
                world_size=inputs.world_size,
                execution_mode=inputs.execution_mode,
                command=inputs.command,
                config_reference=inputs.config_reference,
                previous_execution_id=previous_execution_id,
                resume_checkpoint=inputs.resume_checkpoint,
                parent_execution_id=inputs.parent_execution_id,
                starting_epoch=inputs.starting_epoch,
                starting_global_step=inputs.starting_global_step,
                runtime=_thaw_runtime_metadata(inputs.runtime),
            )
            observer = RunObserver((JsonlEventSink(ExecutionEventWriter.for_runner(context)),))
            current = _ActiveRun(run, lease, context, observer)
            active[run.name] = current
            observer.emit("execution_started")
        return current
    except BaseException:
        if current is None and lease is not None:
            lease.close()
        raise


def _execute_step(
    active: _ActiveRun,
    dispatch: DispatchEntry,
    step: StepConfig,
    hook: PreDispatchHook | None,
    base_environment: Mapping[str, str],
    pending: _PendingDispatch,
) -> StepResult:
    """Run one child while emitting paired phase and task lifecycle records."""
    task_id = step.name
    observer = active.observer
    phase_started = False
    task_started = False
    running_hook = hook is not None
    try:
        with _blocked_interruption_signals():
            phase_started = True
            observer.emit("phase_started", phase=step.name)
            task_started = True
            observer.emit("task_started", phase=step.name, task_id=task_id)
        if hook is not None:
            hook(PreDispatchContext(dispatch, active.run, step, active.context))
        running_hook = False
        process = launch_process(
            command_for_step(step),
            cwd=step.cwd,
            environment=child_environment(
                base_environment,
                run_environment=active.run.environment,
                step_environment=step.environment,
                run_name=active.run.name,
                execution_id=active.context.metadata.execution_id,
                invocation_kind=active.context.metadata.invocation_kind,
                phase=step.name,
            ),
            timeout_seconds=step.timeout_seconds,
        )
        return _finish_process_step(observer, step, task_id, process, pending)
    except _WorkflowInterrupted as error:
        if pending.result is not None:
            pending.result = _with_step_signal(pending.result, error.signal_number)
            return pending.result
        reason = "interrupted"
        _emit_step_terminal(
            observer,
            pending,
            StepResult(step.name, "interrupted", reason=reason, signal=error.signal_number),
            task_event="task_failed" if task_started else None,
            task_fields={"phase": step.name, "task_id": task_id, "message": reason},
            phase_event="phase_failed" if phase_started else None,
            phase_fields={"phase": step.name, "message": reason},
        )
        return _terminal_step_result(pending)
    except KeyboardInterrupt:
        if pending.result is not None:
            pending.result = _with_step_signal(pending.result, signal.SIGINT)
            return pending.result
        reason = "interrupted"
        _emit_step_terminal(
            observer,
            pending,
            StepResult(step.name, "interrupted", reason=reason, signal=signal.SIGINT),
            task_event="task_failed" if task_started else None,
            task_fields={"phase": step.name, "task_id": task_id, "message": reason},
            phase_event="phase_failed" if phase_started else None,
            phase_fields={"phase": step.name, "message": reason},
        )
        return _terminal_step_result(pending)
    except OSError as error:
        reason = f"could not launch: {error}"
        _emit_step_terminal(
            observer,
            pending,
            StepResult(step.name, "failed", reason=reason),
            task_event="task_failed",
            task_fields={
                "phase": step.name,
                "task_id": task_id,
                "message": reason,
                "exit_code": 127,
            },
            phase_event="phase_failed",
            phase_fields={"phase": step.name, "message": reason, "exit_code": 127},
        )
        return _terminal_step_result(pending)
    except Exception as error:
        prefix = "pre-dispatch hook failed" if running_hook else "workflow execution failed"
        reason = f"{prefix}: {error}"
        _emit_step_terminal(
            observer,
            pending,
            StepResult(step.name, "failed", reason=reason),
            task_event="task_failed",
            task_fields={"phase": step.name, "task_id": task_id, "message": reason},
            phase_event="phase_failed",
            phase_fields={"phase": step.name, "message": reason},
        )
        return _terminal_step_result(pending)


def _finish_process_step(
    observer: RunObserver,
    step: StepConfig,
    task_id: str,
    process: ProcessResult,
    pending: _PendingDispatch,
) -> StepResult:
    """Publish one supervised process outcome as an indivisible terminal pair."""
    if process.interrupted:
        reason = "interrupted"
        _emit_step_terminal(
            observer,
            pending,
            StepResult(step.name, "interrupted", process, reason, process.signal),
            task_event="task_failed",
            task_fields={"phase": step.name, "task_id": task_id, "message": reason},
            phase_event="phase_failed",
            phase_fields={
                "phase": step.name,
                "exit_code": process.return_code,
                "duration_seconds": process.duration_seconds,
                "message": reason,
            },
        )
        return _terminal_step_result(pending)
    if process.return_code == 0 and not process.timed_out:
        _emit_step_terminal(
            observer,
            pending,
            StepResult(step.name, "completed", process),
            task_event="task_completed",
            task_fields={"phase": step.name, "task_id": task_id, "exit_code": 0},
            phase_event="phase_completed",
            phase_fields={
                "phase": step.name,
                "exit_code": 0,
                "duration_seconds": process.duration_seconds,
            },
        )
        return _terminal_step_result(pending)
    reason = "timed out" if process.timed_out else f"exit code {process.return_code}"
    _emit_step_terminal(
        observer,
        pending,
        StepResult(step.name, "failed", process, reason),
        task_event="task_failed",
        task_fields={
            "phase": step.name,
            "task_id": task_id,
            "exit_code": process.return_code,
            "message": reason,
        },
        phase_event="phase_failed",
        phase_fields={
            "phase": step.name,
            "exit_code": process.return_code,
            "duration_seconds": process.duration_seconds,
            "message": reason,
        },
    )
    return _terminal_step_result(pending)


def _skip_step(
    active: _ActiveRun,
    step: StepConfig,
    reason: str,
    pending: _PendingDispatch,
) -> StepResult:
    """Record a terminal skip without launching a child process."""
    _emit_step_terminal(
        active.observer,
        pending,
        StepResult(step.name, "skipped", reason=reason),
        task_event="task_skipped",
        task_fields={"phase": step.name, "task_id": step.name, "message": reason},
        phase_event="phase_skipped",
        phase_fields={"phase": step.name, "message": reason},
    )
    return _terminal_step_result(pending)


def _emit_step_terminal(
    observer: RunObserver,
    pending: _PendingDispatch,
    result: StepResult,
    *,
    task_event: EventName | None,
    task_fields: Mapping[str, Any],
    phase_event: EventName | None,
    phase_fields: Mapping[str, Any],
) -> None:
    """Publish a step's terminal task/phase pair without splitting its lifecycle."""
    deferred_signals: list[int] = []
    with _blocked_interruption_signals(
        deliver_deferred=False,
        deferred_signals=deferred_signals,
    ):
        pending.result = result
        if task_event is not None:
            observer.emit(task_event, **task_fields)
        if phase_event is not None:
            observer.emit(phase_event, **phase_fields)
    if deferred_signals:
        pending.result = _with_step_signal(result, deferred_signals[0])


def _terminal_step_result(pending: _PendingDispatch) -> StepResult:
    """Return the terminal result published before its lifecycle pair was emitted."""
    result = pending.result
    if result is None:
        raise RuntimeError("terminal workflow step result was not published")
    return result


def _with_step_signal(result: StepResult, signal_number: int) -> StepResult:
    """Retain a completed terminal step while surfacing a subsequent interruption."""
    return StepResult(
        name=result.name,
        outcome=result.outcome,
        process=result.process,
        reason=result.reason,
        signal=result.signal if result.signal is not None else signal_number,
    )


def _record_pending_result(
    pending: _PendingDispatch,
    result_by_key: dict[tuple[str, str], StepResult],
    dispatch_results: list[DispatchResult],
) -> None:
    """Commit a pre-published terminal step result to both workflow result views."""
    result = _terminal_step_result(pending)
    key = (pending.dispatch.run_name, pending.dispatch.step_name)
    if key in result_by_key:
        return
    result_by_key[key] = result
    dispatch_results.append(DispatchResult(pending.dispatch.run_name, result))


def _finalize_active_runs(
    workflow: ProgrammaticWorkflow,
    active: Mapping[str, _ActiveRun],
    results: Mapping[tuple[str, str], StepResult],
    *,
    interrupted_signal: int | None,
    executor_failed: bool,
) -> tuple[RunResult, ...]:
    """Emit one deterministic execution terminal event and release every resource."""
    finalized: list[RunResult] = []
    for run in workflow.runs:
        current = active.get(run.name)
        if current is None:
            continue
        step_results = tuple(
            results[(run.name, step.name)]
            for step in run.steps
            if (run.name, step.name) in results
        )
        outcome = _run_outcome(
            step_results,
            expected_step_count=len(run.steps),
            interrupted_signal=interrupted_signal,
            executor_failed=executor_failed,
        )
        try:
            if outcome == "completed":
                current.observer.emit("execution_completed", exit_code=0)
            elif outcome == "interrupted":
                terminal_signal = interrupted_signal or _step_interruption_signal(step_results)
                current.observer.emit(
                    "execution_interrupted",
                    signal=terminal_signal,
                )
            else:
                current.observer.emit("execution_failed", exit_code=1)
        finally:
            try:
                current.observer.close()
            finally:
                current.lease.close()
        finalized.append(
            RunResult(
                run_name=run.name,
                outcome=outcome,
                execution_id=current.context.metadata.execution_id,
                steps=step_results,
            )
        )
    return tuple(finalized)


def _run_outcome(
    results: Sequence[StepResult],
    *,
    expected_step_count: int,
    interrupted_signal: int | None,
    executor_failed: bool,
) -> RunOutcome:
    """Classify direct failure, cross-run blocking, and interruption distinctly."""
    if interrupted_signal is not None or any(result.outcome == "interrupted" for result in results):
        return "interrupted"
    if executor_failed and len(results) < expected_step_count:
        return "failed"
    if any(result.outcome == "failed" for result in results):
        return "failed"
    if any(result.outcome == "skipped" for result in results):
        return "blocked"
    return "completed"


def _complete_run_results(
    workflow: ProgrammaticWorkflow,
    finalized: Sequence[RunResult],
    *,
    interrupted_signal: int | None,
) -> tuple[RunResult, ...]:
    """Return one result for every selected run, including unstarted interruptions."""
    finalized_by_name = {result.run_name: result for result in finalized}
    return tuple(
        finalized_by_name.get(
            run.name,
            RunResult(
                run_name=run.name,
                outcome="interrupted" if interrupted_signal is not None else "blocked",
                execution_id=None,
                steps=(),
            ),
        )
        for run in workflow.runs
    )


def _step_interruption_signal(results: Sequence[StepResult]) -> int:
    """Return a run-local interruption signal for its execution terminal event."""
    for result in results:
        if result.outcome == "interrupted":
            if result.signal is not None:
                return result.signal
            if result.process is not None and result.process.signal is not None:
                return result.process.signal
    return signal.SIGINT


def _validate_execution_inputs_for_runs(runs: Sequence[ProgrammaticRun]) -> None:
    """Preflight every immutable execution input using core metadata contracts."""
    for run in runs:
        inputs = run.execution
        if not isinstance(inputs.invocation_kind, str) or not inputs.invocation_kind:
            raise ValueError("execution invocation_kind must be a non-empty string")
        sanitize_command(inputs.command)
        if inputs.config_reference != "":
            sanitize_reference(inputs.config_reference)
        if inputs.resume_checkpoint is not None:
            sanitize_reference(inputs.resume_checkpoint)
        if inputs.runtime is not None:
            runtime = _thaw_runtime_metadata(inputs.runtime)
            if runtime is None:
                raise RuntimeError("frozen execution runtime metadata was unexpectedly absent")
            sanitize_metadata_fields(runtime)
        if (
            not isinstance(inputs.world_size, int)
            or isinstance(inputs.world_size, bool)
            or inputs.world_size < 1
        ):
            raise ValueError("execution world_size must be a positive integer")
        if inputs.execution_mode not in {"single", "distributed"}:
            raise ValueError("execution execution_mode must be 'single' or 'distributed'")
        if (inputs.execution_mode == "single") != (inputs.world_size == 1):
            raise ValueError("execution execution_mode and world_size disagree")
        for field_name in ("starting_epoch", "starting_global_step"):
            value = getattr(inputs, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"execution {field_name} must be a non-negative integer or None")
        for field_name in ("previous_execution_id", "parent_execution_id"):
            value = getattr(inputs, field_name)
            if value is not None:
                validate_execution_id(value)
        _validate_environment(run.environment)
        for step in run.steps:
            _validate_environment(step.environment)


def _validate_programmatic_layouts(runs: Sequence[ProgrammaticRun]) -> None:
    """Reject existing non-directory layout paths before any run is prepared."""
    for run in runs:
        layout = run.layout
        for path in (layout.entry, layout.run_dir):
            if path.exists() and not path.is_dir():
                raise ValueError(f"Programmatic run layout path must be a directory: {path}")
        parent = layout.entry
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Programmatic run layout parent must be a directory: {parent}")


def _freeze_runtime_metadata(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively detach immutable execution metadata from caller-owned values."""
    return MappingProxyType(
        {name: _freeze_runtime_value(value) for name, value in runtime.items()}
    )


def _freeze_runtime_value(value: Any) -> Any:
    """Freeze mappings and sequences retained by :class:`ExecutionInputs`."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: _freeze_runtime_value(item) for name, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_runtime_value(item) for item in value)
    return value


def _thaw_runtime_metadata(runtime: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Copy frozen runtime metadata into JSON-compatible values for core validation."""
    if runtime is None:
        return None
    return {name: _thaw_runtime_value(value) for name, value in runtime.items()}


def _thaw_runtime_value(value: Any) -> Any:
    """Return one mutable JSON-compatible copy of a frozen metadata value."""
    if isinstance(value, Mapping):
        return {name: _thaw_runtime_value(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_runtime_value(item) for item in value]
    return value


def _validate_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Validate explicit child overrides without reading or persisting secrets."""
    if not isinstance(environment, Mapping):
        raise ValueError("workflow environment must be a mapping")
    validated: dict[str, str] = {}
    for name, value in environment.items():
        if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
            raise ValueError("workflow environment has an invalid variable name")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("workflow environment values must be strings")
        validated[name] = value
    return validated


@contextmanager
def _interruption_signals() -> Iterator[_InterruptionState | None]:
    """Translate process termination signals into executor-managed cleanup."""
    if not hasattr(signal, "SIGTERM"):
        yield None
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    state = _InterruptionState()
    state.critical_depth = 1
    state.initial_guard_active = True
    state_token = _INTERRUPTION_STATE.set(state)
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
            # Python allows signal registration only on the main thread.
            # Direct callers on another thread still retain child cleanup from
            # launch.py and do not receive this main-thread translation layer.
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
            _INTERRUPTION_STATE.reset(state_token)


def _release_initial_interrupt_guard(state: _InterruptionState | None) -> None:
    """Deliver an install-window signal only after the executor enters its handler."""
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
    deferred_signals: list[int] | None = None,
) -> Iterator[None]:
    """Defer SIGINT/SIGTERM across resource-registration and release boundaries."""
    state = _INTERRUPTION_STATE.get()
    if state is not None:
        state.critical_depth += 1
    try:
        yield
    finally:
        if state is not None:
            state.critical_depth -= 1
            if state.critical_depth == 0 and state.deferred_signal is not None:
                signal_number = state.deferred_signal
                state.deferred_signal = None
                if deferred_signals is not None:
                    deferred_signals.append(signal_number)
                elif deliver_deferred:
                    raise _WorkflowInterrupted(signal_number)
