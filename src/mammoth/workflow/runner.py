"""Sequential DAG workflow execution with Mammoth lifecycle artifacts.

The CLI passes validated config here. Child commands receive only documented
join variables; their command meaning and all project computation remain local.
"""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mammoth.core import (
    RunLayout,
    claim_logical_run_lease,
    create_execution_context,
    latest_execution_id,
)
from mammoth.core.events import ExecutionEventWriter
from mammoth.core.execution import (
    EXECUTION_ID_ENV,
    INVOCATION_KIND_ENV,
    PHASE_ENV,
    RUN_NAME_ENV,
)
from mammoth.logging import JsonlEventSink, RunObserver
from mammoth.workflow.config import RunConfig, StepConfig, WorkflowConfig
from mammoth.workflow.launch import CommandPlan, ProcessResult, command_for_step, launch_process

StepOutcome = Literal["completed", "failed", "skipped", "interrupted"]
RunOutcome = Literal["completed", "failed", "interrupted", "dry-run"]


@dataclass(frozen=True, slots=True)
class StepResult:
    """One selected step's terminal workflow outcome."""

    name: str
    outcome: StepOutcome
    process: ProcessResult | None = None
    reason: str | None = None


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

    @property
    def successful(self) -> bool:
        """Return whether every selected run completed or was only planned."""
        return all(run.outcome in {"completed", "dry-run"} for run in self.runs)


def plan_workflow(
    workflow: WorkflowConfig,
    *,
    selected_runs: Sequence[str] | None = None,
    selected_steps: Sequence[str] | None = None,
) -> tuple[CommandPlan, ...]:
    """Resolve selected commands without creating entries or execution artifacts."""
    plans: list[CommandPlan] = []
    for run in workflow.select_runs(selected_runs):
        for step in run.ordered_steps(selected_steps):
            plans.append(
                CommandPlan(
                    run_name=run.name,
                    step_name=step.name,
                    command=command_for_step(step),
                    cwd=step.cwd,
                    timeout_seconds=step.timeout_seconds,
                )
            )
    return tuple(plans)


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
    """Plan or execute selected runs while preserving per-run attempt isolation."""
    plans = plan_workflow(
        workflow,
        selected_runs=selected_runs,
        selected_steps=selected_steps,
    )
    selected = workflow.select_runs(selected_runs)
    if dry_run:
        return WorkflowResult(
            runs=tuple(
                RunResult(
                    run_name=run.name,
                    outcome="dry-run",
                    execution_id=None,
                    steps=(),
                )
                for run in selected
            ),
            plans=plans,
        )
    command = tuple(invocation_command or (sys.executable, "-m", "mammoth", "workflow", "run"))
    environment = dict(os.environ if base_environment is None else base_environment)
    results = tuple(
        run_one(
            run,
            entry=Path(entry),
            selected_steps=selected_steps,
            workflow=workflow,
            invocation_command=command,
            base_environment=environment,
        )
        for run in selected
    )
    return WorkflowResult(runs=results, plans=plans)


def run_one(
    run: RunConfig,
    *,
    entry: Path,
    selected_steps: Sequence[str] | None,
    workflow: WorkflowConfig,
    invocation_command: Sequence[str],
    base_environment: Mapping[str, str],
) -> RunResult:
    """Execute one logical run under a process-lifetime producer lease."""
    steps = run.ordered_steps(selected_steps)
    layout = RunLayout(entry, run.name).prepare()
    previous_execution_id = latest_execution_id(layout.run_dir)
    with claim_logical_run_lease(layout.run_dir):
        context = create_execution_context(
            layout.run_dir,
            run_name=run.name,
            invocation_kind="workflow",
            intended_phases=tuple(step.name for step in steps),
            world_size=1,
            execution_mode="single",
            command=invocation_command,
            config_reference=workflow.source,
            previous_execution_id=previous_execution_id,
        )
        observer = RunObserver((JsonlEventSink(ExecutionEventWriter.for_runner(context)),))
        observer.emit("execution_started")
        try:
            step_results = execute_steps(
                run,
                steps,
                observer=observer,
                execution_id=context.metadata.execution_id,
                base_environment=base_environment,
            )
            interrupted = any(result.outcome == "interrupted" for result in step_results)
            failed = any(result.outcome == "failed" for result in step_results)
            if interrupted:
                observer.emit("execution_interrupted", signal=signal.SIGINT)
                outcome: RunOutcome = "interrupted"
            elif failed:
                observer.emit("execution_failed", exit_code=1)
                outcome = "failed"
            else:
                observer.emit("execution_completed", exit_code=0)
                outcome = "completed"
        except BaseException as error:
            observer.emit("execution_failed", message=str(error), exit_code=1)
            raise
        finally:
            observer.close()
    return RunResult(
        run_name=run.name,
        outcome=outcome,
        execution_id=context.metadata.execution_id,
        steps=step_results,
    )


def execute_steps(
    run: RunConfig,
    steps: Sequence[StepConfig],
    *,
    observer: RunObserver,
    execution_id: str,
    base_environment: Mapping[str, str],
) -> tuple[StepResult, ...]:
    """Execute dependency-ordered steps with stop, continue, skip, and run policies."""
    results: list[StepResult] = []
    by_name: dict[str, StepResult] = {}
    stopped = False
    for step in steps:
        dependency_failed = any(
            by_name[dependency].outcome in {"failed", "interrupted", "skipped"}
            for dependency in step.needs
            if dependency in by_name
        )
        reason: str | None = None
        if stopped:
            reason = "stopped after an earlier step failure"
        elif dependency_failed and step.dependency_failure == "skip":
            reason = "dependency did not complete successfully"
        if reason is not None:
            observer.emit("phase_skipped", phase=step.name, message=reason)
            result = StepResult(step.name, "skipped", reason=reason)
            results.append(result)
            by_name[step.name] = result
            continue

        observer.emit("phase_started", phase=step.name)
        try:
            process = launch_process(
                command_for_step(step),
                cwd=step.cwd,
                environment=child_environment(
                    base_environment,
                    run_environment=run.environment,
                    step_environment=step.environment,
                    run_name=run.name,
                    execution_id=execution_id,
                    phase=step.name,
                ),
                timeout_seconds=step.timeout_seconds,
            )
        except OSError as error:
            reason = f"could not launch: {error}"
            observer.emit("phase_failed", phase=step.name, message=reason, exit_code=127)
            result = StepResult(step.name, "failed", reason=reason)
            results.append(result)
            by_name[step.name] = result
            if step.on_failure == "stop":
                stopped = True
            continue
        if process.interrupted:
            observer.emit(
                "phase_failed",
                phase=step.name,
                exit_code=process.return_code,
                duration_seconds=process.duration_seconds,
                message="interrupted",
            )
            result = StepResult(step.name, "interrupted", process, "interrupted")
            stopped = True
        elif process.return_code == 0 and not process.timed_out:
            observer.emit(
                "phase_completed",
                phase=step.name,
                exit_code=0,
                duration_seconds=process.duration_seconds,
            )
            result = StepResult(step.name, "completed", process)
        else:
            reason = "timed out" if process.timed_out else f"exit code {process.return_code}"
            observer.emit(
                "phase_failed",
                phase=step.name,
                exit_code=process.return_code,
                duration_seconds=process.duration_seconds,
                message=reason,
            )
            result = StepResult(step.name, "failed", process, reason)
            if step.on_failure == "stop":
                stopped = True
        results.append(result)
        by_name[step.name] = result
    return tuple(results)


def child_environment(
    base: Mapping[str, str],
    *,
    run_environment: Mapping[str, str],
    step_environment: Mapping[str, str],
    run_name: str,
    execution_id: str,
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
            INVOCATION_KIND_ENV: "workflow",
            PHASE_ENV: phase,
        }
    )
    return environment
