"""Strict declarative workflow parsing, planning, and local execution."""

from __future__ import annotations

from mammoth.workflow.config import (
    RunConfig,
    StepConfig,
    WorkflowConfig,
    load_workflow,
)
from mammoth.workflow.launch import (
    CapturedProcessResult,
    CommandPlan,
    ProcessResult,
    SupervisedProcess,
    command_for_step,
    run_captured_process,
)
from mammoth.workflow.runner import (
    RunResult,
    StepResult,
    WorkflowResult,
    plan_workflow,
    run_workflow,
)

__all__ = [
    "CapturedProcessResult",
    "CommandPlan",
    "ProcessResult",
    "RunConfig",
    "RunResult",
    "StepConfig",
    "StepResult",
    "SupervisedProcess",
    "WorkflowConfig",
    "WorkflowResult",
    "command_for_step",
    "load_workflow",
    "plan_workflow",
    "run_captured_process",
    "run_workflow",
]
