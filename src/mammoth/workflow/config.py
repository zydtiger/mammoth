"""Strict declarative workflow values and YAML parsing.

The workflow runner consumes these project-neutral commands. Mammoth validates
structure and dependencies but never interprets command arguments or step names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import yaml

from mammoth.core.identity import validate_run_name

WORKFLOW_SCHEMA_VERSION = 1

LauncherKind = Literal["local", "torchrun"]
FailurePolicy = Literal["stop", "continue"]
DependencyFailurePolicy = Literal["skip", "run"]

_ROOT_KEYS = frozenset({"schema_version", "defaults", "runs"})
_DEFAULT_KEYS = frozenset(
    {
        "launcher",
        "processes",
        "torchrun_args",
        "timeout_seconds",
        "on_failure",
        "dependency_failure",
        "environment",
        "cwd",
    }
)
_RUN_KEYS = frozenset({"steps", "environment", "defaults"})
_STEP_KEYS = frozenset({"command", "needs", *_DEFAULT_KEYS})


@dataclass(frozen=True, slots=True)
class StepConfig:
    """One opaque command and its generic execution policy."""

    name: str
    command: Sequence[str]
    needs: Sequence[str] = ()
    launcher: LauncherKind = "local"
    processes: int = 1
    torchrun_args: Sequence[str] = ()
    timeout_seconds: float | None = None
    on_failure: FailurePolicy = "stop"
    dependency_failure: DependencyFailurePolicy = "skip"
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | None = None

    def __post_init__(self) -> None:
        validate_run_name(self.name)
        if isinstance(self.command, str) or not isinstance(self.command, Sequence):
            raise ValueError(f"Step {self.name!r} command must be a sequence of arguments")
        command = tuple(self.command)
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError(f"Step {self.name!r} command must contain non-empty strings")
        if isinstance(self.needs, str) or not isinstance(self.needs, Sequence):
            raise ValueError(f"Step {self.name!r} needs must be a sequence of step names")
        needs = tuple(self.needs)
        if any(not isinstance(item, str) or not item for item in needs):
            raise ValueError(f"Step {self.name!r} needs must contain non-empty strings")
        if self.launcher not in {"local", "torchrun"}:
            raise ValueError(f"Unsupported launcher for {self.name!r}: {self.launcher!r}")
        if (
            not isinstance(self.processes, int)
            or isinstance(self.processes, bool)
            or self.processes < 1
        ):
            raise ValueError(f"Step {self.name!r} processes must be a positive integer")
        if self.launcher == "local" and self.processes != 1:
            raise ValueError(f"Local step {self.name!r} must use processes=1")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int | float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError(f"Step {self.name!r} timeout_seconds must be positive")
        if self.on_failure not in {"stop", "continue"}:
            raise ValueError(f"Invalid on_failure policy for step {self.name!r}")
        if self.dependency_failure not in {"skip", "run"}:
            raise ValueError(f"Invalid dependency_failure policy for step {self.name!r}")
        if isinstance(self.torchrun_args, str) or not isinstance(self.torchrun_args, Sequence):
            raise ValueError(f"Step {self.name!r} torchrun_args must be a sequence of arguments")
        torchrun_args = tuple(self.torchrun_args)
        if any(not isinstance(item, str) or not item for item in torchrun_args):
            raise ValueError(f"Step {self.name!r} torchrun_args must contain non-empty strings")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "needs", needs)
        object.__setattr__(self, "torchrun_args", torchrun_args)
        if self.cwd is not None:
            try:
                cwd = Path(self.cwd)
            except TypeError as error:
                raise ValueError(
                    f"Step {self.name!r} cwd must be a path-like value or None"
                ) from error
            object.__setattr__(self, "cwd", cwd)
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(parse_environment(self.environment, f"step {self.name!r}")),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One logical run containing a validated directed acyclic step graph."""

    name: str
    steps: tuple[StepConfig, ...]
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_run_name(self.name)
        if not self.steps:
            raise ValueError(f"Run {self.name!r} must contain at least one step")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        validate_step_graph(self.steps)

    def step(self, name: str) -> StepConfig:
        """Return one exact step name or raise ``KeyError``."""
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"Run {self.name!r} has no step {name!r}")

    def ordered_steps(self, selected: Sequence[str] | None = None) -> tuple[StepConfig, ...]:
        """Return dependency-first declaration-stable steps, including selected needs."""
        ordered = topological_steps(self.steps)
        if selected is None:
            return ordered
        selected_names = tuple(selected)
        missing = sorted(set(selected_names).difference(step.name for step in self.steps))
        if missing:
            raise KeyError(f"Run {self.name!r} has no steps: {', '.join(missing)}")
        required = set(selected_names)
        by_name = {step.name: step for step in self.steps}
        pending = list(selected_names)
        while pending:
            name = pending.pop()
            for dependency in by_name[name].needs:
                if dependency not in required:
                    required.add(dependency)
                    pending.append(dependency)
        return tuple(step for step in ordered if step.name in required)


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Complete schema-v1 workflow document."""

    runs: tuple[RunConfig, ...]
    source: Path
    schema_version: int = WORKFLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKFLOW_SCHEMA_VERSION:
            raise ValueError(f"Unsupported workflow schema version: {self.schema_version!r}")
        names = [run.name for run in self.runs]
        if not names:
            raise ValueError("Workflow must contain at least one run")
        if len(names) != len(set(names)):
            raise ValueError("Workflow run names must be unique")

    def select_runs(self, names: Sequence[str] | None = None) -> tuple[RunConfig, ...]:
        """Return exact requested runs in document order."""
        if names is None:
            return self.runs
        requested = set(names)
        missing = sorted(requested.difference(run.name for run in self.runs))
        if missing:
            raise KeyError(f"Workflow has no runs: {', '.join(missing)}")
        return tuple(run for run in self.runs if run.name in requested)


def load_workflow(path: Path) -> WorkflowConfig:
    """Safely load and strictly validate one YAML workflow document."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Workflow document must be a mapping")
    reject_unknown_keys(payload, _ROOT_KEYS, "workflow")
    schema_version = required_int(payload, "schema_version", minimum=1)
    if schema_version != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(f"Unsupported workflow schema version: {schema_version!r}")
    defaults = optional_mapping(payload, "defaults")
    reject_unknown_keys(defaults, _DEFAULT_KEYS, "workflow defaults")
    runs_payload = required_mapping(payload, "runs")
    runs = tuple(
        parse_run(name, value, defaults=defaults, source_directory=source.parent)
        for name, value in runs_payload.items()
    )
    return WorkflowConfig(runs=runs, source=source, schema_version=schema_version)


def parse_run(
    name: Any,
    payload: Any,
    *,
    defaults: Mapping[str, Any],
    source_directory: Path,
) -> RunConfig:
    """Parse one named run from a strict mapping."""
    if not isinstance(name, str):
        raise ValueError("Workflow run names must be strings")
    validate_run_name(name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Run {name!r} must be a mapping")
    reject_unknown_keys(payload, _RUN_KEYS, f"run {name!r}")
    run_defaults = merge_mappings(defaults, optional_mapping(payload, "defaults"))
    reject_unknown_keys(run_defaults, _DEFAULT_KEYS, f"run {name!r} defaults")
    run_environment = parse_environment(payload.get("environment", {}), f"run {name!r}")
    steps_payload = required_mapping(payload, "steps")
    steps = tuple(
        parse_step(
            step_name,
            step_payload,
            defaults=run_defaults,
            source_directory=source_directory,
        )
        for step_name, step_payload in steps_payload.items()
    )
    return RunConfig(name=name, steps=steps, environment=run_environment)


def parse_step(
    name: Any,
    payload: Any,
    *,
    defaults: Mapping[str, Any],
    source_directory: Path,
) -> StepConfig:
    """Parse one step with document and run defaults applied."""
    if not isinstance(name, str):
        raise ValueError("Workflow step names must be strings")
    validate_run_name(name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Step {name!r} must be a mapping")
    reject_unknown_keys(payload, _STEP_KEYS, f"step {name!r}")
    merged = merge_mappings(defaults, payload)
    command = string_sequence(merged.get("command"), f"step {name!r} command", nonempty=True)
    needs = string_sequence(merged.get("needs", ()), f"step {name!r} needs")
    environment = parse_environment(merged.get("environment", {}), f"step {name!r}")
    launcher = enum_value(merged.get("launcher", "local"), {"local", "torchrun"}, "launcher")
    processes = integer_value(merged.get("processes", 1), "processes", minimum=1)
    torchrun_args = string_sequence(
        merged.get("torchrun_args", ()), f"step {name!r} torchrun_args"
    )
    timeout = optional_positive_number(merged.get("timeout_seconds"), "timeout_seconds")
    on_failure = enum_value(merged.get("on_failure", "stop"), {"stop", "continue"}, "on_failure")
    dependency_failure = enum_value(
        merged.get("dependency_failure", "skip"), {"skip", "run"}, "dependency_failure"
    )
    cwd_value = merged.get("cwd")
    if cwd_value is not None and (not isinstance(cwd_value, str) or not cwd_value):
        raise ValueError(f"Step {name!r} cwd must be a non-empty string or null")
    cwd = None if cwd_value is None else (source_directory / cwd_value).resolve()
    return StepConfig(
        name=name,
        command=command,
        needs=needs,
        launcher=cast(LauncherKind, launcher),
        processes=processes,
        torchrun_args=torchrun_args,
        timeout_seconds=timeout,
        on_failure=cast(FailurePolicy, on_failure),
        dependency_failure=cast(DependencyFailurePolicy, dependency_failure),
        environment=environment,
        cwd=cwd,
    )


def validate_step_graph(steps: Sequence[StepConfig]) -> None:
    """Reject duplicate names, missing edges, self-edges, and cycles."""
    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise ValueError("Step names must be unique within a run")
    known = set(names)
    for step in steps:
        missing = sorted(set(step.needs).difference(known))
        if missing:
            raise ValueError(f"Step {step.name!r} has missing dependencies: {', '.join(missing)}")
        if step.name in step.needs:
            raise ValueError(f"Step {step.name!r} cannot depend on itself")
    topological_steps(steps)


def topological_steps(steps: Sequence[StepConfig]) -> tuple[StepConfig, ...]:
    """Return declaration-stable dependency-first steps or raise on a cycle."""
    remaining = list(steps)
    completed: set[str] = set()
    ordered: list[StepConfig] = []
    while remaining:
        ready = [step for step in remaining if set(step.needs).issubset(completed)]
        if not ready:
            cycle = ", ".join(step.name for step in remaining)
            raise ValueError(f"Workflow step graph contains a cycle involving: {cycle}")
        for step in ready:
            ordered.append(step)
            completed.add(step.name)
            remaining.remove(step)
    return tuple(ordered)


def reject_unknown_keys(payload: Mapping[str, Any], allowed: frozenset[str], scope: str) -> None:
    """Reject misspelled or speculative configuration keys."""
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown keys in {scope}: {', '.join(unknown)}")


def merge_mappings(*mappings: Mapping[str, Any]) -> dict[str, Any]:
    """Merge shallow default mappings in precedence order."""
    merged: dict[str, Any] = {}
    for mapping in mappings:
        merged.update(mapping)
    return merged


def required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one required mapping field."""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return cast(Mapping[str, Any], value)


def optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one optional mapping field, defaulting to empty."""
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return cast(Mapping[str, Any], value)


def required_int(payload: Mapping[str, Any], key: str, *, minimum: int) -> int:
    """Return one required bounded integer field."""
    return integer_value(payload.get(key), key, minimum=minimum)


def integer_value(value: Any, name: str, *, minimum: int) -> int:
    """Validate one bounded integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def string_sequence(value: Any, name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    """Validate a sequence of non-empty strings without accepting a scalar."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list of strings")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{name} must contain non-empty strings")
    if nonempty and not items:
        raise ValueError(f"{name} must not be empty")
    return cast(tuple[str, ...], items)


def parse_environment(value: Any, scope: str) -> dict[str, str]:
    """Validate explicit environment overrides without reading process secrets."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{scope} environment must be a mapping")
    environment: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
            raise ValueError(f"{scope} has an invalid environment variable name")
        if not isinstance(item, str) or "\x00" in item:
            raise ValueError(f"{scope} environment values must be strings")
        environment[name] = item
    return environment


def optional_positive_number(value: Any, name: str) -> float | None:
    """Validate an optional positive finite number."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number or null")
    return float(value)


def enum_value(value: Any, allowed: set[str], name: str) -> str:
    """Validate one string enumeration."""
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value
