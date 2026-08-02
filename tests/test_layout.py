from __future__ import annotations

from pathlib import Path

import pytest

from mammoth.core import RunLayout, validate_run_name


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "../escape", "nested/run", "/absolute", "white space", "-prefix"],
)
def test_run_name_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_run_name(value)


def test_run_layout_resolves_and_prepares_stable_paths(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "entries", "experiment.v1_2").prepare()

    assert layout.run_dir == tmp_path / "entries" / "experiment.v1_2"
    assert layout.manifest_path == layout.run_dir / "manifest.json"
    assert layout.executions_dir == layout.run_dir / "logs" / "executions"
    assert layout.execution_dir("attempt-1") == layout.executions_dir / "attempt-1"
    assert layout.checkpoints_dir.is_dir()
    assert layout.results_dir.is_dir()
    assert layout.visualizations_dir.is_dir()


def test_layouts_are_independent_of_entry_name(tmp_path: Path) -> None:
    first = RunLayout(tmp_path / "runs", "same-name")
    second = RunLayout(tmp_path / "anything", "same-name")

    assert first.run_dir != second.run_dir
    assert first.run_dir.relative_to(first.entry) == second.run_dir.relative_to(second.entry)
