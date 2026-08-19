from __future__ import annotations

from pathlib import Path

import pytest

from mammoth.core import RunLayout, derive_run_name, validate_run_name


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "../escape", "nested/run", "/absolute", "white space", "-prefix"],
)
def test_run_name_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_run_name(value)


def test_derive_run_name_is_stable_for_the_same_target(tmp_path: Path) -> None:
    target = tmp_path / "dataset.tif"

    first = derive_run_name("preprocess", target)
    second = derive_run_name("preprocess", target)

    assert first == second
    assert validate_run_name(first) == first


def test_derive_run_name_distinguishes_same_stem_different_paths(tmp_path: Path) -> None:
    first_target = tmp_path / "one" / "dataset.tif"
    second_target = tmp_path / "two" / "dataset.tif"

    first = derive_run_name("preprocess", first_target)
    second = derive_run_name("preprocess", second_target)

    assert first != second
    assert first.startswith("preprocess-dataset.tif-")
    assert second.startswith("preprocess-dataset.tif-")


def test_derive_run_name_never_embeds_the_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dataset.tif"

    name = derive_run_name("preprocess", target)

    assert str(target) not in name
    assert str(tmp_path) not in name


def test_derive_run_name_sanitizes_unsafe_stem_characters(tmp_path: Path) -> None:
    target = tmp_path / "weird name!!.tif"

    name = derive_run_name("preprocess", target)

    assert validate_run_name(name) == name
    assert "!" not in name
    assert " " not in name


def test_derive_run_name_falls_back_when_the_stem_is_empty(tmp_path: Path) -> None:
    target = tmp_path / "..."

    name = derive_run_name("preprocess", target)

    assert validate_run_name(name) == name
    assert name.startswith("preprocess-")
    assert name.count("-") == 1


def test_derive_run_name_rejects_an_empty_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        derive_run_name("", tmp_path / "dataset.tif")


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
