"""The settings of Steering and its commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from posteriorbench_methods.steering import cli, settings

RUNNER = CliRunner()


def test_the_defaults_are_the_paper_settings() -> None:
    chosen = settings.Settings()
    assert chosen.layers is None and chosen.threshold == 0.10
    assert chosen.strengths == (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)
    assert chosen.selection_draws == 128


def test_the_default_layers_are_those_at_the_paper_depths_of_the_model() -> None:
    """Layer int(n d) - 1 at the depths d = 0.35, 0.40, ..., 0.60 of a model with n layers: of
    Qwen3.5-0.8B's 24, of Qwen3.8-27B's 64 and of Gemma-4-31B-it's 60."""
    chosen = settings.Settings()
    assert chosen.with_layers_of(24).layers == (7, 8, 9, 11, 12, 13)
    assert chosen.with_layers_of(64).layers == (21, 24, 27, 31, 34, 37)
    assert chosen.with_layers_of(60).layers == (20, 23, 26, 29, 32, 35)
    for layers in (24, 60, 64, 48, 100):
        by_float = tuple(int(layers * depth / 100) - 1 for depth in range(35, 65, 5))
        assert settings.layers_at_depths(layers) == by_float
    # Stated layers stay as they are, on any model.
    stated = settings.Settings(layers=(11, 12))
    assert stated.with_layers_of(64) is stated
    # A model with too few layers for six distinct ones needs its layers stated.
    with pytest.raises(settings.LayersError, match="state the layers"):
        chosen.with_layers_of(2)


@pytest.mark.parametrize(
    "changes",
    [
        {"layers": (7, 7)},
        {"layers": (-1,)},
        {"layers": ()},
        {"threshold": 0.05},
        {"strengths": (1.0, 1.0)},
        {"strengths": (0.0,)},
        {"strengths": (float("inf"),)},
        {"selection_draws": 0},
    ],
)
def test_settings_that_the_method_cannot_run_with_are_refused(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        settings.Settings(**changes)  # pyright: ignore[reportArgumentType]


def test_config_prints_the_settings_and_takes_changes() -> None:
    printed = RUNNER.invoke(cli.app, ["config"])
    assert printed.exit_code == 0, printed.output
    assert json.loads(printed.output) == {
        "layers": None,
        "strengths": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0],
        "selection_draws": 128,
        "threshold": 0.1,
    }
    changed = RUNNER.invoke(cli.app, ["config", "--set", "strengths=1,2.5", "--set", "layers=12"])
    assert changed.exit_code == 0, changed.output
    assert json.loads(changed.output)["strengths"] == [1.0, 2.5]
    assert json.loads(changed.output)["layers"] == [12]
    assert RUNNER.invoke(cli.app, ["config", "--set", "strengths=0"]).exit_code != 0


def test_the_command_group_has_run_and_config() -> None:
    printed = RUNNER.invoke(cli.app, ["--help"])
    assert printed.exit_code == 0 and "run" in printed.output and "config" in printed.output
    run_help = RUNNER.invoke(cli.app, ["run", "--help"])
    assert run_help.exit_code == 0 and "--method-data" in run_help.output
