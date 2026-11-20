"""The commands of the SFT family and its settings."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest
from typer.testing import CliRunner

from posteriorbench_methods.common import reselection, run_context, run_record
from posteriorbench_methods.sft import cli, reselect, settings

RUNNER = CliRunner()


def test_the_defaults_are_the_papers_settings() -> None:
    paper = settings.Settings()
    assert paper.budgets == tuple(2**exponent for exponent in range(12, 23))
    assert (paper.budgets[0], paper.budgets[-1], len(paper.budgets)) == (4096, 4194304, 11)
    assert (paper.training_share, paper.learning_rate) == (0.25, 1e-6)
    assert (paper.trainable_last_layers, paper.batch_size, paper.selection_draws) == (4, 32, 128)
    assert (paper.weight_decay, paper.gradient_clip) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("training_share", 1.0),
        ("training_share", 0.0),
        ("budgets", (0,)),
        ("learning_rate", 0.0),
        ("trainable_last_layers", 0),
        ("batch_size", 0),
        ("selection_draws", 0),
    ],
)
def test_settings_the_method_cannot_run_with_are_refused(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(settings.Settings(), **{name: value})


def test_config_prints_the_settings_that_run_would_use() -> None:
    result = RUNNER.invoke(cli.app, ["config"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["training_share"] == 0.25
    changed = RUNNER.invoke(
        cli.app, ["config", "--set", "budgets=1048576,2097152", "--set", "learning_rate=1e-5"]
    )
    printed = json.loads(changed.output)
    assert (printed["budgets"], printed["learning_rate"]) == ([1048576, 2097152], 1e-5)
    assert RUNNER.invoke(cli.app, ["config", "--set", "no_such_setting=1"]).exit_code != 0


def test_the_family_has_the_commands_run_config_and_reselect() -> None:
    result = RUNNER.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert all(command in result.output for command in ("run", "config", "reselect"))
    for command in ("run", "reselect"):
        result = RUNNER.invoke(cli.app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--task" in result.output and "--set" in result.output
    assert (
        RUNNER.invoke(cli.app, ["reselect", "--task", "t", "--out", "o", "--set", "x=1"]).exit_code
        != 0
    )


def test_reselect_runs_the_reselection_with_the_options_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    def run_method(*arguments: Any) -> list[run_record.RunSummary]:
        calls.append(arguments)
        return []

    monkeypatch.setattr(run_context, "run_method", run_method)
    result = RUNNER.invoke(
        cli.app,
        ["reselect", "--task", "wc_1853/goblin_mention", "--out", "runs", "--replicate", "2"]
        + ["--set", "budgets=524288"],
    )
    assert result.exit_code == 0, result.output
    ((family, method, entry, options, chosen),) = calls
    assert (family, method) == ("sft", "sft")
    assert entry.run is reselect.reselect and entry.original_batch_size == 32
    assert (options.task_id, options.out, options.replicate) == (
        "wc_1853/goblin_mention",
        pathlib.Path("runs"),
        2,
    )
    assert chosen == settings.Settings(budgets=(524288,))


def test_reselect_reports_a_missing_training_run_in_one_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_method(*arguments: Any) -> list[run_record.RunSummary]:
        raise reselection.NoTrainingRunError("runs/x holds no training run")

    monkeypatch.setattr(run_context, "run_method", run_method)
    result = RUNNER.invoke(cli.app, ["reselect", "--task", "wc_1853/goblin_mention", "--out", "x"])
    assert result.exit_code == 1
    assert result.output == "Error: runs/x holds no training run\n"
