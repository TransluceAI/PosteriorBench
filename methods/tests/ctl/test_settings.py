"""The settings of CTL and the command that prints them."""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

from posteriorbench_methods.ctl import cli
from posteriorbench_methods.ctl import settings as ctl_settings

RUNNER = CliRunner()


def test_the_checkpoint_schedule() -> None:
    settings = ctl_settings.Settings()
    scheduled = [update for update in range(300) if settings.saves_checkpoint_after(update)]
    assert scheduled == [1, 2, 4, 8, 16, 32, 64, 128, 160, 192, 224, 256, 288]


def test_update_caps_are_checkpoints_of_the_training_run() -> None:
    with pytest.raises(ValueError, match="checkpoint schedule"):
        ctl_settings.Settings(update_caps=(3,))
    # Under one budget of 100,000 passes, training ends after update 22.
    with pytest.raises(ValueError, match="beyond the last update"):
        ctl_settings.Settings(budgets=(100_000,), training_budgets=(100_000,))
    ctl_settings.Settings(budgets=(100_000,), training_budgets=(100_000,), update_caps=(4, 8, 16))


def test_budgets_are_training_budgets() -> None:
    with pytest.raises(ValueError, match="not among the training budgets"):
        ctl_settings.Settings(budgets=(2**19,))
    assert ctl_settings.Settings(budgets=(2**20,)).training_budgets == (2**20, 2**21, 2**22)


@pytest.mark.parametrize(
    "changes",
    [
        {"budgets": (), "update_caps": ()},
        {"training_budgets": ()},
        {"training_budgets": (1912,), "budgets": (1912,), "update_caps": ()},
        {"responses_per_update": 0},
        {"selection_draws": 0},
        {"learning_rate": 0.0},
        {"epsilon": 0.0},
        {"checkpoint_updates": ()},
        {"checkpoint_interval": 0},
    ],
)
def test_settings_that_cannot_run_are_refused(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(
            ctl_settings.Settings(), **changes
        )  # pyright: ignore[reportArgumentType]


def test_config_prints_the_settings_and_set_changes_them() -> None:
    printed = RUNNER.invoke(cli.app, ["config"])
    assert printed.exit_code == 0, printed.output
    assert json.loads(printed.output) == json.loads(
        json.dumps(dataclasses.asdict(ctl_settings.Settings()))
    )
    changed = RUNNER.invoke(
        cli.app,
        ["config", "--set", "budgets=", "--set", "update_caps=4,8", "--set", "epsilon=1e-8"],
    )
    assert changed.exit_code == 0, changed.output
    values = json.loads(changed.output)
    assert (values["budgets"], values["update_caps"], values["epsilon"]) == ([], [4, 8], 1e-8)
    assert RUNNER.invoke(cli.app, ["config", "--set", "update_caps=3"]).exit_code != 0
