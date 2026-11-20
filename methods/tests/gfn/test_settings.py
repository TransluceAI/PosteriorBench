"""The settings of GFlowNet and its command group."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from typer.testing import CliRunner

from posteriorbench_methods.common import cli_options, run_record
from posteriorbench_methods.gfn import cli
from posteriorbench_methods.gfn import settings as gfn_settings


def test_the_checkpoint_schedule() -> None:
    settings = gfn_settings.Settings()
    saved = [update for update in range(1, 260) if settings.saves_checkpoint_after(update)]
    assert saved == [1, 2, 4, 8, 16, 20, 32, 64, 128, 160, 192, 224, 256]


@pytest.mark.parametrize(
    "changes",
    [
        {"budgets": (), "update_caps": ()},
        {"budgets": (2**19,)},  # not a training budget
        {"update_caps": (3,)},  # no checkpoint is saved after update 3
        {"update_caps": (144,)},  # after 128 only every 32nd update
        {"epsilon": 0.0},
        {"warmup_updates": 0},
        {"replay_probability": 1.5},
        {"temperature_low": 3.0},
        {"checkpoint_budget_counts": (4096, 1024)},
        {"generation_batch_size": 16},
    ],
)
def test_settings_that_cannot_run_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        gfn_settings.Settings(**changes)


def test_every_setting_can_be_set_from_the_command_line() -> None:
    changed = cli_options.load_settings(
        gfn_settings.Settings(),
        assignments=["budgets=1048576", "update_caps=", "replay_probability=0.5"],
    )
    assert changed == dataclasses.replace(
        gfn_settings.Settings(), budgets=(1_048_576,), update_caps=(), replay_probability=0.5
    )


def test_config_prints_the_settings() -> None:
    runner = CliRunner()
    printed = runner.invoke(cli.app, ["config", "--set", "learning_rate=3e-5"])
    assert printed.exit_code == 0, printed.output
    expected = dataclasses.replace(gfn_settings.Settings(), learning_rate=3e-5)
    assert json.loads(printed.output) == run_record.settings_json(expected)
    assert runner.invoke(cli.app, ["config", "--set", "update_caps=3"]).exit_code != 0
    assert runner.invoke(cli.app, ["run", "--help"]).exit_code == 0
