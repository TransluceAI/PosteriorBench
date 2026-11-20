"""The settings of KL-DPG, JS-DPG and GDC, and the commands of the family."""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

from posteriorbench_methods.dpg import cli, settings

RUNNER = CliRunner()
PAPER_BUDGETS = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)


def test_the_defaults_are_the_papers_settings() -> None:
    fdpg, gdc = settings.FdpgSettings(), settings.GdcSettings()
    for paper in (fdpg, gdc):
        assert paper.budgets == PAPER_BUDGETS and paper.training_budget == 2**20
        assert (paper.learning_rate, paper.warmup_updates) == (1.41e-5, 100)
        assert (paper.adam_beta1, paper.adam_beta2, paper.adam_epsilon) == (0.9, 0.999, 1e-8)
        assert paper.selection_draws == 128
    assert (fdpg.objective, fdpg.responses_per_update) == ("kl", 256)
    assert (fdpg.baseline_decay, fdpg.baseline_group) == (0.99, 64)
    assert (gdc.responses_per_update, gdc.loss_divisor) == (2048, 256)
    assert (gdc.fit_check_interval, gdc.fit_check_draws) == (10, 5120)


def test_the_settings_name_their_method() -> None:
    assert settings.FdpgSettings(objective="kl").method == "kl-dpg"
    assert settings.FdpgSettings(objective="js").method == "js-dpg"
    assert settings.GdcSettings().method == "gdc"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("objective", "tv"),
        ("budgets", ()),
        ("budgets", (4096, 4096)),
        ("budgets", (1912,)),  # nothing left after the threshold tests
        ("budgets", (2**21,)),  # beyond the training budget
        ("training_budget", 2000),  # does not pay for the first checkpoint
        ("responses_per_update", 0),
        ("learning_rate", 0.0),
        ("warmup_updates", -1),
        ("adam_beta1", 1.0),
        ("baseline_decay", 1.0),
        ("baseline_group", 0),
        ("selection_draws", 0),
    ],
)
def test_fdpg_settings_the_method_cannot_run_with_are_refused(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(settings.FdpgSettings(), **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("budgets", (2**21,)),
        ("responses_per_update", 1000),  # not a multiple of loss_divisor
        ("loss_divisor", 0),
        ("fit_check_interval", 0),
        ("fit_check_draws", 0),
        ("adam_epsilon", 0.0),
    ],
)
def test_gdc_settings_the_method_cannot_run_with_are_refused(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(settings.GdcSettings(), **{name: value})


def test_config_prints_the_settings_that_run_would_use() -> None:
    printed = {
        method: json.loads(RUNNER.invoke(cli.app, ["config", "--method", method]).output)
        for method in ("kl-dpg", "js-dpg", "gdc")
    }
    assert printed["kl-dpg"]["objective"] == "kl" and printed["js-dpg"]["objective"] == "js"
    assert printed["gdc"]["fit_check_draws"] == 5120
    changed = RUNNER.invoke(
        cli.app,
        [
            "config",
            "--method",
            "gdc",
            "--set",
            "budgets=32768,65536",
            "--set",
            "learning_rate=4e-5",
        ],
    )
    values = json.loads(changed.output)
    assert (values["budgets"], values["learning_rate"]) == ([32768, 65536], 4e-5)
    unknown = RUNNER.invoke(cli.app, ["config", "--method", "gdc", "--set", "objective=js"])
    assert unknown.exit_code != 0


def test_the_family_has_the_commands_run_and_config_and_needs_a_method() -> None:
    result = RUNNER.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output and "config" in result.output
    assert RUNNER.invoke(cli.app, ["config"]).exit_code != 0
    assert RUNNER.invoke(cli.app, ["config", "--method", "f-dpg"]).exit_code != 0
