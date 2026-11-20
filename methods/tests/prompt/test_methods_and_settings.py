"""Method names, the settings and the command group of the prompt family."""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

from posteriorbench_methods.prompt import cli, methods, settings

RUNNER = CliRunner()


def test_names_of_the_methods() -> None:
    assert methods.parse("prompt-ebjc") == (methods.Selection("ebjc"), False)
    assert methods.parse("prompt-ce+composition") == (methods.Selection("ce"), True)
    assert methods.parse("mixture-eb16") == (methods.Mixture("eb", 16), False)
    assert methods.parse("mixture-uniform64+composition") == (methods.Mixture("uniform", 64), True)
    assert methods.parse("instruction+composition") == (methods.Instruction(), True)
    for name in ("prompt-ebjc", "prompt-hitrate", "mixture-eb4", "mixture-uniform1", "instruction"):
        assert methods.parse(name)[0].name == name


@pytest.mark.parametrize(
    "name", ["prompt-best", "mixture-eb0", "mixture-16", "ebjc", "instruction+", "prompt-EBJC"]
)
def test_other_names_are_refused(name: str) -> None:
    with pytest.raises(methods.MethodError):
        methods.parse(name)
    with pytest.raises(ValueError):
        settings.Settings(methods=(name,))


def test_the_defaults_are_the_main_comparison() -> None:
    defaults = settings.Settings()
    assert defaults.methods == settings.MAIN_COMPARISON
    assert defaults.pool_sizes == tuple(2**i for i in range(11))
    assert (defaults.selection_draws, defaults.ce_bank_size) == (16, 512)
    plain = [name for name in defaults.methods if not name.endswith(methods.COMPOSITION)]
    assert plain == [
        "prompt-ebjc",
        "prompt-hitrate",
        "prompt-iwae",
        "prompt-ce",
        "mixture-eb16",
        "instruction",
    ]


@pytest.mark.parametrize(
    "changes",
    [{"pool_sizes": (0,)}, {"pool_sizes": (4, 4)}, {"selection_draws": 0}, {"ce_bank_size": 0}],
)
def test_impossible_settings_are_refused(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(settings.Settings(), **changes)


def test_config_prints_the_settings_and_takes_changes() -> None:
    result = RUNNER.invoke(cli.app, ["config"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["methods"] == list(settings.MAIN_COMPARISON)
    result = RUNNER.invoke(
        cli.app,
        ["config", "--set", "pool_sizes=2048,131072", "--set", "methods=prompt-ebjc,mixture-eb16"],
    )
    assert result.exit_code == 0, result.output
    printed = json.loads(result.output)
    assert printed["pool_sizes"] == [2048, 131072]
    assert printed["methods"] == ["prompt-ebjc", "mixture-eb16"]
    result = RUNNER.invoke(cli.app, ["config", "--set", "methods=prompt-best"])
    assert result.exit_code != 0


def test_the_group_has_run_and_config() -> None:
    result = RUNNER.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output and "config" in result.output
