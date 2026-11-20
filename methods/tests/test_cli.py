"""The root command: every family's commands, with the same options and defaults in every family,
the rules by which the families that train choose a checkpoint again, the mistakes that a command
reports as one line, and the threshold test at a rank."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest
import typer.main
from typer.testing import CliRunner

from posteriorbench_methods import cli
from posteriorbench_methods.common import run_context, run_record

RUNNER = CliRunner()
FAMILIES = ("sft", "prompt", "steering", "gfn", "ctl", "dpg")
# The options of `run` that choose the method and its settings; the others are the run options.
METHOD_OPTIONS = {"method", "config", "assignments"}
# The rules of `reselect`, by family and method: the 1% threshold, then the method's own rules.
RULES: dict[str, dict[str, list[str]]] = {
    "sft": {"sft": ["threshold-1pct"]},
    "gfn": {"gflownet": ["threshold-1pct", "highest-mean-log-reward"]},
    "ctl": {"ctl": ["threshold-1pct", "final-checkpoint"]},
    "dpg": {
        "kl-dpg": ["threshold-1pct", "final-checkpoint"],
        "js-dpg": ["threshold-1pct", "final-checkpoint"],
        "gdc": ["threshold-1pct", "last-sampling-distribution", "final-policy"],
    },
}
# The run options that `reselect` does not have: it composes nothing, deletes no checkpoint
# file, trains nothing that it could send to wandb, and the methods that train read no data.
NOT_IN_RESELECT = {"compose", "keep_checkpoints", "wandb_project", "wandb_group", "method_data"}
# The commands with which a family writes the method data that its `run` reads.
METHOD_DATA_COMMANDS: dict[str, list[str]] = {
    "prompt": ["write-candidates"],
    "steering": ["find-refusal-direction", "build-vectors"],
}


def commands(family: str) -> dict[str, Any]:
    """The commands of a family's group, by name, as the command-line library holds them."""
    root: Any = typer.main.get_command(cli.app)
    return dict(root.commands[family].commands)


def test_the_root_command_has_every_family() -> None:
    assert tuple(cli.FAMILIES) == FAMILIES
    result = RUNNER.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert all(family in result.output for family in FAMILIES)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_family_has_the_commands_run_and_config(family: str) -> None:
    found = commands(family)
    # A family that trains has one more command, which chooses a checkpoint again.
    expected = ["run", "config"] + ["reselect"] * (family in RULES)
    assert sorted(found) == sorted(expected + METHOD_DATA_COMMANDS.get(family, []))
    assert {param.name for param in found["config"].params} == METHOD_OPTIONS
    result = RUNNER.invoke(cli.app, [family, "config"] + ["--method", "gdc"] * (family == "dpg"))
    assert result.exit_code == 0, result.output
    assert isinstance(json.loads(result.output), dict)


@pytest.mark.parametrize("family", FAMILIES)
def test_run_has_the_run_options_with_their_defaults(family: str) -> None:
    """One name, one meaning and one default for every option, in every family."""
    params = {param.name: param for param in commands(family)["run"].params}
    fields = {field.name: field for field in dataclasses.fields(run_context.RunOptions)}
    assert set(params) == (set(fields) - {"task_id"}) | {"task"} | METHOD_OPTIONS
    defaults: dict[str, Any] = {
        name: field.default
        for name, field in fields.items()
        if field.default is not dataclasses.MISSING
    }
    assert {name: params[name].default for name in defaults} == defaults
    assert defaults["compose"] is True and defaults["keep_checkpoints"] == "testable"
    assert params["task"].required and params["out"].required
    assert all(param.help for param in params.values())


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["sft", "run", "--task", "wc_9999/nope"], "Error: unknown task 'wc_9999/nope'"),
        (["prompt", "run"], "Error: prompt selection needs the file of candidate prompts"),
        (["steering", "run"], "Error: Steering reads its vectors from the method data"),
        (
            ["prompt", "run", "--no-compose", "--set", "methods=prompt-ebjc+composition"],
            "Error: no method to run",
        ),
        (
            ["prompt", "run", "--set", "methods=prompt-ebjc", "--set", "pool_sizes="],
            "Error: no method to run",
        ),
        # A task of the generalization section needs its data, and a task that a language
        # model judges its two judge models; a task of the package takes neither.
        (
            ["sft", "run", "--task", "gemma-4-31b-it/wc_0670/em_dash"],
            "Error: the data of the generalization tasks was not given",
        ),
        (
            ["sft", "run", "--task", "gemma-4-31b-it/wc_0670/goblin_mention"],
            "Error: unknown task 'gemma-4-31b-it/wc_0670/goblin_mention'",
        ),
        (
            ["sft", "run", "--task", "qwen3.5-0.8b/wc_0670/physical_embodiment", "--data", "."],
            "Error: qwen3.5-0.8b/wc_0670/physical_embodiment is judged by a language model",
        ),
        (
            ["sft", "run", "--judge-url", "http://localhost:1/v1", "--batch-size", "2"],
            "Error: wc_1853/goblin_mention is a task of the package",
        ),
        # The paper runs prompt selection, Steering and SFT on these tasks.
        (
            ["ctl", "run", "--task", "gemma-4-31b-it/wc_0670/em_dash", "--data", "."],
            "Error: ctl runs on the tasks of the package",
        ),
    ],
)
def test_a_mistake_is_reported_as_one_line_before_the_model_is_loaded(
    arguments: list[str], message: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_: Any) -> Any:
        raise AssertionError("the model is not loaded")

    monkeypatch.setattr(run_context, "load_original_model", no_model)
    monkeypatch.setattr(run_context, "load_task_original_model", no_model)
    if "--task" not in arguments:
        arguments = [*arguments, "--task", "wc_1853/goblin_mention"]
    result = RUNNER.invoke(cli.app, [*arguments, "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert result.output.startswith(message) and result.output.count("\n") == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("family", sorted(RULES))
def test_reselect_has_the_options_of_run_that_it_needs(family: str) -> None:
    """The same names, defaults and help as in `run`, and the rule, which is the 1% threshold
    unless given. The help of --out says where the records of `reselect` go."""
    found = commands(family)
    of_run = {param.name: param for param in found["run"].params}
    params = {param.name: param for param in found["reselect"].params}
    assert set(params) == (set(of_run) - NOT_IN_RESELECT) | {"rule"}
    for name in set(params) - {"rule"}:
        same = ("opts", "default", "required") + ("help",) * (name != "out")
        assert all(getattr(params[name], key) == getattr(of_run[name], key) for key in same), name
    assert "<out>/reselected/<rule>/<method>/<setting>/<task>/rep<r>/" in params["out"].help
    assert params["rule"].default == "threshold-1pct" and params["rule"].help
    listed = " ".join(RUNNER.invoke(cli.app, [family, "reselect", "--help"]).output.split())
    assert all(rule in listed for rules in RULES[family].values() for rule in rules)
    # The sentence that names the rules: one rule, several, or those of each method.
    sentence = {
        "sft": "The only rule is threshold-1pct.",
        "gfn": "The rules are threshold-1pct and highest-mean-log-reward.",
        "ctl": "The rules are threshold-1pct and final-checkpoint.",
        "dpg": "The rules of each method are kl-dpg: threshold-1pct, final-checkpoint; js-dpg: "
        "threshold-1pct, final-checkpoint; gdc: threshold-1pct, last-sampling-distribution, "
        "final-policy.",
    }[family]
    assert sentence in found["reselect"].help


@pytest.mark.parametrize(
    ("family", "method"), [(family, method) for family in RULES for method in RULES[family]]
)
def test_reselect_runs_one_of_the_rules_of_the_method(
    family: str, method: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []

    def run_method(*arguments: Any) -> list[Any]:
        calls.append(arguments)
        return []

    monkeypatch.setattr(run_context, "run_method", run_method)
    arguments = [family, "reselect", "--task", "wc_1853/goblin_mention", "--out", str(tmp_path)]
    arguments += ["--method", method] * (len(RULES[family]) > 1)
    functions: list[Any] = []
    for rule in [None, *RULES[family][method]]:
        result = RUNNER.invoke(cli.app, arguments + ([] if rule is None else ["--rule", rule]))
        assert result.exit_code == 0, result.output
        _, name, entry, options, _ = calls[-1]
        assert name == method and options.task_id == "wc_1853/goblin_mention"
        functions.append(entry.run)
    # Without --rule the rule is the 1% threshold; every rule has a function of its own.
    assert functions[0] is functions[1]
    assert len(set(functions[1:])) == len(RULES[family][method])
    assert all(function.__module__.endswith(f".{family}.reselect") for function in functions)
    # A rule that the method does not have is refused, with the rules that it has.
    other = "final-policy" if method != "gdc" else "final-checkpoint"
    refused = RUNNER.invoke(cli.app, [*arguments, "--rule", other])
    assert refused.exit_code == 2 and len(calls) == len(functions)
    message = " ".join(refused.output.replace("│", " ").split())
    assert f"unknown rule '{other}'" in message
    assert all(rule in message for rule in RULES[family][method])


@pytest.mark.parametrize("family", sorted(RULES))
def test_reselect_without_a_training_run_is_one_line_before_the_model_is_loaded(
    family: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_: Any) -> Any:
        raise AssertionError("the model is not loaded")

    monkeypatch.setattr(run_context, "load_original_model", no_model)
    method = next(iter(RULES[family]))
    arguments = [family, "reselect", "--task", "wc_1853/goblin_mention", "--out", str(tmp_path)]
    arguments += ["--method", method] * (len(RULES[family]) > 1)
    result = RUNNER.invoke(cli.app, arguments)
    directory = run_record.training_run_dir(tmp_path, method, "wc_1853/goblin_mention", 0)
    assert result.exit_code == 1
    assert result.output.startswith(f"Error: {directory} holds no training run")
    assert result.output.count("\n") == 1 and not list(tmp_path.iterdir())


def all_options(command: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], set[str]]:
    """The options of every command below `command`, by the words that name the command."""
    if hasattr(command, "commands"):
        found: dict[tuple[str, ...], set[str]] = {}
        for name, member in command.commands.items():
            found.update(all_options(member, (*path, name)))
        return found
    return {path: {option for param in command.params for option in param.opts}}


def test_an_option_has_one_spelling_in_every_command() -> None:
    """The environment variable of a key is `--<whose>-key-env`, and the number of responses
    judged at the same time is `--judge-concurrency`."""
    options = all_options(typer.main.get_command(cli.app))
    assert "--writer-key-env" in options[("prompt", "write-candidates")]
    assert {"--judge-key-env", "--judge-concurrency"} <= options[("steering", "build-vectors")]
    judged = {"--judge-key-env", "--review-key-env", "--judge-concurrency"}
    assert judged <= options[("generalization", "evaluate")]
    for family in FAMILIES:
        assert judged <= options[(family, "run")]
    every: set[str] = set().union(*options.values())
    assert {option for option in every if "key" in option} == {
        "--writer-key-env",
        "--judge-key-env",
        "--review-key-env",
    }
    assert {option for option in every if "concurrency" in option} == {"--judge-concurrency"}


def test_threshold_test_command() -> None:
    result = RUNNER.invoke(cli.app, ["threshold-test", "--rank", "1"])
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert (record["rank"], record["draws"], record["required_hits"]) == (1, 164, 25)
    assert RUNNER.invoke(cli.app, ["threshold-test", "--rank", "0"]).exit_code != 0
