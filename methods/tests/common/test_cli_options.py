"""`cli_options`: settings from --config and --set, and the command group of a family."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import types
from typing import Any, Literal

import posteriorbench
import pytest
import typer
from typer.testing import CliRunner

from posteriorbench_methods.common import cli_options, errors, run_context, run_record

RUNNER = CliRunner()


@dataclasses.dataclass(frozen=True)
class ToySettings:
    """Settings of a toy method (every supported field type)."""

    learning_rate: float = 1e-6
    batch: int = 32
    budgets: tuple[int, ...] = (1048576, 2097152)
    signal: Literal["ebjc", "jc"] = "ebjc"
    compose_plain: bool = True
    cap: int | None = None
    name: str = "toy"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def test_load_settings_from_assignments_and_file(tmp_path: pathlib.Path) -> None:
    changed = cli_options.load_settings(
        ToySettings(),
        assignments=["learning_rate=3e-7", "budgets=4096,8192", "signal=jc", "cap=none",
                     "compose_plain=false", "batch=8"],
    )  # fmt: skip
    assert changed == ToySettings(3e-7, 8, (4096, 8192), "jc", False, None)
    assert cli_options.load_settings(ToySettings(), assignments=["budgets="]).budgets == ()
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"learning_rate": 1, "budgets": [16384], "cap": 7}))
    from_file = cli_options.load_settings(ToySettings(), path, ["batch=4"])
    assert (from_file.learning_rate, from_file.budgets, from_file.cap, from_file.batch) == (
        1.0, (16384,), 7, 4
    )  # fmt: skip
    assert run_record.settings_json(from_file)["budgets"] == [16384]


@pytest.mark.parametrize(
    "assignment", ["unknown=1", "batch=1.5", "signal=iwae", "compose_plain=maybe", "batch"]
)
def test_bad_assignments_raise(assignment: str) -> None:
    with pytest.raises(cli_options.SettingsError):
        cli_options.load_settings(ToySettings(), assignments=[assignment])


def test_dataclass_checks_run() -> None:
    with pytest.raises(ValueError, match="positive"):
        cli_options.load_settings(ToySettings(), assignments=["learning_rate=0"])


def toy_app(captured: dict[str, Any]) -> Any:
    def run(context: run_context.RunContext, settings: ToySettings) -> list[run_record.RunSummary]:
        raise AssertionError("not called: run_method is replaced in these tests")

    entry = run_context.MethodEntry(ToySettings(), run)
    return cli_options.family_app(
        "toy", help="A toy family.", methods={"toy": entry, "toy-b": entry}
    )


def test_a_family_whose_methods_have_rules_gets_the_command_reselect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """`reselect` runs the function of the rule that --rule names, the 1% threshold unless given,
    after it has checked that the output root holds a training run of the method."""
    group: Any = typer.main.get_command(toy_app({}))
    assert sorted(group.commands) == ["config", "run"]
    ran: list[tuple[str, ToySettings]] = []

    def rule(name: str) -> Any:
        def choose(
            context: run_context.RunContext, settings: ToySettings
        ) -> list[run_record.RunSummary]:
            ran.append((name, settings))
            return []

        return choose

    def run(context: run_context.RunContext, settings: ToySettings) -> list[run_record.RunSummary]:
        raise AssertionError("reselect does not run the method")

    def check(options: run_context.RunOptions, settings: ToySettings) -> None:
        raise AssertionError("reselect does not make the check of `run`")

    rules = {"threshold-1pct": rule("threshold-1pct"), "final-checkpoint": rule("own")}
    entry = run_context.MethodEntry(ToySettings(), run, check=check, reselect=rules)
    app = cli_options.family_app("toy", help="A toy family.", methods={"toy": entry})
    group = typer.main.get_command(app)
    assert sorted(group.commands) == ["config", "reselect", "run"]
    listed = " ".join(group.commands["reselect"].help.split())
    assert "The rules are threshold-1pct and final-checkpoint." in listed
    opened: list[run_context.RunOptions] = []

    def open_context(family: str, method: str, entry: Any, options: Any, settings: Any) -> Any:
        opened.append(options)
        return types.SimpleNamespace(tracker=types.SimpleNamespace(finish=lambda: None))

    monkeypatch.setattr(run_context, "open_context", open_context)
    arguments = ["reselect", "--task", "wc_1853/goblin_mention", "--out", str(tmp_path)]
    missing = RUNNER.invoke(app, arguments)
    directory = run_record.training_run_dir(tmp_path, "toy", "wc_1853/goblin_mention", 0)
    assert missing.exit_code == 1 and not opened and not ran
    assert missing.output.startswith(f"Error: {directory} holds no training run")
    directory.mkdir(parents=True)
    assert RUNNER.invoke(app, arguments).exit_code == 0
    changed = [*arguments, "--rule", "final-checkpoint", "--set", "batch=4", "--replicate", "1"]
    assert RUNNER.invoke(app, changed).exit_code == 1  # replicate 1 has no training run
    run_record.training_run_dir(tmp_path, "toy", "wc_1853/goblin_mention", 1).mkdir()
    assert RUNNER.invoke(app, changed).exit_code == 0
    assert ran == [
        ("threshold-1pct", ToySettings()),
        ("own", dataclasses.replace(ToySettings(), batch=4)),
    ]
    assert [options.replicate for options in opened] == [0, 1]
    assert opened[0] == run_context.RunOptions(task_id="wc_1853/goblin_mention", out=tmp_path)
    refused = RUNNER.invoke(app, [*arguments, "--rule", "no-such-rule"])
    assert refused.exit_code == 2
    assert "unknown rule 'no-such-rule'" in " ".join(refused.output.replace("│", " ").split())


def test_group_without_arguments_prints_help() -> None:
    result = RUNNER.invoke(toy_app({}), [])
    assert "run" in result.output and "config" in result.output


def test_run_has_an_option_for_every_run_option_with_help() -> None:
    group: Any = typer.main.get_command(toy_app({}))
    params = group.commands["run"].params
    names = {opt for param in params for opt in param.opts}
    fields = {field.name for field in dataclasses.fields(run_context.RunOptions)}
    assert {param.name for param in params} >= fields - {"task_id"}
    assert {"--task", "--out", "--compose", "--method", "--config", "--set"} <= names
    assert all(param.help for param in params)
    assert RUNNER.invoke(toy_app({}), ["run", "--help"]).exit_code == 0


def test_config_prints_effective_settings() -> None:
    result = RUNNER.invoke(toy_app({}), ["config", "--method", "toy", "--set", "batch=4"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["batch"] == 4
    missing = RUNNER.invoke(toy_app({}), ["config"])
    assert missing.exit_code != 0  # two methods: --method is required


def test_run_builds_options_with_the_defaults_of_run_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_run_method(
        family: str, method: str, entry: Any, options: run_context.RunOptions, settings: Any
    ) -> list[run_record.RunSummary]:
        seen.update(family=family, method=method, options=options, settings=settings)
        return []

    monkeypatch.setattr(run_context, "run_method", fake_run_method)
    result = RUNNER.invoke(
        toy_app({}),
        ["run", "--task", "wc_1853/goblin_mention", "--out", str(tmp_path), "--method", "toy-b",
         "--set", "batch=2"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    options: run_context.RunOptions = seen["options"]
    assert seen["method"] == "toy-b" and seen["settings"].batch == 2
    assert options == run_context.RunOptions(task_id="wc_1853/goblin_mention", out=tmp_path)
    assert options.replicate == 0 and options.compose
    assert options.keep_checkpoints == "testable"
    assert options.evaluation_draws == 1024
    changed = RUNNER.invoke(
        toy_app({}),
        ["run", "--task", "wc_1853/goblin_mention", "--out", str(tmp_path), "--method", "toy",
         "--no-compose", "--keep-checkpoints", "all", "--replicate", "2"],
    )  # fmt: skip
    assert changed.exit_code == 0, changed.output
    options = seen["options"]
    assert (options.compose, options.keep_checkpoints, options.replicate) == (False, "all", 2)
    # Method data that does not exist is refused before anything is loaded.
    seen.clear()
    missing = RUNNER.invoke(
        toy_app({}),
        ["run", "--task", "wc_1853/goblin_mention", "--out", str(tmp_path), "--method", "toy",
         "--method-data", str(tmp_path / "no_such_file")],
    )  # fmt: skip
    assert missing.exit_code != 0 and not seen


def test_keep_checkpoints_shows_its_choices() -> None:
    choices = [choice.value for choice in cli_options.KeepCheckpointsChoice]
    assert choices == list(run_context.KEEP_CHECKPOINTS)
    group: Any = typer.main.get_command(toy_app({}))
    (option,) = [p for p in group.commands["run"].params if p.name == "keep_checkpoints"]
    assert list(option.type.choices) == choices  # what --help lists
    refused = RUNNER.invoke(
        toy_app({}),
        ["run", "--task", "t", "--out", "x", "--method", "toy", "--keep-checkpoints", "some"],
    )
    assert refused.exit_code == 2 and "'some' is not one of" in refused.output


@pytest.mark.parametrize(
    "error",
    [
        errors.MethodsError("no file of candidate prompts"),
        posteriorbench.PosteriorBenchError("unknown task 'wc_9999/nope'"),
    ],
)
def test_an_error_that_the_user_can_repair_is_one_line(
    error: Exception, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    def failing_run_method(*_: Any) -> list[run_record.RunSummary]:
        raise error

    monkeypatch.setattr(run_context, "run_method", failing_run_method)
    arguments = ["run", "--task", "wc_9999/nope", "--out", str(tmp_path), "--method", "toy"]
    result = RUNNER.invoke(toy_app({}), arguments)
    assert result.exit_code == 1
    assert result.output == f"Error: {error}\n"


def test_another_error_is_not_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    def failing_run_method(*_: Any) -> list[run_record.RunSummary]:
        raise ZeroDivisionError("a bug")

    monkeypatch.setattr(run_context, "run_method", failing_run_method)
    arguments = ["run", "--task", "t", "--out", str(tmp_path), "--method", "toy"]
    result = RUNNER.invoke(toy_app({}), arguments)
    assert isinstance(result.exception, ZeroDivisionError)


def test_run_options_checks() -> None:
    with pytest.raises(run_context.OptionsError):
        run_context.RunOptions(task_id="t", out=pathlib.Path("x"), replicate=-1)
    with pytest.raises(run_context.OptionsError):
        run_context.RunOptions(task_id="t", out=pathlib.Path("x"), keep_checkpoints="some")  # type: ignore
    assert run_context.RunOptions(task_id="t", out=pathlib.Path("x")).to_json()["out"] == "x"
