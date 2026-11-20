"""The command group of a family (`family_app`): `run`, `config`, and `reselect` for a family that
trains.

The options of `run` are those of `run_context.RunOptions` in every family. A method's settings
are a frozen dataclass whose defaults are the paper's settings; `--config FILE` (a JSON object) and
then `--set NAME=VALUE` change them, and `config` prints the result. A field of the settings may be
a bool, an int, a float, a str, a Literal of strs, or a tuple of ints, floats or strs, each
optionally None: `--set budgets=1048576,2097152`, `--set budgets=` (the empty tuple),
`--set update_cap=none`.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import pathlib
import sys
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypeVar

import typer

from posteriorbench_methods.common import errors, reselection, run_context, run_record

SettingsT = TypeVar("SettingsT")

TaskOption = Annotated[
    str,
    typer.Option(
        "--task",
        help="Task id of the package, e.g. wc_1853/goblin_mention, or of a task of the "
        "generalization section, e.g. gemma-4-31b-it/wc_0670/em_dash.",
    ),
]
OutOption = Annotated[
    pathlib.Path,
    typer.Option(
        "--out", help="Output root: records go to <out>/<method>/<setting>/<task>/rep<r>/."
    ),
]
ReselectOutOption = Annotated[
    pathlib.Path,
    typer.Option(
        "--out",
        help="Output root of the `run` whose training run is read; the records of `reselect` "
        "go to <out>/reselected/<rule>/<method>/<setting>/<task>/rep<r>/.",
    ),
]
ReplicateOption = Annotated[
    int,
    typer.Option(
        "--replicate",
        min=0,
        help="Replicate r: it enters the seed of every random stream that the replicates of a "
        "task do not share, and is the seed of the evaluation (the paper uses 0, 1 and 2).",
    ),
]
DataOption = Annotated[
    str | None,
    typer.Option(
        "--data",
        help="Dataset directory or Hub id. Default: the task table that comes with the package "
        "and the hit files of the dataset on the Hub. A task of the generalization section "
        "needs the data of those tasks here.",
    ),
]
DataRevisionOption = Annotated[
    str | None, typer.Option("--data-revision", help="Revision of a Hub dataset.")
]
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        help="Where the original model is loaded from: a Hub id or a local directory. Default: "
        "the benchmark's model, or the original model of a task of the generalization section.",
    ),
]
ModelRevisionOption = Annotated[
    str | None,
    typer.Option(
        "--model-revision",
        help="Commit of the model. Default: the benchmark's commit for the benchmark's model.",
    ),
]
DeviceOption = Annotated[
    str | None,
    typer.Option("--device", help="Torch device, e.g. cuda or cpu. Default: cuda if available."),
]
ComposeOption = Annotated[
    bool,
    typer.Option("--compose/--no-compose", help="Also run <method>+composition at every setting."),
]


class KeepCheckpointsChoice(str, enum.Enum):
    """`run_context.KeepCheckpoints` as choices of --keep-checkpoints."""

    ALL = "all"
    TESTABLE = "testable"
    RETURNED = "returned"


KeepCheckpointsOption = Annotated[
    KeepCheckpointsChoice,
    typer.Option(
        "--keep-checkpoints",
        help="Checkpoint files that remain after a run: all; testable, those that a threshold "
        "test can reach and the submitted ones; or returned, the submitted ones.",
    ),
]
EvaluationDrawsOption = Annotated[
    int,
    typer.Option(
        "--evaluation-draws",
        min=1,
        help="Evaluation draws of a sampler. The benchmark's number is 1,024; a result with "
        "another number is not a benchmark result.",
    ),
]
WandbProjectOption = Annotated[
    str | None,
    typer.Option(
        "--wandb-project",
        help="Send live metrics to this wandb project (entity/project); needs the wandb extra. "
        "Default: nothing is sent.",
    ),
]
WandbGroupOption = Annotated[str | None, typer.Option("--wandb-group", help="wandb run group.")]
MethodDataOption = Annotated[
    pathlib.Path | None,
    typer.Option(
        "--method-data",
        exists=True,
        help="The data the method reads, for the families that need some: the file of candidate "
        "prompts (Prompt selection), the directory of steering vectors (Steering).",
    ),
]
JudgeUrlOption = Annotated[
    str | None,
    typer.Option(
        "--judge-url",
        help="For a task that a language model judges: the address of the rubric judge, a "
        "server with the OpenAI chat-completions interface, e.g. http://localhost:8000/v1.",
    ),
]
JudgeModelOption = Annotated[
    str | None, typer.Option("--judge-model", help="The rubric judge's model name on its server.")
]
JudgeKeyEnvOption = Annotated[
    str | None,
    typer.Option(
        "--judge-key-env",
        help="The environment variable that holds the rubric judge's key, if it has one.",
    ),
]
ReviewUrlOption = Annotated[
    str | None,
    typer.Option(
        "--review-url",
        help="The address of the model that reviews each response that the rubric judge calls "
        "a match.",
    ),
]
ReviewModelOption = Annotated[
    str | None, typer.Option("--review-model", help="The reviewing model's name on its server.")
]
ReviewKeyEnvOption = Annotated[
    str | None,
    typer.Option(
        "--review-key-env", help="The environment variable that holds the reviewing model's key."
    ),
]
LabelsOption = Annotated[
    pathlib.Path | None,
    typer.Option(
        "--labels",
        dir_okay=False,
        help="The file that keeps the labels of the two judge models, so that a response is "
        "judged once. Default: they are kept for the run only.",
    ),
]
JudgeConcurrencyOption = Annotated[
    int,
    typer.Option(
        "--judge-concurrency",
        min=1,
        help="Responses judged at the same time.",
    ),
]
BatchSizeOption = Annotated[
    int | None,
    typer.Option(
        "--batch-size",
        min=1,
        help="For a task of the generalization section: the most responses in one forward pass "
        "that scores given responses. Default: that of the task's model and response limit.",
    ),
]
GenerationBatchSizeOption = Annotated[
    int | None,
    typer.Option(
        "--generation-batch-size",
        min=1,
        help="For a task of the generalization section: the most responses in one forward pass "
        "that generates. The responses that a seed draws depend on it. Default: that of the "
        "task's model and response limit.",
    ),
]
MethodOption = Annotated[
    str | None,
    typer.Option(
        "--method",
        help="The method to run, in the family dpg (kl-dpg, js-dpg or gdc). The prompt family "
        "takes its methods as a setting, --set methods=...",
    ),
]
RuleOption = Annotated[
    str,
    typer.Option(
        "--rule",
        help="The rule that chooses the checkpoint: threshold-1pct, the submission rule with a "
        "hit-rate threshold of 1% in place of 10%, or one of the method's own rules.",
    ),
]
ConfigOption = Annotated[
    pathlib.Path | None,
    typer.Option(
        "--config",
        exists=True,
        dir_okay=False,
        help="JSON object of the method settings to change, with the names that `config` prints.",
    ),
]
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        help="Change one method setting, NAME=VALUE (repeatable; applied after --config).",
    ),
]


class SettingsError(ValueError):
    """A settings file or a --set names an unknown setting or gives a value of the wrong type."""


def _is_optional(kind: Any) -> tuple[bool, Any]:
    """(True, X) for `X | None`, else (False, kind)."""
    if typing.get_origin(kind) in (typing.Union, types.UnionType):
        arguments = [a for a in typing.get_args(kind) if a is not type(None)]
        if len(arguments) == 1 and len(typing.get_args(kind)) == 2:
            return True, arguments[0]
    return False, kind


def _scalar_from_text(text: str, kind: Any, name: str) -> object:
    """A bool, int, float, str or Literal value from command-line text."""
    if kind is bool:
        lowered = text.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise SettingsError(f"{name}: expected true or false, got {text!r}")
    if kind is int:
        try:
            return int(text)
        except ValueError as error:
            raise SettingsError(f"{name}: expected an integer, got {text!r}") from error
    if kind is float:
        try:
            return float(text)
        except ValueError as error:
            raise SettingsError(f"{name}: expected a number, got {text!r}") from error
    if kind is str:
        return text
    if typing.get_origin(kind) is typing.Literal:
        choices = typing.get_args(kind)
        if text not in choices:
            raise SettingsError(f"{name}: expected one of {', '.join(choices)}; got {text!r}")
        return text
    raise SettingsError(f"{name}: settings of type {kind} cannot be set from the command line")


def value_from_text(text: str, kind: Any, name: str) -> object:
    """The value of setting `name` of type `kind` from the text of `--set NAME=VALUE`."""
    optional, inner = _is_optional(kind)
    if optional and text.strip().lower() in ("none", "null"):
        return None
    if typing.get_origin(inner) is tuple:
        arguments = typing.get_args(inner)
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise SettingsError(f"{name}: only tuples of one element type (tuple[T, ...])")
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return tuple(_scalar_from_text(part, arguments[0], name) for part in parts)
    return _scalar_from_text(text, inner, name)


def _scalar_from_json(value: object, kind: Any, name: str) -> object:
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind is bool and isinstance(value, bool):
        return value
    if kind in (int, float, str) and isinstance(value, kind) and not isinstance(value, bool):
        return value
    if typing.get_origin(kind) is typing.Literal and value in typing.get_args(kind):
        return value
    raise SettingsError(f"{name}: {value!r} is not a value of type {kind}")


def value_from_json(value: object, kind: Any, name: str) -> object:
    """The value of setting `name` of type `kind` from a JSON settings file (lists for tuples)."""
    optional, inner = _is_optional(kind)
    if optional and value is None:
        return None
    if typing.get_origin(inner) is tuple:
        arguments = typing.get_args(inner)
        if len(arguments) != 2 or arguments[1] is not Ellipsis or not isinstance(value, list):
            raise SettingsError(f"{name}: expected a list for {inner}")
        elements: list[object] = value  # pyright: ignore[reportUnknownVariableType]
        return tuple(_scalar_from_json(item, arguments[0], name) for item in elements)
    return _scalar_from_json(value, inner, name)


def load_settings(
    defaults: SettingsT,
    config_file: pathlib.Path | None = None,
    assignments: Sequence[str] = (),
) -> SettingsT:
    """`defaults` (an instance of a frozen dataclass) changed by a JSON settings file and then by
    NAME=VALUE assignments. The dataclass checks the result in its `__post_init__`."""
    if not dataclasses.is_dataclass(defaults) or isinstance(defaults, type):
        raise SettingsError("settings are a dataclass instance")
    kinds = typing.get_type_hints(type(defaults))
    names = [field.name for field in dataclasses.fields(defaults)]
    changes: dict[str, object] = {}
    if config_file is not None:
        loaded: object = json.loads(config_file.read_text())
        if not isinstance(loaded, dict):
            raise SettingsError(f"{config_file}: a settings file holds one JSON object")
        mapping: dict[str, object] = loaded  # pyright: ignore[reportUnknownVariableType]
        for name, value in mapping.items():
            if name not in names:
                raise SettingsError(
                    f"unknown setting {name!r}; the settings are {', '.join(names)}"
                )
            changes[name] = value_from_json(value, kinds[name], name)
    for assignment in assignments:
        name, separator, text = assignment.partition("=")
        name = name.strip()
        if not separator:
            raise SettingsError(f"--set takes NAME=VALUE, got {assignment!r}")
        if name not in names:
            raise SettingsError(f"unknown setting {name!r}; the settings are {', '.join(names)}")
        changes[name] = value_from_text(text, kinds[name], name)
    return dataclasses.replace(defaults, **changes)  # pyright: ignore[reportArgumentType]


def choose_method(methods: Mapping[str, run_context.MethodEntry[Any]], method: str | None) -> str:
    """The method to run: `method`, or the family's only method."""
    if method is None:
        if len(methods) == 1:
            return next(iter(methods))
        raise typer.BadParameter(f"--method is required: one of {', '.join(sorted(methods))}")
    if method not in methods:
        raise typer.BadParameter(
            f"unknown method {method!r}; the methods are {', '.join(sorted(methods))}"
        )
    return method


def _settings_or_exit(
    entry: run_context.MethodEntry[Any], config: pathlib.Path | None, assignments: list[str] | None
) -> Any:
    try:
        return load_settings(entry.settings, config, assignments or [])
    except (SettingsError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


def run_and_print(
    family: str,
    method: str,
    entry: run_context.MethodEntry[Any],
    options: run_context.RunOptions,
    settings: Any,
) -> None:
    """Run a method and print the summaries of its records as JSON; progress goes to stderr, and
    errors as one line (`errors.OneLineErrors`)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    with errors.OneLineErrors():
        summaries = run_context.run_method(family, method, entry, options, settings)
    typer.echo(json.dumps([summary.to_json() for summary in summaries], indent=1))


def family_app(
    family: str,
    *,
    help: str,
    methods: Mapping[str, run_context.MethodEntry[Any]],
) -> typer.Typer:
    """The command group of a family: `run`, `config`, and `reselect` if a method has rules to
    choose a checkpoint again. `methods` maps the name of each method to its entry."""
    if not methods:
        raise ValueError("a family has at least one method")
    app = typer.Typer(
        name=family,
        help=help,
        no_args_is_help=True,
        add_completion=False,
        pretty_exceptions_show_locals=False,
    )
    names = ", ".join(methods)
    defaults = run_context.RunOptions  # its class attributes are the defaults of the options

    @app.command("run", help=f"Run a method of this family ({names}) on one task and replicate.")
    def run_command(  # pyright: ignore[reportUnusedFunction]
        task: TaskOption,
        out: OutOption,
        method: MethodOption = None,
        replicate: ReplicateOption = defaults.replicate,
        data: DataOption = defaults.data,
        data_revision: DataRevisionOption = defaults.data_revision,
        model: ModelOption = defaults.model,
        model_revision: ModelRevisionOption = defaults.model_revision,
        device: DeviceOption = defaults.device,
        compose: ComposeOption = defaults.compose,
        keep_checkpoints: KeepCheckpointsOption = KeepCheckpointsChoice(defaults.keep_checkpoints),
        evaluation_draws: EvaluationDrawsOption = defaults.evaluation_draws,
        wandb_project: WandbProjectOption = defaults.wandb_project,
        wandb_group: WandbGroupOption = defaults.wandb_group,
        method_data: MethodDataOption = defaults.method_data,
        judge_url: JudgeUrlOption = defaults.judge_url,
        judge_model: JudgeModelOption = defaults.judge_model,
        judge_key_env: JudgeKeyEnvOption = defaults.judge_key_env,
        review_url: ReviewUrlOption = defaults.review_url,
        review_model: ReviewModelOption = defaults.review_model,
        review_key_env: ReviewKeyEnvOption = defaults.review_key_env,
        labels: LabelsOption = defaults.labels,
        judge_concurrency: JudgeConcurrencyOption = defaults.judge_concurrency,
        batch_size: BatchSizeOption = defaults.batch_size,
        generation_batch_size: GenerationBatchSizeOption = defaults.generation_batch_size,
        config: ConfigOption = None,
        assignments: SetOption = None,
    ) -> None:
        name = choose_method(methods, method)
        entry = methods[name]
        settings = _settings_or_exit(entry, config, assignments)
        options = run_context.RunOptions(
            task_id=task,
            out=out,
            replicate=replicate,
            data=data,
            data_revision=data_revision,
            model=model,
            model_revision=model_revision,
            device=device,
            compose=compose,
            keep_checkpoints=keep_checkpoints.value,
            evaluation_draws=evaluation_draws,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            method_data=method_data,
            judge_url=judge_url,
            judge_model=judge_model,
            judge_key_env=judge_key_env,
            review_url=review_url,
            review_model=review_model,
            review_key_env=review_key_env,
            labels=labels,
            judge_concurrency=judge_concurrency,
            batch_size=batch_size,
            generation_batch_size=generation_batch_size,
        )
        run_and_print(family, name, entry, options, settings)

    @app.command("config", help="Print a method's settings as JSON: the paper's, unless changed.")
    def config_command(  # pyright: ignore[reportUnusedFunction]
        method: MethodOption = None,
        config: ConfigOption = None,
        assignments: SetOption = None,
    ) -> None:
        entry = methods[choose_method(methods, method)]
        typer.echo(
            json.dumps(
                run_record.settings_json(_settings_or_exit(entry, config, assignments)), indent=1
            )
        )

    if any(entry.reselect for entry in methods.values()):
        _add_reselect_command(app, family, methods)
    return app


def rules_sentence(methods: Mapping[str, run_context.MethodEntry[Any]]) -> str:
    """The sentence of `reselect`'s help that names the rules of each method of the family."""
    with_rules = {name: list(entry.reselect) for name, entry in methods.items() if entry.reselect}
    if len(methods) > 1:
        listed = "; ".join(f"{name}: {', '.join(rules)}" for name, rules in with_rules.items())
        return f"The rules of each method are {listed}."
    (rules,) = with_rules.values()
    if len(rules) == 1:
        return f"The only rule is {rules[0]}."
    return f"The rules are {', '.join(rules[:-1])} and {rules[-1]}."


def _add_reselect_command(
    app: typer.Typer, family: str, methods: Mapping[str, run_context.MethodEntry[Any]]
) -> None:
    """Add `reselect`, which chooses the checkpoint of a finished training run again by `--rule`
    (`MethodEntry.reselect`; by default the 1% threshold). It checks that the output root holds a
    training run of the method before loading the task and the model."""
    defaults = run_context.RunOptions

    @app.command(
        "reselect",
        help="Choose the checkpoint of a finished training run again, by another rule, and "
        "evaluate it. Reads the training run under --out, which `run` must have written with "
        "--keep-checkpoints all and the same settings, and writes its records under "
        f"<out>/reselected/<rule>/. {rules_sentence(methods)}",
    )
    def reselect_command(  # pyright: ignore[reportUnusedFunction]
        task: TaskOption,
        out: ReselectOutOption,
        rule: RuleOption = reselection.THRESHOLD_1PCT,
        method: MethodOption = None,
        replicate: ReplicateOption = defaults.replicate,
        data: DataOption = defaults.data,
        data_revision: DataRevisionOption = defaults.data_revision,
        model: ModelOption = defaults.model,
        model_revision: ModelRevisionOption = defaults.model_revision,
        device: DeviceOption = defaults.device,
        evaluation_draws: EvaluationDrawsOption = defaults.evaluation_draws,
        judge_url: JudgeUrlOption = defaults.judge_url,
        judge_model: JudgeModelOption = defaults.judge_model,
        judge_key_env: JudgeKeyEnvOption = defaults.judge_key_env,
        review_url: ReviewUrlOption = defaults.review_url,
        review_model: ReviewModelOption = defaults.review_model,
        review_key_env: ReviewKeyEnvOption = defaults.review_key_env,
        labels: LabelsOption = defaults.labels,
        judge_concurrency: JudgeConcurrencyOption = defaults.judge_concurrency,
        batch_size: BatchSizeOption = defaults.batch_size,
        generation_batch_size: GenerationBatchSizeOption = defaults.generation_batch_size,
        config: ConfigOption = None,
        assignments: SetOption = None,
    ) -> None:
        name = choose_method(methods, method)
        entry = methods[name]
        if rule not in entry.reselect:
            raise typer.BadParameter(
                f"unknown rule {rule!r}; the rules of {name} are {', '.join(entry.reselect)}"
            )
        settings = _settings_or_exit(entry, config, assignments)
        options = run_context.RunOptions(
            task_id=task,
            out=out,
            replicate=replicate,
            data=data,
            data_revision=data_revision,
            model=model,
            model_revision=model_revision,
            device=device,
            evaluation_draws=evaluation_draws,
            judge_url=judge_url,
            judge_model=judge_model,
            judge_key_env=judge_key_env,
            review_url=review_url,
            review_model=review_model,
            review_key_env=review_key_env,
            labels=labels,
            judge_concurrency=judge_concurrency,
            batch_size=batch_size,
            generation_batch_size=generation_batch_size,
        )

        def check(options: run_context.RunOptions, settings: Any) -> None:
            reselection.require_training_run(options, name)

        chosen = dataclasses.replace(entry, run=entry.reselect[rule], check=check)
        run_and_print(family, name, chosen, options, settings)
