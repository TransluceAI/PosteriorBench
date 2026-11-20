"""`posteriorbench-methods steering`: `run` and `config`, and `find-refusal-direction` and
`build-vectors`, which write the vectors directory that `run` reads, on the benchmark's model or
with `--task` on a generalization task's original model.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from typing import Annotated

import typer
from posteriorbench import benchmark, original_model

from posteriorbench_methods.common import cli_options, data_source, errors, run_context, tracking
from posteriorbench_methods.generalization import tasks as generalization_tasks
from posteriorbench_methods.steering import (
    coherence,
    construction,
    construction_prompts,
    refusal,
    run,
    settings,
)

ORIGINAL_BATCH_SIZE = 32  # responses in one forward pass of the original model
JUDGE_KEY_VARIABLE = "JUDGE_API_KEY"  # the environment variable of the judge's key

app = cli_options.family_app(
    "steering",
    help="Steering and Steering + composition, and building their steering vectors.",
    methods={
        "steering": run_context.MethodEntry(
            settings.Settings(),
            run.run,
            original_batch_size=ORIGINAL_BATCH_SIZE,
            check=run.check,
            generalization_tasks=True,
        )
    },
)

MethodDataDirectoryOption = Annotated[
    pathlib.Path,
    typer.Option(
        "--method-data",
        file_okay=False,
        help="The directory of steering vectors. The command writes into it and creates it if "
        "it does not exist; `run` reads it as --method-data.",
    ),
]
CHECKOUT_HELP = (
    f"A checkout of {refusal.REPOSITORY} at commit {refusal.COMMIT}, which holds the "
    "refusal-direction pipeline of Angell et al."
)
RUN_DEFAULTS = run_context.RunOptions  # its class attributes are the defaults of a run's options
GeneralizationTaskOption = Annotated[
    str | None,
    typer.Option(
        "--task",
        help="A task of the generalization section, e.g. qwen3.8-27b/wc_0670/chinese_character: "
        "the command then works on the task's original model. Default: the benchmark's model.",
    ),
]


def _generalization_task(task_id: str | None) -> generalization_tasks.GeneralizationTask | None:
    return None if task_id is None else generalization_tasks.get_task(task_id)


def _load_original_model(
    task: generalization_tasks.GeneralizationTask | None,
    model: str | None,
    model_revision: str | None,
    device: str | None,
) -> original_model.OriginalModel:
    """The original model as a run loads it: the benchmark's in float32, or `task`'s."""
    options = run_context.RunOptions(
        task_id="" if task is None else task.task_id,
        out=pathlib.Path(),
        model=model,
        model_revision=model_revision,
    )
    resolved = run_context.resolve_device(device)
    if task is None:
        return run_context.load_original_model(options, resolved, ORIGINAL_BATCH_SIZE)
    return run_context.load_task_original_model(task, options, resolved)


@app.command(
    "find-refusal-direction",
    help="Find the refusal direction of the original model with the pipeline of Angell et al. "
    "and write it into the directory of steering vectors, with the number of forward passes "
    "that the search took. `build-vectors` ablates this direction for one of the two vectors of "
    "a behavior.",
)
def find_refusal_direction_command(
    method_data: MethodDataDirectoryOption,
    lmtailrisk_checkout: Annotated[
        pathlib.Path,
        typer.Option("--lmtailrisk-checkout", exists=True, file_okay=False, help=CHECKOUT_HELP),
    ],
    task: GeneralizationTaskOption = None,
    model: cli_options.ModelOption = RUN_DEFAULTS.model,
    model_revision: cli_options.ModelRevisionOption = RUN_DEFAULTS.model_revision,
    device: cli_options.DeviceOption = RUN_DEFAULTS.device,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    with errors.OneLineErrors():
        # Fail on a checkout that cannot be imported before the model is loaded.
        refusal.import_from_checkout(lmtailrisk_checkout, "pipeline.submodules.select_direction")
        original = _load_original_model(_generalization_task(task), model, model_revision, device)
        record = refusal.find_and_save(original, lmtailrisk_checkout, method_data)
    typer.echo(json.dumps(record, indent=1))


@app.command(
    "build-vectors",
    help="Build the steering vectors of one behavior and write them, with what they cost to "
    "build, into the directory of steering vectors. The original model answers the questions of "
    "the behavior's construction prompts under instructions for and against the behavior; the "
    "behavior's rule and a coherence judge label the responses; a vector is the mean activation "
    "of the kept positive responses minus that of the kept negative ones. A behavior whose "
    "construction prompts hold a judge prompt is labeled by the judge in place of a rule.",
)
def build_vectors_command(
    prompts: Annotated[
        pathlib.Path,
        typer.Option(
            "--prompts",
            help="The file of the behavior's construction prompts: JSON with name, "
            "instruction_pairs and questions.",
        ),
    ],
    method_data: MethodDataDirectoryOption,
    judge_url: Annotated[
        str,
        typer.Option(
            "--judge-url",
            help="The coherence judge: the address of a server with the OpenAI chat-completions "
            "interface, such as http://localhost:8000/v1.",
        ),
    ],
    judge_model: Annotated[
        str, typer.Option("--judge-model", help="The judge's model name on that server.")
    ],
    behavior: Annotated[
        str | None,
        typer.Option(
            "--behavior",
            help="The behavior, as in the task ids of its tasks: goblin_mention. Not given with "
            "--task, which names the behavior.",
        ),
    ] = None,
    task: GeneralizationTaskOption = None,
    judge_key_env: Annotated[
        str,
        typer.Option(
            "--judge-key-env",
            help="The environment variable that holds the judge's key. If it is not set, no key "
            "is sent.",
        ),
    ] = JUDGE_KEY_VARIABLE,
    judge_concurrency: Annotated[
        int,
        typer.Option("--judge-concurrency", min=1, help="Responses judged at the same time."),
    ] = 100,
    lmtailrisk_checkout: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--lmtailrisk-checkout",
            exists=True,
            file_okay=False,
            help=CHECKOUT_HELP + " The vector with ablation needs its hooks, and the judge's "
            "prompt is read from it unless --coherence-prompt is given.",
        ),
    ] = None,
    coherence_prompt: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--coherence-prompt",
            exists=True,
            dir_okay=False,
            help="A text file with the judge's prompt, with the placeholders {question} and "
            "{answer}. Default: the prompt coherence_0_100 of the checkout.",
        ),
    ] = None,
    model: cli_options.ModelOption = RUN_DEFAULTS.model,
    model_revision: cli_options.ModelRevisionOption = RUN_DEFAULTS.model_revision,
    device: cli_options.DeviceOption = RUN_DEFAULTS.device,
    data: cli_options.DataOption = RUN_DEFAULTS.data,
    data_revision: cli_options.DataRevisionOption = RUN_DEFAULTS.data_revision,
    config: cli_options.ConfigOption = None,
    assignments: cli_options.SetOption = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        chosen = cli_options.load_settings(construction.Settings(), config, assignments or [])
    except ValueError as error:  # an unknown setting, or a value outside its range
        raise typer.BadParameter(str(error)) from error
    if (behavior is None) == (task is None):
        raise typer.BadParameter("give one of --behavior and --task")
    with errors.OneLineErrors():
        # Everything that can fail without the model fails before it is loaded.
        of_task = _generalization_task(task)
        behavior = of_task.behavior if of_task is not None else behavior
        assert behavior is not None
        behavior_prompts = construction_prompts.load(prompts, behavior)
        rule: str | None = None
        if behavior_prompts.judge_prompt is None:
            # The rule of a task of the generalization section is the benchmark's for its
            # behavior, so the vectors of either serve the tasks of both that the model has.
            rule = of_task.rule if of_task is not None else None
            if not rule:
                bench = benchmark.Benchmark(data, revision=data_revision)
                with data_source.reading(data):
                    rule = construction.behavior_rule(bench, behavior)
        judge_prompt = coherence.read_prompt(lmtailrisk_checkout, coherence_prompt)
        key = os.environ.get(judge_key_env) or None
        judge = coherence.ChatServer(judge_url, judge_model, key)
        model_name = model
        if model_name is None:
            model_name = original_model.DEFAULT_MODEL if of_task is None else of_task.model.name
        ablation, passes = construction.open_ablation(
            chosen, method_data, lmtailrisk_checkout, model_name
        )
        summary = construction.build(
            _load_original_model(of_task, model, model_revision, device),
            behavior_prompts,
            rule,
            judge,
            judge_prompt,
            directory=method_data,
            settings=chosen,
            ablation=ablation,
            refusal_direction_passes=passes,
            judge_model=judge_model,
            judge_concurrency=judge_concurrency,
            tracker=tracking.Tracker(labels={"family": "steering", "behavior": behavior}),
        )
    typer.echo(json.dumps(summary, indent=1))
