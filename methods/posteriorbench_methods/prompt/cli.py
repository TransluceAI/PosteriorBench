"""`posteriorbench-methods prompt`: the commands `run` and `config` of the prompt family, and
`write-candidates`, which writes the candidate prompts that `run` reads.

    posteriorbench-methods prompt write-candidates --task TASK [--task TASK ...] \\
        --pool-file FILE --candidates N --writer-url URL --writer-model NAME
    posteriorbench-methods prompt run --task TASK --out OUT --method-data FILE
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from typing import Annotated

import typer
from posteriorbench import benchmark

from posteriorbench_methods.common import cli_options, data_source, errors, run_context, tracking
from posteriorbench_methods.generalization import tasks as generalization_tasks
from posteriorbench_methods.prompt import candidate_writing, run, settings, writer

FAMILY = "prompt"
WRITER_KEY_VARIABLE = "WRITER_API_KEY"

app = cli_options.family_app(
    FAMILY,
    help="Prompt selection, prompt mixtures, Instruction, and their compositions.",
    methods={
        "prompt": run_context.MethodEntry(
            settings.Settings(),
            run.run,
            original_batch_size=64,
            check=run.check,
            generalization_tasks=True,
        )
    },
)


@app.command(
    "write-candidates",
    help="Write candidate system prompts for tasks with a writer LLM, into a pool file that "
    "`run` reads as --method-data. The writer is any server with the chat-completions interface "
    "of the OpenAI API. One call asks once for the candidates that each task lacks, and can "
    "write fewer; call it again to write the rest. Prints what it did for each task as JSON.",
)
def write_candidates_command(
    tasks: Annotated[
        list[str],
        typer.Option(
            "--task",
            help="Task id of the package, e.g. wc_1853/goblin_mention, or of a task of the "
            "generalization section (repeatable; the tasks are written one after another).",
        ),
    ],
    pool_file: Annotated[
        pathlib.Path,
        typer.Option(
            "--pool-file",
            dir_okay=False,
            help="The pool file. It is created if it does not exist; candidates of other tasks "
            "and earlier candidates of these tasks are kept.",
        ),
    ],
    candidates: Annotated[
        int,
        typer.Option(
            "--candidates",
            min=1,
            help="The number of candidates that each task should have in the pool file. The "
            "writer is asked for what a task lacks.",
        ),
    ],
    writer_url: Annotated[
        str,
        typer.Option(
            "--writer-url",
            help="The address of the writer's chat-completions interface, e.g. "
            "http://localhost:8000/v1; requests go to <address>/chat/completions.",
        ),
    ],
    writer_model: Annotated[
        str, typer.Option("--writer-model", help="The model name that the writer's server knows.")
    ],
    writer_key_env: Annotated[
        str,
        typer.Option(
            "--writer-key-env",
            help="The environment variable that holds the writer's key. If it is not set, no "
            "key is sent.",
        ),
    ] = WRITER_KEY_VARIABLE,
    new_batch: Annotated[
        bool,
        typer.Option(
            "--new-batch/--same-batch",
            help="Whether the candidates that this call adds to a task form a new batch, after "
            "the task's last one, or join its last batch. A new batch keeps the order that "
            "earlier runs gave the task's candidates.",
        ),
    ] = False,
    explanations: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--explanations",
            dir_okay=False,
            help="The file that keeps the writer's explanation of each behavior. Default: next "
            "to the pool file, with the ending .explanations.json.",
        ),
    ] = None,
    data: cli_options.DataOption = None,
    data_revision: cli_options.DataRevisionOption = None,
    assignments: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="Change one setting of the writing, NAME=VALUE (repeatable): "
            "scenarios_per_call, temperature, explanation_tokens, scenario_tokens, "
            "system_prompt_tokens, concurrency, attempts, timeout.",
        ),
    ] = None,
) -> None:
    try:
        writing = cli_options.load_settings(
            candidate_writing.Settings(), assignments=assignments or []
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    tracker = tracking.Tracker(labels={"family": FAMILY, "command": "write-candidates"})
    with errors.OneLineErrors():
        bench = benchmark.Benchmark(data, revision=data_revision)
        with data_source.reading(data):
            chosen = [
                (
                    generalization_tasks.get_task(task_id)
                    if generalization_tasks.has_task_id_form(task_id)
                    else bench.task(task_id)
                )
                for task_id in tasks
            ]
        llm = writer.ChatCompletionsWriter(
            writer_url,
            writer_model,
            key=writer.key_from_environment(writer_key_env),
            temperature=writing.temperature,
            attempts=writing.attempts,
            timeout=writing.timeout,
        )
        reports = [
            candidate_writing.add_to_pool_file(
                pool_file,
                task,
                llm,
                candidates=candidates,
                new_batch=new_batch,
                settings=writing,
                explanations=explanations,
                tracker=tracker,
            )
            for task in chosen
        ]
    typer.echo(json.dumps([report.to_json() for report in reports], indent=1))
