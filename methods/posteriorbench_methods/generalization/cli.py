"""The commands of the generalization tasks: `list-tasks` and `evaluate`.

`evaluate` scores a reference sampler (the original model, Instruction or the fallback). The
paper's methods run on these tasks through their families' commands, `posteriorbench-methods
<family> run --task TASK --data DIR`.
"""

from __future__ import annotations

import enum
import json
import pathlib
from typing import Annotated, Any

import typer
from posteriorbench import ledger, model_samplers, results

from posteriorbench_methods.common import errors
from posteriorbench_methods.generalization import instructions
from posteriorbench_methods.generalization import judge as judges
from posteriorbench_methods.generalization import task_set, tasks

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="The six tasks of the paper's section on generalization to larger models and "
    "complicated behaviors: listing them and evaluating a sampler.",
)


class SamplerChoice(enum.StrEnum):
    ORIGINAL = "original"
    INSTRUCTION = "instruction"
    FALLBACK = "fallback"


TaskOption = Annotated[str, typer.Option("--task", help="A task id of `list-tasks`")]
DataOption = Annotated[
    str, typer.Option("--data", help="The tasks' data: a directory or a Hub dataset id")
]
RevisionOption = Annotated[
    str | None, typer.Option("--data-revision", help="Revision of a Hub dataset id")
]
ModelOption = Annotated[
    str | None,
    typer.Option("--model", help="Another location of the task's original model (a directory)"),
]
ModelRevisionOption = Annotated[str | None, typer.Option(help="Revision of --model")]
DtypeOption = Annotated[
    str | None, typer.Option(help="float32, bfloat16 or float16 (default: the model's)")
]
DeviceOption = Annotated[str | None, typer.Option(help="Device (default: cuda if available)")]
BatchOption = Annotated[int | None, typer.Option(help="Responses scored per forward pass")]
GenerationBatchOption = Annotated[
    int | None, typer.Option(help="Responses generated per forward pass (the draws depend on it)")
]
JudgeUrlOption = Annotated[
    str | None,
    typer.Option(help="The rubric judge's address, such as http://localhost:8000/v1"),
]
JudgeModelOption = Annotated[str | None, typer.Option(help="The rubric judge's model name")]
JudgeKeyOption = Annotated[
    str | None, typer.Option(help="Environment variable with the rubric judge's key, if any")
]
ReviewUrlOption = Annotated[str | None, typer.Option(help="The reviewing model's address")]
ReviewModelOption = Annotated[str | None, typer.Option(help="The reviewing model's name")]
ReviewKeyOption = Annotated[
    str | None, typer.Option(help="Environment variable with the reviewing model's key")
]
LabelsOption = Annotated[
    pathlib.Path | None,
    typer.Option(help="The file that keeps labels, so that a response is judged once"),
]
ConcurrencyOption = Annotated[
    int, typer.Option("--judge-concurrency", min=1, help="Responses judged at the same time")
]


def _write(path: pathlib.Path | None, value: Any) -> None:
    text = json.dumps(value, indent=1, allow_nan=True)
    if path is None:
        typer.echo(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def _judge_models(
    judge_url: str | None,
    judge_model: str | None,
    judge_key_env: str | None,
    review_url: str | None,
    review_model: str | None,
    review_key_env: str | None,
) -> tuple[judges.ChatModel | None, judges.ChatModel | None]:
    """The two judge models from the command line; None where no address and name are given."""
    rubric_judge = (
        None
        if judge_url is None or judge_model is None
        else judges.rubric_judge_model(judge_url, judge_model, judge_key_env)
    )
    review = (
        None
        if review_url is None or review_model is None
        else judges.review_model(review_url, review_model, review_key_env)
    )
    return rubric_judge, review


@app.command("list-tasks")
@errors.OneLineErrors()
def list_tasks() -> None:
    """List the six tasks, one line each: id, original model, response limit, how the behavior
    is judged, hit count, p̂(S), references, and what an evaluation needs."""
    for task in tasks.all_tasks():
        needs = task.model.paper_name
        if task.rubric is not None:
            needs += ", a rubric judge and a reviewing model"
        typer.echo(
            f"{task.task_id}\t{task.model.paper_name}\t{task.max_response_tokens}\t"
            f"{task.judged_by}\t{task.hit_count}\t{task.probability:.3e}\t{task.references}\t"
            f"needs: {needs}"
        )


@app.command("evaluate")
@errors.OneLineErrors()
def evaluate(
    task_id: TaskOption,
    data_source: DataOption,
    sampler_kind: Annotated[
        SamplerChoice,
        typer.Option(
            "--sampler",
            help="original (the original model itself), instruction (the Instruction text as "
            "system prompt) or fallback (the fallback's scores; no model and no judge)",
        ),
    ],
    out: Annotated[
        pathlib.Path | None, typer.Option(help="Result JSON (stdout if omitted)")
    ] = None,
    replicate: Annotated[int, typer.Option(help="Replicate index (also the default seed)")] = 0,
    seed: Annotated[int | None, typer.Option(help="Evaluation seed (default: --replicate)")] = None,
    reference_logp: Annotated[
        str, typer.Option(help="Log p of the references: shipped or recomputed")
    ] = "shipped",
    data_revision: RevisionOption = None,
    model: ModelOption = None,
    model_revision: ModelRevisionOption = None,
    dtype: DtypeOption = None,
    device: DeviceOption = None,
    batch_size: BatchOption = None,
    generation_batch_size: GenerationBatchOption = None,
    judge_url: JudgeUrlOption = None,
    judge_model: JudgeModelOption = None,
    judge_key_env: JudgeKeyOption = None,
    review_url: ReviewUrlOption = None,
    review_model: ReviewModelOption = None,
    review_key_env: ReviewKeyOption = None,
    labels: LabelsOption = None,
    judge_concurrency: ConcurrencyOption = 32,
) -> None:
    """Evaluate a reference sampler on one task and write the result as JSON. A task with a
    rubric needs the two judge models; they are checked before the model is loaded."""
    if reference_logp not in results.REFERENCE_LOGP_CHOICES:
        raise typer.BadParameter(f"--reference-logp is one of {results.REFERENCE_LOGP_CHOICES}")
    choice: results.ReferenceLogp = "shipped" if reference_logp == "shipped" else "recomputed"
    rubric_judge, review = _judge_models(
        judge_url, judge_model, judge_key_env, review_url, review_model, review_key_env
    )
    gen = task_set.GeneralizationTasks(
        data_source,
        revision=data_revision,
        judge_model=rubric_judge,
        review_model=review,
        labels=labels,
        concurrency=judge_concurrency,
    )
    task = gen.task(task_id)
    evaluation_seed = replicate if seed is None else seed
    if sampler_kind is SamplerChoice.FALLBACK:
        result = gen.fallback_result(task, seed=evaluation_seed, replicate=replicate)
    else:
        gen.judge(task)  # fails before the model loads if a task with a rubric has no judges
        original = task_set.load_original_model(
            task,
            name=model,
            revision=model_revision,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            generation_batch_size=generation_batch_size,
        )
        candidate = (
            model_samplers.original_model_sampler(original, task)
            if sampler_kind is SamplerChoice.ORIGINAL
            else instructions.instruction_sampler(original, task)
        )
        result = gen.evaluate(
            task,
            candidate,
            original,
            reference_logp=choice,
            ledger=ledger.LedgerTotals.zero(),
            seed=evaluation_seed,
            replicate=replicate,
            method="original-model" if sampler_kind is SamplerChoice.ORIGINAL else "instruction",
        )
    _write(out, result.to_dict())
    if out is not None:
        typer.echo(
            f"{result.task_id} replicate {result.replicate}: native hit rate "
            f"{result.native_hit_rate:.3f}, capped forward KL {result.capped_forward_kl:.2f}, "
            f"reverse KL {result.reverse_kl:.2f} -> {out}",
            err=True,
        )


if __name__ == "__main__":
    app()
