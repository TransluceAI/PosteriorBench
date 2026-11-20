"""The commands of the analyses of finished runs: compute-per-inference, time-operations,
cap-sensitivity and estimate-ps. Each prints JSON, or writes it to `--out`.
"""

from __future__ import annotations

import json
import pathlib
from typing import Annotated, Any

import typer
from posteriorbench import benchmark, original_model

from posteriorbench_methods.analysis import (
    cap_sensitivity,
    compute,
    flops,
    gpu_times,
    ps_estimate,
    records,
)
from posteriorbench_methods.common import cli_options, data_source, errors, hf_runtime, run_context

RecordsArgument = Annotated[
    list[pathlib.Path],
    typer.Argument(
        metavar="RECORDS...",
        help="Output roots, record directories or result.json files of finished runs.",
    ),
]
OutOption = Annotated[
    pathlib.Path | None, typer.Option("--out", help="Write the JSON to this file (default: print).")
]
RunsOption = Annotated[
    bool, typer.Option("--runs/--no-runs", help="Also list the values of every single run.")
]
# The task whose prompt the operations are timed on: its 24 tokens are the median length of the
# prompts of the benchmark's 21 tasks.
TIMED_TASK = "wc_0532/profanity"


def _write(out: pathlib.Path | None, value: Any) -> None:
    """Print `value` as JSON, or write it to `out`. An infinite value is written as Infinity."""
    text = json.dumps(value, indent=1, allow_nan=True)
    if out is None:
        typer.echo(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


ModelOption = Annotated[
    str, typer.Option("--model", help="The original model: a Hub id or a local directory.")
]


def compute_per_inference(
    paths: RecordsArgument,
    times: Annotated[
        pathlib.Path,
        typer.Option(
            "--times",
            show_default=False,  # the default is a file of the installed package
            help="The measured times that price GPU time: a file of `time-operations`. "
            "Default: the paper's measurements on one NVIDIA H100.",
        ),
    ] = gpu_times.PAPER_TIMES,
    flops_only: Annotated[
        bool, typer.Option("--flops-only", help="Leave out the GPU time.")
    ] = False,
    model: ModelOption = original_model.DEFAULT_MODEL,
    model_revision: cli_options.ModelRevisionOption = None,
    data: cli_options.DataOption = None,
    data_revision: cli_options.DataRevisionOption = None,
    runs: RunsOption = False,
    out: OutOption = None,
) -> None:
    """The FLOPs and the GPU time of finished runs for each inference that they are charged.

    Given for every point, a method at a setting, and for every method as the highest value among
    its points."""
    with errors.OneLineErrors():
        import transformers  # here, so that --help does not load it

        found = records.find(paths)
        revision = original_model.resolve_model_revision(model, model_revision)
        shape = flops.Shape.from_config(
            transformers.AutoConfig.from_pretrained(model, revision=revision)
        )
        bench = benchmark.Benchmark(data, revision=data_revision)
        with data_source.reading(data):
            computed, not_covered = compute.of_records(
                found,
                shape,
                prompt_tokens=lambda task_id: len(bench.task(task_id).prompt_token_ids),
                times=None if flops_only else gpu_times.read(times),
            )
        result: dict[str, Any] = {**compute.summarize(computed), "not_covered": not_covered}
        if runs:
            result["runs"] = [run.to_json() for run in computed]
        _write(out, result)


def time_operations_command(
    out: Annotated[pathlib.Path, typer.Option("--out", help="The times file to write.")],
    task: Annotated[
        str, typer.Option("--task", help="The task whose prompt the operations are timed on.")
    ] = TIMED_TASK,
    model: ModelOption = original_model.DEFAULT_MODEL,
    model_revision: cli_options.ModelRevisionOption = None,
    data: cli_options.DataOption = None,
    data_revision: cli_options.DataRevisionOption = None,
    device: cli_options.DeviceOption = None,
    repetitions: Annotated[
        int, typer.Option(min=1, help="Timed calls of each operation; a time is their median.")
    ] = 6,
    warmup: Annotated[
        int, typer.Option(min=0, help="Calls of each operation before the timed ones.")
    ] = 2,
) -> None:
    """Time the operations that `compute-per-inference` prices GPU time from, on this GPU.

    Writes them as a times file for the option --times of `compute-per-inference`."""
    with errors.OneLineErrors():
        from posteriorbench_methods.analysis import time_operations

        with data_source.reading(data):
            prompt = benchmark.Benchmark(data, revision=data_revision).task(task).prompt_token_ids

        def load() -> original_model.OriginalModel:
            return original_model.OriginalModel.from_pretrained(
                model,
                revision=model_revision,
                dtype="float32",
                device=run_context.resolve_device(device),
            )

        original = load()
        response_format = hf_runtime.ResponseFormat.of(original)
        conditions = {
            **original.describe(),
            "task_id": task,
            "prompt_tokens": len(prompt),
            "response_tokens": response_format.max_tokens,
            "warmup": warmup,
        }
        del original
        session = time_operations.measure(
            lambda: load().model,
            prompt,
            response_tokens=response_format.max_tokens,
            stop_token_ids=response_format.stop_token_ids,
            repetitions=repetitions,
            warmup=warmup,
            conditions=conditions,
        )
        gpu_times.write(out, [session])
        typer.echo(f"wrote {out}")


def cap_sensitivity_command(
    paths: RecordsArgument,
    caps: Annotated[
        str, typer.Option("--caps", help="The caps C of the forward KL, comma-separated.")
    ] = ",".join(f"{cap:g}" for cap in cap_sensitivity.CAPS),
    allow_unequal_replicates: Annotated[
        bool, typer.Option(help="Allow tasks with different numbers of replicates.")
    ] = False,
    runs: RunsOption = False,
    out: OutOption = None,
) -> None:
    """The capped forward KL of finished runs at other caps C.

    Averaged as `posteriorbench aggregate` averages, with the share of the references of passing
    samplers at each cap."""
    try:
        values = [float(cap) for cap in caps.split(",") if cap.strip()]
    except ValueError as error:
        raise typer.BadParameter("--caps takes numbers, e.g. 10,20,50,100,200") from error
    if not values or min(values) <= 0:
        raise typer.BadParameter("--caps takes at least one cap, and every cap is positive")

    with errors.OneLineErrors():
        analysis = cap_sensitivity.analyze(
            records.find(paths),
            values,
            require_equal_replicates=not allow_unequal_replicates,
            with_runs=runs,
        )
        _write(out, analysis)


def estimate_ps(
    paths: RecordsArgument,
    draws: Annotated[
        int,
        typer.Option(
            "--draws",
            min=1,
            help="The evaluation draws that the estimate reads, from the first one on. "
            "The paper's estimate reads 960 of the 1,024.",
        ),
    ] = ps_estimate.DRAWS,
    allow_unequal_replicates: Annotated[
        bool, typer.Option(help="Allow tasks with different numbers of replicates.")
    ] = False,
    samplers: Annotated[
        bool,
        typer.Option(
            "--samplers/--no-samplers", help="Also list every sampler that passed the threshold."
        ),
    ] = False,
    runs: RunsOption = False,
    out: OutOption = None,
) -> None:
    """The importance-sampling estimate of p(S) from the evaluation draws of finished runs.

    Gives the average absolute log error of every point, and the log error of every passing
    sampler with the scores of its evaluation."""
    with errors.OneLineErrors():
        analysis = ps_estimate.analyze(
            records.find(paths),
            draws,
            require_equal_replicates=not allow_unequal_replicates,
            with_runs=runs,
            with_samplers=samplers,
        )
        _write(out, analysis)


def register(app: typer.Typer) -> None:
    """Add the commands of the analyses to the root command."""
    app.command("compute-per-inference")(compute_per_inference)
    app.command("time-operations")(time_operations_command)
    app.command("cap-sensitivity")(cap_sensitivity_command)
    app.command("estimate-ps")(estimate_ps)
