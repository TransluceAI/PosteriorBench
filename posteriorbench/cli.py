"""Command-line interface: `posteriorbench <command> --help` (or `python -m posteriorbench`)."""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import functools
import importlib
import json
import pathlib
import time
from typing import Annotated, Any, ParamSpec, TypeVar

import typer

from posteriorbench import (
    _version,
    aggregate,
    benchmark,
    data,
    errors,
    estimators,
    ledger,
    model_samplers,
    original_model,
    results,
    sampler,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)


class Subset(enum.StrEnum):
    BENCHMARK = "benchmark"
    SCORED = "scored"
    ALL = "all"


class SamplerChoice(enum.StrEnum):
    ORIGINAL = "original"
    INSTRUCTION = "instruction"
    FALLBACK = "fallback"
    CUSTOM = "custom"


class ReferenceLogpChoice(enum.StrEnum):
    SHIPPED = "shipped"
    RECOMPUTED = "recomputed"
    BOTH = "both"


class Dtype(enum.StrEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


# The method label recorded for the package's own samplers unless --method is given.
REFERENCE_METHODS = {
    SamplerChoice.ORIGINAL: "original-model",
    SamplerChoice.INSTRUCTION: "instruction",
    SamplerChoice.FALLBACK: "fallback",
}

DataOption = Annotated[
    str | None,
    typer.Option(
        "--data",
        help="Local dataset directory or Hugging Face dataset id (default: the task table and "
        "the evaluation file bundled with the package)",
    ),
]
DataRevisionOption = Annotated[
    str | None, typer.Option("--revision", help="Dataset revision (Hub dataset ids only)")
]
Factory = collections.abc.Callable[..., Any]
P = ParamSpec("P")
R = TypeVar("R")


def _one_line_errors(command: collections.abc.Callable[P, R]) -> collections.abc.Callable[P, R]:
    """Report the package's own errors (a missing dataset file, an unknown or unscored task, a
    sampler that breaks the rules) as one line and exit with status 1."""

    @functools.wraps(command)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return command(*args, **kwargs)
        except errors.PosteriorBenchError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error

    return wrapped


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"posteriorbench {_version.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Evaluate samplers on PosteriorBench."""


def _write(path: pathlib.Path | None, value: Any) -> None:
    """Write `value` as JSON to `path` (stdout if None). Infinite values are written as JSON's
    Infinity / -Infinity."""
    text = json.dumps(value, indent=1, allow_nan=True)
    if path is None:
        typer.echo(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def _summary(result: results.EvaluationResult, path: pathlib.Path) -> str:
    """One line for the terminal: the scores of `result` and the file it was written to."""
    below = result.kind == "sampler" and result.fallback
    fallback = " (below the threshold: the fallback's scores)" if below else ""
    return (
        f"{result.task_id} replicate {result.replicate}: "
        f"native hit rate {result.native_hit_rate:.3f}{fallback}, "
        f"capped forward KL {result.capped_forward_kl:.2f}, reverse KL {result.reverse_kl:.2f}, "
        f"construction cost {result.construction_inferences:,.0f} inferences -> {path}"
    )


def _read_results(paths: list[pathlib.Path]) -> list[results.EvaluationResult]:
    """Results from JSON files (one result each) and JSON-lines files (one result per line; split
    on "\\n" only, since texts may hold other line separators)."""
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".jsonl":
            lines = path.read_text().split("\n")
            records.extend(json.loads(line) for line in lines if line.strip())
        else:
            records.append(json.loads(path.read_text()))
    try:
        return [results.EvaluationResult.from_dict(record) for record in records]
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


@app.command("list-tasks")
@_one_line_errors
def list_tasks(
    data_source: DataOption = None,
    revision: DataRevisionOption = None,
    subset: Annotated[
        Subset,
        typer.Option(
            help="benchmark (the benchmark's 21 tasks), scored (the 108 scored pairs) or all "
            "(all 140 pairs)"
        ),
    ] = Subset.BENCHMARK,
) -> None:
    """List tasks, one line per pair: id, hit count, p̂(S) and a flag.

    The flag is `benchmark_task` for one of the benchmark's 21 tasks, `scored` for another scored
    pair, and empty for an unscored pair."""
    bench = benchmark.Benchmark(data_source, revision=revision)
    for task in bench.tasks(subset.value):
        kind = "benchmark_task" if task.benchmark_task else "scored" if task.scored else ""
        typer.echo(f"{task.task_id}\t{task.hit_count}\t{task.probability:.3e}\t{kind}")


def _load_factory(factory: str) -> Factory:
    """The function named by `module:function`; the module must be importable."""
    if ":" not in factory:
        raise typer.BadParameter("--factory takes module:function")
    module_name, function_name = factory.split(":", 1)
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise typer.BadParameter(f"cannot load --factory {factory}: {error}") from error
    if not callable(function):
        raise typer.BadParameter(f"--factory {factory} is not a function")
    return function


def _build_custom(
    factory: Factory, name: str, task: data.Task, model: Any, bench: benchmark.Benchmark
) -> tuple[sampler.Sampler, ledger.LedgerTotals]:
    """Call the factory, `function(task=, model=, benchmark=)`, and check what it returns: the
    sampler and its construction ledger (or the ledger's totals)."""
    built = factory(task=task, model=model, benchmark=bench)
    if not (isinstance(built, tuple) and len(built) == 2):
        raise typer.BadParameter(f"{name} must return (sampler, ledger)")
    candidate, spend = built
    if not isinstance(candidate, sampler.Sampler):
        raise typer.BadParameter(f"{name} did not return a posteriorbench Sampler")
    if isinstance(spend, ledger.Ledger):
        spend = spend.totals()
    if not isinstance(spend, ledger.LedgerTotals):
        raise typer.BadParameter(f"{name} did not return a Ledger or LedgerTotals")
    return candidate, spend


def _candidate(
    kind: SamplerChoice,
    factory: Factory | None,
    factory_name: str,
    task: data.Task,
    model: original_model.OriginalModel,
    bench: benchmark.Benchmark,
) -> tuple[sampler.Sampler, ledger.LedgerTotals]:
    """The sampler to evaluate and what building it cost (nothing for the reference samplers)."""
    if factory is not None:
        return _build_custom(factory, factory_name, task, model, bench)
    if kind is SamplerChoice.ORIGINAL:
        return model_samplers.original_model_sampler(model, task), ledger.LedgerTotals.zero()
    return model_samplers.instruction_sampler(model, task), ledger.LedgerTotals.zero()


@app.command("evaluate")
@_one_line_errors
def evaluate(
    task_id: Annotated[str, typer.Option("--task", help="Task id, e.g. wc_1853/goblin_mention")],
    sampler_kind: Annotated[
        SamplerChoice,
        typer.Option(
            "--sampler",
            help="original (the original model itself), instruction (the handwritten instruction "
            "as system prompt), fallback (the fallback's scores; no model is loaded) or custom "
            "(your sampler, built by --factory)",
        ),
    ],
    factory: Annotated[
        str | None,
        typer.Option(
            help="For --sampler custom: module:function, called as function(task=, model=, "
            "benchmark=), which returns (Sampler, Ledger); the module must be importable"
        ),
    ] = None,
    out: Annotated[
        pathlib.Path | None, typer.Option(help="Result JSON (stdout if omitted)")
    ] = None,
    reference_logp: Annotated[
        ReferenceLogpChoice,
        typer.Option(
            help="Log p of the references: shipped (the values stored with the dataset), "
            "recomputed (scored by --model), or both (one set of draws; needs --out and "
            "--out-recomputed)"
        ),
    ] = ReferenceLogpChoice.SHIPPED,
    out_recomputed: Annotated[
        pathlib.Path | None,
        typer.Option(help="Result JSON with the recomputed log p, for --reference-logp both"),
    ] = None,
    method: Annotated[str | None, typer.Option(help="Method name recorded in the result")] = None,
    setting: Annotated[
        str | None, typer.Option(help="Setting recorded in the result (e.g. K=16, B=1048576)")
    ] = None,
    replicate: Annotated[int, typer.Option(help="Replicate index (also the default seed)")] = 0,
    seed: Annotated[
        int | None, typer.Option(help="Evaluation seed in [0, 2^31) (default: --replicate)")
    ] = None,
    cap: Annotated[float, typer.Option(help="The cap C of the forward KL")] = estimators.CAP,
    details: Annotated[bool, typer.Option(help="Keep draws and per-reference terms")] = False,
    data_source: DataOption = None,
    revision: DataRevisionOption = None,
    hits_source: Annotated[
        str | None,
        typer.Option(
            "--hits",
            help="Where a custom sampler's factory reads corpus hits: a local directory that holds "
            "hits/<prompt_id>/<behavior>.parquet, or a Hub dataset id (default: --data if given, "
            "else the Hub dataset)",
        ),
    ] = None,
    model_name: Annotated[
        str, typer.Option("--model", help="Hub model id or local directory")
    ] = original_model.DEFAULT_MODEL,
    model_revision: Annotated[
        str | None,
        typer.Option(help="Model revision (default: the dataset's for the benchmark model)"),
    ] = None,
    dtype: Annotated[Dtype, typer.Option(help="Weights dtype")] = Dtype.FLOAT32,
    device: Annotated[str | None, typer.Option(help="Device (default: cuda if available)")] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Responses generated or scored per forward pass (the draws depend on it)"
        ),
    ] = 16,
) -> None:
    """Evaluate one sampler on one task and write the result as JSON."""
    both = reference_logp is ReferenceLogpChoice.BOTH
    custom = sampler_kind is SamplerChoice.CUSTOM
    if both and (out is None or out_recomputed is None or out == out_recomputed):
        raise typer.BadParameter("--reference-logp both needs --out and --out-recomputed")
    if not both and out_recomputed is not None:
        raise typer.BadParameter("--out-recomputed is for --reference-logp both")
    if both and sampler_kind is SamplerChoice.FALLBACK:
        raise typer.BadParameter("the fallback's scores do not use the log p of the references")
    if custom != (factory is not None):
        raise typer.BadParameter("--factory is required with --sampler custom, and only then")
    build = None if factory is None else _load_factory(factory)
    bench = benchmark.Benchmark(data_source, revision=revision, hits_source=hits_source)
    task = bench.task(task_id)
    task.require_scored()
    try:
        settings = results.EvaluationSettings(cap=cap)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    evaluation_seed = replicate if seed is None else seed
    method = method if method is not None else REFERENCE_METHODS.get(sampler_kind)
    started = time.time()
    written: dict[pathlib.Path | None, results.EvaluationResult]
    if sampler_kind is SamplerChoice.FALLBACK:
        result = bench.fallback_result(
            task, seed=evaluation_seed, replicate=replicate, settings=settings
        )
        written = {out: dataclasses.replace(result, method=method, setting=setting)}
    else:
        model = original_model.OriginalModel.from_pretrained(
            model_name,
            revision=model_revision,
            dtype=dtype.value,
            device=device,
            batch_size=batch_size,
        )
        candidate, spend = _candidate(sampler_kind, build, str(factory), task, model, bench)
        options: results.EvaluationOptions = {
            "ledger": spend,
            "seed": evaluation_seed,
            "replicate": replicate,
            "settings": settings,
            "method": method,
            "setting": setting,
            "keep_details": details,
        }
        if both:
            pair = bench.evaluate_both(task, candidate, model, **options)
            written = {out: pair["shipped"], out_recomputed: pair["recomputed"]}
        else:
            choice: results.ReferenceLogp = (
                "shipped" if reference_logp is ReferenceLogpChoice.SHIPPED else "recomputed"
            )
            written = {
                out: bench.evaluate(task, candidate, model, reference_logp=choice, **options)
            }
    run = {
        "threshold": str(settings.threshold),
        "reverse_hit_limit": settings.reverse_hit_limit,
        "seconds": round(time.time() - started, 1),
    }
    for path, result in written.items():
        _write(path, {**result.to_dict(), "run": run})
        if path is not None:
            typer.echo(_summary(result, path), err=True)


@app.command("aggregate")
@_one_line_errors
def aggregate_results(
    paths: Annotated[
        list[pathlib.Path],
        typer.Argument(metavar="RESULTS...", help="Result JSON files (or JSON-lines files)"),
    ],
    expect_replicates: Annotated[
        str | None,
        typer.Option(help="Replicates every task must have, comma-separated, e.g. 0,1,2"),
    ] = None,
    allow_unequal_replicates: Annotated[
        bool, typer.Option(help="Allow tasks with different numbers of replicates")
    ] = False,
    expect_tasks: Annotated[
        str | None,
        typer.Option(
            help='Tasks every group must cover: "benchmark" (the benchmark\'s 21 tasks) or '
            '"scored" (the 108 scored pairs), read from --data, or comma-separated task ids'
        ),
    ] = None,
    data_source: DataOption = None,
    out: Annotated[
        pathlib.Path | None, typer.Option(help="Summary JSON (stdout if omitted)")
    ] = None,
) -> None:
    """Average results over replicates within a task, then over tasks, for each (method, setting).

    Gives the mean and the standard error. Fails if a group mixes caps, shipped and recomputed
    reference log p, or draw counts, and, with --expect-replicates / --expect-tasks, if a task or
    replicate is missing."""
    expected = None
    if expect_replicates is not None:
        try:
            expected = {int(r) for r in expect_replicates.split(",") if r.strip()}
        except ValueError as error:
            raise typer.BadParameter("--expect-replicates takes integers, e.g. 0,1,2") from error
    tasks: set[str] | None = None
    if expect_tasks in ("benchmark", "scored"):
        bench = benchmark.Benchmark(data_source)
        subset: benchmark.TaskSubset = "benchmark" if expect_tasks == "benchmark" else "scored"
        tasks = {t.task_id for t in bench.tasks(subset)}
    elif expect_tasks is not None:
        tasks = {t.strip() for t in expect_tasks.split(",") if t.strip()}
    summaries: list[dict[str, Any]] = []
    for key, group in aggregate.group_results(_read_results(paths)).items():
        try:
            summary = aggregate.aggregate(
                group,
                require_equal_replicates=not allow_unequal_replicates,
                expected_replicates=expected,
                expected_tasks=tasks,
            )
        except ValueError as error:
            raise typer.BadParameter(f"method {key[0]}, setting {key[1]}: {error}") from error
        summaries.append({"method": key[0], "setting": key[1], **summary.to_dict()})
    _write(out, summaries)


if __name__ == "__main__":
    app()
