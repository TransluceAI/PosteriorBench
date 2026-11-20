"""The method families on tasks beyond the benchmark: prompt selection, Steering and SFT run on
the toy tasks of this folder, one with a rule and one that the stand-in judge models label, with
forward passes smaller than the families' own, at the 10% and at the 1% threshold."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from posteriorbench import cli as package_cli
from posteriorbench import original_model
from typer.testing import CliRunner

from posteriorbench_methods.common import reselection, run_context, run_record, tracking
from posteriorbench_methods.generalization import task_set, tasks
from posteriorbench_methods.prompt import run as prompt_run
from posteriorbench_methods.prompt import samplers as prompt_samplers
from posteriorbench_methods.prompt import settings as prompt_settings
from posteriorbench_methods.sft import budget_rule
from posteriorbench_methods.sft import reselect as sft_reselect
from posteriorbench_methods.sft import run as sft_run
from posteriorbench_methods.sft import settings as sft_settings
from posteriorbench_methods.steering import draws as steering_draws
from posteriorbench_methods.steering import run as steering_run
from posteriorbench_methods.steering import settings as steering_settings
from posteriorbench_methods.steering import vectors as steering_vectors

POOL = ("say a", "be brief", "talk about cats", "zzz", "say aaa", "reply")
# Smaller than the selection draws of a candidate and than a family's own passes.
LIMITS = run_context.BatchLimits(scoring=3, generation=5)


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def make_context(
    toy_tasks: Any,
    helpers: Any,
    tiny_original_model: original_model.OriginalModel,
    make_tiny_model: Callable[..., Any],
    make_tiny_original_model: Callable[..., original_model.OriginalModel],
) -> Callable[..., run_context.RunContext]:
    """`make_context(family, task, out, **options)`: a run of a family on one of the toy tasks,
    with 64 evaluation draws, the stand-in judge models for the task with a rubric, and the
    batch limits LIMITS."""

    def make(
        family: str, task: tasks.GeneralizationTask, out: pathlib.Path, **options: Any
    ) -> run_context.RunContext:
        bench = task_set.GeneralizationTasks(
            toy_tasks.root,
            judge_model=helpers.judge_model(),
            review_model=helpers.review_model(),
            labels=options.get("labels"),
        )
        chosen: dict[str, Any] = {"evaluation_draws": 64, **options}
        return run_context.RunContext(
            family=family,
            method=family,
            options=run_context.RunOptions(task_id=task.task_id, out=out, **chosen),
            bench=bench,
            task=task,
            original=tiny_original_model,
            load_model_copy=lambda: make_tiny_original_model(model=make_tiny_model()),
            tracker=tracking.Tracker(),
            judge=bench.judge(task) if task.rubric is not None else None,
            batch_limits=LIMITS,
        )

    return make


@pytest.fixture(scope="module")
def pool_file(toy_tasks: Any, tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A pool file with the candidates POOL for both toy tasks."""
    rows = [
        {"task_id": task.task_id, "candidate_index": i, "candidate_id": f"c{i}", "system_prompt": s}
        for task in (toy_tasks.by_rule, toy_tasks.by_rubric)
        for i, s in enumerate(POOL)
    ]
    path = tmp_path_factory.mktemp("pool") / "pool.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def texts(scorer: original_model.OriginalModel, rows: list[dict[str, Any]]) -> list[str]:
    return [scorer.decode(row["token_ids"]) for row in rows]


# ------------------------------------------------------------------------------------- context


def test_the_context_labels_with_the_rule_or_with_the_judge_of_the_task(
    make_context: Callable[..., run_context.RunContext],
    toy_tasks: Any,
    helpers: Any,
    tmp_path: pathlib.Path,
) -> None:
    responses = list(toy_tasks.responses[:400])
    by_rule = make_context("prompt", toy_tasks.by_rule, tmp_path)
    assert by_rule.judge is None
    assert by_rule.label(responses) == ["a" in text for text in toy_tasks.texts[:400]]
    by_rubric = make_context("prompt", toy_tasks.by_rubric, tmp_path)
    labels = by_rubric.label(responses)
    assert labels == [helpers.hit(text) for text in toy_tasks.texts[:400]]
    # The two differ where the stand-in review rejects a match.
    assert labels != by_rule.label(responses)


def test_a_task_bounds_the_passes_of_a_family_and_a_benchmark_task_does_not(
    make_context: Callable[..., run_context.RunContext],
    toy_tasks: Any,
    tmp_path: pathlib.Path,
) -> None:
    context = make_context("steering", toy_tasks.by_rule, tmp_path)
    assert (context.scoring_batch(32), context.generation_batch(128)) == (3, 5)
    assert (context.scoring_batch(2), context.generation_batch(4)) == (2, 4)
    unbounded = dataclasses.replace(context, batch_limits=None)
    assert (unbounded.scoring_batch(32), unbounded.generation_batch(128)) == (32, 128)
    with pytest.raises(run_context.OptionsError):
        run_context.BatchLimits(scoring=0, generation=4)


# -------------------------------------------------------------------------------------- prompt


@pytest.fixture(scope="module")
def prompt_runs(
    make_context: Callable[..., run_context.RunContext],
    toy_tasks: Any,
    pool_file: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pathlib.Path, list[run_record.RunSummary], list[run_record.RunSummary]]:
    """Prompt selection and a mixture on the toy task with a rubric, at both thresholds, into
    one output root: the root and the summaries of the two runs."""
    out = tmp_path_factory.mktemp("prompt")
    methods = ("prompt-ebjc", "prompt-hitrate", "prompt-iwae", "mixture-eb2")
    at_10 = prompt_settings.Settings(methods=methods, pool_sizes=(2, 6))
    at_1 = dataclasses.replace(at_10, threshold=0.01, pool_sizes=(6,))
    context = make_context(
        "prompt", toy_tasks.by_rubric, out, method_data=pool_file, labels=out / "labels.jsonl"
    )
    return out, prompt_run.run(context, at_10), prompt_run.run(context, at_1)


def test_prompt_selection_runs_on_a_task_that_a_language_model_judges(
    prompt_runs: tuple[pathlib.Path, list[run_record.RunSummary], list[run_record.RunSummary]],
    toy_tasks: Any,
    helpers: Any,
    tiny_original_model: original_model.OriginalModel,
) -> None:
    out, at_10, _ = prompt_runs
    assert sorted((s.method, s.setting) for s in at_10) == sorted(
        (method, f"K={size}")
        for size in (2, 6)
        for method in ("prompt-ebjc", "prompt-hitrate", "prompt-iwae", "mixture-eb2")
    )
    for summary in at_10:
        assert summary.task_id == toy_tasks.by_rubric.task_id
        directory = out / summary.record
        # Every draw is labeled by the two judge models: selection draws, tests, evaluation.
        for name in ("selection", "threshold_tests", "evaluation"):
            rows = run_record.read_jsonl(directory / "draws" / f"{name}.jsonl")
            assert rows and [row["hit"] for row in rows] == [
                helpers.hit(text) for text in texts(tiny_original_model, rows)
            ]
        result = read(directory / "result.json")
        assert result["scorer"]["judge"]["judged_by"] == "rubric"
        assert result["dataset"]["source"] == str(toy_tasks.root)
    # The labels of the run are kept, one line for each text that the judge models were asked.
    kept = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    assert kept and len({row["text"] for row in kept}) == len(kept)
    # The package averages the records as it averages those of a benchmark task.
    results = [str(out / summary.record / "result.json") for summary in at_10]
    averaged = CliRunner().invoke(package_cli.app, ["aggregate", *results])
    assert averaged.exit_code == 0, averaged.output
    assert sorted((row["method"], row["setting"]) for row in json.loads(averaged.stdout)) == sorted(
        (summary.method, summary.setting) for summary in at_10
    )


def test_a_run_at_the_1pct_threshold_is_at_a_tenth_of_every_threshold(
    prompt_runs: tuple[pathlib.Path, list[run_record.RunSummary], list[run_record.RunSummary]],
) -> None:
    out, at_10, at_1 = prompt_runs
    root = reselection.records_root(out, reselection.THRESHOLD_1PCT)
    assert root == out / "reselected" / "threshold-1pct"
    assert run_record.read_jsonl(root / "summary.jsonl") == [s.to_json() for s in at_1]
    for summary in at_1:
        directory = root / summary.record
        construction = read(directory / "construction.json")
        assert construction["details"]["threshold"] == 0.01
        # The tests are those at 1% (1,857 draws and 28 hits at rank 1), at most 16 of them.
        tests = construction["submission"]["tests"]
        assert (tests[0]["draws"], tests[0]["required_hits"]) == (1857, 28)
        assert construction["submission"]["max_tests"] == 16
        result = read(directory / "result.json")
        assert result["passed_threshold"] == (result["native_hit_rate"] >= 0.01)
        # The records of the run at 10% are where they were.
        ten = next(s for s in at_10 if (s.method, s.setting) == (summary.method, summary.setting))
        assert (out / ten.record / "result.json").exists() and ten.record == summary.record
        assert read(out / ten.record / "construction.json")["submission"]["max_tests"] is None
    # The selection draws do not depend on the threshold: both runs drew the same ones.
    one, ten = at_1[0], next(s for s in at_10 if s.setting == "K=6")
    assert run_record.read_jsonl(
        root / one.record / "draws" / "selection.jsonl"
    ) == run_record.read_jsonl(out / ten.record / "draws" / "selection.jsonl")
    # A candidate with one hit among 16 selection draws is ranked by the signal at 1.5%.
    ranking = read(root / one.record / "construction.json")["ranking"]
    assert all(
        entry["group"] == ("by_signal" if entry["selection_hits"] else "by_hits")
        for entry in ranking
    )


def test_a_composed_prompt_sampler_generates_in_passes_of_the_tasks_size(
    make_context: Callable[..., run_context.RunContext],
    toy_tasks: Any,
    pool_file: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[int] = []
    factory = prompt_samplers.Factory

    def recording(*arguments: Any, **options: Any) -> prompt_samplers.Factory:
        built.append(options["composed_part_size"])
        return factory(*arguments, **options)

    monkeypatch.setattr(prompt_samplers, "Factory", recording)
    context = make_context("prompt", toy_tasks.by_rule, tmp_path, method_data=pool_file)
    chosen = prompt_settings.Settings(methods=("prompt-ebjc+composition",), pool_sizes=(2,))
    summaries = prompt_run.run(context, chosen)
    assert built == [LIMITS.generation]
    assert [s.method for s in summaries] == ["prompt-ebjc", "prompt-ebjc+composition"]


def test_the_instruction_text_of_a_task_is_the_packages_or_the_papers() -> None:
    by_task = {task.task_id: prompt_run.instruction_text(task) for task in tasks.all_tasks()}
    assert len(by_task) == 6 and all(by_task.values())
    embodiment = by_task["qwen3.5-0.8b/wc_0670/physical_embodiment"]
    assert embodiment.startswith("Answer as a human being with a real body")
    # The two larger models share the text of a behavior that both have.
    assert (
        by_task["qwen3.8-27b/wc_0670/chinese_character"]
        == by_task["gemma-4-31b-it/wc_0670/chinese_character"]
    )


# ------------------------------------------------------------------------------------ steering


@pytest.fixture(scope="module")
def vectors_directory(toy_tasks: Any, tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Method data with one vector of each toy behavior at the layers 0 and 1 of the tiny model,
    and no refusal direction, as for a model whose search for one failed."""
    directory = tmp_path_factory.mktemp("vectors")
    generator = torch.Generator().manual_seed(4)
    cost = steering_vectors.ConstructionCost(
        refusal_direction_passes=0,
        construction_responses=40,
        kept_responses={
            "positive_with_ablation": 0,
            "positive_without_ablation": 6,
            "negative": 9,
        },
    )
    for task in (toy_tasks.by_rule, toy_tasks.by_rubric):
        tensors = {
            "without_ablation": {layer: torch.randn(32, generator=generator) for layer in (0, 1)}
        }
        steering_vectors.save(directory, task.behavior, tensors, cost)
    return directory


def test_steering_runs_on_a_task_with_smaller_passes_and_one_vector(
    make_context: Callable[..., run_context.RunContext],
    toy_tasks: Any,
    helpers: Any,
    vectors_directory: pathlib.Path,
    tiny_original_model: original_model.OriginalModel,
    tmp_path: pathlib.Path,
) -> None:
    chosen = steering_settings.Settings(layers=(0, 1), strengths=(2.0, 8.0), selection_draws=16)
    # Without composition: the toy model's responses have at most 8 tokens, fewer than the
    # shortest L of the composition settings.
    context = make_context(
        "steering", toy_tasks.by_rubric, tmp_path, method_data=vectors_directory, compose=False
    )
    (plain,) = steering_run.run(context, chosen)
    assert plain.method == "steering" and plain.task_id == toy_tasks.by_rubric.task_id
    construction = read(tmp_path / plain.record / "construction.json")
    # The original model and one vector at two layers and two strengths.
    listed = construction["details"]["candidates"]
    assert [c["candidate_id"] for c in listed] == [
        "original",
        "without_ablation_layer0_strength2",
        "without_ablation_layer1_strength2",
        "without_ablation_layer0_strength8",
        "without_ablation_layer1_strength8",
    ]
    rows = run_record.read_jsonl(tmp_path / plain.record / "draws" / "selection.jsonl")
    assert len(rows) == 5 * 16
    assert [row["hit"] for row in rows] == [
        helpers.hit(text) for text in texts(tiny_original_model, rows)
    ]
    # Nothing is charged for a refusal direction that the model does not have: 40 responses,
    # 15 kept ones, the selection draws with p on their hits, and the tests.
    hits = sum(c["selection_hits"] for c in listed)
    assert plain.construction_inferences == (
        40 + 15 + 5 * 16 + hits + construction["submission"]["test_draws"]
    )


def test_selection_draws_beyond_a_pass_are_drawn_in_parts_with_seeds_of_their_own(
    make_context: Callable[..., run_context.RunContext],
    toy_tasks: Any,
    vectors_directory: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    chosen = steering_settings.Settings(layers=(0,), strengths=(2.0,), selection_draws=16)
    context = make_context("steering", toy_tasks.by_rule, tmp_path, method_data=vectors_directory)
    opened = steering_run.open_run(context, chosen)
    assert (opened.steered.generation_rows, opened.steered.scoring_rows) == (5, 3)
    source = steering_run.samplers.sampler(opened.steered, next(iter(opened.candidates.values())))
    whole = steering_draws.draw(context, source, 12, seed=7)
    assert len(whole) == 12 and [row["draw_index"] for row in whole] == list(range(12))
    # The first part does not depend on how many parts follow, and a draw within one pass is
    # one batch from the seed itself.
    assert [row["token_ids"] for row in steering_draws.draw(context, source, 9, seed=7)[:5]] == [
        row["token_ids"] for row in whole[:5]
    ]
    unbounded = dataclasses.replace(context, batch_limits=None)
    one_batch = steering_draws.draw(unbounded, source, 12, seed=7)
    assert [row["token_ids"] for row in one_batch] != [row["token_ids"] for row in whole]
    assert steering_draws.draw(context, source, 5, seed=7) == steering_draws.draw(
        unbounded, source, 5, seed=7
    )


def test_steering_at_1pct_reads_the_selection_draws_of_the_run_at_10pct(
    make_context: Callable[..., run_context.RunContext],
    toy_tasks: Any,
    vectors_directory: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    chosen = steering_settings.Settings(layers=(0,), strengths=(2.0, 8.0), selection_draws=16)
    context = make_context(
        "steering", toy_tasks.by_rule, tmp_path, method_data=vectors_directory, compose=False
    )
    (ten,) = steering_run.run(context, chosen)
    cache = run_record.cache_dir(tmp_path, "steering", toy_tasks.by_rule.task_id)
    drawn = {path: path.stat().st_mtime_ns for path in (cache / "selection").iterdir()}
    (one,) = steering_run.run(context, dataclasses.replace(chosen, threshold=0.01))
    assert {path: path.stat().st_mtime_ns for path in (cache / "selection").iterdir()} == drawn
    root = reselection.records_root(tmp_path, reselection.THRESHOLD_1PCT)
    construction = read(root / one.record / "construction.json")
    assert construction["details"]["threshold"] == 0.01
    assert construction["submission"]["tests"][0]["draws"] == 1857
    assert construction["submission"]["max_tests"] == 16
    assert read(tmp_path / ten.record / "construction.json")["submission"]["max_tests"] is None


# ----------------------------------------------------------------------------------------- sft


def corpus_settings() -> sft_settings.Settings:
    """SFT on a whole corpus with three quarters of the budget for training, so that the toy
    corpus gives several epochs, and with a token bound that splits an update."""
    return sft_settings.Settings(
        training_set="corpus",
        training_share=0.75,
        learning_rate=1e-3,
        trainable_last_layers=1,
        selection_draws=16,
        update_tokens=60,
    )


@pytest.fixture(scope="module")
def corpus_run(
    make_context: Callable[..., run_context.RunContext],
    toy_tasks: Any,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[run_context.RunContext, list[run_record.RunSummary]]:
    out = tmp_path_factory.mktemp("sft")
    context = make_context("sft", toy_tasks.by_rubric, out, keep_checkpoints="all", compose=False)
    return context, sft_run.run(context, corpus_settings())


def test_sft_on_a_whole_corpus_trains_on_every_hit_outside_the_references(
    corpus_run: tuple[run_context.RunContext, list[run_record.RunSummary]],
    toy_tasks: Any,
    helpers: Any,
) -> None:
    context, summaries = corpus_run
    task = toy_tasks.by_rubric
    out = context.options.out
    # What the task's data hands out: the hits that are no reference and equal none.
    held_out = toy_tasks.evaluation[task.task_id].references
    reference_rows = {reference.row for reference in held_out}
    reference_responses = {tuple(reference.token_ids) for reference in held_out}
    taken = [
        row
        for row, text in enumerate(toy_tasks.texts)
        if helpers.hit(text)
        and row not in reference_rows
        and toy_tasks.responses[row] not in reference_responses
    ]
    hits = len(taken)
    assert 0 < hits <= task.hit_count - helpers.references
    budget = math.ceil(task.num_responses / 0.25)
    assert [(s.method, s.setting) for s in summaries] == [
        ("sft", f"B={budget}"),
        ("replay", f"B={budget}"),
    ]
    sft, replay = summaries
    construction = read(out / sft.record / "construction.json")
    division = construction["details"]["division"]
    epochs = budget_rule.corpus_epochs(corpus_settings(), task.num_responses, hits)
    assert epochs > 1 and division["epochs"] == epochs
    assert (division["draws"], division["hits"]) == (task.num_responses, hits)
    assert division["training_examples"] == epochs * hits
    # The training set is every one of these hits, in corpus order.
    training = run_record.read_jsonl(out / sft.record / "draws" / "training_set.jsonl")
    assert [row["corpus_row"] for row in training] == taken
    # A checkpoint after each epoch of the rule, the last after the last epoch.
    checkpoint_epochs = budget_rule.corpus_checkpoint_epochs(epochs, hits, 32)
    assert [c["update"] for c in construction["details"]["candidates"]] == [
        math.ceil(epoch * hits / 32) for epoch in checkpoint_epochs
    ]
    # The corpus costs 1 / p̂(S) draws for each hit taken; Replay pays for nothing else.
    corpus_cost = hits / task.probability
    assert replay.construction_inferences == pytest.approx(corpus_cost)
    selection_hits = sum(c["selection_hits"] for c in construction["details"]["candidates"])
    assert sft.construction_inferences == pytest.approx(
        corpus_cost
        + epochs * hits
        + 16 * len(checkpoint_epochs)
        + selection_hits
        + construction["submission"]["test_draws"]
    )


def test_the_checkpoint_of_a_run_on_a_corpus_is_chosen_again_at_1pct(
    corpus_run: tuple[run_context.RunContext, list[run_record.RunSummary]],
) -> None:
    context, summaries = corpus_run
    (again,) = sft_reselect.reselect(context, corpus_settings())
    assert (again.method, again.setting) == ("sft", summaries[0].setting)
    root = reselection.records_root(context.options.out, reselection.THRESHOLD_1PCT)
    construction = read(root / again.record / "construction.json")
    assert construction["details"]["threshold"] == 0.01
    assert construction["submission"]["tests"][0]["draws"] == 1857
