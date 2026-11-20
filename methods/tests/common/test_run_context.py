"""`RunContext`: its methods fill in the task, the replicate and the options."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from collections.abc import Callable
from typing import Any

import posteriorbench
import pytest
from posteriorbench import data as package_data
from posteriorbench import ledger

from posteriorbench_methods.common import (
    errors,
    run_context,
    run_record,
    seeds,
    submission,
    tracking,
)
from posteriorbench_methods.generalization import models, task_data, tasks

TASK_ID = "wc_1853/goblin_mention"
ToyContext = Callable[[pathlib.Path], run_context.RunContext]


def test_methods_fill_in_task_and_replicate(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    run = toy_context(tmp_path)
    assert run.stream_seed("toy", "selection", "c1") == seeds.stream_seed(
        "toy", TASK_ID, 2, "selection", "c1"
    )
    assert run.stream_seed("toy", "selection", "c1", shared_by_replicates=True) == (
        seeds.stream_seed("toy", TASK_ID, None, "selection", "c1")
    )
    assert (
        run.run_dir("toy", "none") == tmp_path / "toy" / "none" / "wc_1853__goblin_mention" / "rep2"
    )
    assert run.training_run_dir("toy").parts[-4:] == (
        "training_runs",
        "toy",
        "wc_1853__goblin_mention",
        "rep2",
    )
    assert run.training_run_dir("toy", budget=4096).parts[-2:] == ("rep2", "B=4096")
    assert run.cache_dir().parts[-3:] == ("cache", "toy", "wc_1853__goblin_mention")


@dataclasses.dataclass(frozen=True)
class ToySettings:
    learning_rate: float = 1e-6
    budgets: tuple[int, ...] = (4096,)


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def test_evaluate_and_record_writes_one_record_from_the_pieces(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    run = toy_context(tmp_path)
    sampler = posteriorbench.FiniteSampler(
        [([1, 0], 0.5), ([2, 0], 0.5)], inferences_per_response=1
    )
    rule = submission.Rule(max_tests=None, ties="order")
    ranking = submission.rank_candidates(
        [
            submission.Candidate("first", 0, "s0", 16, 8, score=2.0),
            submission.Candidate("second", 1, "s1", 16, 8, score=1.0),
        ],
        rule,
    )
    result = submission.submit(ranking, lambda candidate, test: test.required_hits, rule)
    spend = ledger.Ledger()
    spend.charge(ledger.Category.THRESHOLD_TEST, inferences=164, note="rank 1")
    recorded = run.evaluate_and_record(
        sampler,
        method="toy",
        setting="B=4096",
        settings=ToySettings(),
        spend=spend,
        ranking=ranking,
        submission=result,
        submitted={"update": 12},
        details={"candidates": 2},
        draws={"training_set": [{"draw_index": 0, "token_ids": [1, 0], "hit": True}]},
    )
    assert recorded.shipped.draws == 64 and recorded.shipped.replicate == 2
    assert recorded.shipped.construction_inferences == 164.0
    assert recorded.directory == run.run_dir("toy", "B=4096")
    construction = read(recorded.directory / "construction.json")
    assert list(construction) == [
        "method", "setting", "task_id", "replicate", "status", "submitted", "ranking",
        "submission", "ledger", "details",
    ]  # fmt: skip
    assert (construction["method"], construction["setting"]) == ("toy", "B=4096")
    assert (construction["task_id"], construction["replicate"]) == (TASK_ID, 2)
    assert construction["status"] == "passed"
    assert construction["submitted"] == {**result.tests[0].to_json(), "update": 12}
    assert construction["submitted"]["candidate_id"] == "first"
    assert construction["ranking"] == [entry.to_json() for entry in ranking]
    assert construction["submission"] == result.to_json()
    assert construction["ledger"]["totals"]["inferences"] == 164.0
    assert construction["details"] == {"candidates": 2}
    config = read(recorded.directory / "config.json")
    assert (config["family"], config["method"], config["setting"]) == ("toy", "toy", "B=4096")
    assert config["settings"] == {"learning_rate": 1e-6, "budgets": [4096]}
    assert config["options"]["replicate"] == 2 and config["scorer"] == {"backend": "fake"}
    assert (recorded.directory / "draws" / "training_set.jsonl").exists()


def test_records_without_a_submission_and_reused_records(
    toy_context: ToyContext, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=tracking.LOGGER_NAME)
    run = toy_context(tmp_path)
    sampler = posteriorbench.FiniteSampler([([1, 0], 1.0)], inferences_per_response=1)
    plain = run.evaluate_and_record(
        sampler,
        method="toy",
        setting="none",
        settings={"x": 1},
        spend=ledger.Ledger(),
        submitted={"prompt": "say goblin"},
    )
    construction = read(plain.directory / "construction.json")
    assert construction["status"] == "no_selection"  # a sampler returned without the rule
    assert construction["submitted"] == {"prompt": "say goblin"}
    assert construction["ranking"] is None and construction["submission"] is None
    nothing = run.record_no_sampler(
        method="toy", setting="B=4096", settings={"x": 1}, budget=4096.0, spend=None
    )
    assert nothing.shipped.construction_inferences == 4096.0
    construction = read(nothing.directory / "construction.json")
    assert construction["status"] == "no_sampler" and construction["ledger"] is None
    assert construction["submitted"] is None
    composed = ledger.Ledger.resume(ledger.LedgerTotals.zero())
    composed.charge(ledger.Category.GENERATION, inferences=496, note="composition")
    reused = run.record_reused(
        plain.results,
        method="toy+composition",
        setting="none",
        settings={"x": 1},
        spend=composed,
        status="not_composed",
        details={"plain": "no_selection"},
    )
    assert reused.shipped.construction_inferences == 496.0
    assert reused.shipped.native_hit_rate == plain.shipped.native_hit_rate
    construction = read(reused.directory / "construction.json")
    assert construction["status"] == "not_composed"
    assert construction["details"] == {"plain": "no_selection"}
    assert read(reused.directory / "config.json")["method"] == "toy+composition"
    with pytest.raises(run_record.RecordError):
        run.record_reused(
            plain.results, method="toy+composition", setting="none", settings={}, spend=composed
        )
    # Every record that is written is logged, with its status and its summary.
    events = [json.loads(record.getMessage()) for record in caplog.records]
    assert [(event["event"], event["method"], event["status"]) for event in events] == [
        ("record", "toy", "no_selection"),
        ("record", "toy", "no_sampler"),
        ("record", "toy+composition", "not_composed"),
    ]
    assert events[0]["native_hit_rate"] == plain.shipped.native_hit_rate
    assert events[2]["record"] == reused.summary.record


def test_a_method_checks_its_options_before_anything_is_loaded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(run_context, "open_context", lambda *_: opened.append("context"))

    def check(options: run_context.RunOptions, settings: dict[str, Any]) -> None:
        if options.method_data is None:
            raise errors.MethodsError("no method data")

    entry = run_context.MethodEntry(settings={}, run=lambda context, settings: [], check=check)
    options = run_context.RunOptions(task_id=TASK_ID, out=tmp_path)
    with pytest.raises(errors.MethodsError, match="no method data"):
        run_context.run_method("toy", "toy", entry, options, {})
    assert not opened


def test_a_run_needs_a_scored_pair(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pair without references cannot be evaluated: it is refused before the model is loaded,
    in every family."""
    unscored = next(task for task in posteriorbench.Benchmark().tasks("all") if not task.scored)
    loaded: list[str] = []
    monkeypatch.setattr(run_context, "load_original_model", lambda *_: loaded.append("model"))
    entry = run_context.MethodEntry(settings={}, run=lambda context, settings: [])
    options = run_context.RunOptions(task_id=unscored.task_id, out=tmp_path)
    with pytest.raises(posteriorbench.PosteriorBenchError, match="not a scored pair"):
        run_context.open_context("toy", "toy", entry, options, {})
    assert not loaded and not list(tmp_path.iterdir())


def test_a_run_at_the_1pct_threshold_has_its_rule_its_evaluation_and_its_records_root(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    run = toy_context(tmp_path)
    assert run.threshold == 0.10 and run.records_root == tmp_path
    rule = submission.Rule(max_tests=None, ties="order")
    assert run.rule_at_threshold(rule) is rule
    one = run.at_threshold(0.01)
    assert one.threshold == 0.01 and run.threshold == 0.10
    assert one.records_root == tmp_path / "reselected" / "threshold-1pct"
    assert one.run_dir("toy", "none").parent.parent.parent.parent == one.records_root
    # A rule that tests until a candidate passes tests at most 16 at 1%; a training method
    # keeps its 8.
    at_1 = one.rule_at_threshold(rule)
    assert (at_1.threshold, at_1.signal_hit_fraction, at_1.max_tests) == (0.01, 0.015, 16)
    assert one.rule_at_threshold(submission.Rule(max_tests=8)).max_tests == 8
    with pytest.raises(run_context.OptionsError, match="hit-rate threshold"):
        run.at_threshold(0.05)
    # A sampler that hits on about 5% of its draws is the fallback at 10% and passes at 1%.
    sampler = posteriorbench.FiniteSampler(
        [([1, 0], 0.05), ([2, 0], 0.95)], inferences_per_response=1
    )
    record: dict[str, Any] = {
        "method": "toy",
        "setting": "none",
        "settings": {},
        "spend": ledger.Ledger(),
    }
    at_10pct = run.evaluate_and_record(sampler, details={"x": 1}, **record)
    at_1pct = one.evaluate_and_record(sampler, details={"x": 1}, **record)
    assert at_10pct.shipped.fallback and not at_1pct.shipped.fallback
    assert at_1pct.directory == one.run_dir("toy", "none") != at_10pct.directory
    assert read(at_1pct.directory / "construction.json")["details"] == {"threshold": 0.01, "x": 1}
    assert read(at_10pct.directory / "construction.json")["details"] == {"x": 1}
    # The other two kinds of record go to the same root.
    nothing = one.record_no_sampler(
        method="toy", setting="B=4096", settings={}, budget=4096.0, spend=None
    )
    reused = one.record_reused(
        at_1pct.results,
        method="toy+composition",
        setting="none",
        settings={},
        spend=ledger.Ledger(),
        status="not_composed",
    )
    assert one.records_root in nothing.directory.parents
    assert one.records_root in reused.directory.parents
    assert len(run_record.read_jsonl(one.records_root / "summary.jsonl")) == 3
    assert len(run_record.read_jsonl(tmp_path / "summary.jsonl")) == 1


GENERALIZATION_TASK = "qwen3.8-27b/wc_0670/chinese_character"
JUDGED_TASK = "qwen3.5-0.8b/wc_0670/physical_embodiment"


def test_a_generalization_task_is_opened_with_its_own_model_and_limits(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_model: Any,
    tiny_tokenizer: Any,
) -> None:
    """A task id of the generalization section opens the task's data and the task's original
    model; the batch sizes of that model bound the family's passes."""
    loaded: list[tuple[str, str | None, int | None]] = []

    def load(task: Any, options: run_context.RunOptions, device: str) -> Any:
        loaded.append((task.task_id, options.model, options.batch_size))
        return models.TaskOriginalModel(
            tiny_model,
            tiny_tokenizer,
            generation_batch_size=6,
            batch_size=4 if options.batch_size is None else options.batch_size,
            max_response_tokens=task.max_response_tokens,
            name=task.model.name if options.model is None else options.model,
        )

    monkeypatch.setattr(run_context, "load_task_original_model", load)
    entry = run_context.MethodEntry(
        settings={}, run=lambda context, settings: [], generalization_tasks=True
    )
    # The data holds the task: a run looks for the task's evaluation file before it loads a model.
    task_directory = tmp_path / task_data.task_dir(tasks.get_task(GENERALIZATION_TASK))
    evaluation_file = task_directory / package_data.EVALUATION_FILE
    evaluation_file.parent.mkdir(parents=True)
    evaluation_file.touch()
    options = run_context.RunOptions(
        task_id=GENERALIZATION_TASK, out=tmp_path / "out", data=str(tmp_path)
    )
    context = run_context.open_context("toy", "toy", entry, options, {})
    assert context.task.task_id == GENERALIZATION_TASK and context.judge is None
    assert context.batch_limits == run_context.BatchLimits(scoring=4, generation=6)
    assert (context.scoring_batch(32), context.generation_batch(128)) == (4, 6)
    assert context.original.max_response_tokens == 50
    # The model is loaded from where the options say, or from the task's own location, which
    # the record of the run then names.
    assert loaded == [(GENERALIZATION_TASK, None, None)]
    assert context.options.model == "Qwen/Qwen3.8-27B"
    context.load_model_copy()
    assert loaded[-1] == (GENERALIZATION_TASK, None, None)
    smaller = dataclasses.replace(options, model="/weights/qwen", batch_size=2)
    context = run_context.open_context("toy", "toy", entry, smaller, {})
    assert loaded[-1] == (GENERALIZATION_TASK, "/weights/qwen", 2)
    assert context.options.model == "/weights/qwen" and context.batch_limits is not None
    assert context.batch_limits.scoring == 2


def test_what_a_generalization_task_lacks_is_reported_before_its_model_is_loaded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_: Any) -> Any:
        raise AssertionError("the model is not loaded")

    monkeypatch.setattr(run_context, "load_task_original_model", no_model)
    monkeypatch.setattr(run_context, "load_original_model", no_model)
    entry = run_context.MethodEntry(
        settings={}, run=lambda context, settings: [], generalization_tasks=True
    )

    def open_with(**options: Any) -> None:
        chosen = run_context.RunOptions(out=tmp_path / "out", **options)
        run_context.open_context("toy", "toy", entry, chosen, {})

    # A method that does not run on these tasks says so.
    of_the_package = dataclasses.replace(entry, generalization_tasks=False)
    with pytest.raises(run_context.TaskOptionsError, match="runs on the tasks of the package"):
        run_context.open_context(
            "toy",
            "toy",
            of_the_package,
            run_context.RunOptions(task_id=GENERALIZATION_TASK, out=tmp_path, data=str(tmp_path)),
            {},
        )

    with pytest.raises(errors.MethodsError, match="unknown task"):
        open_with(task_id="qwen3.8-27b/wc_0670/goblin_mention", data=str(tmp_path))
    with pytest.raises(errors.MethodsError, match="data of the generalization tasks"):
        open_with(task_id=GENERALIZATION_TASK)
    # A directory that does not hold the task.
    with pytest.raises(task_data.TaskDataError, match=f"{GENERALIZATION_TASK}: no "):
        open_with(task_id=GENERALIZATION_TASK, data=str(tmp_path))
    # A task that a language model judges needs the two judge models.
    with pytest.raises(errors.MethodsError, match="needs a rubric judge .* and a reviewing model"):
        open_with(task_id=JUDGED_TASK, data=str(tmp_path))
    with pytest.raises(errors.MethodsError, match="needs a rubric judge .* and a reviewing model"):
        open_with(
            task_id=JUDGED_TASK,
            data=str(tmp_path),
            judge_url="http://localhost:1/v1",
            judge_model="judge",
        )
    # The options of these tasks are refused for a task of the package.
    with pytest.raises(run_context.TaskOptionsError, match="section: --judge-url, --batch-size$"):
        open_with(task_id=TASK_ID, judge_url="http://localhost:1/v1", batch_size=2)
    assert not (tmp_path / "out").exists()


def test_the_judge_models_of_the_options_have_the_papers_request_settings() -> None:
    options = run_context.RunOptions(
        task_id=JUDGED_TASK,
        out=pathlib.Path("out"),
        judge_url="http://localhost:8000/v1",
        judge_model="judge",
        review_url="https://example.org/v1",
        review_model="review",
        review_key_env="REVIEW_KEY",
    )
    rubric_judge, review = run_context.judge_models(options)
    assert rubric_judge is not None and review is not None
    assert rubric_judge.describe()["request"]["reasoning_effort"] == "medium"
    assert review.describe()["request"] == {"reasoning": {"effort": "high"}}
    assert (rubric_judge.api_key_env, review.api_key_env) == (None, "REVIEW_KEY")
    assert run_context.judge_models(dataclasses.replace(options, review_model=None))[1] is None
    assert options.generalization_options() == [
        "judge_url", "judge_model", "review_url", "review_model", "review_key_env",
    ]  # fmt: skip
