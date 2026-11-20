"""Choosing the checkpoint of a finished GFlowNet training run again: with a 1% threshold, and by
the highest mean log reward."""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from posteriorbench_methods.common import (
    rank_table,
    reselection,
    run_context,
    run_record,
    seeds,
    training_run,
)
from posteriorbench_methods.gfn import reselect
from posteriorbench_methods.gfn import run as gfn_run
from posteriorbench_methods.gfn import settings as gfn_settings

Context = Callable[..., run_context.RunContext]
Quick = Callable[..., gfn_settings.Settings]
TASK_DIR = "wc_1853__goblin_mention"


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def files(root: pathlib.Path) -> dict[str, bytes]:
    """Every file under `root`, by its path."""
    return {str(path): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_reselect_repeats_the_submission_rule_with_a_1pct_threshold(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(update_caps=(1, 4), budgets=(4096,))
    context = make_context(tmp_path)
    of_run = gfn_run.run(context, settings)
    before = files(tmp_path)
    summaries = reselect.reselect(context, settings)
    assert [(s.method, s.setting) for s in summaries] == [(s.method, s.setting) for s in of_run]
    # The run's records and its training run are as they were.
    assert {path: data for path, data in files(tmp_path).items() if path in before} == before
    root = reselection.records_root(tmp_path, "threshold-1pct")
    assert run_record.read_jsonl(root / "summary.jsonl") == [s.to_json() for s in summaries]
    at_rank_1 = rank_table.threshold_test(1, threshold=0.01)
    for plain, summary in zip(of_run, summaries, strict=True):
        assert summary.record == plain.record
        directory = root / summary.record
        built, built_by_run = (
            read(directory / "construction.json"),
            read(tmp_path / plain.record / "construction.json"),
        )
        details = built["details"]
        assert details.pop("threshold") == 0.01 and details == built_by_run["details"]
        # The same candidates, ranked by the signal where the hit fraction is above 1.5%.
        candidates = details["candidates"]
        assert {entry["candidate_id"] for entry in built["ranking"]} | {
            alias for entry in built["ranking"] for alias in entry["aliases"]
        } == {row["checkpoint_id"] for row in candidates}
        for entry in built["ranking"]:
            above = entry["selection_hits"] / entry["selection_draws"] > 0.015
            assert entry["group"] == ("by_signal" if above else "by_hits")
        first = built["submission"]["tests"][0]
        assert (first["draws"], first["required_hits"]) == (
            at_rank_1.draws,
            at_rank_1.required_hits,
        )
        assert built["submission"]["max_tests"] == 8
        # The cost: the training up to the last candidate and the tests at 1%, beyond the budget.
        tests = built["submission"]["test_draws"]
        result = read(directory / "result.json")
        assert built["ledger"]["resumed_from"] == candidates[-1]["ledger"]
        assert result["construction_inferences"] == candidates[-1]["ledger"]["inferences"] + tests
        assert result["budget_passes"] == candidates[-1]["budget_spent"] + tests
        assert (directory / "draws" / "selection.jsonl").read_bytes() == (
            tmp_path / plain.record / "draws" / "selection.jsonl"
        ).read_bytes()
        test_rows = run_record.read_jsonl(directory / "draws" / "threshold_tests.jsonl")
        assert len(test_rows) == tests
        assert built["submitted"]["checkpoint_id"] == built["submitted"]["candidate_id"]
        # The tests at 1% draw from streams of their own.
        by_run = run_record.read_jsonl(tmp_path / plain.record / "draws" / "threshold_tests.jsonl")
        assert {row["seed"] for row in test_rows}.isdisjoint(row["seed"] for row in by_run)
        assert len(run_record.read_jsonl(directory / "draws" / "evaluation.jsonl")) == 64


def test_the_evaluation_at_1pct_keeps_a_sampler_that_the_benchmark_scores_as_the_fallback(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that no checkpoint passes: the best-observed checkpoint is submitted, and the
    evaluation applies the 1% threshold."""
    context = make_context(tmp_path)
    settings = quick(update_caps=(2,))
    gfn_run.run(context, settings)
    # No response of a threshold test is a hit.
    monkeypatch.setattr(
        reselect.selection.CheckpointDraws,
        "draw",
        lambda self, checkpoint, count, seed: [
            {"draw_index": i, "token_ids": [5], "text": "", "hit": False, "logq": 0.0, "seed": seed}
            for i in range(count)
        ],
    )
    (summary,) = reselect.reselect(context, settings)
    built = read(
        reselection.records_root(tmp_path, "threshold-1pct") / summary.record / "construction.json"
    )
    assert built["status"] == "best_observed"
    assert [test["rank"] for test in built["submission"]["tests"]] == list(
        range(1, len(built["ranking"]) + 1)
    )
    assert built["submitted"]["rank"] == 1 and built["submitted"]["hits"] == 0
    assert summary.fallback == (summary.native_hit_rate < 0.01)


def test_mean_log_reward() -> None:
    draws = [
        {"hit": True, "logp": -2.0},
        {"hit": False, "logp": -1.0},
        {"hit": False, "logp": -3.0},
    ]
    assert reselect.mean_log_reward(draws, 1e-8) == pytest.approx(
        (-2.0 - 1.0 - 3.0 + 2 * math.log(1e-8)) / 3
    )
    assert reselect.mean_log_reward(draws[:1], 1e-8) == -2.0


def test_the_own_rule_submits_the_checkpoint_with_the_highest_mean_log_reward(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(update_caps=(1, 4), budgets=(4096,))
    context = make_context(tmp_path)
    of_run = gfn_run.run(context, settings)
    before = files(tmp_path)
    summaries = reselect.highest_mean_log_reward(context, settings)
    assert [(s.method, s.setting) for s in summaries] == [(s.method, s.setting) for s in of_run]
    assert {path: data for path, data in files(tmp_path).items() if path in before} == before
    root = reselection.records_root(tmp_path, "highest-mean-log-reward")
    training = tmp_path / "training_runs" / "gflownet" / TASK_DIR / "rep0"
    for plain, summary in zip(of_run, summaries, strict=True):
        directory = root / summary.record
        built = read(directory / "construction.json")
        details = built["details"]
        candidates = [row["checkpoint_id"] for row in details["candidates"]]
        assert details["candidates"] == (
            read(tmp_path / plain.record / "construction.json")["details"]["candidates"]
        )
        assert details["own_rule"] == "highest-mean-log-reward"
        means = {
            name: reselect.mean_log_reward(
                training_run.read_selection_draws(training, name), settings.epsilon
            )
            for name in candidates
        }
        assert details["mean_log_reward"] == means
        best = max(means.values())
        # The earliest of the checkpoints with the highest mean.
        expected = next(name for name in candidates if means[name] == best)
        assert built["submitted"] == {
            "checkpoint_id": expected,
            "update": int(expected.removeprefix("update_")),
        }
        # No ranking and no threshold test: the cost is the training up to the last candidate.
        assert built["status"] == "own_rule"
        assert built["ranking"] is None and built["submission"] is None
        result = read(directory / "result.json")
        last = details["candidates"][-1]
        assert result["construction_inferences"] == last["ledger"]["inferences"]
        assert result["budget_passes"] == last["budget_spent"]
        assert built["ledger"]["charges"] == []
        names = {path.name for path in (directory / "draws").iterdir()}
        assert names == {"selection.jsonl", "evaluation.jsonl"}
        # The benchmark's threshold decides whether the sampler is scored as the fallback.
        assert result["fallback"] == (result["native_hit_rate"] < 0.10)


def test_checkpoints_with_the_same_mean_go_to_the_earliest(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = make_context(tmp_path)
    settings = quick(update_caps=(4,))
    gfn_run.run(context, settings)
    monkeypatch.setattr(reselect, "mean_log_reward", lambda draws, epsilon: -1.0)
    (summary,) = reselect.highest_mean_log_reward(context, settings)
    root = reselection.records_root(tmp_path, "highest-mean-log-reward")
    built = read(root / summary.record / "construction.json")
    assert built["submitted"]["checkpoint_id"] == "update_000001"


def test_the_random_streams_of_reselect(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tests at 1% draw from streams of their own, indexed by the point and the rank like
    those of a run. The own rule draws nothing but the evaluation. Both build a new policy,
    whose adapters the checkpoints then replace."""
    context = make_context(tmp_path)
    settings = quick(update_caps=(2,), budgets=(4096,))
    gfn_run.run(context, settings)
    used: set[tuple[Any, ...]] = set()
    stream_seed = seeds.stream_seed

    def recording(
        owner: str, task_id: str, replicate: int | None, stream: str, *index: str | int
    ) -> int:
        used.add((owner, stream, *(type(part) for part in index)))
        return stream_seed(owner, task_id, replicate, stream, *index)

    monkeypatch.setattr(seeds, "stream_seed", recording)
    reselect.reselect(context, settings)
    assert used == {
        ("gflownet", "initialization"),
        ("gflownet", "threshold-test-1pct", str, int),
        ("gflownet", "threshold-test-1pct", int, int),
    }
    used.clear()
    reselect.highest_mean_log_reward(context, settings)
    assert used == {("gflownet", "initialization")}


def test_reselect_needs_a_training_run_that_went_as_far_as_its_points(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    with pytest.raises(reselection.NoTrainingRunError, match="holds no training run"):
        reselect.reselect(context, quick(update_caps=(2,)))
    assert not list(tmp_path.iterdir())
    gfn_run.run(context, quick(update_caps=(2,)))
    before = files(tmp_path)
    for rule in (reselect.reselect, reselect.highest_mean_log_reward):
        with pytest.raises(reselection.NoTrainingRunError, match="has not trained as far"):
            rule(context, quick(update_caps=(4,)))
        with pytest.raises(training_run.TrainingRunError, match="other settings"):
            rule(context, quick(update_caps=(2,), lora_rank=8))
    assert files(tmp_path) == before


def test_a_checkpoint_file_that_a_run_deleted_is_reported(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    settings = quick(update_caps=(4,))
    gfn_run.run(context, settings)
    training = tmp_path / "training_runs" / "gflownet" / TASK_DIR / "rep0"
    for path in (training / "checkpoints").iterdir():
        path.unlink()
    for rule in (reselect.reselect, reselect.highest_mean_log_reward):
        with pytest.raises(training_run.TrainingRunError, match="--keep-checkpoints all"):
            rule(context, settings)
