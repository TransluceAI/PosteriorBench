"""`run_record`: the layout of a record and strict JSON, and the files that `RunContext` writes
with it when it evaluates a sampler, reuses an evaluation at another cost or records no sampler."""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import posteriorbench
import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import run_context, run_record

TASK_ID = "wc_1853/goblin_mention"
HIT, MISS = [1, 0], [2, 0]  # token 0 is the stop token
ToyContext = Callable[[pathlib.Path], run_context.RunContext]


class FakeScorer:
    """A scorer without a model: token 1 decodes to a goblin mention; every log p is -5."""

    stop_token_ids: tuple[int, ...] = (0,)
    max_response_tokens: int = 50

    def logprobs(
        self, prompt_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
    ) -> list[float]:
        return [-5.0] * len(responses)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "a goblin" if 1 in token_ids else "nothing"

    def describe(self) -> dict[str, Any]:
        return {"backend": "fake"}


def test_locations_and_labels(tmp_path: pathlib.Path) -> None:
    assert run_record.run_dir(tmp_path, "sft", "B=16384", TASK_ID, 2) == (
        tmp_path / "sft" / "B=16384" / "wc_1853__goblin_mention" / "rep2"
    )
    assert run_record.training_run_dir(tmp_path, "ctl", TASK_ID, 0) == (
        tmp_path / "training_runs" / "ctl" / "wc_1853__goblin_mention" / "rep0"
    )
    assert run_record.training_run_dir(tmp_path, "sft", TASK_ID, 1, budget=1048576) == (
        tmp_path / "training_runs" / "sft" / "wc_1853__goblin_mention" / "rep1" / "B=1048576"
    )
    assert run_record.cache_dir(tmp_path, "steering", TASK_ID).parts[-3:] == (
        "cache", "steering", "wc_1853__goblin_mention"
    )  # fmt: skip
    assert run_record.setting_label(budget=1048576) == "B=1048576"
    assert run_record.setting_label(update_cap=32) == "u=32"
    assert run_record.setting_label(pool_size=16) == "K=16"
    assert run_record.setting_label() == "none"
    with pytest.raises(run_record.RecordError):
        run_record.setting_label(budget=1, pool_size=2)


def test_strict_json(tmp_path: pathlib.Path) -> None:
    value = {"a": math.inf, "b": [np.float64(1.5), np.int64(2)], "c": np.arange(2), 3: (math.nan,)}
    run_record.write_json(tmp_path / "x.json", value)
    assert run_record.read_json(tmp_path / "x.json") == {
        "a": "inf", "b": [1.5, 2], "c": [0, 1], "3": ["nan"]
    }  # fmt: skip
    rows = [{"token_ids": [1, 0], "text": "a b"}]
    run_record.write_jsonl(tmp_path / "r.jsonl", rows)
    assert run_record.read_jsonl(tmp_path / "r.jsonl") == rows
    with pytest.raises(run_record.RecordError):
        run_record.write_draws(tmp_path, "bad", [{"hit": True}])


def test_the_draws_of_the_submission_rule_have_the_same_keys(tmp_path: pathlib.Path) -> None:
    """A row of a file named in DRAW_KEYS has those keys, which are written first."""
    test_row = {"token_ids": [1, 0], "text": "a", "hit": True, "seed": 3, "draw_index": 0}
    with pytest.raises(run_record.RecordError, match="candidate_id"):
        run_record.write_draws(tmp_path, "threshold_tests", [test_row])
    run_record.write_draws(
        tmp_path, "threshold_tests", [{**test_row, "candidate_id": "c", "rank": 1}]
    )
    (written,) = run_record.read_jsonl(tmp_path / "draws" / "threshold_tests.jsonl")
    assert list(written) == [*run_record.DRAW_KEYS["threshold_tests"], "text"]
    assert written == {**test_row, "candidate_id": "c", "rank": 1}
    # A file of a family's own has the keys that the family gives it.
    run_record.write_draws(tmp_path, "training_set", [{"token_ids": [1, 0]}])
    selection = run_record.DRAW_KEYS["selection"]
    assert run_record.DRAW_KEYS["composition_selection"] == ("setting", *selection[1:])
    assert {"logq", "logp"} <= set(selection)


def evaluate(context: run_context.RunContext) -> run_record.Recorded:
    candidate = posteriorbench.FiniteSampler([(HIT, 0.6), (MISS, 0.4)], inferences_per_response=1)
    spend = ledger.Ledger()
    spend.charge(ledger.Category.GENERATION, inferences=100, note="x")
    return context.evaluate_and_record(
        candidate, method="m", setting="none", settings={"family": "f"}, spend=spend,
        status="passed", draws={"training_set": [{"draw_index": 0, "token_ids": HIT, "hit": True}]},
    )  # fmt: skip


def test_evaluate_and_record_writes_the_layout(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    recorded = evaluate(toy_context(tmp_path))
    directory = recorded.directory
    assert directory == run_record.run_dir(tmp_path, "m", "none", TASK_ID, 2)
    names = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())
    assert names == [
        "config.json", "construction.json", "draws/evaluation.jsonl", "draws/training_set.jsonl",
        "evaluation_details.json", "evaluation_hits.json", "result.json", "result_recomputed.json",
    ]  # fmt: skip
    result = json.loads((directory / "result.json").read_text())
    assert "details" not in result and result["replicate"] == 2 and result["seed"] == 2
    assert result["construction_inferences"] == 100.0
    details = json.loads((directory / "evaluation_details.json").read_text())
    assert set(details) == {"shipped", "recomputed"}
    assert "draw_token_ids" not in details["shipped"]
    rows = run_record.read_jsonl(directory / "draws" / "evaluation.jsonl")
    assert len(rows) == 64 and all("token_ids" in row for row in rows)
    assert sum(row.get("scored_for_reverse_kl", False) for row in rows) == sum(
        row["hit"] for row in rows
    )
    summary = run_record.read_jsonl(tmp_path / "summary.jsonl")
    assert summary == [recorded.summary.to_json()]
    assert summary[0]["record"] == "m/none/wc_1853__goblin_mention/rep2"


def evaluated(
    hit_probability: float, reverse_hit_limit: int
) -> tuple[posteriorbench.FiniteSampler, posteriorbench.EvaluationResult]:
    """A sampler that returns HIT with `hit_probability`, and its evaluation on 64 draws, with the
    details."""
    bench = posteriorbench.Benchmark()
    candidate = posteriorbench.FiniteSampler(
        [(HIT, hit_probability), (MISS, 1 - hit_probability)], inferences_per_response=1
    )
    both = bench.evaluate_both(
        bench.task(TASK_ID), candidate, FakeScorer(), ledger=ledger.Ledger(), seed=1, replicate=1,
        settings=posteriorbench.EvaluationSettings(draws=64, reverse_hit_limit=reverse_hit_limit),
        keep_details=True,
    )  # fmt: skip
    return candidate, both["shipped"]


def test_a_record_has_the_scores_of_every_hit_of_its_evaluation(tmp_path: pathlib.Path) -> None:
    """The evaluation scores as many hits as its reverse KL is averaged over; the record has
    log q and log p of all of them."""
    candidate, result = evaluated(hit_probability=0.6, reverse_hit_limit=4)
    task = posteriorbench.Benchmark().task(TASK_ID)
    scores = run_record.score_every_hit(task, candidate, FakeScorer(), result)
    rows = run_record.evaluation_draw_rows(result)
    hits = [row["draw_index"] for row in rows if row["hit"]]
    scored_by_the_evaluation = [row for row in rows if row.get("scored_for_reverse_kl")]
    assert len(scored_by_the_evaluation) == 4 < len(hits)
    assert scores.draw_index == tuple(hits)
    assert scores.logq == pytest.approx([math.log(0.6)] * len(hits))
    assert scores.logp == (-5.0,) * len(hits)
    by_index = dict(zip(scores.draw_index, zip(scores.logq, scores.logp)))
    for row in scored_by_the_evaluation:  # the values of the evaluation are kept
        assert by_index[row["draw_index"]] == (row["logq"], row["logp"])
    run_record.write_record(
        tmp_path, results={"shipped": result}, construction={}, config={}, draws={},
        hit_scores=scores,
    )  # fmt: skip
    written = run_record.read_json(tmp_path / "evaluation_hits.json")
    assert list(written) == ["draw_index", "logq", "logp"]
    assert run_record.HitScores.from_json(written) == scores


def test_the_hits_of_a_sampler_below_the_threshold_are_scored_too(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    candidate = posteriorbench.FiniteSampler([(HIT, 0.05), (MISS, 0.95)], inferences_per_response=1)
    recorded = toy_context(tmp_path).evaluate_and_record(
        candidate, method="m", setting="none", settings={}, spend=ledger.Ledger()
    )
    assert recorded.shipped.fallback and recorded.shipped.hits
    rows = run_record.read_jsonl(recorded.directory / "draws" / "evaluation.jsonl")
    assert not any("logq" in row for row in rows)
    scores = recorded.hit_scores
    assert scores is not None and len(scores.draw_index) == recorded.shipped.hits
    assert scores.logq == pytest.approx([math.log(0.05)] * len(scores.draw_index))


def test_a_sampler_or_scorer_that_returns_too_few_values_is_an_error() -> None:
    class ShortScorer(FakeScorer):
        def logprobs(
            self, prompt_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
        ) -> list[float]:
            return [-5.0] * (len(responses) - 1)

    candidate, result = evaluated(hit_probability=0.05, reverse_hit_limit=128)
    task = posteriorbench.Benchmark().task(TASK_ID)
    with pytest.raises(run_record.RecordError, match="log p for"):
        run_record.score_every_hit(task, candidate, ShortScorer(), result)


def test_reuse_and_no_sampler(toy_context: ToyContext, tmp_path: pathlib.Path) -> None:
    context = toy_context(tmp_path)
    recorded = evaluate(context)
    composed = ledger.Ledger.resume(ledger.LedgerTotals.from_dict(recorded.shipped.ledger or {}))
    composed.charge(ledger.Category.GENERATION, inferences=496, note="search")
    reused = context.record_reused(
        recorded.results, method="m+composition", setting="none", settings={}, spend=composed,
        status="not_composed", hit_scores=recorded.hit_scores,
    )  # fmt: skip
    assert reused.shipped.construction_inferences == 596.0
    assert reused.shipped.native_hit_rate == recorded.shipped.native_hit_rate
    assert (reused.directory / "draws" / "evaluation.jsonl").exists()
    assert reused.hit_scores == recorded.hit_scores is not None
    assert run_record.read_json(reused.directory / "evaluation_hits.json") == run_record.read_json(
        recorded.directory / "evaluation_hits.json"
    )
    nothing = context.record_no_sampler(
        method="m", setting="B=4096", settings={}, budget=4096.0, spend=None
    )
    assert nothing.shipped.kind == "no_sampler" and nothing.shipped.construction_inferences == 4096
    assert not (nothing.directory / "result_recomputed.json").exists()
    assert nothing.hit_scores is None
    assert not (nothing.directory / "evaluation_hits.json").exists()
    kept = run_record.relabel(nothing.results, method="x", setting="B=4096", spend=composed)
    assert kept["shipped"].construction_inferences == 4096  # costed at its budget: unchanged
    assert len(run_record.read_jsonl(tmp_path / "summary.jsonl")) == 3


def test_construction_and_config_records() -> None:
    spend = ledger.Ledger.resume(ledger.LedgerTotals.zero())
    spend.charge(ledger.Category.THRESHOLD_TEST, inferences=164, note="rank 1")
    record = run_record.construction_record(
        method="m", setting="none", task_id=TASK_ID, replicate=0, status="passed", spend=spend
    )
    assert list(record) == [
        "method", "setting", "task_id", "replicate", "status", "submitted", "ranking",
        "submission", "ledger", "details",
    ]  # fmt: skip
    assert record["ledger"]["charges"][0]["inferences"] == 164
    assert record["ledger"]["resumed_from"]["inferences"] == 0.0
    without = run_record.construction_record(
        method="m", setting="B=4096", task_id=TASK_ID, replicate=0, status="no_sampler", spend=None
    )
    assert without["ledger"] is None and without["submitted"] is None
    config = run_record.config_record(
        family="f", method="m", setting="none", task_id=TASK_ID, replicate=0,
        settings={"lr": 1e-6}, options={"out": "x"}, scorer=FakeScorer(),
    )  # fmt: skip
    assert config["versions"]["posteriorbench"] == posteriorbench.__version__
    assert config["scorer"] == {"backend": "fake"}
