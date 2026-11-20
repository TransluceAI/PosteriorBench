"""Choosing the checkpoint of a finished training run of KL-DPG, JS-DPG or GDC again: with a 1%
threshold, and by each method's own rule."""

from __future__ import annotations

import dataclasses
import json
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
from posteriorbench_methods.dpg import reselect, run, settings

TASK_DIR = "wc_1853__goblin_mention"
# 3,000 and 3,100 have the same candidates; 6,000 is the training budget.
BUDGETS = (3000, 3100, 6000)
SHARED: dict[str, Any] = {
    "budgets": BUDGETS,
    "training_budget": 6000,
    "learning_rate": 3e-3,
    "warmup_updates": 2,
    "selection_draws": 32,
}
KL_DPG = settings.FdpgSettings(objective="kl", responses_per_update=32, **SHARED)
SETTINGS: dict[str, settings.Settings] = {
    "kl-dpg": KL_DPG,
    "js-dpg": settings.FdpgSettings(
        objective="js", responses_per_update=32, baseline_group=12, **SHARED
    ),
    "gdc": settings.GdcSettings(
        responses_per_update=64,
        loss_divisor=16,
        fit_check_interval=2,
        fit_check_draws=96,
        **SHARED,
    ),
}
METHODS = sorted(SETTINGS)
Context = Callable[..., run_context.RunContext]


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def files(root: pathlib.Path) -> dict[str, bytes]:
    """Every file that `run` wrote under the output root `root`, by its path: all but the records
    of `reselect`."""
    return {
        str(path): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "reselected" not in path.relative_to(root).parts
    }


@dataclasses.dataclass(frozen=True)
class Finished:
    """A finished run of one method that kept every checkpoint file: its context and the
    checkpoints of its training run."""

    context: run_context.RunContext
    method: str
    checkpoints: list[training_run.Checkpoint]

    @property
    def out(self) -> pathlib.Path:
        return self.context.options.out

    def candidates(self, budget: int) -> list[training_run.Checkpoint]:
        return [row for row in self.checkpoints if row.budget_spent <= budget - 1912]

    def record(self, rule: str, budget: int) -> pathlib.Path:
        root = reselection.records_root(self.out, rule)
        return root / self.method / f"B={budget}" / TASK_DIR / "rep0"


@pytest.fixture(scope="module")
def runs(make_context: Context, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Finished]:
    """A run of each method at the three budgets, without composition."""
    finished: dict[str, Finished] = {}
    for method in METHODS:
        out = tmp_path_factory.mktemp(f"dpg_{method}")
        context = make_context(out, method, keep_checkpoints="all", compose=False)
        run.run(context, SETTINGS[method])
        checkpoints = training_run.read_index(context.training_run_dir(method))
        finished[method] = Finished(context, method, checkpoints)
    return finished


@pytest.mark.parametrize("method", METHODS)
def test_reselect_repeats_the_submission_rule_with_a_1pct_threshold(
    method: str, runs: dict[str, Finished]
) -> None:
    finished = runs[method]
    before = files(finished.out)
    summaries = reselect.reselect(finished.context, SETTINGS[method])
    assert [(s.method, s.setting) for s in summaries] == [(method, f"B={b}") for b in BUDGETS]
    # The run's records and its training run are as they were.
    assert files(finished.out) == before
    root = reselection.records_root(finished.out, "threshold-1pct")
    assert run_record.read_jsonl(root / "summary.jsonl")[-3:] == [s.to_json() for s in summaries]
    at_rank_1 = rank_table.threshold_test(1, threshold=0.01)
    for budget in BUDGETS:
        directory = finished.record("threshold-1pct", budget)
        built = read(directory / "construction.json")
        candidates = finished.candidates(budget)
        assert built["details"] == {
            "threshold": 0.01,
            "candidates": [row.to_json() for row in candidates],
        }
        ranked = {entry["candidate_id"] for entry in built["ranking"]} | {
            alias for entry in built["ranking"] for alias in entry["aliases"]
        }
        assert ranked == {row.checkpoint_id for row in candidates}
        for entry in built["ranking"]:
            above = entry["selection_hits"] / entry["selection_draws"] > 0.015
            assert entry["group"] == ("by_signal" if above else "by_hits")
        first = built["submission"]["tests"][0]
        assert (first["draws"], first["required_hits"]) == (
            at_rank_1.draws,
            at_rank_1.required_hits,
        )
        # The cost: the training up to the last candidate and the tests at 1%.
        tests = built["submission"]["test_draws"]
        result = read(directory / "result.json")
        assert built["ledger"]["resumed_from"] == candidates[-1].ledger
        assert result["construction_inferences"] == candidates[-1].ledger["inferences"] + tests
        assert result["budget_passes"] == candidates[-1].budget_spent + tests
        of_run = finished.out / method / f"B={budget}" / TASK_DIR / "rep0"
        assert (directory / "draws" / "selection.jsonl").read_bytes() == (
            of_run / "draws" / "selection.jsonl"
        ).read_bytes()
        test_rows = run_record.read_jsonl(directory / "draws" / "threshold_tests.jsonl")
        assert len(test_rows) == tests
        by_run = run_record.read_jsonl(of_run / "draws" / "threshold_tests.jsonl")
        assert {row["seed"] for row in test_rows}.isdisjoint(row["seed"] for row in by_run)
        submitted = built["submitted"]
        assert submitted["checkpoint_id"] == submitted["candidate_id"]
        assert result["fallback"] == (result["native_hit_rate"] < 0.01)
    # Budgets with the same candidates get the same choice, from one set of tests.
    same = [read(finished.record("threshold-1pct", b) / "construction.json") for b in (3000, 3100)]
    assert same[0]["submission"] == same[1]["submission"]
    assert same[0]["ranking"] == same[1]["ranking"]
    rows = [
        (finished.record("threshold-1pct", b) / "draws" / "threshold_tests.jsonl").read_bytes()
        for b in (3000, 3100)
    ]
    assert rows[0] == rows[1]
    evaluations = [
        (finished.record("threshold-1pct", b) / "draws" / "evaluation.jsonl").read_bytes()
        for b in (3000, 3100)
    ]
    assert evaluations[0] == evaluations[1]


def own_rule_records(
    finished: Finished, rule: str, summaries: list[run_record.RunSummary]
) -> list[dict[str, Any]]:
    """Check what every record of an own rule has in common; returns what each submitted."""
    assert [(s.method, s.setting) for s in summaries] == [
        (finished.method, f"B={b}") for b in BUDGETS
    ]
    submitted: list[dict[str, Any]] = []
    for budget in BUDGETS:
        directory = finished.record(rule, budget)
        built = read(directory / "construction.json")
        candidates = finished.candidates(budget)
        assert built["details"] == {
            "own_rule": rule,
            "candidates": [row.to_json() for row in candidates],
        }
        # No ranking and no threshold test: the cost is the training up to the last candidate.
        assert built["status"] == "own_rule"
        assert built["ranking"] is None and built["submission"] is None
        result = read(directory / "result.json")
        assert result["construction_inferences"] == candidates[-1].ledger["inferences"]
        assert result["budget_passes"] == candidates[-1].budget_spent
        assert built["ledger"]["charges"] == []
        assert {path.name for path in (directory / "draws").iterdir()} == {"evaluation.jsonl"}
        # The benchmark's threshold decides whether the sampler is scored as the fallback.
        assert result["fallback"] == (result["native_hit_rate"] < 0.10)
        submitted.append(built["submitted"])
    return submitted


@pytest.mark.parametrize("method", ["kl-dpg", "js-dpg"])
def test_the_own_rule_of_kl_dpg_and_js_dpg_submits_the_final_checkpoint(
    method: str, runs: dict[str, Finished]
) -> None:
    finished = runs[method]
    chosen = SETTINGS[method]
    assert isinstance(chosen, settings.FdpgSettings)
    before = files(finished.out)
    summaries = reselect.final_checkpoint(finished.context, chosen)
    assert files(finished.out) == before
    submitted = own_rule_records(finished, "final-checkpoint", summaries)
    for budget, chose in zip(BUDGETS, submitted, strict=True):
        last = finished.candidates(budget)[-1]
        assert chose == {"checkpoint_id": last.checkpoint_id, "update": last.update}
    # At the training budget the final checkpoint is the policy after the last update.
    assert submitted[-1]["checkpoint_id"] == finished.checkpoints[-1].checkpoint_id


def test_gdc_has_an_own_rule_for_each_of_its_two_models(runs: dict[str, Finished]) -> None:
    finished = runs["gdc"]
    chosen = SETTINGS["gdc"]
    assert isinstance(chosen, settings.GdcSettings)
    before = files(finished.out)
    of_r = own_rule_records(
        finished,
        "last-sampling-distribution",
        reselect.last_sampling_distribution(finished.context, chosen),
    )
    of_pi = own_rule_records(
        finished, "final-policy", reselect.final_policy(finished.context, chosen)
    )
    assert files(finished.out) == before
    for budget, r, pi in zip(BUDGETS, of_r, of_pi, strict=True):
        candidates = finished.candidates(budget)
        sampling = [row for row in candidates if row.extra["model"] == "r"]
        assert r == {"checkpoint_id": sampling[-1].checkpoint_id, "update": sampling[-1].update}
        assert r["checkpoint_id"].startswith("r_update_")
        assert pi == {
            "checkpoint_id": candidates[-1].checkpoint_id,
            "update": candidates[-1].update,
        }
    # The original model is the first sampling distribution. At the training budget the final
    # policy is the last checkpoint of the training run, and the two rules agree only if a fit
    # check has just made that policy the sampling distribution.
    last = finished.checkpoints[-1]
    assert finished.checkpoints[0].checkpoint_id == "r_update_000000"
    assert of_pi[-1]["checkpoint_id"] == last.checkpoint_id
    assert (of_r[-1] == of_pi[-1]) == (last.extra["model"] == "r")


def test_the_random_streams_of_reselect(
    runs: dict[str, Finished], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tests at 1% draw from streams of their own, indexed like those of a run: by the budget
    for the training budget and by the last candidate for a smaller budget. An own rule draws
    nothing but the evaluation."""
    used: set[tuple[Any, ...]] = set()
    stream_seed = seeds.stream_seed

    def recording(
        owner: str, task_id: str, replicate: int | None, stream: str, *index: str | int
    ) -> int:
        used.add((owner, stream, *(type(part) for part in index)))
        return stream_seed(owner, task_id, replicate, stream, *index)

    monkeypatch.setattr(seeds, "stream_seed", recording)
    finished = runs["gdc"]
    chosen = SETTINGS["gdc"]
    assert isinstance(chosen, settings.GdcSettings)
    reselect.reselect(finished.context, chosen)
    assert used == {
        ("dpg", "threshold-test-1pct", int, int),
        ("dpg", "threshold-test-1pct", str, int),
    }
    used.clear()
    reselect.last_sampling_distribution(finished.context, chosen)
    reselect.final_policy(finished.context, chosen)
    assert not used


def test_the_last_sampling_distribution_among_candidates() -> None:
    def candidate(name: str, model: str) -> training_run.Checkpoint:
        return training_run.Checkpoint(name, 0, 0.0, {}, name, 32, 0, extra={"model": model})

    candidates = [
        candidate("r_update_000000", "r"),
        candidate("pi_update_000002", "pi"),
        candidate("r_update_000004", "r"),
        candidate("pi_update_000006", "pi"),
    ]
    assert reselect.last_sampling_distribution_among(candidates) is candidates[2]
    assert reselect.last_sampling_distribution_among(candidates[:2]) is candidates[0]
    assert reselection.final_checkpoint(candidates) is candidates[3]


def test_reselect_needs_a_training_run_that_went_as_far_as_its_budgets(
    make_context: Context, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path, "kl-dpg", compose=False)
    to_3000 = dataclasses.replace(KL_DPG, budgets=(3000,))
    rules = (reselect.reselect, reselect.final_checkpoint)
    for rule in rules:
        with pytest.raises(reselection.NoTrainingRunError, match="holds no training run"):
            rule(context, to_3000)
    assert not list(tmp_path.iterdir())
    run.run(context, to_3000)
    before = files(tmp_path)
    for rule in rules:
        with pytest.raises(reselection.NoTrainingRunError, match="has not trained as far"):
            rule(context, dataclasses.replace(to_3000, budgets=(6000,)))
        with pytest.raises(training_run.TrainingRunError, match="other settings"):
            rule(context, dataclasses.replace(to_3000, adam_beta1=0.8))
        # 2,000 - 1,912 passes do not pay for the first checkpoint: nothing to choose.
        with pytest.raises(training_run.CandidatesError, match="too small for any candidate"):
            rule(context, dataclasses.replace(to_3000, budgets=(2000, 3000)))
        with pytest.raises(settings.MethodMismatchError):
            rule(context, dataclasses.replace(to_3000, objective="js"))
    assert files(tmp_path) == before
    for path in (context.training_run_dir("kl-dpg") / "checkpoints").iterdir():
        path.unlink()
    for rule in rules:
        with pytest.raises(training_run.TrainingRunError, match="--keep-checkpoints all"):
            rule(context, to_3000)
