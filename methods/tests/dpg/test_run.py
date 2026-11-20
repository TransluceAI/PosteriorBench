"""KL-DPG, JS-DPG and GDC end to end on the tiny model: the records at each budget, their costs,
composition, and the checkpoint files that remain."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from posteriorbench_methods.common import run_context, run_record, seeds, training_run
from posteriorbench_methods.dpg import run, settings

TASK_DIR = "wc_1853__goblin_mention"
# 2,000 is too small for a candidate; 3,000 and 3,100 have the same candidates; 6,000 is the
# training budget.
BUDGETS = (2000, 3000, 3100, 6000)
SHARED: dict[str, Any] = {
    "budgets": BUDGETS,
    "training_budget": 6000,
    "learning_rate": 3e-3,
    "warmup_updates": 2,
    "selection_draws": 32,
}
SETTINGS: dict[str, settings.Settings] = {
    "kl-dpg": settings.FdpgSettings(objective="kl", responses_per_update=32, **SHARED),
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
RUNS = ["kl-dpg", "js-dpg", "gdc", "kl-dpg on a rare behavior"]


@dataclasses.dataclass(frozen=True)
class Finished:
    """One finished run: its output root, method, summaries, checkpoints and replicate."""

    out: pathlib.Path
    method: str
    summaries: list[run_record.RunSummary]
    checkpoints: list[training_run.Checkpoint]
    replicate: int

    def record(self, budget: int, *, composed: bool = False) -> pathlib.Path:
        method = f"{self.method}+composition" if composed else self.method
        return self.out / method / f"B={budget}" / TASK_DIR / f"rep{self.replicate}"

    def construction(self, budget: int, *, composed: bool = False) -> dict[str, Any]:
        return read(self.record(budget, composed=composed) / "construction.json")

    def result(self, budget: int, *, composed: bool = False) -> dict[str, Any]:
        return read(self.record(budget, composed=composed) / "result.json")


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def finish(context: run_context.RunContext, chosen: settings.Settings) -> Finished:
    summaries = run.run(context, chosen)
    checkpoints = training_run.read_index(context.training_run_dir(chosen.method))
    return Finished(context.options.out, chosen.method, summaries, checkpoints, context.replicate)


@pytest.fixture(scope="module")
def runs(
    make_context: Callable[..., run_context.RunContext],
    rare_dataset: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Finished]:
    """A run of each method at the four budgets, and one of KL-DPG on a rare behavior. They keep
    every checkpoint file."""
    finished: dict[str, Finished] = {}
    for name in RUNS:
        method, _, rare = name.partition(" on ")
        out = tmp_path_factory.mktemp("dpg_run")
        dataset = rare_dataset if rare else None
        context = make_context(out, method, dataset=dataset, keep_checkpoints="all")
        finished[name] = finish(context, SETTINGS[method])
    return finished


@pytest.mark.parametrize("name", RUNS)
def test_a_run_writes_the_method_and_its_composition_at_each_budget(
    name: str, runs: dict[str, Finished]
) -> None:
    finished = runs[name]
    method = finished.method
    assert [(s.method, s.setting) for s in finished.summaries] == [
        (label, f"B={budget}") for budget in BUDGETS for label in (method, f"{method}+composition")
    ]
    lines = (finished.out / "summary.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [s.to_json() for s in finished.summaries]
    for summary in finished.summaries:
        directory = finished.out / summary.record
        files = {path.name for path in directory.iterdir()}
        assert {"result.json", "construction.json", "config.json"} <= files
        assert ("result_recomputed.json" in files) == (summary.kind == "sampler")
        config = read(directory / "config.json")
        assert (config["family"], config["method"]) == ("dpg", summary.method)
        assert config["settings"]["budgets"] == list(BUDGETS)
        assert config["settings"]["learning_rate"] == 3e-3
        for path in (directory / "draws").glob("*.jsonl"):
            assert all("token_ids" in row for row in run_record.read_jsonl(path))
    draws = {path.stem for path in (finished.record(6000) / "draws").iterdir()}
    assert draws == {"selection", "threshold_tests", "evaluation"}


@pytest.mark.parametrize("name", RUNS)
def test_a_budget_too_small_for_a_candidate_is_the_fallback_at_cost_b(
    name: str, runs: dict[str, Finished]
) -> None:
    """2,000 - 1,912 passes do not pay for the 96 of the first checkpoint."""
    finished = runs[name]
    plain, composed = (
        read(finished.out / summary.record / "construction.json")
        for summary in finished.summaries[:2]
    )
    for summary in finished.summaries[:2]:
        assert summary.fallback and summary.kind == "no_sampler"
        assert summary.construction_inferences == 2000
    assert plain["status"] == "no_sampler" and plain["details"] == {"candidates": []}
    assert composed["status"] == "not_composed"
    assert composed["details"]["composition"] == {"reason": "plain_not_passed"}
    assert plain["ledger"] is None and composed["ledger"] is None


@pytest.mark.parametrize("name", RUNS)
def test_a_budget_chooses_among_the_checkpoints_saved_within_it(
    name: str, runs: dict[str, Finished]
) -> None:
    finished = runs[name]
    checkpoints = finished.checkpoints
    for budget in BUDGETS[1:]:
        construction = finished.construction(budget)
        candidates = [c for c in checkpoints if c.budget_spent <= budget - 1912]
        assert construction["details"]["candidates"] == [c.to_json() for c in candidates]
        within = [c.checkpoint_id for c in candidates]
        assert within
        # The record holds the selection draws of its candidates, from the training run.
        selection = run_record.read_jsonl(finished.record(budget) / "draws" / "selection.jsonl")
        assert [row["candidate_id"] for row in selection[::32]] == within
        assert sum(row["hit"] for row in selection) == sum(c.selection_hits for c in candidates)
        ranked = [entry["candidate_id"] for entry in construction["ranking"]]
        aliases = [alias for entry in construction["ranking"] for alias in entry["aliases"]]
        assert sorted(ranked + aliases) == sorted(within)
        assert construction["submitted"]["checkpoint_id"] in ranked
        assert construction["status"] in ("passed", "best_observed")
    last = finished.construction(6000)["details"]["candidates"][-1]
    assert last["checkpoint_id"] == checkpoints[-1].checkpoint_id


@pytest.mark.parametrize("name", RUNS)
def test_a_run_costs_the_training_until_its_last_candidate_and_its_tests(
    name: str, runs: dict[str, Finished]
) -> None:
    finished = runs[name]
    by_id = {c.checkpoint_id: c for c in finished.checkpoints}
    for budget in BUDGETS[1:]:
        construction, result = finished.construction(budget), finished.result(budget)
        last = by_id[construction["details"]["candidates"][-1]["checkpoint_id"]]
        ledger = construction["ledger"]
        tests = construction["submission"]["test_draws"]
        assert ledger["resumed_from"] == last.ledger
        assert ledger["totals"]["budget"] == last.budget_spent + tests <= budget
        assert ledger["totals"]["inferences"] == last.ledger["inferences"] + tests
        assert result["construction_inferences"] == ledger["totals"]["inferences"]
        assert result["budget_passes"] == ledger["totals"]["budget"]
        drawn = run_record.read_jsonl(finished.record(budget) / "draws" / "threshold_tests.jsonl")
        assert len(drawn) == tests
        for test in construction["submission"]["tests"]:
            mine = [row for row in drawn if row["rank"] == test["rank"]]
            assert (len(mine), sum(row["hit"] for row in mine)) == (test["draws"], test["hits"])
            assert {row["candidate_id"] for row in mine} == {test["candidate_id"]}


@pytest.mark.parametrize("name", RUNS)
def test_budgets_with_the_same_candidates_differ_in_nothing_but_their_label(
    name: str, runs: dict[str, Finished]
) -> None:
    finished = runs[name]
    for composed in (False, True):
        first = finished.construction(3000, composed=composed)
        second = finished.construction(3100, composed=composed)
        assert {**first, "setting": "B=3100"} == second
        assert {**finished.result(3000, composed=composed), "setting": "B=3100"} == (
            finished.result(3100, composed=composed)
        )


def test_composition(runs: dict[str, Finished]) -> None:
    """A composed record is the method's record again if the submitted checkpoint did not pass or
    is the original model; else it adds the cost of composition to the method's. The four runs
    cover the three cases."""
    seen: set[str] = set()
    for finished in runs.values():
        original_model = finished.checkpoints[0].sampler_id
        for budget in BUDGETS[1:]:
            plain = finished.construction(budget)
            composed = finished.construction(budget, composed=True)
            results = finished.result(budget), finished.result(budget, composed=True)
            assert composed["details"]["plain"] == {
                "status": plain["status"],
                "submitted": plain["submitted"],
            }
            if composed["status"] == "not_composed":
                reason = composed["details"]["composition"]["reason"]
                seen.add(reason)
                is_original_model = plain["submitted"]["sampler_id"] == original_model
                if plain["status"] == "passed":
                    assert reason == "original_model" and is_original_model
                else:
                    assert reason == "plain_not_passed"
                assert composed["ledger"]["totals"] == plain["ledger"]["totals"]
                assert {**results[0], "method": results[1]["method"]} == results[1]
                continue
            seen.add("composed")
            assert plain["status"] == "passed"
            assert plain["submitted"]["sampler_id"] != original_model
            outcome = composed["details"]["composition"]
            hits, tests = sum(outcome["selection_hits"]), composed["submission"]["tests"]
            # A test draw costs 1 under a plain setting (alpha = 0 and L at the limit of 8
            # tokens or beyond) and 2 under any other.
            test_cost = 0
            for test in tests:
                tested = outcome["settings"][int(test["candidate_id"])]
                is_plain = tested["alpha"] == 0 and tested["switch"] >= 8
                test_cost += test["draws"] * (1 if is_plain else 2)
            before, after = plain["ledger"]["totals"], composed["ledger"]["totals"]
            assert composed["ledger"]["resumed_from"] == before
            assert after["inferences"] - before["inferences"] == 496 + hits + test_cost
            assert after["budget"] - before["budget"] == 496 + 256 + 512 + test_cost
            assert composed["submitted"]["setting"] == outcome["winner"]["setting"]
            directory = finished.record(budget, composed=True)
            selection = run_record.read_jsonl(directory / "draws" / "composition_selection.jsonl")
            assert len(selection) == 256 and sum(row["hit"] for row in selection) == hits
            assert len({row["setting"] for row in selection}) == 16
            drawn = run_record.read_jsonl(directory / "draws" / "composition_threshold_tests.jsonl")
            assert len(drawn) == sum(test["draws"] for test in tests)
            if outcome["winner_is_plain"]:
                assert results[1]["capped_forward_kl"] == results[0]["capped_forward_kl"]
            else:
                assert results[1]["inferences_per_response"] == 2.0
    assert seen == {"plain_not_passed", "original_model", "composed"}


def test_a_checkpoint_that_several_budgets_submit_is_evaluated_once_and_composed_once(
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated: list[tuple[str, str]] = []
    evaluate = run_context.RunContext.evaluate_and_record

    def counting(self: run_context.RunContext, candidate: Any, **arguments: Any) -> Any:
        evaluated.append((arguments["method"], arguments["setting"]))
        return evaluate(self, candidate, **arguments)

    monkeypatch.setattr(run_context.RunContext, "evaluate_and_record", counting)
    finished = finish(make_context(tmp_path, "js-dpg", replicate=2), SETTINGS["js-dpg"])
    submitted = {
        budget: finished.construction(budget)["submitted"]["sampler_id"] for budget in BUDGETS[1:]
    }
    # The budgets 3,000 and 3,100 have the same candidates and submit the same checkpoint. Which
    # checkpoint the budget 6,000 submits, and which setting a composition chooses, can differ
    # between machines. In every case a checkpoint is evaluated at the first budget that submits
    # it, and its composition is evaluated there too unless it chooses the plain setting.
    assert submitted[3000] == submitted[3100]
    expected: list[tuple[str, str]] = []
    for budget in BUDGETS[1:]:
        if any(
            submitted[earlier] == submitted[budget] for earlier in BUDGETS[1:] if earlier < budget
        ):
            continue
        expected.append(("js-dpg", f"B={budget}"))
        composition = finished.construction(budget, composed=True)["details"]["composition"]
        if composition.get("winner_is_plain") is False:
            expected.append(("js-dpg+composition", f"B={budget}"))
    assert evaluated == expected
    if submitted[6000] != submitted[3000]:
        return
    # The budgets 3,000 and 6,000 have different candidates. Their records of one checkpoint
    # differ in cost alone.
    for composed in (False, True):
        first = finished.result(3000, composed=composed)
        last = finished.result(6000, composed=composed)
        assert last["construction_inferences"] > first["construction_inferences"]
        cost = ("setting", "construction_inferences", "budget_passes", "ledger")
        assert {**first, **{key: last[key] for key in cost}} == last
        draws = [
            run_record.read_jsonl(finished.record(budget, composed=composed) / "draws" / file)
            for budget in (3000, 6000)
            for file in ["evaluation.jsonl"]
        ]
        assert draws[0] == draws[1]


def test_the_objective_is_set_by_the_method(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """`--method js-dpg --set objective=kl` is refused: it would run KL-DPG under JS-DPG's name."""
    with pytest.raises(settings.MethodMismatchError, match="set by --method"):
        run.run(make_context(tmp_path, "js-dpg"), SETTINGS["kl-dpg"])
    assert not list(tmp_path.iterdir())


def test_without_compose_a_run_writes_the_method_alone(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    chosen = dataclasses.replace(SETTINGS["gdc"], budgets=(2000, 3000))
    finished = finish(make_context(tmp_path, "gdc", compose=False), chosen)
    assert [(s.method, s.setting) for s in finished.summaries] == [
        ("gdc", "B=2000"),
        ("gdc", "B=3000"),
    ]
    assert not (tmp_path / "gdc+composition").exists()


@pytest.mark.parametrize("keep", ["all", "testable", "returned"])
def test_the_checkpoint_files_that_remain(
    keep: str, make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """Few responses per update, so that the largest budget has more than 8 distinct
    candidates."""
    chosen = dataclasses.replace(SETTINGS["kl-dpg"], budgets=(3000, 6000), responses_per_update=8)
    context = make_context(tmp_path, "kl-dpg", compose=False, keep_checkpoints=keep)
    finished = finish(context, chosen)
    where = context.training_run_dir("kl-dpg")
    remaining = {
        c.sampler_id
        for c in finished.checkpoints
        if training_run.checkpoint_path(where, c.checkpoint_id).exists()
    }
    rankings = [finished.construction(budget)["ranking"] for budget in (3000, 6000)]
    assert len(rankings[1]) > 8
    returned = {finished.construction(budget)["submitted"]["sampler_id"] for budget in (3000, 6000)}
    testable = {entry["sampler_id"] for ranking in rankings for entry in ranking[:8]}
    every = {c.sampler_id for c in finished.checkpoints}
    assert remaining == {"all": every, "testable": testable | returned, "returned": returned}[keep]
    assert returned < testable | returned < every


def test_the_random_streams_of_a_run(
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KL-DPG and JS-DPG share the stream of their initialization, and the three methods share
    the streams of their threshold tests: indexed by the budget for the training budget, and by
    the last candidate for a smaller budget. A method uses the streams of the composition exactly
    when one of its budgets composes a checkpoint."""
    used: dict[str, set[tuple[Any, ...]]] = {}
    stream_seed = seeds.stream_seed

    def recording(
        owner: str, task_id: str, replicate: int | None, stream: str, *index: str | int
    ) -> int:
        assert task_id == "wc_1853/goblin_mention" and replicate == 2
        used[method].add((owner, stream, *(type(part) for part in index)))
        return stream_seed(owner, task_id, replicate, stream, *index)

    monkeypatch.setattr(seeds, "stream_seed", recording)
    composes: dict[str, bool] = {}
    for method in ("kl-dpg", "js-dpg", "gdc"):
        used[method] = set()
        chosen = dataclasses.replace(SETTINGS[method], budgets=(3000, 6000))
        finished = finish(make_context(tmp_path / method, method, replicate=2), chosen)
        composes[method] = any(
            finished.construction(budget, composed=True)["status"] != "not_composed"
            for budget in (3000, 6000)
        )
        last_candidate = finished.construction(3000)["details"]["candidates"][-1]
        seed = run_record.read_jsonl(finished.record(3000) / "draws" / "threshold_tests.jsonl")[0]
        assert seed["seed"] == stream_seed(
            "dpg", "wc_1853/goblin_mention", 2, "threshold-test", last_candidate["checkpoint_id"], 1
        )
    tests = {("dpg", "threshold-test", int, int), ("dpg", "threshold-test", str, int)}
    for method in ("kl-dpg", "js-dpg", "gdc"):
        training = tests | {
            ("gdc" if method == "gdc" else "dpg", "initialization"),
            (method, "training", int),
            (method, "selection", str),
        }
        if method == "gdc":
            training.add(("gdc", "fit-check", int, int))
        composition = {
            (method, "composition-selection", str, str),
            (method, "composition-threshold-test", str, str),
        }
        assert training <= used[method] <= training | composition
        assert ((method, "composition-selection", str, str) in used[method]) == composes[method]
