"""Composition: the settings, their ranking, `choose_setting` and its charges."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import posteriorbench
import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import (
    charges,
    composition,
    rank_table,
    run_context,
    run_record,
    signals,
    submission,
)

ToyContext = Callable[[pathlib.Path], run_context.RunContext]
PLAIN = composition.Setting(0.0, 50)
MIXED = composition.Setting(0.25, 16)


def test_grid_and_settings() -> None:
    grid = composition.settings_grid()
    assert [(s.alpha, s.switch) for s in grid] == [
        (alpha, switch) for alpha in (0.0, 0.25, 0.5, 0.75) for switch in (4, 16, 32, 50)
    ]
    assert [s.is_plain() for s in grid].count(True) == 1
    assert sum(16 * s.inferences_per_response() for s in grid) == 496
    assert composition.settings_grid(response_limit=40)[3].switch == 40
    assert composition.Setting(0.25, 16).setting_id == "a0.25_t16"
    # A setting in which the original model writes every token still counts two inferences.
    assert composition.Setting(1.0, 50).inferences_per_response() == 2
    assert composition.Setting(0.5, 0).inferences_per_response() == 2


def test_choose_setting_with_charges() -> None:
    rng = np.random.default_rng(3)
    drawn: list[str] = []

    def draw(setting: composition.Setting) -> signals.SelectionDraws:
        drawn.append(setting.setting_id)
        hit = rng.random(16) < (0.4 if setting.alpha == 0.25 else 0.05)
        return signals.SelectionDraws(
            hit, rng.normal(-10, 1, 16), np.where(hit, rng.normal(-12, 1, 16), 0)
        )

    tested: list[tuple[str, int]] = []

    def test(
        setting: composition.Setting,
        entry: submission.RankedCandidate,
        threshold_test: rank_table.ThresholdTest,
    ) -> int:
        tested.append((setting.setting_id, threshold_test.draws))
        return threshold_test.required_hits if len(tested) == 2 else 0

    outcome = composition.choose_setting(draw, test)
    assert drawn == [s.setting_id for s in composition.settings_grid()]
    assert outcome.submission.status == "passed" and len(tested) == 2
    assert outcome.winner == outcome.setting_of(outcome.submission.tests[1])
    target = ledger.Ledger()
    charges.apply(target, composition.composition_charges(outcome))
    expected = (
        496
        + outcome.total_selection_hits
        + sum(
            t.draws * outcome.setting_of(t).inferences_per_response()
            for t in outcome.submission.tests
        )
    )
    assert target.inferences == expected
    record = outcome.to_json()
    assert record["winner"]["setting"] == outcome.winner.setting_id
    assert list(record) == [
        "settings", "selection_hits", "ranking", "submission", "winner", "winner_is_plain"
    ]  # fmt: skip


def test_should_compose() -> None:
    assert composition.should_compose("passed")
    assert not composition.should_compose("passed", returned_is_original_model=True)
    assert not composition.should_compose("best_observed")


def plain_record(run: run_context.RunContext, *, passes: bool) -> run_record.Recorded:
    """The record of a method "toy" that submitted its one candidate, with a cost of 164."""
    rule = submission.Rule(max_tests=None, ties="order")
    ranking = submission.rank_candidates(
        [submission.Candidate("first", 0, "s0", 16, 8, score=2.0)], rule
    )
    result = submission.submit(ranking, lambda c, test: test.required_hits * passes, rule)
    spend = ledger.Ledger()
    charges.apply(spend, charges.threshold_test(164))
    return run.evaluate_and_record(
        posteriorbench.FiniteSampler([([1, 0], 0.5), ([2, 0], 0.5)], inferences_per_response=1),
        method="toy",
        setting="B=4096",
        settings={"x": 1},
        spend=spend,
        ranking=ranking,
        submission=result,
        submitted={"update": 3},
    )


def rows(count: int, hits: int) -> list[dict[str, Any]]:
    """`count` draws of which the first `hits` are hits, as rows."""
    return [
        {"draw_index": i, "token_ids": [1, 0], "hit": i < hits, "logq": -1.0, "logp": -2.0}
        | {"seed": 7}
        for i in range(count)
    ]


class Composition:
    """The functions that a family gives `compose`, for a sampler whose hits are all under the
    setting `good`. It counts the samplers that are built for the evaluation."""

    def __init__(self, good: composition.Setting) -> None:
        self.good = good
        self.built: list[composition.Setting] = []

    def selection_draws(self, setting: composition.Setting) -> list[dict[str, Any]]:
        return rows(16, 8 if setting == self.good else 0)

    def threshold_test(self, setting: composition.Setting, count: int) -> list[dict[str, Any]]:
        return rows(count, count if setting == self.good else 0)

    def sampler(self, setting: composition.Setting) -> posteriorbench.Sampler:
        self.built.append(setting)
        return posteriorbench.FiniteSampler([([1, 0], 1.0)], inferences_per_response=2)

    def compose(
        self, run: run_context.RunContext, plain: run_record.Recorded, **more: Any
    ) -> run_record.Recorded:
        return composition.compose(
            run,
            plain,
            settings={"x": 1},
            spend=ledger.LedgerTotals.from_dict(plain.shipped.ledger or {}),
            selection_draws=self.selection_draws,
            threshold_test=self.threshold_test,
            sampler=self.sampler,
            **more,
        )


def read(recorded: run_record.Recorded, name: str) -> Any:
    return json.loads((recorded.directory / name).read_text())


def test_compose_evaluates_the_sampler_under_the_chosen_setting(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    run = toy_context(tmp_path)
    plain = plain_record(run, passes=True)
    family = Composition(MIXED)
    composed = family.compose(run, plain, details={"note": "kept"})
    assert composed.summary.method == "toy+composition" and composed.summary.setting == "B=4096"
    assert family.built == [MIXED] and composed.shipped.inferences_per_response == 2
    # The method's cost, the selection draws of the 16 settings with p on 8 hits, and one test
    # of 164 draws at two inferences each.
    assert composed.shipped.construction_inferences == 164 + 496 + 8 + 2 * 164
    construction = read(composed, "construction.json")
    assert construction["status"] == "passed"
    assert construction["submitted"]["setting"] == MIXED.setting_id
    assert construction["ledger"]["resumed_from"]["inferences"] == 164
    details = construction["details"]
    assert list(details) == ["note", "plain", "composition"]
    assert details["plain"] == {"status": "passed", "submitted": plain.construction["submitted"]}
    assert details["composition"]["winner"]["setting"] == MIXED.setting_id
    selection = run_record.read_jsonl(composed.directory / "draws" / "composition_selection.jsonl")
    assert [row["setting"] for row in selection[::16]] == [
        setting.setting_id for setting in composition.settings_grid()
    ]
    tested = run_record.read_jsonl(
        composed.directory / "draws" / "composition_threshold_tests.jsonl"
    )
    assert len(tested) == 164 and {(row["setting"], row["rank"]) for row in tested} == {
        (MIXED.setting_id, 1)
    }


def test_compose_repeats_the_evaluation_when_the_plain_setting_is_chosen(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    run = toy_context(tmp_path)
    plain = plain_record(run, passes=True)
    family = Composition(PLAIN)
    # An evaluation of the composed sampler is not used either: the sampler is the method's own.
    composed = family.compose(
        run, plain, evaluated=plain_record(toy_context(tmp_path / "other"), passes=True)
    )
    assert not family.built
    assert composed.shipped.native_hit_rate == plain.shipped.native_hit_rate
    assert composed.shipped.inferences_per_response == 1
    assert composed.shipped.construction_inferences == 164 + 496 + 8 + 164
    assert read(composed, "construction.json")["details"]["composition"]["winner_is_plain"]
    assert run_record.read_jsonl(composed.directory / "draws" / "evaluation.jsonl") == (
        run_record.read_jsonl(plain.directory / "draws" / "evaluation.jsonl")
    )


def test_compose_uses_an_evaluation_that_exists(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    run = toy_context(tmp_path)
    plain = plain_record(run, passes=True)
    first = Composition(MIXED).compose(run, plain)
    family = Composition(MIXED)
    again = family.compose(run, plain, evaluated=first)
    assert not family.built
    assert again.shipped == first.shipped


@pytest.mark.parametrize(
    ("passes", "is_original_model", "reason"),
    [(False, False, "plain_not_passed"), (True, True, "original_model")],
)
def test_a_sampler_that_is_not_composed_repeats_the_methods_record(
    passes: bool, is_original_model: bool, reason: str, toy_context: ToyContext,
    tmp_path: pathlib.Path,
) -> None:  # fmt: skip
    run = toy_context(tmp_path)
    plain = plain_record(run, passes=passes)
    family = Composition(MIXED)
    composed = family.compose(run, plain, is_original_model=is_original_model)
    assert not family.built
    construction = read(composed, "construction.json")
    assert construction["status"] == "not_composed" and construction["submitted"] is None
    assert construction["ranking"] is None and construction["submission"] is None
    assert construction["details"] == {
        "plain": {
            "status": plain.construction["status"],
            "submitted": plain.construction["submitted"],
        },
        "composition": {"reason": reason},
    }
    assert composed.shipped.construction_inferences == plain.shipped.construction_inferences
    assert not (composed.directory / "draws" / "composition_selection.jsonl").exists()


def test_a_method_without_a_sampler_is_not_composed(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    run = toy_context(tmp_path)
    plain = run.record_no_sampler(
        method="toy", setting="B=4096", settings={"x": 1}, budget=4096.0, spend=None
    )
    composed = composition.record_not_composed(run, plain, settings={"x": 1}, spend=None)
    assert composed.summary.method == "toy+composition" and composed.shipped.fallback
    assert composed.shipped.construction_inferences == 4096.0
    construction = read(composed, "construction.json")
    assert construction["status"] == "not_composed" and construction["ledger"] is None
    assert construction["details"] == {
        "plain": {"status": "no_sampler", "submitted": None},
        "composition": {"reason": "plain_not_passed"},
    }
