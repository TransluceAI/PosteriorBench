"""Composition of a sampler with the original model.

Under the setting (alpha, L), each of the first L tokens of a response is drawn from
(1 - alpha) q + alpha p, where q is the sampler and p the original model, and p writes the rest. The
16 settings are alpha in {0, 0.25, 0.5, 0.75} x L in {4, 16, 32, the response limit}. The setting
alpha = 0 with L at the response limit is q itself: the plain setting. The response limit is the
task's: 50 tokens on the benchmark, up to 2,048 on the tasks of the generalization section. A draw
of a composed setting runs q and p and costs two inferences; a draw of the plain setting costs one.

The submission rule chooses the setting (`choose_setting`), and `compose` is the whole step of a
run. log q of a response under a setting is `hf_runtime.composed_logprobs`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from posteriorbench import ledger, original_model
from posteriorbench import sampler as pb_sampler

from posteriorbench_methods.common import charges, rank_table, run_context, run_record, signals
from posteriorbench_methods.common import submission as submission_rule

Rows = Sequence[Mapping[str, Any]]

ALPHAS = (0.0, 0.25, 0.5, 0.75)
SHORT_SWITCHES = (4, 16, 32)  # the values of L below the response limit
RESPONSE_LIMIT = original_model.MAX_RESPONSE_TOKENS  # 50
SELECTION_DRAWS = 16  # per setting
# The settings are ranked by EB-JC with ties in grid order, and tested until one passes.
RULE = submission_rule.Rule(max_tests=None, ties="order", representative="earliest")


class CompositionError(ValueError):
    """Settings or selection draws that composition cannot use."""


@dataclasses.dataclass(frozen=True)
class Setting:
    """One composition setting: the original model has weight `alpha` in each of the first
    `switch` tokens of a response (the paper's L) and writes the tokens after them alone."""

    alpha: float
    switch: int

    @property
    def setting_id(self) -> str:
        """For example "a0.25_t16"."""
        return f"a{self.alpha:g}_t{self.switch}"

    def is_plain(self, response_limit: int = RESPONSE_LIMIT) -> bool:
        """Whether the composed sampler is the sampler itself: alpha = 0 and L at least the
        response limit, so the original model is never read."""
        return self.alpha == 0.0 and self.switch >= response_limit

    def inferences_per_response(self, response_limit: int = RESPONSE_LIMIT) -> int:
        """What one draw costs: 1 for the plain setting, 2 for every other, including alpha = 1 or
        L = 0 (none in the grid)."""
        return 1 if self.is_plain(response_limit) else 2


def settings_grid(response_limit: int = RESPONSE_LIMIT) -> tuple[Setting, ...]:
    """The 16 settings in grid order: alpha outside, L inside. The fourth is the plain setting, the
    only one if the response limit is above 32."""
    switches = (*SHORT_SWITCHES, response_limit)
    return tuple(Setting(float(a), int(s)) for a in ALPHAS for s in switches)


def rank_settings(
    candidates: Sequence[submission_rule.CandidateDraws], rule: submission_rule.Rule = RULE
) -> list[submission_rule.RankedCandidate]:
    """The ranking of the 16 settings from their selection draws: one candidate per setting, in
    grid order, with `order` its index in the grid. `rule` is `RULE` at the run's threshold."""
    if len(candidates) != len(ALPHAS) * (len(SHORT_SWITCHES) + 1):
        raise CompositionError("composition ranks one candidate per setting")
    if any(c.draws.hit.size != SELECTION_DRAWS for c in candidates):
        raise CompositionError(f"a setting has {SELECTION_DRAWS} selection draws")
    ranking = submission_rule.rank_from_selection_draws(candidates, rule)
    if len(ranking) != len(candidates):
        raise CompositionError("the settings must be distinct samplers")
    return ranking


@dataclasses.dataclass(frozen=True)
class CompositionOutcome:
    """The composition of one sampler. `settings` and `selection_hits` are in grid order; in
    `ranking` a candidate's id is its setting's index in the grid."""

    settings: tuple[Setting, ...]
    selection_hits: tuple[int, ...]
    ranking: tuple[submission_rule.RankedCandidate, ...]
    submission: submission_rule.Submission
    winner: Setting
    winner_is_plain: bool

    @property
    def total_selection_hits(self) -> int:
        """The hits among the 256 selection draws."""
        return sum(self.selection_hits)

    def setting_of(self, test: submission_rule.TestOutcome) -> Setting:
        """The setting that `test` tested."""
        return self.settings[int(test.candidate_id)]

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return {
            "settings": [
                {"setting": s.setting_id, "alpha": s.alpha, "switch": s.switch}
                for s in self.settings
            ],
            "selection_hits": list(self.selection_hits),
            "ranking": [entry.to_json() for entry in self.ranking],
            "submission": self.submission.to_json(),
            "winner": {
                "setting": self.winner.setting_id,
                "alpha": self.winner.alpha,
                "switch": self.winner.switch,
            },
            "winner_is_plain": self.winner_is_plain,
        }


def choose_setting(
    draw: Callable[[Setting], signals.SelectionDraws],
    test: Callable[[Setting, submission_rule.RankedCandidate, rank_table.ThresholdTest], int],
    *,
    sampler_id: Callable[[Setting], str] | None = None,
    response_limit: int = RESPONSE_LIMIT,
    rule: submission_rule.Rule = RULE,
) -> CompositionOutcome:
    """Choose the composition setting of one sampler by the submission rule.

    `draw(setting)` draws the setting's 16 selection draws (hit flags, composed log q, and log p on
    the hits); it is called for the 16 settings in grid order. `test(setting, candidate,
    threshold_test)` draws `threshold_test.draws` new responses of the composed sampler and returns
    the number of hits; it is called in rank order until a setting passes. `sampler_id` defaults to
    the setting id. The caller charges `composition_charges(outcome)`."""
    settings = settings_grid(response_limit)
    candidates = [
        submission_rule.CandidateDraws(
            candidate_id=str(index),
            order=index,
            sampler_id=setting.setting_id if sampler_id is None else sampler_id(setting),
            draws=draw(setting),
        )
        for index, setting in enumerate(settings)
    ]
    ranking = rank_settings(candidates, rule)
    result = submission_rule.submit(
        ranking,
        lambda entry, threshold_test: test(
            settings[int(entry.candidate_id)], entry, threshold_test
        ),
        rule,
    )
    assert result.submitted is not None, "a ranking of 16 settings always submits one"
    winner = settings[int(result.submitted.candidate_id)]
    return CompositionOutcome(
        settings=settings,
        selection_hits=tuple(int(c.draws.hit.sum()) for c in candidates),
        ranking=tuple(ranking),
        submission=result,
        winner=winner,
        winner_is_plain=winner.is_plain(response_limit),
    )


def composition_charges(
    outcome: CompositionOutcome, *, components: int = 1, response_limit: int = RESPONSE_LIMIT
) -> list[charges.ChargeSpec]:
    """What composition adds to the cost of the method it composes: the selection draws of the 16
    settings, then every threshold test at its setting's cost per draw. `components` is the number
    of components of a mixture, 1 for any other sampler."""
    specs = charges.composition_selection_draws(outcome.total_selection_hits, components=components)
    for test in outcome.submission.tests:
        setting = outcome.setting_of(test)
        specs += charges.threshold_test(
            test.draws,
            inferences_per_response=setting.inferences_per_response(response_limit),
            note=f"composition test, rank {test.rank}, {setting.setting_id}",
        )
    return specs


def should_compose(plain_status: str, *, returned_is_original_model: bool = False) -> bool:
    """Whether a method's sampler is composed: only if it passed its own threshold test
    (`plain_status` "passed") and is not the original model itself (a training run's first
    checkpoint, the unsteered setting of Steering), since every setting would then be the
    original model again."""
    return plain_status == "passed" and not returned_is_original_model


def record_not_composed(
    context: run_context.RunContext,
    plain: run_record.Recorded,
    *,
    settings: Any,
    spend: ledger.Ledger | ledger.LedgerTotals | None,
    plain_status: str | None = None,
    details: Mapping[str, Any] | None = None,
    draws: Mapping[str, Rows] | None = None,
) -> run_record.Recorded:
    """Write the record of `<method>+composition` for a sampler that is not composed: the
    method's evaluation again, at the method's cost, with the status "not_composed". A family
    calls it itself where the method has no sampler (a budget that pays for no candidate); `spend`
    is None for a method that kept no ledger. The other arguments are those of `compose`."""
    status = plain.construction["status"] if plain_status is None else plain_status
    return context.record_reused(
        plain.results,
        hit_scores=plain.hit_scores,
        method=f"{plain.summary.method}+composition",
        setting=plain.summary.setting,
        settings=settings,
        spend=spend,
        status="not_composed",
        details={
            **(details or {}),
            "plain": {"status": status, "submitted": plain.construction["submitted"]},
            "composition": {
                "reason": "original_model" if status == "passed" else "plain_not_passed"
            },
        },
        draws=draws,
    )


def compose(
    context: run_context.RunContext,
    plain: run_record.Recorded,
    *,
    settings: Any,
    spend: ledger.Ledger | ledger.LedgerTotals,
    selection_draws: Callable[[Setting], Rows],
    threshold_test: Callable[[Setting, int], Rows],
    sampler: Callable[[Setting], pb_sampler.Sampler],
    sampler_id: Callable[[Setting], str] | None = None,
    plain_status: str | None = None,
    is_original_model: bool = False,
    charge: Callable[[CompositionOutcome], list[charges.ChargeSpec]] | None = None,
    evaluated: run_record.Recorded | None = None,
    details: Mapping[str, Any] | None = None,
    draws: Mapping[str, Rows] | None = None,
) -> run_record.Recorded:
    """Compose the sampler that a method returned with the original model, and write the record
    of `<method>+composition` at the method's setting, charged on top of `spend`. A sampler that
    `should_compose` leaves out goes to `record_not_composed`; a chosen plain setting reuses the
    method's evaluation.

    plain: the method's record; `spend` and `settings` are the method's.
    selection_draws(setting): the 16 selection draws of the sampler composed under `setting`, as
        rows with the keys of `run_record.DRAW_KEYS["composition_selection"]` but `setting`;
        `logq` is the composed log q.
    threshold_test(setting, n): n new draws of the composed sampler, as rows with the keys of
        `DRAW_KEYS["composition_threshold_tests"]` but `setting` and `rank`.
    sampler(setting): the composed sampler, for the evaluation.
    plain_status: the status of the method's submission, where it is not that of `plain`.
    charge(outcome): composition's charges, by default `composition_charges`.
    evaluated: a record that already holds the evaluation of the sampler at the chosen setting,
        for a method that submits one sampler at several settings."""
    status = plain.construction["status"] if plain_status is None else plain_status
    if not should_compose(status, returned_is_original_model=is_original_model):
        return record_not_composed(
            context,
            plain,
            settings=settings,
            spend=spend,
            plain_status=status,
            details=details,
            draws=draws,
        )
    method = f"{plain.summary.method}+composition"
    setting = plain.summary.setting
    limit = int(context.original.max_response_tokens)
    selection_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    def draw(composed: Setting) -> signals.SelectionDraws:
        rows = selection_draws(composed)
        selection_rows.extend({**row, "setting": composed.setting_id} for row in rows)
        return signals.SelectionDraws.from_rows(rows)

    def test(
        composed: Setting,
        candidate: submission_rule.RankedCandidate,
        test_at_rank: rank_table.ThresholdTest,
    ) -> int:
        rows = threshold_test(composed, test_at_rank.draws)
        test_rows.extend(
            {**row, "setting": composed.setting_id, "rank": candidate.rank} for row in rows
        )
        return sum(bool(row["hit"]) for row in rows)

    outcome = choose_setting(
        draw,
        test,
        sampler_id=sampler_id,
        response_limit=limit,
        rule=context.rule_at_threshold(RULE),
    )
    context.tracker.event(
        "composition", method=method, setting=setting, chosen=outcome.winner.setting_id
    )
    composed_spend = ledger.Ledger.resume(
        spend.totals() if isinstance(spend, ledger.Ledger) else spend
    )
    charges.apply(
        composed_spend,
        composition_charges(outcome, response_limit=limit) if charge is None else charge(outcome),
    )
    record: dict[str, Any] = {
        "method": method,
        "setting": setting,
        "settings": settings,
        "spend": composed_spend,
        "ranking": outcome.ranking,
        "submission": outcome.submission,
        "submitted": {"setting": outcome.winner.setting_id},
        "details": {
            **(details or {}),
            "plain": {"status": status, "submitted": plain.construction["submitted"]},
            "composition": outcome.to_json(),
        },
        "draws": {
            **(draws or {}),
            "composition_selection": selection_rows,
            "composition_threshold_tests": test_rows,
        },
    }
    if outcome.winner_is_plain:
        return context.record_reused(plain.results, hit_scores=plain.hit_scores, **record)
    if evaluated is not None:
        return context.record_reused(evaluated.results, hit_scores=evaluated.hit_scores, **record)
    return context.evaluate_and_record(sampler(outcome.winner), **record)
