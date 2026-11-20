"""The checkpoint that a run submits at each update cap and budget, by the submission rule
(`common.submission`), among the checkpoints of the task and replicate's one training run. The
candidates at an update cap u are the checkpoints saved in the first u updates; those at a budget
B, the ones saved within `training_run.candidate_limit(B)`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from posteriorbench import ledger

from posteriorbench_methods.common import (
    charges,
    rank_table,
    run_context,
    run_record,
    signals,
    submission,
    training_run,
)
from posteriorbench_methods.ctl import budget_rule
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import training, twisted_model

# Of several checkpoints with the same weights, the earliest stands for them in the EB-JC fit.
RULE = submission.Rule(max_tests=submission.TRAINING_MAX_TESTS, representative="earliest")
# A threshold test draws its responses in parts of this size. The part that starts at draw i of
# the test draws from a generator with the test's seed + i.
TEST_PART = 32


@dataclasses.dataclass(frozen=True)
class Point:
    """One point at which a run submits a sampler: an update cap or a training budget."""

    update_cap: int | None = None
    budget: int | None = None

    def __post_init__(self) -> None:
        if (self.update_cap is None) == (self.budget is None):
            raise ValueError("a point is an update cap or a budget")

    @property
    def setting(self) -> str:
        """The setting in the record's path: "u=<update cap>" or "B=<budget>"."""
        return run_record.setting_label(update_cap=self.update_cap, budget=self.budget)

    @property
    def stream_index(self) -> int | str:
        """What tells this point's threshold-test streams from those of the other points: the
        budget, or "u<update cap, five digits>" (an update cap and a budget may be the same
        number)."""
        return self.budget if self.budget is not None else f"u{self.update_cap:05d}"


@dataclasses.dataclass(frozen=True)
class Candidates:
    """The candidates at one point, in the order saved. `budget` is the run's budget at the
    point; for an update cap, the budget that the cap stands for
    (`budget_rule.update_cap_budget`)."""

    point: Point
    checkpoints: list[training_run.Checkpoint]
    budget: int

    def describe(self) -> dict[str, Any]:
        """The point and its candidates in a record: update_cap or budget, for an update cap the
        budget it stands for, and the candidates as rows of the training run's index."""
        point: dict[str, Any] = (
            {"budget": self.budget}
            if self.point.update_cap is None
            else {"update_cap": self.point.update_cap, "equivalent_budget": self.budget}
        )
        return {**point, "candidates": [row.to_json() for row in self.checkpoints]}


def candidates_at(
    point: Point, checkpoints: Sequence[training_run.Checkpoint], settings: ctl_settings.Settings
) -> Candidates:
    """The candidates at `point`. At a budget, the checkpoint that closed it is the last of them;
    at an update cap, they must also be the candidates at the budget that the cap stands for
    (training_run.CandidatesError otherwise)."""
    reserve = submission.THRESHOLD_TEST_RESERVE
    if point.budget is not None:
        at_budget = training_run.candidates_at_budget(
            checkpoints, point.budget, reserve=reserve, require_boundary=True
        )
        return Candidates(point, at_budget, point.budget)
    assert point.update_cap is not None
    at_cap = training_run.candidates_at_update_cap(checkpoints, point.update_cap)
    budget = budget_rule.update_cap_budget(point.update_cap, len(at_cap), settings.passes)
    if at_cap != training_run.candidates_at_budget(
        checkpoints, budget, reserve=reserve, require_boundary=False
    ):
        raise training_run.CandidatesError(
            f"the candidates at update cap {point.update_cap} are not those at budget {budget}"
        )
    return Candidates(point, at_cap, budget)


def points(settings: ctl_settings.Settings) -> list[Point]:
    """The points of `settings`: its update caps, then its budgets."""
    return [Point(update_cap=cap) for cap in settings.update_caps] + [
        Point(budget=budget) for budget in settings.budgets
    ]


def rank(
    context: run_context.RunContext,
    checkpoints: Sequence[training_run.Checkpoint],
    rule: submission.Rule = RULE,
) -> list[submission.RankedCandidate]:
    """The ranking of the candidates from their selection draws."""
    candidates: list[submission.CandidateDraws] = []
    for checkpoint in checkpoints:
        rows = training_run.read_selection_draws(
            training.directory(context), checkpoint.checkpoint_id
        )
        candidates.append(
            submission.CandidateDraws(
                candidate_id=checkpoint.checkpoint_id,
                order=checkpoint.update,
                sampler_id=checkpoint.sampler_id,
                draws=signals.SelectionDraws(
                    hit=np.array([row["hit"] for row in rows], dtype=bool),
                    logq=np.array([row["logq"] for row in rows], dtype=float),
                    logp=np.array([row["logp"] for row in rows], dtype=float),
                ),
            )
        )
    return submission.rank_from_selection_draws(candidates, rule)


def submit(
    context: run_context.RunContext,
    twisted: twisted_model.TwistedModel,
    candidates: Candidates,
    ranking: Sequence[submission.RankedCandidate],
    rule: submission.Rule = RULE,
    *,
    stream: str = "threshold-test",
    within_budget: bool = True,
) -> tuple[submission.Submission, ledger.Ledger, list[dict[str, Any]]]:
    """Test the candidates in rank order and submit one (`submission.submit`). Tests draw with
    seeds of the random stream `stream`. Returns the submission, what the run at this point spent
    (the training up to its last candidate, whichever it submits, plus the tests), and the test
    draws as rows with their `candidate_id` and `rank`. With `within_budget` the ledger is limited
    to the point's budget. Leaves the twisted model with the submitted checkpoint's weights."""
    last = candidates.checkpoints[-1]
    spend = ledger.Ledger.resume(
        ledger.LedgerTotals.from_dict(last.ledger),
        budget_limit=candidates.budget if within_budget else None,
    )
    test_rows: list[dict[str, Any]] = []

    def test(
        candidate: submission.RankedCandidate, threshold_test: rank_table.ThresholdTest
    ) -> int:
        training.load_checkpoint(context, twisted, candidate.candidate_id)
        seed = context.stream_seed("ctl", stream, candidates.point.stream_index, candidate.rank)
        rows: list[dict[str, Any]] = []
        for first in range(0, threshold_test.draws, TEST_PART):
            count = min(TEST_PART, threshold_test.draws - first)
            draws = twisted_model.sample(twisted, count, seed + first)
            part = twisted_model.label(context, draws.token_ids(), seed=seed + first)
            for row, token_logq in zip(part, draws.token_logq(), strict=True):
                rows.append({**row, "draw_index": len(rows), "logq": math.fsum(token_logq)})
        charges.apply(
            spend,
            charges.threshold_test(
                threshold_test.draws, note=f"threshold test, rank {candidate.rank}"
            ),
        )
        test_rows.extend(
            {**row, "candidate_id": candidate.candidate_id, "rank": candidate.rank} for row in rows
        )
        return sum(row["hit"] for row in rows)

    result = submission.submit(ranking, test, rule)
    assert result.submitted is not None, "a point has a candidate, so one is submitted"
    training.load_checkpoint(context, twisted, result.submitted.candidate_id)
    return result, spend, test_rows
