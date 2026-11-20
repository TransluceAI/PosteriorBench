"""Which checkpoint a run submits at each of its budgets, by the submission rule
(`common.submission`). Budgets with the same candidates (`budget_rule`) form a group and submit
the same checkpoint.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from posteriorbench import ledger

from posteriorbench_methods.common import (
    charges,
    rank_table,
    run_context,
    signals,
    submission,
    training_run,
)
from posteriorbench_methods.dpg import samplers
from posteriorbench_methods.dpg import settings as dpg_settings

RULE = submission.Rule(max_tests=submission.TRAINING_MAX_TESTS)


@dataclasses.dataclass(frozen=True)
class Group:
    """Budgets of a run, in increasing order, that have the same candidates: the first
    checkpoints of the training run. `test_stream` tells the group's threshold tests apart in the
    seeds of their draws: the training budget for its own group, and for a group of smaller
    budgets the id of its last candidate."""

    budgets: tuple[int, ...]
    candidates: tuple[training_run.Checkpoint, ...]
    test_stream: int | str


def groups(
    checkpoints: list[training_run.Checkpoint], settings: dpg_settings.Settings
) -> tuple[list[Group], list[int]]:
    """The budgets of `settings` in groups with the same candidates, from the fewest candidates
    to the most, and the budgets that are too small for any candidate. The training budget, if
    it is one of the budgets, is a group of its own, and its candidates include the checkpoint
    that closed it."""
    reserve = submission.THRESHOLD_TEST_RESERVE
    smaller = sorted(budget for budget in settings.budgets if budget < settings.training_budget)
    found = [
        Group(group.budgets, tuple(checkpoints[: group.count]), group.last_checkpoint)
        for group in training_run.group_budgets(checkpoints, smaller, reserve=reserve)
    ]
    grouped = {budget for group in found for budget in group.budgets}
    if settings.training_budget in settings.budgets:
        candidates = training_run.candidates_at_budget(
            checkpoints, settings.training_budget, reserve=reserve, require_boundary=True
        )
        found.append(
            Group((settings.training_budget,), tuple(candidates), settings.training_budget)
        )
    return found, [budget for budget in smaller if budget not in grouped]


@dataclasses.dataclass
class Policy:
    """A copy of the original model that holds the weights of one checkpoint of the training run
    in `directory` at a time. `holds` is the sampler id of the checkpoint loaded last; None before
    the first."""

    directory: pathlib.Path
    models: samplers.Models
    holds: str | None = None

    def load(self, checkpoint: training_run.Checkpoint) -> None:
        """Load the weights of `checkpoint` into the model, unless it holds them already. Raises
        training_run.TrainingRunError if an earlier run deleted the checkpoint's file."""
        if self.holds == checkpoint.sampler_id:
            return
        tensors = training_run.load_checkpoint(self.directory, checkpoint.checkpoint_id)
        model = self.models.q.model
        model.load_state_dict(tensors, strict=True)
        model.eval()
        self.holds = checkpoint.sampler_id


@dataclasses.dataclass(frozen=True)
class Submitted:
    """What a group of budgets submits, with the candidates' selection draws and the threshold
    tests' draws as rows (each with its `candidate_id`, and a test row with its `rank`)."""

    group: Group
    ranking: list[submission.RankedCandidate]
    result: submission.Submission
    checkpoint: training_run.Checkpoint
    selection_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]

    def spend(self, budget: int | None) -> ledger.Ledger:
        """What the run at one of the group's budgets spent: the training until its last
        candidate was saved, whichever candidate it submits, and its threshold tests. `budget`
        is the limit of the ledger (None for no limit)."""
        totals = ledger.LedgerTotals.from_dict(self.group.candidates[-1].ledger)
        spend = ledger.Ledger.resume(totals, budget_limit=budget)
        charges.apply(spend, charges.threshold_test(self.result.test_draws))
        return spend


def rank(
    directory: pathlib.Path,
    candidates: tuple[training_run.Checkpoint, ...],
    rule: submission.Rule = RULE,
) -> list[submission.RankedCandidate]:
    """The ranking of `candidates` from their selection draws in the training run's
    `directory`."""
    return submission.rank_from_selection_draws(
        [
            submission.CandidateDraws(
                candidate_id=checkpoint.checkpoint_id,
                order=order,
                sampler_id=checkpoint.sampler_id,
                draws=signals.SelectionDraws.from_rows(
                    training_run.read_selection_draws(directory, checkpoint.checkpoint_id)
                ),
            )
            for order, checkpoint in enumerate(candidates)
        ],
        rule,
    )


def submit(
    context: run_context.RunContext,
    policy: Policy,
    group: Group,
    rule: submission.Rule = RULE,
    *,
    stream: str = "threshold-test",
) -> Submitted:
    """Rank the candidates of `group`, test them in rank order and submit one
    (`submission.submit`). Tests draw with seeds of the random stream `stream`, which KL-DPG,
    JS-DPG and GDC share."""
    by_id = {checkpoint.checkpoint_id: checkpoint for checkpoint in group.candidates}
    ranking = rank(policy.directory, group.candidates, rule)
    source = samplers.test_sampler(policy.models)
    rows: list[dict[str, Any]] = []

    def test(
        candidate: submission.RankedCandidate, threshold_test: rank_table.ThresholdTest
    ) -> int:
        policy.load(by_id[candidate.candidate_id])
        seed = context.stream_seed("dpg", stream, group.test_stream, candidate.rank)
        drawn = samplers.draw(context, source, threshold_test.draws, seed=seed)
        rows.extend(
            {**row, "candidate_id": candidate.candidate_id, "rank": candidate.rank} for row in drawn
        )
        return sum(row["hit"] for row in drawn)

    result = submission.submit(ranking, test, rule)
    assert result.submitted is not None, "a group has a candidate, so one is submitted"
    return Submitted(
        group=group,
        ranking=ranking,
        result=result,
        checkpoint=by_id[result.submitted.candidate_id],
        selection_rows=training_run.selection_rows(policy.directory, group.candidates),
        test_rows=rows,
    )
