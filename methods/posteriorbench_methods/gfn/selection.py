"""The checkpoint that a run submits at each update cap and budget, by the submission rule
(`common.submission`), among the checkpoints of the task and replicate's one training run. The
candidates at an update cap u are the checkpoints saved in the first u updates; those at a budget
B, the ones saved within `training_run.candidate_limit(B)`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from posteriorbench import ledger as pb_ledger
from posteriorbench import rules

from posteriorbench_methods.common import (
    charges,
    rank_table,
    run_context,
    run_record,
    signals,
    submission,
    training_run,
)
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import settings as gfn_settings

METHOD = gfn_policy.METHOD
RULE = submission.Rule(max_tests=submission.TRAINING_MAX_TESTS)


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

    def describe(self) -> dict[str, int]:
        """The point in a record: {"update_cap": u} or {"budget": B}."""
        if self.budget is not None:
            return {"budget": self.budget}
        assert self.update_cap is not None
        return {"update_cap": self.update_cap}

    @property
    def stream_index(self) -> int | str:
        """What tells this point's threshold-test streams from those of the other points: the
        budget, or "U<update cap>" (an update cap and a budget may be the same number)."""
        return self.budget if self.budget is not None else f"U{self.update_cap}"

    def candidates(
        self, checkpoints: Sequence[training_run.Checkpoint]
    ) -> list[training_run.Checkpoint]:
        """The candidates at this point. At a budget, they must include the checkpoint that
        closed it; at an update cap, none may be a checkpoint that closed a budget
        (training_run.CandidatesError otherwise)."""
        if self.budget is not None:
            return training_run.candidates_at_budget(
                checkpoints,
                self.budget,
                reserve=submission.THRESHOLD_TEST_RESERVE,
                require_boundary=True,
            )
        assert self.update_cap is not None
        return training_run.candidates_at_update_cap(
            checkpoints, self.update_cap, allow_boundary=False
        )


def points(settings: gfn_settings.Settings) -> list[Point]:
    """The points of `settings`: its update caps, then its budgets."""
    return [Point(update_cap=cap) for cap in settings.update_caps] + [
        Point(budget=budget) for budget in settings.budgets
    ]


def rank(
    directory: pathlib.Path,
    candidates: Sequence[training_run.Checkpoint],
    rule: submission.Rule = RULE,
) -> list[submission.RankedCandidate]:
    """The ranking of the candidates from their selection draws in the training run
    `directory`."""
    with_draws: list[submission.CandidateDraws] = []
    for checkpoint in candidates:
        rows = training_run.read_selection_draws(directory, checkpoint.checkpoint_id)
        draws = signals.SelectionDraws(
            hit=np.array([bool(row["hit"]) for row in rows]),
            logq=np.array([float(row["logq"]) for row in rows]),
            logp=np.array([float(row["logp"]) for row in rows]),
        )
        with_draws.append(
            submission.CandidateDraws(
                candidate_id=checkpoint.checkpoint_id,
                order=checkpoint.update,
                sampler_id=checkpoint.sampler_id,
                draws=draws,
            )
        )
    return submission.rank_from_selection_draws(with_draws, rule)


@dataclasses.dataclass
class CheckpointDraws:
    """Draws from the checkpoints of one training run, loaded into `policy` one at a time.
    `loaded` is the sampler id of the checkpoint that the policy holds; None before the first."""

    directory: pathlib.Path
    policy: gfn_policy.Policy
    decode: Callable[[Sequence[int]], str]
    is_hit: Callable[[str], bool]
    loaded: str | None = None

    @classmethod
    def of(cls, context: run_context.RunContext, policy: gfn_policy.Policy) -> CheckpointDraws:
        """The draws from the checkpoints of the training run of the context's task and
        replicate, loaded into `policy`."""
        return cls(
            context.training_run_dir(METHOD),
            policy,
            context.original.decode,
            rules.parse_rule(context.task.rule),
        )

    def load(self, checkpoint: training_run.Checkpoint) -> None:
        """Put the weights of `checkpoint` into the policy (changes the policy). Raises
        training_run.TrainingRunError if an earlier run deleted the checkpoint's file."""
        if self.loaded != checkpoint.sampler_id:
            tensors = training_run.load_checkpoint(self.directory, checkpoint.checkpoint_id)
            training_run.restore_trainable_tensors(self.policy.model, tensors)
            self.loaded = checkpoint.sampler_id

    def draw(
        self, checkpoint: training_run.Checkpoint, count: int, seed: int
    ) -> list[dict[str, Any]]:
        """`count` new responses of `checkpoint` at temperature 1, one row each: draw_index,
        token_ids, text, hit, logq (as generated) and seed (of the row's part)."""
        self.load(checkpoint)
        groups = self.policy.parts(count, seed)
        rows: list[dict[str, Any]] = []
        for group, draws in zip(groups, self.policy.draw(groups), strict=True):
            for draw in draws:
                text = self.decode(draw.token_ids)
                rows.append(
                    {
                        "draw_index": len(rows),
                        "token_ids": draw.token_ids,
                        "text": text,
                        "hit": self.is_hit(text),
                        "logq": draw.logq,
                        "seed": group.seed,
                    }
                )
        return rows


def submit(
    context: run_context.RunContext,
    draws: CheckpointDraws,
    point: Point,
    candidates: Sequence[training_run.Checkpoint],
    ranking: Sequence[submission.RankedCandidate],
    rule: submission.Rule = RULE,
    *,
    stream: str = "threshold-test",
) -> tuple[submission.Submission, pb_ledger.Ledger, list[dict[str, Any]]]:
    """Test the candidates at `point` in rank order and submit one (`submission.submit`). Tests
    draw with seeds of the random stream `stream`. Returns the submission, what the run at this
    point spent (the training up to its last candidate, whichever it submits, plus the tests),
    and the test draws as rows with their `candidate_id` and `rank`."""
    by_id = {checkpoint.checkpoint_id: checkpoint for checkpoint in candidates}
    test_rows: list[dict[str, Any]] = []

    def test(
        candidate: submission.RankedCandidate, threshold_test: rank_table.ThresholdTest
    ) -> int:
        seed = context.stream_seed(METHOD, stream, point.stream_index, candidate.rank)
        rows = draws.draw(by_id[candidate.candidate_id], threshold_test.draws, seed)
        test_rows.extend(
            row | {"candidate_id": candidate.candidate_id, "rank": candidate.rank} for row in rows
        )
        return sum(bool(row["hit"]) for row in rows)

    result = submission.submit(ranking, test, rule)
    spend = pb_ledger.Ledger.resume(pb_ledger.LedgerTotals.from_dict(candidates[-1].ledger))
    charges.apply(spend, charges.threshold_test(result.test_draws))
    return result, spend, test_rows
