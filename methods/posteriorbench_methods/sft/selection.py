"""Which checkpoint SFT submits: the submission rule (`common.submission`) over the checkpoints
of a run's training, testing at most the first 8 in rank order (`budget_rule` sets their tests
aside)."""

from __future__ import annotations

from typing import Any

from posteriorbench import ledger

from posteriorbench_methods.common import charges, rank_table, signals, submission, training_run
from posteriorbench_methods.sft import samplers, train

RULE = submission.Rule(max_tests=submission.TRAINING_MAX_TESTS)


def rank(
    run: train.SftRun,
    checkpoints: list[training_run.Checkpoint],
    rule: submission.Rule = RULE,
) -> list[submission.RankedCandidate]:
    """The ranking of the checkpoints from their selection draws."""
    candidates = [
        submission.CandidateDraws(
            candidate_id=checkpoint.checkpoint_id,
            order=order,
            sampler_id=checkpoint.sampler_id,
            draws=signals.SelectionDraws.from_rows(
                training_run.read_selection_draws(run.directory, checkpoint.checkpoint_id)
            ),
        )
        for order, checkpoint in enumerate(checkpoints)
    ]
    return submission.rank_from_selection_draws(candidates, rule)


def submit(
    run: train.SftRun,
    ranking: list[submission.RankedCandidate],
    spend: ledger.Ledger,
    rule: submission.Rule = RULE,
    *,
    stream: str = "threshold-test",
) -> tuple[submission.Submission, list[dict[str, Any]]]:
    """Test the checkpoints in rank order and submit one (`submission.submit`). Tests draw with
    seeds of the random stream `stream` and are charged to `spend`. Returns the submission and
    the test draws as rows with their `candidate_id` and `rank`. Leaves the run's model with the
    submitted checkpoint's weights, in eval mode."""
    source = samplers.sampler(run.models)
    rows: list[dict[str, Any]] = []

    def test(
        candidate: submission.RankedCandidate, threshold_test: rank_table.ThresholdTest
    ) -> int:
        train.load_checkpoint(run, candidate.candidate_id)
        seed = run.context.stream_seed("sft", stream, run.budget, candidate.rank)
        drawn = samplers.draw(run.context, source, threshold_test.draws, seed=seed)
        charges.apply(
            spend,
            charges.threshold_test(
                threshold_test.draws, note=f"threshold test, rank {candidate.rank}"
            ),
        )
        rows.extend(
            {**row, "candidate_id": candidate.candidate_id, "rank": candidate.rank} for row in drawn
        )
        return sum(row["hit"] for row in drawn)

    result = submission.submit(ranking, test, rule)
    assert result.submitted is not None, "a training run has a checkpoint, so one is submitted"
    train.load_checkpoint(run, result.submitted.candidate_id)
    return result, rows


def delete_checkpoint_files(
    run: train.SftRun,
    checkpoints: list[training_run.Checkpoint],
    ranking: list[submission.RankedCandidate],
    result: submission.Submission,
) -> None:
    """Delete the checkpoint files that `keep_checkpoints` does not keep: with "testable" those
    beyond the tested ranks, with "returned" all but the submitted one."""
    assert result.submitted is not None
    keep = training_run.samplers_to_keep(
        run.context.options.keep_checkpoints,
        testable=[entry.sampler_id for entry in ranking[: RULE.max_tests]],
        returned=[result.submitted.sampler_id],
    )
    training_run.prune_checkpoints(run.directory, checkpoints, keep)
