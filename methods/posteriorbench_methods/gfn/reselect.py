"""GFlowNet's checkpoint of a finished training run, chosen again (`common.reselection`) among
the candidates that `run` had at each update cap and budget: at the 1% threshold (`reselect`),
or by GFlowNet's own rule, the highest mean log reward over the selection draws, without a
threshold test (`highest_mean_log_reward`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from posteriorbench_methods.common import (
    reselection,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import selection
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.gfn import training

METHOD = gfn_policy.METHOD
HIGHEST_MEAN_LOG_REWARD = "highest-mean-log-reward"
RULE_1PCT = submission.at_one_percent(selection.RULE)


def open_training_run(
    context: run_context.RunContext, settings: gfn_settings.Settings
) -> tuple[list[training_run.Checkpoint], selection.CheckpointDraws]:
    """The checkpoints of the finished training run of the context's task and replicate, and a
    new policy to load them into."""
    checkpoints = reselection.read_training_run(
        context.training_run_dir(METHOD),
        training.training_run_settings(context, settings),
        training.request(settings),
    )
    policy = gfn_policy.build(context, settings)
    return checkpoints, selection.CheckpointDraws.of(context, policy)


def reselect(
    context: run_context.RunContext, settings: gfn_settings.Settings
) -> list[run_record.RunSummary]:
    """Choose a checkpoint at the 1% threshold and evaluate it, at each update cap and then each
    budget of `settings`. Writes one record of `gflownet` for each under
    `<out>/reselected/threshold-1pct/` and returns their summaries."""
    checkpoints, draws = open_training_run(context, settings)
    summaries: list[run_record.RunSummary] = []
    for point in selection.points(settings):
        candidates = point.candidates(checkpoints)
        ranking = selection.rank(draws.directory, candidates, RULE_1PCT)
        result, spend, test_rows = selection.submit(
            context,
            draws,
            point,
            candidates,
            ranking,
            RULE_1PCT,
            stream=reselection.TEST_STREAM_1PCT,
        )
        assert result.submitted is not None, "a point has at least one candidate"
        checkpoint = next(
            row for row in candidates if row.checkpoint_id == result.submitted.candidate_id
        )
        draws.load(checkpoint)
        recorded = reselection.record_threshold_1pct(
            context,
            draws.policy.sampler(),
            method=METHOD,
            setting=point.setting,
            settings=settings,
            spend=spend,
            ranking=ranking,
            submission=result,
            checkpoint=checkpoint,
            details=point.describe() | {"candidates": [row.to_json() for row in candidates]},
            selection_rows=training_run.selection_rows(draws.directory, candidates),
            test_rows=test_rows,
        )
        summaries.append(recorded.summary)
    return summaries


def mean_log_reward(selection_draws: Sequence[Mapping[str, Any]], epsilon: float) -> float:
    """The mean of log R(y) over a checkpoint's selection draws (rows with `hit` and `logp`):
    log p(y) for a hit and log p(y) + log `epsilon` for a miss."""
    miss = math.log(epsilon)
    rewards = [row["logp"] + (0.0 if row["hit"] else miss) for row in selection_draws]
    return sum(rewards) / len(rewards)


def highest_mean_log_reward(
    context: run_context.RunContext, settings: gfn_settings.Settings
) -> list[run_record.RunSummary]:
    """Submit the checkpoint with the highest mean log reward, the earliest of equal means, and
    evaluate it, at each update cap and then each budget of `settings`. Writes one record of
    `gflownet` for each under `<out>/reselected/highest-mean-log-reward/`, whose details hold the
    mean of every candidate, and returns their summaries."""
    checkpoints, draws = open_training_run(context, settings)
    summaries: list[run_record.RunSummary] = []
    for point in selection.points(settings):
        candidates = point.candidates(checkpoints)
        means = {
            row.checkpoint_id: mean_log_reward(
                training_run.read_selection_draws(draws.directory, row.checkpoint_id),
                settings.epsilon,
            )
            for row in candidates
        }
        # `max` returns the first of the candidates with the highest mean, which is the earliest.
        checkpoint = max(candidates, key=lambda row: means[row.checkpoint_id])
        draws.load(checkpoint)
        recorded = reselection.record_own_rule(
            context,
            draws.policy.sampler(),
            rule=HIGHEST_MEAN_LOG_REWARD,
            method=METHOD,
            setting=point.setting,
            settings=settings,
            checkpoint=checkpoint,
            candidates=candidates,
            details=point.describe()
            | {"mean_log_reward": means, "candidates": [row.to_json() for row in candidates]},
            draws={"selection": training_run.selection_rows(draws.directory, candidates)},
        )
        summaries.append(recorded.summary)
    return summaries
