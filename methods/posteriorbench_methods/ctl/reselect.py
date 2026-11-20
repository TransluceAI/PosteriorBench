"""CTL's checkpoint of a finished training run, chosen again (`common.reselection`) among the
candidates that `run` had at each update cap and budget: at the 1% threshold (`reselect`), or by
CTL's own rule, the twist as it is when training ends, which is the last candidate, without a
threshold test (`final_checkpoint`).
"""

from __future__ import annotations

from posteriorbench_methods.common import (
    reselection,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.ctl import run, selection
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import training, twisted_model

RULE_1PCT = submission.at_one_percent(selection.RULE)


def open_training_run(
    context: run_context.RunContext, settings: ctl_settings.Settings
) -> tuple[list[training_run.Checkpoint], twisted_model.TwistedModel]:
    """The checkpoints of the finished training run of the context's task and replicate, and a
    new twisted model to load them into."""
    checkpoints = reselection.read_training_run(
        training.directory(context),
        training.training_run_settings(context, settings),
        training.request(settings),
    )
    return checkpoints, twisted_model.build(context)


def reselect(
    context: run_context.RunContext, settings: ctl_settings.Settings
) -> list[run_record.RunSummary]:
    """Choose a checkpoint at the 1% threshold and evaluate it, at each update cap and then each
    budget of `settings`. Writes one record of `ctl` for each under
    `<out>/reselected/threshold-1pct/` and returns their summaries."""
    checkpoints, twisted = open_training_run(context, settings)
    summaries: list[run_record.RunSummary] = []
    for point in selection.points(settings):
        candidates = selection.candidates_at(point, checkpoints, settings)
        ranking = selection.rank(context, candidates.checkpoints, RULE_1PCT)
        result, spend, test_rows = selection.submit(
            context,
            twisted,
            candidates,
            ranking,
            RULE_1PCT,
            stream=reselection.TEST_STREAM_1PCT,
            within_budget=False,
        )
        assert result.submitted is not None
        checkpoint = next(
            row
            for row in candidates.checkpoints
            if row.checkpoint_id == result.submitted.candidate_id
        )
        recorded = reselection.record_threshold_1pct(
            context,
            twisted_model.TwistedSampler(twisted),
            method=run.CTL,
            setting=point.setting,
            settings=settings,
            spend=spend,
            ranking=ranking,
            submission=result,
            checkpoint=checkpoint,
            details=candidates.describe(),
            selection_rows=training_run.selection_rows(
                training.directory(context), candidates.checkpoints
            ),
            test_rows=test_rows,
        )
        summaries.append(recorded.summary)
    return summaries


def final_checkpoint(
    context: run_context.RunContext, settings: ctl_settings.Settings
) -> list[run_record.RunSummary]:
    """Submit the last candidate and evaluate it, at each update cap and then each budget of
    `settings`. Writes one record of `ctl` for each under `<out>/reselected/final-checkpoint/`
    and returns their summaries."""
    checkpoints, twisted = open_training_run(context, settings)
    summaries: list[run_record.RunSummary] = []
    for point in selection.points(settings):
        candidates = selection.candidates_at(point, checkpoints, settings)
        checkpoint = reselection.final_checkpoint(candidates.checkpoints)
        training.load_checkpoint(context, twisted, checkpoint.checkpoint_id)
        recorded = reselection.record_own_rule(
            context,
            twisted_model.TwistedSampler(twisted),
            rule=reselection.FINAL_CHECKPOINT,
            method=run.CTL,
            setting=point.setting,
            settings=settings,
            checkpoint=checkpoint,
            candidates=candidates.checkpoints,
            details=candidates.describe(),
        )
        summaries.append(recorded.summary)
    return summaries
