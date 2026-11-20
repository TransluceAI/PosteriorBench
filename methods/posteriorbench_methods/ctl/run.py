"""CTL and CTL + composition on one task and replicate.

A run trains the twist once (`training`), as far as its points need. At every update cap and
every budget of the settings it then submits one of the training run's checkpoints by the
submission rule (`selection`), evaluates it, and composes it with the original model
(`composition`).
"""

from __future__ import annotations

import dataclasses

from posteriorbench import ledger

from posteriorbench_methods.common import (
    composition,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.ctl import composition as ctl_composition
from posteriorbench_methods.ctl import selection
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import training, twisted_model

CTL = "ctl"


@dataclasses.dataclass(frozen=True)
class Submitted:
    """What CTL submitted at one point, for its composition. `spend` is what the point cost: the
    training up to its last candidate, plus the threshold tests."""

    checkpoint: training_run.Checkpoint
    ranking: list[submission.RankedCandidate]
    spend: ledger.Ledger
    recorded: run_record.Recorded


def run(
    context: run_context.RunContext, settings: ctl_settings.Settings
) -> list[run_record.RunSummary]:
    """Run CTL on the context's task and replicate. Writes one record of `ctl` at each update cap
    and then each budget of `settings` (settings u=<update cap> and B=<budget>), each followed,
    with `compose`, by one of `ctl+composition`; returns their summaries in that order."""
    twisted = twisted_model.build(context)
    checkpoints = training.train(context, settings, twisted)
    summaries: list[run_record.RunSummary] = []
    testable: set[str] = set()
    returned: set[str] = set()
    for point in selection.points(settings):
        submitted = submit_checkpoint(context, settings, twisted, checkpoints, point)
        summaries.append(submitted.recorded.summary)
        testable |= {entry.sampler_id for entry in submitted.ranking[: selection.RULE.max_tests]}
        returned.add(submitted.checkpoint.sampler_id)
        if context.options.compose:
            summaries.append(compose(context, settings, twisted, submitted).summary)
    keep = training_run.samplers_to_keep(
        context.options.keep_checkpoints, testable=testable, returned=returned
    )
    training_run.prune_checkpoints(training.directory(context), checkpoints, keep)
    return summaries


def submit_checkpoint(
    context: run_context.RunContext,
    settings: ctl_settings.Settings,
    twisted: twisted_model.TwistedModel,
    checkpoints: list[training_run.Checkpoint],
    point: selection.Point,
) -> Submitted:
    """Submit one of the candidates at `point` by the submission rule, evaluate it and write its
    record. Leaves the submitted checkpoint's weights in the twisted model."""
    candidates = selection.candidates_at(point, checkpoints, settings)
    ranking = selection.rank(context, candidates.checkpoints)
    result, spend, test_rows = selection.submit(context, twisted, candidates, ranking)
    assert result.submitted is not None
    checkpoint = next(
        row for row in candidates.checkpoints if row.checkpoint_id == result.submitted.candidate_id
    )
    recorded = context.evaluate_and_record(
        twisted_model.TwistedSampler(twisted),
        method=CTL,
        setting=point.setting,
        settings=settings,
        spend=spend,
        ranking=ranking,
        submission=result,
        submitted={"checkpoint_id": checkpoint.checkpoint_id, "update": checkpoint.update},
        details=candidates.describe(),
        draws={
            "selection": training_run.selection_rows(
                training.directory(context), candidates.checkpoints
            ),
            "threshold_tests": test_rows,
        },
    )
    return Submitted(checkpoint, ranking, spend, recorded)


def compose(
    context: run_context.RunContext,
    settings: ctl_settings.Settings,
    twisted: twisted_model.TwistedModel,
    plain: Submitted,
) -> run_record.Recorded:
    """Write the record of `ctl+composition` at a point (`composition.compose`) for the
    checkpoint that the twisted model holds."""
    sampler_id = plain.checkpoint.sampler_id
    return composition.compose(
        context,
        plain.recorded,
        settings=settings,
        spend=plain.spend,
        selection_draws=lambda setting: ctl_composition.selection_draws(
            context, twisted, setting, sampler_id
        ),
        threshold_test=lambda setting, count: ctl_composition.threshold_test(
            context, twisted, setting, count, sampler_id
        ),
        sampler=lambda setting: ctl_composition.sampler(twisted, setting),
    )
