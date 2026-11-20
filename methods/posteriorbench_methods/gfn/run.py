"""GFlowNet and GFlowNet + composition on one task and replicate.

A run trains the policy once (`training`), as far as its points need. At every update cap and
every budget of the settings it then submits one of the training run's checkpoints by the
submission rule (`selection`), evaluates it, and composes it with the original model
(`composition`).
"""

from __future__ import annotations

import dataclasses
import functools

from posteriorbench import ledger as pb_ledger

from posteriorbench_methods.common import (
    composition,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.gfn import composition as gfn_composition
from posteriorbench_methods.gfn import cost
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import selection
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.gfn import training

METHOD = gfn_policy.METHOD


@dataclasses.dataclass(frozen=True)
class Submitted:
    """What GFlowNet submitted at one point, for its composition. `spend` is what the point cost:
    the training up to its last candidate, plus the threshold tests."""

    point: selection.Point
    checkpoint: training_run.Checkpoint
    spend: pb_ledger.Ledger
    recorded: run_record.Recorded
    ranking: list[submission.RankedCandidate]


def run(
    context: run_context.RunContext, settings: gfn_settings.Settings
) -> list[run_record.RunSummary]:
    """Run GFlowNet on the context's task and replicate. Writes one record of `gflownet` at each
    update cap and then each budget of `settings`, each followed, with `compose`, by one of
    `gflownet+composition`; returns their summaries in that order."""
    policy = gfn_policy.build(context, settings)
    checkpoints = training.train(context, settings, policy)
    draws = selection.CheckpointDraws.of(context, policy)
    summaries: list[run_record.RunSummary] = []
    testable: set[str] = set()
    returned: set[str] = set()
    for point in selection.points(settings):
        submitted = submit_checkpoint(context, settings, checkpoints, draws, point)
        summaries.append(submitted.recorded.summary)
        testable |= {entry.sampler_id for entry in submitted.ranking[: selection.RULE.max_tests]}
        returned.add(submitted.checkpoint.sampler_id)
        if context.options.compose:
            summaries.append(compose(context, settings, draws, submitted).summary)
    keep = training_run.samplers_to_keep(
        context.options.keep_checkpoints, testable=testable, returned=returned
    )
    training_run.prune_checkpoints(draws.directory, checkpoints, keep)
    return summaries


def submit_checkpoint(
    context: run_context.RunContext,
    settings: gfn_settings.Settings,
    checkpoints: list[training_run.Checkpoint],
    draws: selection.CheckpointDraws,
    point: selection.Point,
) -> Submitted:
    """Submit one of the candidates at `point` by the submission rule, evaluate it and write its
    record. Leaves the submitted checkpoint's weights in the policy."""
    candidates = point.candidates(checkpoints)
    ranking = selection.rank(draws.directory, candidates)
    result, spend, test_rows = selection.submit(context, draws, point, candidates, ranking)
    assert result.submitted is not None, "a point has at least one candidate"
    assert point.budget is None or spend.budget <= point.budget, "a run stays within its budget"
    checkpoint = next(
        row for row in candidates if row.checkpoint_id == result.submitted.candidate_id
    )
    draws.load(checkpoint)
    recorded = context.evaluate_and_record(
        draws.policy.sampler(),
        method=METHOD,
        setting=point.setting,
        settings=settings,
        spend=spend,
        ranking=ranking,
        submission=result,
        submitted={"checkpoint_id": checkpoint.checkpoint_id, "update": checkpoint.update},
        details=point.describe() | {"candidates": [row.to_json() for row in candidates]},
        draws={
            "selection": training_run.selection_rows(draws.directory, candidates),
            "threshold_tests": test_rows,
        },
    )
    return Submitted(point, checkpoint, spend, recorded, ranking)


def compose(
    context: run_context.RunContext,
    settings: gfn_settings.Settings,
    draws: selection.CheckpointDraws,
    plain: Submitted,
) -> run_record.Recorded:
    """Write the record of `gflownet+composition` at a point (`composition.compose`) for the
    checkpoint that the policy holds. The checkpoint after the first update is the original
    model and is not composed."""
    composed = gfn_composition.composed_policy(context, draws.policy)
    sampler_id = plain.checkpoint.sampler_id
    return composition.compose(
        context,
        plain.recorded,
        settings=settings,
        spend=plain.spend,
        is_original_model=training.is_original_model(plain.checkpoint),
        selection_draws=lambda setting: gfn_composition.selection_draws(
            context, composed, setting, sampler_id=sampler_id, is_hit=draws.is_hit
        ),
        threshold_test=lambda setting, count: gfn_composition.threshold_test(
            context, composed, setting, count, sampler_id=sampler_id, is_hit=draws.is_hit
        ),
        sampler=composed.sampler,
        charge=functools.partial(cost.composition_charges, response_limit=composed.response_limit),
        details=plain.point.describe(),
    )
