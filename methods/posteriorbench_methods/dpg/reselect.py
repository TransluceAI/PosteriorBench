"""The checkpoint of a finished training run of KL-DPG, JS-DPG or GDC, chosen again
(`common.reselection`) among the candidates that `run` had at each budget; budgets with the same
candidates get the same choice.

- `reselect`: the 1% threshold, for each of the three methods.
- `final_checkpoint`, the own rule of KL-DPG and JS-DPG: the policy when training ends, which at
  a budget is the last candidate.
- `last_sampling_distribution` and `final_policy`, GDC's own rules for its two models: the last
  candidate that is an r, and the last candidate. A sampling distribution that a fit check has
  just replaced is a copy of the policy, so the last candidate is the policy whichever of the two
  it was saved as.

An own rule has no threshold test.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from posteriorbench_methods.common import (
    reselection,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.dpg import run, samplers, selection
from posteriorbench_methods.dpg import settings as dpg_settings
from posteriorbench_methods.dpg import training

LAST_SAMPLING_DISTRIBUTION = "last-sampling-distribution"
FINAL_POLICY = "final-policy"
RULE_1PCT = submission.at_one_percent(selection.RULE)


def open_training_run(
    context: run_context.RunContext, settings: dpg_settings.Settings
) -> tuple[list[selection.Group], selection.Policy]:
    """The budgets of `settings` in groups with the same candidates, from the finished training
    run of the method on the context's task and replicate, and a new copy of the original model
    to load its checkpoints into. Raises training_run.CandidatesError if a budget is too small
    for any candidate."""
    run.check_method(context, settings)
    trained = training.Training(context, settings)
    checkpoints = reselection.read_training_run(
        trained.directory, training.training_run_settings(trained), trained.request
    )
    groups, without_candidates = selection.groups(checkpoints, settings)
    if without_candidates:
        raise training_run.CandidatesError(
            f"the budgets {without_candidates} are too small for any candidate, so there is no "
            "checkpoint to choose"
        )
    models = samplers.Models.of(context, context.load_model_copy().model)
    return groups, selection.Policy(trained.directory, models)


def reselect(
    context: run_context.RunContext, settings: dpg_settings.Settings
) -> list[run_record.RunSummary]:
    """Choose a checkpoint at the 1% threshold and evaluate it, at each budget of `settings` in
    increasing order. Writes one record of the method for each budget under
    `<out>/reselected/threshold-1pct/` and returns their summaries."""
    groups, policy = open_training_run(context, settings)
    summaries: list[run_record.RunSummary] = []
    for group in groups:
        submitted = selection.submit(
            context, policy, group, RULE_1PCT, stream=reselection.TEST_STREAM_1PCT
        )
        policy.load(submitted.checkpoint)
        for budget in group.budgets:
            recorded = reselection.record_threshold_1pct(
                context,
                samplers.returned_sampler(policy.models, settings.method),
                method=settings.method,
                setting=run_record.setting_label(budget=budget),
                settings=settings,
                spend=submitted.spend(None),
                ranking=submitted.ranking,
                submission=submitted.result,
                checkpoint=submitted.checkpoint,
                details={"candidates": [row.to_json() for row in group.candidates]},
                selection_rows=submitted.selection_rows,
                test_rows=submitted.test_rows,
            )
            summaries.append(recorded.summary)
    return summaries


def _submit_own_choice(
    context: run_context.RunContext,
    settings: dpg_settings.Settings,
    rule: str,
    choose: Callable[[Sequence[training_run.Checkpoint]], training_run.Checkpoint],
) -> list[run_record.RunSummary]:
    """Submit the candidate that `choose` picks among the candidates of each budget, and
    evaluate it. Writes one record of the method for each budget of `settings`, in increasing
    order, under `<out>/reselected/<rule>/` and returns their summaries."""
    groups, policy = open_training_run(context, settings)
    summaries: list[run_record.RunSummary] = []
    for group in groups:
        checkpoint = choose(group.candidates)
        policy.load(checkpoint)
        for budget in group.budgets:
            recorded = reselection.record_own_rule(
                context,
                samplers.returned_sampler(policy.models, settings.method),
                rule=rule,
                method=settings.method,
                setting=run_record.setting_label(budget=budget),
                settings=settings,
                checkpoint=checkpoint,
                candidates=group.candidates,
                details={"candidates": [row.to_json() for row in group.candidates]},
            )
            summaries.append(recorded.summary)
    return summaries


def final_checkpoint(
    context: run_context.RunContext, settings: dpg_settings.FdpgSettings
) -> list[run_record.RunSummary]:
    """KL-DPG's and JS-DPG's own rule: submit the last candidate at each budget of `settings`.
    The records go under `<out>/reselected/final-checkpoint/`."""
    return _submit_own_choice(
        context, settings, reselection.FINAL_CHECKPOINT, reselection.final_checkpoint
    )


def last_sampling_distribution_among(
    candidates: Sequence[training_run.Checkpoint],
) -> training_run.Checkpoint:
    """The last of GDC's candidates that is a sampling distribution r. The first candidate, the
    original model, is one."""
    return [row for row in candidates if row.extra["model"] == "r"][-1]


def last_sampling_distribution(
    context: run_context.RunContext, settings: dpg_settings.GdcSettings
) -> list[run_record.RunSummary]:
    """GDC's own rule for its sampling distribution: submit the last candidate that is an r, at
    each budget of `settings`. The records go under
    `<out>/reselected/last-sampling-distribution/`."""
    return _submit_own_choice(
        context, settings, LAST_SAMPLING_DISTRIBUTION, last_sampling_distribution_among
    )


def final_policy(
    context: run_context.RunContext, settings: dpg_settings.GdcSettings
) -> list[run_record.RunSummary]:
    """GDC's own rule for its policy: submit the last candidate at each budget of `settings`.
    The records go under `<out>/reselected/final-policy/`."""
    return _submit_own_choice(context, settings, FINAL_POLICY, reselection.final_checkpoint)
