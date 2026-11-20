"""SFT's `reselect`: the checkpoint of a finished run chosen again at the 1% threshold
(`common.reselection`), from the training run of each budget."""

from __future__ import annotations

from posteriorbench_methods.common import (
    reselection,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.sft import budget_rule, draws_from_p, run, samplers, selection
from posteriorbench_methods.sft import settings as sft_settings
from posteriorbench_methods.sft import train

RULE_1PCT = submission.at_one_percent(selection.RULE)


def reselect(
    context: run_context.RunContext, settings: sft_settings.Settings
) -> list[run_record.RunSummary]:
    """Choose a checkpoint at the 1% threshold and evaluate it, at every budget of `settings` (or
    once, on the whole corpus), from the training runs that `run` wrote with the same settings
    and `--keep-checkpoints all`. Writes one record of `sft` per budget under
    `<out>/reselected/threshold-1pct/` and returns their summaries. A budget whose draws held no
    hit raises reselection.NoTrainingRunError."""
    draws = draws_from_p.open_draws(context)
    if settings.training_set == "corpus":
        division, _, corpus_cost = run.corpus_division(settings, draws)
        return [reselect_division(context, settings, division, corpus_cost).summary]
    return [
        reselect_budget(context, settings, draws, budget).summary for budget in settings.budgets
    ]


def reselect_budget(
    context: run_context.RunContext,
    settings: sft_settings.Settings,
    draws: draws_from_p.DrawsFromP,
    budget: int,
) -> run_record.Recorded:
    """`reselect` at one budget."""
    division = budget_rule.divide(budget, settings, draws.arrival)
    if division.hits == 0:
        raise reselection.NoTrainingRunError(
            f"the budget {budget} pays for no hit, so `run` trained nothing there and there is "
            "no checkpoint to choose"
        )
    return reselect_division(context, settings, division, None)


def reselect_division(
    context: run_context.RunContext,
    settings: sft_settings.Settings,
    division: budget_rule.Division,
    corpus_cost: float | None,
) -> run_record.Recorded:
    """`reselect` for the training run of one division of `run` (`corpus_cost` as in
    `train.open_run`)."""
    budget = division.budget
    directory = context.training_run_dir(run.SFT, budget=budget)
    checkpoints = reselection.read_training_run(
        directory,
        train.training_run_settings(context, settings, budget),
        training_run.Request(to_end=True),
    )
    sft_run = train.open_run(context, settings, division, corpus_cost=corpus_cost)
    spend = train.spend_after(checkpoints[-1], None)
    ranking = selection.rank(sft_run, checkpoints, RULE_1PCT)
    result, test_rows = selection.submit(
        sft_run, ranking, spend, RULE_1PCT, stream=reselection.TEST_STREAM_1PCT
    )
    assert result.submitted is not None
    submitted = next(
        row for row in checkpoints if row.checkpoint_id == result.submitted.candidate_id
    )
    return reselection.record_threshold_1pct(
        context,
        samplers.sampler(sft_run.models),
        method=run.SFT,
        setting=run_record.setting_label(budget=budget),
        settings=settings,
        spend=spend,
        ranking=ranking,
        submission=result,
        checkpoint=submitted,
        details={
            "division": division.to_json(),
            "candidates": [checkpoint.to_json() for checkpoint in checkpoints],
        },
        selection_rows=training_run.selection_rows(directory, checkpoints),
        test_rows=test_rows,
    )
