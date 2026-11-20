"""SFT, Replay and SFT + composition on one task and replicate; README.md lists the steps.

At each budget a run divides the budget (`budget_rule`), takes its draws (`draws_from_p`), trains
on their hits (`train`), submits a checkpoint (`selection`), and writes the records of `sft`,
`replay` and `sft+composition`. A budget that pays for no hit records every method as the
fallback, at a cost of the budget. With `training_set` "corpus" the run has one budget, which
follows from the size of the corpus (`budget_rule.divide_corpus`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from posteriorbench import ledger
from posteriorbench import sampler as pb_sampler

from posteriorbench_methods.common import (
    charges,
    composition,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.sft import budget_rule, draws_from_p, samplers, selection
from posteriorbench_methods.sft import settings as sft_settings
from posteriorbench_methods.sft import train

SFT = "sft"
REPLAY = "replay"
Rows = list[dict[str, Any]]


def run(
    context: run_context.RunContext, settings: sft_settings.Settings
) -> list[run_record.RunSummary]:
    """Run SFT at every budget of `settings.budgets`, or once on the whole corpus. Each budget
    writes the records of `sft`, `replay` and, with `compose`, `sft+composition`, in this order,
    under the setting B=<budget>. Returns their summaries in the order written."""
    draws = draws_from_p.open_draws(context)
    if settings.training_set == "corpus":
        return [record.summary for record in run_corpus(context, settings, draws)]
    summaries: list[run_record.RunSummary] = []
    for budget in settings.budgets:
        records = run_budget(context, settings, draws, budget)
        summaries += [record.summary for record in records]
    return summaries


def run_budget(
    context: run_context.RunContext,
    settings: sft_settings.Settings,
    draws: draws_from_p.DrawsFromP,
    budget: int,
) -> list[run_record.Recorded]:
    """The run at one budget; returns the records it wrote."""
    division = budget_rule.divide(budget, settings, draws.arrival)
    context.tracker.event("division", **division.to_json())
    if division.hits == 0:
        records = [
            context.record_no_sampler(
                method=method,
                setting=run_record.setting_label(budget=budget),
                settings=settings,
                budget=budget,
                spend=None,
                details={"division": division.to_json()},
            )
            for method in (SFT, REPLAY)
        ]
        if context.options.compose:
            records.append(
                composition.record_not_composed(context, records[0], settings=settings, spend=None)
            )
        return records
    # `draw_index` is the hit's place among the run's draws, counted from 0.
    training_set: list[dict[str, Any]] = [
        {
            "draw_index": draws.arrival(number) - 1,
            "token_ids": list(hit.token_ids),
            "hit": True,
            "corpus_row": hit.row,
        }
        for number, hit in enumerate(draws.take(division.hits), start=1)
    ]
    return run_division(train.open_run(context, settings, division), training_set)


def corpus_division(
    settings: sft_settings.Settings, draws: draws_from_p.DrawsFromP
) -> tuple[budget_rule.Division, list[dict[str, Any]], float]:
    """A run on the task's whole corpus: its division, its training set as rows of a draws file
    (`draw_index` is the row in the corpus), and what the corpus costs."""
    hits = draws.take_all()
    division = budget_rule.divide_corpus(settings, corpus_draws=draws.corpus_draws, hits=len(hits))
    training_set: list[dict[str, Any]] = [
        {
            "draw_index": hit.row,
            "token_ids": list(hit.token_ids),
            "hit": True,
            "corpus_row": hit.row,
        }
        for hit in hits
    ]
    return division, training_set, draws.corpus_cost(len(hits))


def run_corpus(
    context: run_context.RunContext,
    settings: sft_settings.Settings,
    draws: draws_from_p.DrawsFromP,
) -> list[run_record.Recorded]:
    """The run on the task's whole corpus; returns the records it wrote."""
    division, training_set, corpus_cost = corpus_division(settings, draws)
    context.tracker.event("division", **division.to_json())
    sft_run = train.open_run(context, settings, division, corpus_cost=corpus_cost)
    return run_division(sft_run, training_set)


def run_division(
    sft_run: train.SftRun, training_set: list[dict[str, Any]]
) -> list[run_record.Recorded]:
    """Train on `training_set` (rows with `token_ids`), submit a checkpoint, and write the
    records of the three methods; returns them."""
    context = sft_run.context
    checkpoints, spend = train.train(sft_run, [row["token_ids"] for row in training_set])
    ranking = selection.rank(sft_run, checkpoints)
    result, test_rows = selection.submit(sft_run, ranking, spend)
    selection.delete_checkpoint_files(sft_run, checkpoints, ranking, result)
    sft = record_sft(
        sft_run,
        checkpoints,
        ranking,
        result,
        spend,
        draws={
            "training_set": training_set,
            "selection": training_run.selection_rows(sft_run.directory, checkpoints),
            "threshold_tests": test_rows,
        },
    )
    records = [sft, record_replay(sft_run, training_set)]
    if context.options.compose:
        records.append(compose(sft_run, sft, spend))
    return records


def record_sft(
    sft_run: train.SftRun,
    checkpoints: Sequence[training_run.Checkpoint],
    ranking: Sequence[submission.RankedCandidate],
    result: submission.Submission,
    spend: ledger.Ledger,
    draws: Mapping[str, Sequence[Mapping[str, Any]]],
) -> run_record.Recorded:
    """Evaluate the submitted checkpoint, which the run's model holds, and write its record."""
    assert result.submitted is not None
    submitted = next(
        row for row in checkpoints if row.checkpoint_id == result.submitted.candidate_id
    )
    model = sft_run.models.q.model
    return sft_run.context.evaluate_and_record(
        samplers.sampler(sft_run.models),
        method=SFT,
        setting=run_record.setting_label(budget=sft_run.budget),
        settings=sft_run.settings,
        spend=spend,
        ranking=ranking,
        submission=result,
        submitted={"checkpoint_id": submitted.checkpoint_id, "update": submitted.update},
        details={
            "division": sft_run.division.to_json(),
            "trained_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "candidates": [checkpoint.to_json() for checkpoint in checkpoints],
        },
        draws=draws,
    )


def record_replay(
    sft_run: train.SftRun, training_set: Sequence[Mapping[str, Any]]
) -> run_record.Recorded:
    """Evaluate Replay and write its record. Replay returns one of the rows of `training_set`
    uniformly at random, so a repeated response is that many times as likely; it runs no model
    and costs the run's draws."""
    probability = 1.0 / len(training_set)
    replay = pb_sampler.FiniteSampler(
        [(row["token_ids"], probability) for row in training_set],
        inferences_per_response=0.0,
        name=REPLAY,
    )
    spend = ledger.Ledger(budget_limit=sft_run.budget_limit)
    charges.apply(spend, sft_run.draws_charges())
    return sft_run.context.evaluate_and_record(
        replay,
        method=REPLAY,
        setting=run_record.setting_label(budget=sft_run.budget),
        settings=sft_run.settings,
        spend=spend,
        details={"division": sft_run.division.to_json()},
        draws={"training_set": training_set},
    )


def compose(
    sft_run: train.SftRun, plain: run_record.Recorded, plain_spend: ledger.Ledger
) -> run_record.Recorded:
    """Write the record of `sft+composition` (`composition.compose`) for the checkpoint that the
    run's model holds. `plain` and `plain_spend` are SFT's record and what SFT spent."""
    context, budget, models = sft_run.context, sft_run.budget, sft_run.models

    def selection_draws(composed: composition.Setting) -> Rows:
        seed = context.stream_seed("sft", "composition-selection", budget, composed.setting_id)
        source = samplers.sampler(models, composed)
        return samplers.selection_draws(context, source, composition.SELECTION_DRAWS, seed=seed)

    def threshold_test(composed: composition.Setting, draws: int) -> Rows:
        seed = context.stream_seed("sft", "composition-threshold-test", budget, composed.setting_id)
        return samplers.draw(context, samplers.sampler(models, composed), draws, seed=seed)

    return composition.compose(
        context,
        plain,
        settings=sft_run.settings,
        spend=plain_spend,
        selection_draws=selection_draws,
        threshold_test=threshold_test,
        sampler=lambda composed: samplers.sampler(models, composed),
    )
