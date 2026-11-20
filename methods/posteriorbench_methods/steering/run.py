"""Steering and Steering + composition on one task and replicate (the steps are in the README).

The context gives the task's original model, whose layer count sets the default layers, its
response limit and its labels. The selection draws do not depend on the hit-rate threshold, so
runs at 10% and at 1% read the same ones from the cache directory.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from posteriorbench import ledger

from posteriorbench_methods.common import (
    charges,
    composition,
    hf_runtime,
    rank_table,
    run_context,
    run_record,
    signals,
    submission,
)
from posteriorbench_methods.steering import candidates as steering_candidates
from posteriorbench_methods.steering import cost, draws, hook, samplers
from posteriorbench_methods.steering import settings as steering_settings
from posteriorbench_methods.steering import vectors as steering_vectors

STEERING = "steering"
SETTING = run_record.setting_label()  # "none": Steering has no budget and no pool size
# Ties go to the earlier candidate, a candidate without a selection hit is not ranked, and the
# tests go on until a candidate passes or the ranking ends.
RULE = submission.Rule(max_tests=None, min_hits=1, ties="order", representative="earliest")

Rows = list[dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class SteeringRun:
    """What the steps of one run share. `settings` have their layers stated; `candidates` are
    keyed by `Candidate.key`, in candidate order."""

    context: run_context.RunContext
    settings: steering_settings.Settings
    vectors: steering_vectors.BehaviorVectors
    steered: samplers.SteeredModel
    candidates: Mapping[str, steering_candidates.Candidate]
    streams: draws.ThresholdTestStreams

    @property
    def rule(self) -> submission.Rule:
        """`RULE` at the run's hit-rate threshold."""
        return self.context.rule_at_threshold(RULE)


def check(options: run_context.RunOptions, settings: steering_settings.Settings) -> None:
    """Raise VectorError before the model is loaded if no vectors directory was given."""
    steering_vectors.vectors_directory(options.method_data)


def open_run(context: run_context.RunContext, settings: steering_settings.Settings) -> SteeringRun:
    """Load the behavior's vectors and set up the run at the settings' threshold. Settings
    without layers get `layers_at_depths` of the original model."""
    context = context.at_threshold(settings.threshold)
    settings = settings.with_layers_of(len(hook.decoder_layers(context.original.model)))
    assert settings.layers is not None
    vectors = steering_vectors.load(
        context.options.method_data, context.task.behavior, settings.layers
    )
    steered = samplers.SteeredModel.of(context, vectors)
    draws.check_cache(context, settings.selection_draws)
    found = steering_candidates.candidates(settings, list(vectors.tensors))
    return SteeringRun(
        context=context,
        settings=settings,
        vectors=vectors,
        steered=steered,
        candidates={candidate.key: candidate for candidate in found},
        streams=draws.ThresholdTestStreams(context),
    )


def run(
    context: run_context.RunContext, settings: steering_settings.Settings
) -> list[run_record.RunSummary]:
    """Write the record of `steering` and, unless --no-compose, of `steering+composition`, both
    at setting `none`; return their summaries in that order."""
    steering_run = open_run(context, settings)
    context = steering_run.context
    spend = ledger.Ledger()
    charges.apply(spend, cost.vector_construction(steering_run.vectors.cost))
    selection = selection_draws(steering_run, spend)
    ranking = rank(steering_run, selection)
    result, test_rows = submit(steering_run, ranking, spend)
    plain = record_steering(steering_run, selection, ranking, result, test_rows, spend)
    records = [plain]
    if context.options.compose:
        records.append(compose(steering_run, plain, result, spend))
    return [record.summary for record in records]


def selection_draws(steering_run: SteeringRun, spend: ledger.Ledger) -> dict[str, Rows]:
    """The selection draws of every candidate, by key. Each replicate is charged for them in
    full, also when it reads them from the cache."""
    context = steering_run.context
    selection: dict[str, Rows] = {}
    for key, candidate in steering_run.candidates.items():
        rows = draws.shared_selection_draws(
            context,
            draws.SELECTION,
            key,
            samplers.sampler(steering_run.steered, candidate),
            steering_run.settings.selection_draws,
        )
        hits = sum(row["hit"] for row in rows)
        charges.apply(
            spend, charges.selection_draws(len(rows), hits, note=f"selection draws of {key}")
        )
        context.tracker.event("selection_draws", candidate_id=key, hits=hits)
        selection[key] = rows
    return selection


def rank(
    steering_run: SteeringRun, selection: Mapping[str, Rows]
) -> list[submission.RankedCandidate]:
    """`submission.rank_from_selection_draws` under the run's rule."""
    return submission.rank_from_selection_draws(
        [
            submission.CandidateDraws(
                candidate_id=key,
                order=order,
                sampler_id=key,
                draws=signals.SelectionDraws.from_rows(selection[key]),
            )
            for order, key in enumerate(steering_run.candidates)
        ],
        steering_run.rule,
    )


def submit(
    steering_run: SteeringRun, ranking: Sequence[submission.RankedCandidate], spend: ledger.Ledger
) -> tuple[submission.Submission, Rows]:
    """Test in rank order and submit (`submission.submit`), charging each test to `spend`.
    Returns the submission and the test rows, labeled with `candidate_id` and `rank`. Nothing is
    submitted if no candidate has a selection hit."""
    rows: Rows = []

    def test(ranked: submission.RankedCandidate, threshold_test: rank_table.ThresholdTest) -> int:
        candidate = steering_run.candidates[ranked.candidate_id]
        source = samplers.sampler(steering_run.steered, candidate)
        drawn = steering_run.streams.first(candidate.key, source, threshold_test.draws)
        charges.apply(
            spend,
            charges.threshold_test(
                threshold_test.draws, note=f"threshold test, rank {ranked.rank}"
            ),
        )
        rows.extend({**row, "candidate_id": candidate.key, "rank": ranked.rank} for row in drawn)
        return sum(row["hit"] for row in drawn)

    return submission.submit(ranking, test, steering_run.rule), rows


def record_steering(
    steering_run: SteeringRun,
    selection: Mapping[str, Rows],
    ranking: Sequence[submission.RankedCandidate],
    result: submission.Submission,
    test_rows: Rows,
    spend: ledger.Ledger,
) -> run_record.Recorded:
    """Write the record of `steering`: the evaluation of the submitted candidate or, if nothing
    was submitted, the fallback at the cost that the run spent."""
    context = steering_run.context
    details = {
        "construction_cost": steering_run.vectors.cost.to_json(),
        "candidates": [
            {
                "candidate_id": key,
                **candidate.to_json(),
                "selection_hits": sum(row["hit"] for row in selection[key]),
            }
            for key, candidate in steering_run.candidates.items()
        ],
    }
    stored = {
        "selection": [{**row, "candidate_id": key} for key in selection for row in selection[key]],
        "threshold_tests": test_rows,
    }
    if result.submitted is None:
        return context.record_no_sampler(
            method=STEERING,
            setting=SETTING,
            settings=steering_run.settings,
            budget=None,
            spend=spend,
            ranking=ranking,
            submission=result,
            details=details,
            draws=stored,
        )
    submitted = steering_run.candidates[result.submitted.candidate_id]
    return context.evaluate_and_record(
        samplers.sampler(steering_run.steered, submitted),
        method=STEERING,
        setting=SETTING,
        settings=steering_run.settings,
        spend=spend,
        ranking=ranking,
        submission=result,
        submitted=submitted.to_json(),
        details=details,
        draws=stored,
    )


def compose(
    steering_run: SteeringRun,
    plain: run_record.Recorded,
    plain_result: submission.Submission,
    plain_spend: ledger.Ledger,
) -> run_record.Recorded:
    """Write the record of `steering+composition` (`composition.compose`) for Steering's
    submitted candidate. The settings' selection draws are shared by the replicates; the plain
    setting is the candidate itself, so its test reads the stream Steering's own test read."""
    steered = steering_run.steered
    test = plain_result.submitted
    candidate = None if test is None else steering_run.candidates[test.candidate_id]

    def source(composed: composition.Setting) -> hf_runtime.ModelSampler:
        assert candidate is not None, "a run that submitted nothing is not composed"
        return samplers.sampler(steered, candidate, composed)

    def key(composed: composition.Setting) -> str:
        assert candidate is not None, "a run that submitted nothing is not composed"
        return steering_candidates.sampler_key(candidate, composed, steered.response_limit)

    return composition.compose(
        steering_run.context,
        plain,
        settings=steering_run.settings,
        spend=plain_spend,
        is_original_model=candidate is not None and candidate.is_original_model,
        selection_draws=lambda composed: draws.shared_selection_draws(
            steering_run.context,
            draws.COMPOSITION_SELECTION,
            key(composed),
            source(composed),
            composition.SELECTION_DRAWS,
        ),
        threshold_test=lambda composed, count: steering_run.streams.first(
            key(composed), source(composed), count
        ),
        sampler=source,
        sampler_id=key,
    )
