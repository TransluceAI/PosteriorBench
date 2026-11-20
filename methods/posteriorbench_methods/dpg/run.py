"""KL-DPG, JS-DPG or GDC, and its composition, on one task and replicate.

A run trains the policy once, as far as its budgets need (`fdpg`, `gdc`). At each budget it then
submits one of the checkpoints saved within that budget by the submission rule (`selection`),
evaluates it, and composes it with the original model (`common.composition`) unless it is the
original model itself. A checkpoint that several budgets submit is evaluated once and composed
once.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import torch
from posteriorbench import ledger

from posteriorbench_methods.common import (
    composition,
    hf_runtime,
    run_context,
    run_record,
    training_run,
)
from posteriorbench_methods.dpg import cost, fdpg, gdc, samplers, selection
from posteriorbench_methods.dpg import settings as dpg_settings
from posteriorbench_methods.dpg import training

Rows = list[dict[str, Any]]


@dataclasses.dataclass
class _Run:
    """One run, and what it has evaluated so far.

    original_model: the sampler id of the first checkpoint, the original model.
    evaluated: the record that holds the evaluation of each submitted checkpoint, by sampler id.
    composed: the same for the checkpoint composed under its chosen setting.
    composition_draws: the draws of composition, by stream, sampler id and setting id."""

    context: run_context.RunContext
    settings: dpg_settings.Settings
    policy: selection.Policy
    original_model: str
    evaluated: dict[str, run_record.Recorded] = dataclasses.field(default_factory=lambda: {})
    composed: dict[str, run_record.Recorded] = dataclasses.field(default_factory=lambda: {})
    composition_draws: dict[tuple[str, str, str], Rows] = dataclasses.field(
        default_factory=lambda: {}
    )

    @property
    def method(self) -> str:
        return self.settings.method


def check_method(context: run_context.RunContext, settings: dpg_settings.Settings) -> None:
    """Raise MethodMismatchError if `settings` are those of another method than the context's."""
    if settings.method != context.method:
        raise dpg_settings.MethodMismatchError(
            f"--method {context.method} got the settings of {settings.method}: the objective is "
            "set by --method, not by --set objective=... or a settings file"
        )


def train(
    context: run_context.RunContext, settings: dpg_settings.Settings
) -> tuple[list[training_run.Checkpoint], samplers.Models]:
    """The checkpoints of the method's training run for the context's task and replicate,
    trained as far as the budgets of `settings` need, and a copy of the original model, under the
    task's prompt, to load them into. Trains only what the training run on disk lacks."""
    run = training.Training(context, settings)
    checkpoints = training.check_directory(run)
    if checkpoints is not None:
        return checkpoints, samplers.Models.of(context, context.load_model_copy().model)
    trained = gdc.train(run) if isinstance(settings, dpg_settings.GdcSettings) else fdpg.train(run)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # the memory of the optimizer and of GDC's second model
    return trained


def run(
    context: run_context.RunContext, settings: dpg_settings.Settings
) -> list[run_record.RunSummary]:
    """Run KL-DPG, JS-DPG or GDC, whichever `settings` are for, on the context's task and
    replicate. Writes, for each budget in increasing order, the record of the method and, with
    `compose`, of `<method>+composition`, under the setting B=<budget>; returns their summaries
    in the order written."""
    check_method(context, settings)
    checkpoints, models = train(context, settings)
    directory = context.training_run_dir(settings.method)
    state = _Run(context, settings, selection.Policy(directory, models), checkpoints[0].sampler_id)
    groups, without_candidates = selection.groups(checkpoints, settings)
    records: list[run_record.Recorded] = []
    for budget in without_candidates:
        records += record_no_sampler(state, budget)
    testable: set[str] = set()
    returned: set[str] = set()
    for group in groups:
        submitted = selection.submit(context, state.policy, group)
        testable |= {entry.sampler_id for entry in submitted.ranking[: selection.RULE.max_tests]}
        returned.add(submitted.checkpoint.sampler_id)
        for budget in group.budgets:
            spend = submitted.spend(budget)
            plain = record_method(state, submitted, budget, spend)
            records.append(plain)
            if context.options.compose:
                records.append(record_composition(state, submitted.checkpoint, plain, spend))
    keep = training_run.samplers_to_keep(
        context.options.keep_checkpoints, testable=testable, returned=returned
    )
    training_run.prune_checkpoints(directory, checkpoints, keep)
    return [record.summary for record in records]


def record_no_sampler(state: _Run, budget: int) -> list[run_record.Recorded]:
    """Write the records at a budget that is too small for any candidate: the fallback, at a
    cost of the budget."""
    context = state.context
    plain = context.record_no_sampler(
        method=state.method,
        setting=run_record.setting_label(budget=budget),
        settings=state.settings,
        budget=budget,
        spend=None,
        details={"candidates": []},
    )
    if not context.options.compose:
        return [plain]
    return [
        plain,
        composition.record_not_composed(context, plain, settings=state.settings, spend=None),
    ]


def record_method(
    state: _Run, submitted: selection.Submitted, budget: int, spend: ledger.Ledger
) -> run_record.Recorded:
    """Write the method's record at `budget`: the evaluation of the submitted checkpoint, at the
    cost `spend`. The checkpoint is evaluated the first time a budget submits it."""
    context, checkpoint = state.context, submitted.checkpoint
    shared: dict[str, Any] = {
        "method": state.method,
        "setting": run_record.setting_label(budget=budget),
        "settings": state.settings,
        "spend": spend,
        "ranking": submitted.ranking,
        "submission": submitted.result,
        "submitted": {"checkpoint_id": checkpoint.checkpoint_id, "update": checkpoint.update},
        "details": {"candidates": [c.to_json() for c in submitted.group.candidates]},
        "draws": {"selection": submitted.selection_rows, "threshold_tests": submitted.test_rows},
    }
    earlier = state.evaluated.get(checkpoint.sampler_id)
    if earlier is not None:
        return context.record_reused(earlier.results, hit_scores=earlier.hit_scores, **shared)
    state.policy.load(checkpoint)
    source = samplers.returned_sampler(state.policy.models, state.method)
    recorded = context.evaluate_and_record(source, **shared)
    state.evaluated[checkpoint.sampler_id] = recorded
    return recorded


def record_composition(
    state: _Run,
    checkpoint: training_run.Checkpoint,
    plain: run_record.Recorded,
    plain_spend: ledger.Ledger,
) -> run_record.Recorded:
    """Write the record of `<method>+composition` at a budget (`composition.compose`) for the
    submitted `checkpoint`. `plain` and `plain_spend` are the method's record at the budget and
    what it spent. The draws of composition are made, and the composed sampler evaluated, the
    first time a budget submits the checkpoint; a later budget records them again at its own
    cost."""
    context, models = state.context, state.policy.models
    name = f"{state.method}+composition"

    def source(composed: composition.Setting) -> hf_runtime.ModelSampler:
        state.policy.load(checkpoint)
        return samplers.composed_sampler(models, composed, name=name)

    def drawn(
        stream: str, composed: composition.Setting, draw: Callable[..., Rows], count: int
    ) -> Rows:
        key = (stream, checkpoint.sampler_id, composed.setting_id)
        if key not in state.composition_draws:
            seed = context.stream_seed(state.method, stream, *key[1:])
            state.composition_draws[key] = draw(context, source(composed), count, seed=seed)
        assert len(state.composition_draws[key]) == count, "one checkpoint, one composition"
        return state.composition_draws[key]

    limit = models.generation.response_format.max_tokens
    recorded = composition.compose(
        context,
        plain,
        settings=state.settings,
        spend=plain_spend,
        is_original_model=checkpoint.sampler_id == state.original_model,
        selection_draws=lambda composed: drawn(
            "composition-selection", composed, samplers.selection_draws, composition.SELECTION_DRAWS
        ),
        threshold_test=lambda composed, count: drawn(
            "composition-threshold-test", composed, samplers.draw, count
        ),
        sampler=source,
        charge=functools.partial(cost.composition_charges, response_limit=limit),
        evaluated=state.composed.get(checkpoint.sampler_id),
    )
    # Keep the record that holds the evaluation of the composed sampler, for the budgets that
    # submit this checkpoint later. A record that repeats the method's evaluation (not composed,
    # or the plain setting chosen) is not one.
    if recorded.construction["details"]["composition"].get("winner_is_plain") is False:
        state.composed.setdefault(checkpoint.sampler_id, recorded)
    return recorded
