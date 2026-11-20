"""Which checkpoints a budget chooses from, and what a run at a budget costs."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import torch
from posteriorbench import ledger

from posteriorbench_methods.common import run_context, submission, training_run
from posteriorbench_methods.dpg import samplers, selection, settings, training

# KL-DPG with the paper's settings, when every update takes a gradient step: a checkpoint takes
# 384 passes and an update 1,024.
UPDATES = (0, 1, 2, 4, 8, 16, 32, 64, 96)
SPENT = (384, 1792, 3200, 5632, 10112, 18688, 35456, 68608, 101760)


def checkpoint(update: int, spent: int, **changes: Any) -> training_run.Checkpoint:
    totals = ledger.Ledger()
    totals.charge(ledger.Category.GENERATION, inferences=spent / 2, budget=spent)
    saved = training_run.Checkpoint(
        checkpoint_id=training_run.checkpoint_id(update),
        update=update,
        budget_spent=spent,
        ledger=totals.totals().to_dict(),
        sampler_id=f"sampler-{update}",
        selection_draws=128,
        selection_hits=0,
    )
    return dataclasses.replace(saved, **changes)


CHECKPOINTS = [checkpoint(update, spent) for update, spent in zip(UPDATES, SPENT, strict=True)]


def test_budgets_with_the_same_candidates_form_a_group() -> None:
    chosen = settings.FdpgSettings(budgets=(2000, *(2**k for k in range(12, 20))))
    groups, without_candidates = selection.groups(CHECKPOINTS, chosen)
    assert without_candidates == [2000]  # 2,000 - 1,912 passes do not pay for a checkpoint
    assert [(group.budgets, group.test_stream) for group in groups] == [
        ((4096,), "update_000001"),
        ((8192,), "update_000004"),
        ((16384,), "update_000008"),
        ((32768,), "update_000016"),
        ((65536,), "update_000032"),
        ((131072, 262144, 524288), "update_000096"),
    ]
    for group in groups:
        within = [c for c in CHECKPOINTS if c.budget_spent <= group.budgets[0] - 1912]
        assert list(group.candidates) == within
        assert group.candidates[-1].checkpoint_id == group.test_stream


def test_the_training_budget_is_a_group_of_its_own_closed_by_a_checkpoint() -> None:
    chosen = settings.FdpgSettings(budgets=(65536, 131072), training_budget=131072)
    with pytest.raises(training_run.CandidatesError):
        selection.groups(CHECKPOINTS, chosen)  # no checkpoint closed the training budget
    closed = training_run.mark_boundary(CHECKPOINTS, 131072)
    groups, without_candidates = selection.groups(closed, chosen)
    assert without_candidates == []
    assert [(group.budgets, group.test_stream) for group in groups] == [
        ((65536,), "update_000032"),
        ((131072,), 131072),
    ]
    assert list(groups[1].candidates) == closed


def test_a_run_pays_for_the_training_until_its_last_candidate_and_for_its_tests() -> None:
    group = selection.Group((32768, 40000), tuple(CHECKPOINTS[:6]), "update_000016")
    ranking = [
        submission.RankedCandidate(
            1, "by_hits", "update_000002", 2, "sampler-2", 128, 0, None, None, ()
        )
    ]
    passed = submission.submit(ranking, lambda candidate, test: test.draws, selection.RULE)
    submitted = selection.Submitted(group, ranking, passed, CHECKPOINTS[2], [], [])
    spend = submitted.spend(32768)
    assert spend.resumed_from == ledger.LedgerTotals.from_dict(CHECKPOINTS[5].ledger)
    assert (spend.budget, spend.inferences) == (18688 + 164, 9344 + 164)
    assert spend.budget_limit == 32768


def test_the_policy_loads_a_checkpoint_unless_it_holds_its_weights(
    make_context: Callable[..., run_context.RunContext],
    make_tiny_model: Callable[..., Any],
    tmp_path: pathlib.Path,
) -> None:
    context = make_context(tmp_path, "kl-dpg")
    saved = []
    for update, seed in ((0, 0), (1, 0), (2, 1)):
        tensors = training.weights(make_tiny_model(seed=seed))
        name = training_run.checkpoint_id(update)
        sampler_id = training_run.save_checkpoint(tmp_path, name, tensors)
        saved.append(checkpoint(update, 384, sampler_id=sampler_id))
    assert saved[0].sampler_id == saved[1].sampler_id != saved[2].sampler_id
    model = make_tiny_model(seed=2)
    policy = selection.Policy(tmp_path, samplers.Models.of(context, model))
    policy.load(saved[2])
    assert policy.holds == saved[2].sampler_id
    loaded = training_run.load_checkpoint(tmp_path, "update_000002")
    assert all(torch.equal(value, loaded[key]) for key, value in model.state_dict().items())
    assert not model.training
    policy.load(saved[0])
    training_run.checkpoint_path(tmp_path, "update_000001").unlink()
    policy.load(saved[1])  # the same weights: nothing to load
    with pytest.raises(training_run.TrainingRunError):
        policy.load(dataclasses.replace(saved[1], sampler_id="other"))
