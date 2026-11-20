"""The CTL training run on the tiny model: updates, checkpoints, budgets, and continuing."""

from __future__ import annotations

import math
import pathlib
from collections.abc import Callable

import pytest
import torch
from posteriorbench import data, original_model

from posteriorbench_methods.common import run_context, run_record, training_run
from posteriorbench_methods.ctl import p_scores
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import training, twist, twisted_model

Context = Callable[..., run_context.RunContext]
Quick = Callable[..., ctl_settings.Settings]
TENSORS = ["fc1.bias", "fc1.weight", "fc2.bias", "fc2.weight", "out.bias", "out.weight"]


def train(
    context: run_context.RunContext, settings: ctl_settings.Settings
) -> list[training_run.Checkpoint]:
    return training.train(context, settings, twisted_model.build(context))


def test_the_loss_of_an_update_reads_p_q_and_the_hits_of_its_draws(
    twisted: twisted_model.TwistedModel,
    original: original_model.OriginalModel,
    tiny_task: data.Task,
) -> None:
    draws = twisted_model.sample(twisted, 40, 3)
    responses = draws.token_ids()
    token_logp = p_scores.token_logprobs(original, tiny_task.prompt_token_ids, responses)
    hits = ["a" in original.decode(response) for response in responses]
    loss, negative, positive = training.update_loss(twisted.head, draws, token_logp, hits, 1e-16)
    assert torch.allclose(negative.sum(0), torch.ones(negative.shape[1]))
    assert float(positive.sum()) == pytest.approx(1.0)
    # omega^+ of a miss is negligible, and among the hits it is proportional to p / q.
    logp = torch.tensor([math.fsum(values) for values in token_logp])
    logq = torch.tensor([math.fsum(values) for values in draws.token_logq()])
    flags = torch.tensor(hits)
    assert flags.any() and float(positive[~flags].sum()) < 1e-10
    expected = torch.softmax((logp - logq)[flags], dim=0)
    assert torch.allclose(positive[flags], expected, atol=1e-5)
    # The loss has a gradient in the twist network and nowhere else.
    loss.backward()
    with_gradient = [
        name for name, value in twisted.model.named_parameters() if value.grad is not None
    ]
    assert sorted(name.removeprefix("lm_head.") for name in with_gradient) == TENSORS
    lengths = torch.tensor(draws.lengths)
    log_psi = twist.absorb(twisted.head.log_psi_of(draws.hidden, draws.tokens), lengths)
    assert float(loss.detach()) == pytest.approx(
        float(twist.contrastive_loss(log_psi, negative, positive).detach())
    )


def test_checkpoints_follow_the_schedule_and_close_the_budgets(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    settings = quick(budgets=(3500, 4300))
    checkpoints = train(context, settings)
    directory = training.directory(context)
    assert directory == tmp_path / "training_runs" / "ctl" / "wc_1853__goblin_mention" / "rep0"
    assert [(row.update, row.forced, row.boundary_for) for row in checkpoints] == [
        (1, False, ()),
        (2, False, ()),
        (4, False, (3500,)),
        (7, True, (4300,)),
    ]
    assert checkpoints == training_run.read_index(directory)
    assert len({row.sampler_id for row in checkpoints}) == 4
    selection_inferences = 0.0
    for count, row in enumerate(checkpoints, start=1):
        assert row.checkpoint_id == f"update_{row.update:06d}"
        # An update takes 256 passes and a checkpoint 96, whatever was drawn.
        assert row.budget_spent == 256 * row.update + 96 * count
        selection_inferences += 32 + row.selection_hits
        assert row.ledger["inferences"] == 2 * 64 * row.update + selection_inferences
        for budget in row.boundary_for:
            assert row.budget_spent <= training_run.candidate_limit(budget)
        draws = training_run.read_selection_draws(directory, row.checkpoint_id)
        assert len(draws) == row.selection_draws == 32
        assert sum(draw["hit"] for draw in draws) == row.selection_hits
        assert all(
            math.isfinite(draw["logq"]) and math.isfinite(draw["logp"]) and draw["token_ids"]
            for draw in draws
        )
        tensors = training_run.load_checkpoint(directory, row.checkpoint_id)
        assert sorted(tensors) == TENSORS
        assert training_run.sampler_id(tensors) == row.sampler_id
    done = training_run.read_done(directory)
    assert done is not None and (done.status, done.updates) == ("complete", 7)
    assert not (directory / "resume.pt").exists()
    # Every update leaves its draws, with the values that its weights read.
    assert sorted(path.name for path in (directory / "training").iterdir()) == [
        f"update_{update:06d}.jsonl" for update in range(1, 8)
    ]
    rows = run_record.read_jsonl(training_run.training_draws_path(directory, 7))
    assert len(rows) == 64
    for row in rows:
        assert len(row["token_logq"]) == len(row["token_logp"]) == len(row["token_ids"])
        assert row["logq"] == math.fsum(row["token_logq"])
        assert row["logp"] == math.fsum(row["token_logp"])
        assert row["hit"] == ("a" in row["text"])


def test_training_goes_as_far_as_a_run_needs_and_continues_from_there(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path, deterministic: None
) -> None:
    """Stopping and continuing gives the checkpoints of training once."""
    context = make_context(tmp_path / "in_steps")
    directory = training.directory(context)
    assert [row.update for row in train(context, quick(update_caps=(2,)))] == [1, 2]
    done = training_run.read_done(directory)
    assert done is not None and (done.status, done.updates) == ("stopped", 2)
    assert (directory / "resume.pt").exists()
    # The budget of 3,500 passes is closed after update 4: nothing beyond it is trained.
    in_steps = train(context, quick(update_caps=(4,), budgets=(3500,)))
    assert [row.update for row in in_steps] == [1, 2, 4]
    done = training_run.read_done(directory)
    assert done is not None and (done.status, done.updates) == ("stopped", 4)
    in_steps = train(context, quick(budgets=(4300,)))

    at_once = make_context(tmp_path / "at_once")
    assert train(at_once, quick(budgets=(4300,))) == in_steps
    for row in in_steps:
        mine = training_run.load_checkpoint(directory, row.checkpoint_id)
        theirs = training_run.load_checkpoint(training.directory(at_once), row.checkpoint_id)
        assert all(torch.equal(mine[name], theirs[name]) for name in TENSORS)


def test_a_training_run_that_holds_what_a_run_needs_is_not_trained_again(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    checkpoints = train(context, quick(update_caps=(2,)))
    twisted = twisted_model.build(context)
    before = twisted_model.twist_weights(twisted)
    assert training.train(context, quick(update_caps=(1,)), twisted) == checkpoints
    after = twisted_model.twist_weights(twisted)
    assert all(torch.equal(before[name], after[name]) for name in TENSORS)
    assert not training_run.training_draws_path(training.directory(context), 3).exists()


def test_a_training_run_with_other_settings_needs_another_directory(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    train(context, quick(update_caps=(1,)))
    with pytest.raises(training_run.TrainingRunError):
        train(context, quick(update_caps=(1,), learning_rate=0.02))
    with pytest.raises(training_run.TrainingRunError):
        train(context, quick(update_caps=(1,), epsilon=1e-8))
    # Another replicate has a training run of its own.
    other = make_context(tmp_path, replicate=1)
    assert [row.update for row in train(other, quick(update_caps=(1,)))] == [1]
    assert training.directory(other) != training.directory(context)


def test_a_checkpoint_is_loaded_into_the_twist_network(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    twisted = twisted_model.build(context)
    first, second = training.train(context, quick(update_caps=(2,)), twisted)
    training.load_checkpoint(context, twisted, first.checkpoint_id)
    assert training_run.sampler_id(twisted_model.twist_weights(twisted)) == first.sampler_id
    training.load_checkpoint(context, twisted, second.checkpoint_id)
    assert training_run.sampler_id(twisted_model.twist_weights(twisted)) == second.sampler_id
    with pytest.raises(ValueError):
        twisted_model.load_twist_weights(twisted, {"weight": torch.zeros(1)})
