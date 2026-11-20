"""SFT's training: the trained layers, the order of the training responses, the checkpoints, and
continuing a training run."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import pytest
import torch

from posteriorbench_methods.common import hf_runtime, run_context, training_run
from posteriorbench_methods.sft import budget_rule, draws_from_p, settings, train

BUDGET = 6000
SETTINGS = settings.Settings(budgets=(BUDGET,), trainable_last_layers=1, learning_rate=1e-3)


def open_run(
    context: run_context.RunContext, chosen: settings.Settings = SETTINGS
) -> tuple[train.SftRun, list[tuple[int, ...]]]:
    """A run at BUDGET with its training set."""
    draws = draws_from_p.open_draws(context)
    division = budget_rule.divide(BUDGET, chosen, draws.arrival)
    training_set = [hit.token_ids for hit in draws.take(division.hits)]
    return train.open_run(context, chosen, division), training_set


def test_only_the_last_layers_are_trainable(make_tiny_model: Callable[..., Any]) -> None:
    model = make_tiny_model()
    train.freeze_all_but_last_layers(model, 1)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable and all(".layers.1." in name for name in trainable)
    train.freeze_all_but_last_layers(model, 2)
    assert any(".layers.0." in name for name, v in model.named_parameters() if v.requires_grad)
    assert not model.get_input_embeddings().weight.requires_grad
    with pytest.raises(ValueError, match="2 decoder layers"):
        train.freeze_all_but_last_layers(model, 3)


def test_training_reads_the_training_set_epoch_after_epoch(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    run, _ = open_run(make_context(tmp_path))
    division = run.division
    epochs = [train.epoch_order(run, epoch) for epoch in (1, 2, 3, 4)]
    assert all(sorted(order) == list(range(division.hits)) for order in epochs)
    assert epochs[0] != epochs[1] and epochs[0] == train.epoch_order(run, 1)
    read = [index for batch in train.batches(run) for index in batch]
    assert read == [index for order in epochs for index in order][: division.training_examples]
    sizes = [len(batch) for batch in train.batches(run)]
    assert len(sizes) == division.updates and set(sizes[:-1]) == {division.batch}
    assert list(train.batches(run, 5)) == list(train.batches(run))[4:]


def test_train_saves_the_checkpoints_of_the_division(
    make_context: Callable[..., run_context.RunContext],
    make_tiny_model: Callable[..., Any],
    tmp_path: pathlib.Path,
) -> None:
    run, training_set = open_run(make_context(tmp_path))
    checkpoints, spend = train.train(run, training_set)
    division, draws = run.division, SETTINGS.selection_draws
    assert tuple(row.update for row in checkpoints) == division.checkpoint_updates
    assert [row.checkpoint_id for row in checkpoints][:2] == ["update_000001", "update_000002"]
    assert training_run.read_index(run.directory) == checkpoints
    done = training_run.read_done(run.directory)
    assert done is not None and done.status == "complete" and done.updates == division.updates
    assert training_run.load_resume(run.directory) is None
    # Every checkpoint has other weights, and its own selection draws.
    assert len({row.sampler_id for row in checkpoints}) == len(checkpoints)
    for row in checkpoints:
        selection = training_run.read_selection_draws(run.directory, row.checkpoint_id)
        assert len(selection) == draws == row.selection_draws
        assert sum(r["hit"] for r in selection) == row.selection_hits
        assert all((r["logp"] is not None) == r["hit"] for r in selection)
    # The run has paid for its draws, its updates and the selection draws.
    hits = sum(row.selection_hits for row in checkpoints)
    assert spend.inferences == (
        division.draws + division.training_examples + len(checkpoints) * draws + hits
    )
    assert spend.budget == (
        division.draws + 2 * division.training_examples + 2 * len(checkpoints) * draws + hits
    )
    assert spend.budget == checkpoints[-1].budget_spent <= BUDGET - 1912
    assert checkpoints[-1].extra["examples"] == division.training_examples
    # Only the last layer has changed.
    original = dict(make_tiny_model().named_parameters())
    for name, value in run.models.q.model.named_parameters():
        assert torch.equal(value, original[name]) != value.requires_grad, name


class Interrupted(Exception):
    pass


def interrupt_at_update(monkeypatch: pytest.MonkeyPatch, update: int) -> None:
    """Make the `update`-th update of the next training raise Interrupted. An update is the only
    call of the teacher forcing with gradients."""
    compute = hf_runtime.sequence_logprobs_train
    updates = 0

    def interrupting(lm: hf_runtime.CausalLM, responses: Sequence[Sequence[int]]) -> torch.Tensor:
        nonlocal updates
        updates += torch.is_grad_enabled()
        if torch.is_grad_enabled() and updates == update:
            raise Interrupted
        return compute(lm, responses)

    monkeypatch.setattr(hf_runtime, "sequence_logprobs_train", interrupting)


def test_an_interrupted_training_run_continues_to_the_same_checkpoints(
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    once, training_set = open_run(make_context(tmp_path / "once"))
    expected, expected_spend = train.train(once, training_set)
    context = make_context(tmp_path / "interrupted")
    # Update 7 lies between the checkpoints after updates 4 and 8.
    with monkeypatch.context() as patch:
        interrupt_at_update(patch, 7)
        with pytest.raises(Interrupted):
            train.train(open_run(context)[0], training_set)
    directory = open_run(context)[0].directory
    assert [row.update for row in training_run.read_index(directory)] == [1, 2, 4]
    assert training_run.read_done(directory) is None
    continued, spend = train.train(open_run(context)[0], training_set)
    assert [row.sampler_id for row in continued] == [row.sampler_id for row in expected]
    assert [row.ledger for row in continued] == [row.ledger for row in expected]
    assert spend.totals() == expected_spend.totals()
    for row in expected:
        assert training_run.read_selection_draws(
            directory, row.checkpoint_id
        ) == training_run.read_selection_draws(once.directory, row.checkpoint_id)


def test_a_complete_training_run_is_not_trained_again(
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context(tmp_path)
    run, training_set = open_run(context)
    checkpoints, spend = train.train(run, training_set)
    interrupt_at_update(monkeypatch, 1)
    again, spend_again = train.train(open_run(context)[0], training_set)
    assert again == checkpoints and spend_again.totals() == spend.totals()
    assert spend_again.budget_limit == BUDGET
    # The same directory with another setting of the method is another training run.
    other = settings.Settings(budgets=(BUDGET,), trainable_last_layers=1, learning_rate=1e-4)
    with pytest.raises(training_run.TrainingRunError):
        train.train(open_run(context, other)[0], training_set)
    # The other budgets of a command do not matter.
    more_budgets = settings.Settings(
        budgets=(BUDGET, 2 * BUDGET), trainable_last_layers=1, learning_rate=1e-3
    )
    assert train.train(open_run(context, more_budgets)[0], training_set)[0] == checkpoints


def test_an_update_of_long_responses_passes_through_the_model_in_parts() -> None:
    """A part holds the next responses while its responses times the longest of them, with the
    prompt, stay within the limit."""
    responses = [[1] * length for length in (3, 3, 8, 2, 2, 2, 30, 1)]
    parts = train.update_parts(responses, prompt_tokens=2, token_limit=20)
    # With the prompt the responses have 5, 5, 10, 4, 4, 4, 32 and 3 tokens. The one of 32 is
    # beyond the limit by itself and is a part of its own.
    assert [[len(r) for r in part] for part in parts] == [[3, 3], [8, 2], [2, 2], [30], [1]]
    # In the order of the batch, and nothing lost.
    assert [r for part in parts for r in part] == responses
    for part in parts:
        size = len(part) * max(2 + len(r) for r in part)
        assert size <= 20 or len(part) == 1
    # The next response would not have fitted.
    for part, following in zip(parts, parts[1:], strict=False):
        longest = max(2 + len(r) for r in [*part, following[0]])
        assert (len(part) + 1) * longest > 20
    # The benchmark's batches are one part: 32 responses of at most 50 tokens.
    benchmark_batch = [[1] * 50] * 32
    assert train.update_parts(benchmark_batch, prompt_tokens=34, token_limit=11104) == [
        benchmark_batch
    ]
    # At 2,048 tokens a part holds 5 responses.
    long = [[1] * 2048] * 32
    assert {len(p) for p in train.update_parts(long, prompt_tokens=22, token_limit=11104)} == {
        5,
        2,
    }


def test_an_update_in_parts_is_the_update_of_the_whole_batch(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """The gradients of the parts add up to the gradient of the batch, so the weights after one
    update agree to rounding."""
    after: list[dict[str, torch.Tensor]] = []
    losses: list[float] = []
    for tokens in (11104, 40):
        chosen = dataclasses.replace(SETTINGS, update_tokens=tokens)
        run, training_set = open_run(make_context(tmp_path / str(tokens)), chosen)
        model = run.models.q.model
        parameters = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        batch = [training_set[i] for i in next(train.batches(run))]
        prompt = len(run.models.q.prompt_ids)
        parts = train.update_parts(batch, prompt_tokens=prompt, token_limit=tokens)
        assert (len(parts) == 1) == (tokens == 11104)
        torch.manual_seed(0)
        model.train()
        update = train._update  # pyright: ignore[reportPrivateUsage]
        losses.append(update(run, optimizer, parameters, batch))
        after.append(training_run.trainable_tensors(model))
    assert losses[0] == pytest.approx(losses[1], rel=1e-6)
    for name, whole in after[0].items():
        torch.testing.assert_close(after[1][name], whole, rtol=1e-4, atol=1e-6)
