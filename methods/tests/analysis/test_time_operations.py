"""Timing the operations on a tiny model with the layers of the paper's model, on the CPU."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

import pytest
import torch

from posteriorbench_methods.analysis import gpu_times, operations, time_operations
from posteriorbench_methods.common import hf_runtime
from posteriorbench_methods.ctl import training as ctl_training
from posteriorbench_methods.ctl import twisted_model
from posteriorbench_methods.sft import samplers as sft_samplers
from posteriorbench_methods.sft import settings as sft_settings
from posteriorbench_methods.sft import train as sft_train


def test_every_operation_that_pricing_reads_is_timed_once(tiny_session: gpu_times.Session) -> None:
    keys = [measurement.key for measurement in tiny_session.measurements]
    assert sorted(keys) == sorted(gpu_times.every_key())
    assert all(measurement.seconds > 0 for measurement in tiny_session.measurements)
    assert {measurement.repetitions for measurement in tiny_session.measurements} == {1}
    assert tiny_session.conditions == {"device": "cpu"}


def test_the_timed_session_prices_a_run_of_every_method(tiny_session: gpu_times.Session) -> None:
    times = gpu_times.Times((tiny_session,))
    done = operations.Operations(
        draws_from_p=10, sampler_draws=10, scored=10, training_forwards=10,
        gradient_responses=10, updates=1, update_batch=4,
    )  # fmt: skip
    for kind in ("nothing", "last_layers", "adapters", "twist", "all_weights"):
        assert times.session_for(kind) is tiny_session
        assert gpu_times.run_seconds(kind, done, tiny_session) > 0


def test_the_copies_that_timing_loads(
    tiny_timing: tuple[gpu_times.Session, list[Any]], tiny_hybrid_model: Any
) -> None:
    """Four copies are loaded. SFT's updates are timed first, on a copy of their own, which
    they train: its last four layers move and the others do not. The copy that the measurements
    of the original model share with the gradient step on all weights keeps its weights."""
    session, loaded = tiny_timing
    assert len(loaded) == 4
    first = session.measurements[0]
    assert (first.operation, first.trains, first.batch) == ("update", "last_layers", 1)
    original = dict(tiny_hybrid_model.named_parameters())
    moved = {
        name
        for name, value in loaded[0].named_parameters()
        if not torch.equal(value, original[name])
    }
    layers = {int(re.findall(r"\.layers\.(\d+)\.", name)[0]) for name in moved}
    assert layers == {20, 21, 22, 23}
    for name, value in loaded[1].named_parameters():
        assert torch.equal(value, original[name]), name


def test_the_timed_update_of_sft_is_the_family_s(hybrid_copy: Callable[[], Any]) -> None:
    """From the same weights and the same state of the optimizer, the update that is timed and
    the update of `sft.train` leave the same weights and return the same loss."""
    settings = sft_settings.Settings(learning_rate=1e-3)
    rows = [[5, 9, 7, 11], [6, 8, 2]]
    after: list[tuple[float, dict[str, torch.Tensor]]] = []
    for timed in (True, False):
        model = hybrid_copy()
        lm = hf_runtime.CausalLM(model, (1, 5, 6))
        sft_train.freeze_all_but_last_layers(model, settings.trainable_last_layers)
        parameters = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate)
        model.train()
        if timed:
            loss = time_operations.sft_update(
                lm, optimizer, parameters, rows, settings.gradient_clip
            )
        else:
            generation = hf_runtime.Generation(hf_runtime.ResponseFormat((2,), 8))
            run = sft_train.SftRun(
                context=cast(Any, None),
                settings=settings,
                division=cast(Any, None),
                models=sft_samplers.Models(q=lm, p=lm, generation=generation),
            )
            loss = sft_train._update(  # pyright: ignore[reportPrivateUsage]
                run, optimizer, parameters, rows
            )
        after.append(
            (loss, {name: value.detach().clone() for name, value in model.named_parameters()})
        )
    (timed_loss, timed_weights), (family_loss, family_weights) = after
    assert timed_loss == family_loss
    assert all(torch.equal(timed_weights[name], family_weights[name]) for name in timed_weights)


def test_the_timed_loss_of_the_twist_is_the_family_s(hybrid_copy: Callable[[], Any]) -> None:
    """`twist_loss` on tensors is the loss of `ctl.training.update_loss` on the same values as
    lists, the form in which an update of CTL has log p of its responses."""
    model = hybrid_copy()
    twisted_model.attach_twist_head(model, seed=0)
    head = model.lm_head
    generator = torch.Generator().manual_seed(1)
    responses, tokens, hidden = 6, 5, head.fc1.in_features
    lengths = [5, 5, 3, 5, 2, 5]
    token_logp = -3 * torch.rand((responses, tokens), generator=generator)
    for row, length in enumerate(lengths):
        token_logp[row, length:] = 0
    draws = twisted_model.Draws(
        tokens=torch.randint(4, 30, (responses, tokens), generator=generator),
        hidden=torch.randn((responses, tokens, hidden), generator=generator),
        prefix_logq=(-3 * torch.rand((responses, tokens), generator=generator)).cumsum(1),
        lengths=lengths,
    )
    hits = [True, False, False, True, False, False]
    family, _, _ = ctl_training.update_loss(
        head,
        draws,
        [token_logp[row, :length].tolist() for row, length in enumerate(lengths)],
        hits,
        1e-16,
    )
    timed = time_operations.twist_loss(
        head, draws, torch.tensor(lengths), token_logp, torch.tensor(hits), 1e-16
    )
    assert torch.equal(timed, family) and timed.requires_grad


def test_timed_generation_runs_to_the_response_limit(hybrid_copy: Callable[[], Any]) -> None:
    """No token ends a response, so every timed response has the full number of tokens."""
    lm = hf_runtime.CausalLM(hybrid_copy(), (1, 5, 6))
    generation = hf_runtime.Generation(
        hf_runtime.ResponseFormat((time_operations.NO_STOP_TOKEN,), 6)
    )
    (draws,) = hf_runtime.sample_groups(
        hf_runtime.TwoModels(lm), [hf_runtime.DrawGroup(40, seed=0)], generation
    )
    assert {len(draw.token_ids) for draw in draws} == {6}


def test_a_measurement_needs_a_repetition(hybrid_copy: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match="at least one repetition"):
        time_operations.measure(
            hybrid_copy, (1, 5), response_tokens=4, stop_token_ids=(2, 0), repetitions=0
        )
