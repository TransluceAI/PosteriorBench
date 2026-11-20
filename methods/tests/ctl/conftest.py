"""Fixtures of the CTL tests: a twisted model and a run context on the shared tiny model and
dataset, and settings that make a run take seconds."""

from __future__ import annotations

import copy
import pathlib
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from posteriorbench import benchmark, data, original_model

from posteriorbench_methods.common import hf_runtime, run_context, tracking
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import twisted_model

BATCH = 64  # responses in a forward pass of the original model (the family's value)


@pytest.fixture
def original(
    make_tiny_original_model: Callable[..., original_model.OriginalModel],
) -> original_model.OriginalModel:
    """The tiny original model, with responses of at most 8 tokens."""
    return make_tiny_original_model(batch_size=BATCH)


@pytest.fixture
def twisted(
    tiny_model: Any, tiny_task: data.Task, original: original_model.OriginalModel
) -> twisted_model.TwistedModel:
    """A copy of the tiny model with a new twist head (seed 7), under the tiny task's prompt."""
    model = copy.deepcopy(tiny_model)
    twisted_model.attach_twist_head(model, 7)
    return twisted_model.TwistedModel(
        model, tuple(tiny_task.prompt_token_ids), hf_runtime.ResponseFormat.of(original)
    )


@pytest.fixture
def make_context(
    tiny_dataset: pathlib.Path,
    tiny_task: data.Task,
    tiny_model: Any,
    original: original_model.OriginalModel,
    make_tiny_original_model: Callable[..., original_model.OriginalModel],
) -> Callable[..., run_context.RunContext]:
    """A factory of run contexts on the tiny task: `make_context(out, replicate=0, compose=False,
    keep_checkpoints="all", task=tiny_task)`."""

    def make(
        out: pathlib.Path,
        *,
        replicate: int = 0,
        compose: bool = False,
        keep_checkpoints: run_context.KeepCheckpoints = "all",
        task: data.Task = tiny_task,
    ) -> run_context.RunContext:
        options = run_context.RunOptions(
            task_id=task.task_id,
            out=out,
            replicate=replicate,
            compose=compose,
            keep_checkpoints=keep_checkpoints,
            evaluation_draws=64,
        )
        return run_context.RunContext(
            family="ctl",
            method="ctl",
            options=options,
            bench=benchmark.Benchmark(tiny_dataset),
            task=task,
            original=original,
            load_model_copy=lambda: make_tiny_original_model(
                model=copy.deepcopy(tiny_model), batch_size=BATCH
            ),
            tracker=tracking.Tracker(),
        )

    return make


@pytest.fixture
def quick() -> Callable[..., ctl_settings.Settings]:
    """A factory of settings for quick runs: `quick(update_caps=(1, 2))`. 64 responses per
    update and 32 selection draws, so an update takes 256 passes and a checkpoint 96. Under the
    training budgets 3,500 and 4,300, training saves checkpoints after updates 1, 2 and 4, the
    last of which closes the first budget, and one after update 7 only to close the second."""

    def make(**changes: Any) -> ctl_settings.Settings:
        values: dict[str, Any] = {
            "budgets": (),
            "update_caps": (),
            "training_budgets": (3500, 4300),
            "responses_per_update": 64,
            "selection_draws": 32,
            "learning_rate": 0.01,
        }
        return ctl_settings.Settings(**(values | changes))

    return make


@pytest.fixture
def deterministic() -> Iterator[None]:
    """Makes torch repeat an update bit for bit. The gradient of an update is accumulated into
    the rows of the twist network's last layer (`weight[tokens]`), which torch does in an order
    that can differ from run to run unless it is asked to keep it."""
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(before)
