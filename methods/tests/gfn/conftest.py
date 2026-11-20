"""Fixtures of the GFlowNet tests: a run context on the shared tiny model and dataset, and settings
that make a run take seconds."""

from __future__ import annotations

import copy
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from posteriorbench import benchmark, data, original_model

from posteriorbench_methods.common import run_context, tracking
from posteriorbench_methods.gfn import settings as gfn_settings


@pytest.fixture
def make_context(
    tiny_dataset: pathlib.Path,
    tiny_task: data.Task,
    tiny_model: Any,
    tiny_original_model: original_model.OriginalModel,
    make_tiny_original_model: Callable[..., original_model.OriginalModel],
) -> Callable[..., run_context.RunContext]:
    """A factory of run contexts on the tiny task: `make_context(out, replicate=0,
    compose=False, keep_checkpoints="all")`. The copy of the model that a run trains is a deep
    copy of the tiny model, which leaves torch's generator as the run set it."""

    def make(
        out: pathlib.Path,
        *,
        replicate: int = 0,
        compose: bool = False,
        keep_checkpoints: run_context.KeepCheckpoints = "all",
    ) -> run_context.RunContext:
        options = run_context.RunOptions(
            task_id=tiny_task.task_id,
            out=out,
            replicate=replicate,
            compose=compose,
            keep_checkpoints=keep_checkpoints,
            evaluation_draws=64,
        )
        return run_context.RunContext(
            family="gfn",
            method="gflownet",
            options=options,
            bench=benchmark.Benchmark(tiny_dataset),
            task=tiny_task,
            original=tiny_original_model,
            load_model_copy=lambda: make_tiny_original_model(model=copy.deepcopy(tiny_model)),
            tracker=tracking.Tracker(),
        )

    return make


@pytest.fixture
def quick() -> Callable[..., gfn_settings.Settings]:
    """A factory of settings for quick runs: `quick(update_caps=(1, 2))`. Two groups of eight
    responses per update and 32 selection draws; training budgets of a few thousand passes."""

    def make(**changes: Any) -> gfn_settings.Settings:
        values: dict[str, Any] = {
            "budgets": (),
            "update_caps": (),
            "training_budgets": (4096, 8192),
            "groups_per_update": 2,
            "responses_per_group": 8,
            "selection_draws": 32,
            "generation_batch_size": 32,
            "forward_batch_size": 5,
        }
        return gfn_settings.Settings(**(values | changes))

    return make
