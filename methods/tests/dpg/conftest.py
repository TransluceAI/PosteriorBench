"""Fixtures of the dpg tests: run contexts on the shared tiny model and dataset."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from posteriorbench import benchmark, original_model

from posteriorbench_methods.common import run_context, tracking

TASK_ID = "wc_1853/goblin_mention"
# About 5% of the tiny model's responses: its checkpoints fail the threshold tests.
RARE_RULE = 're.search(r"[a-c][a-c]", y) is not None'


@pytest.fixture(scope="session")
def rare_dataset(
    tiny_original_model: original_model.OriginalModel,
    make_tiny_dataset: Callable[..., pathlib.Path],
) -> pathlib.Path:
    """A dataset of the tiny model whose behavior is rare."""
    return make_tiny_dataset(tiny_original_model, task_id=TASK_ID, rule=RARE_RULE, draws=8000)


@pytest.fixture(scope="session")
def make_context(
    tiny_dataset: pathlib.Path,
    tiny_original_model: original_model.OriginalModel,
    make_tiny_model: Callable[..., Any],
    make_tiny_original_model: Callable[..., original_model.OriginalModel],
) -> Callable[..., run_context.RunContext]:
    """`make_context(out, method, dataset=None, **options)`: a run of `method` on the tiny task
    (of `tiny_dataset`, or of another dataset of the tiny model) with 128 evaluation draws, whose
    copies of the original model are new tiny models with the same weights."""

    def make(
        out: pathlib.Path, method: str, dataset: pathlib.Path | None = None, **options: Any
    ) -> run_context.RunContext:
        bench = benchmark.Benchmark(tiny_dataset if dataset is None else dataset)
        chosen: dict[str, Any] = {"evaluation_draws": 128, **options}
        return run_context.RunContext(
            family="dpg",
            method=method,
            options=run_context.RunOptions(task_id=TASK_ID, out=out, **chosen),
            bench=bench,
            task=bench.task(TASK_ID),
            original=tiny_original_model,
            load_model_copy=lambda: make_tiny_original_model(model=make_tiny_model()),
            tracker=tracking.Tracker(),
        )

    return make
