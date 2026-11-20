"""Fixtures of the analysis tests: the records of small runs of the families that train, a tiny
model with the layers of the paper's model, and a times file measured on it."""

from __future__ import annotations

import copy
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner

from posteriorbench_methods import cli
from posteriorbench_methods.analysis import gpu_times, records, time_operations

TASK_ID = "wc_1853/goblin_mention"
RESPONSE_TOKENS = 8  # of the responses that the tiny times are measured on
# Small runs: the family, the method, and the settings that make the run take seconds.
RUNS: list[tuple[str, str, list[str]]] = [
    ("sft", "sft", ["budgets=2048,6000", "trainable_last_layers=1", "learning_rate=1e-3"]),
    ("gfn", "gflownet", ["budgets=", "update_caps=2", "groups_per_update=2", "selection_draws=32"]),
    ("ctl", "ctl", ["budgets=", "update_caps=2", "responses_per_update=64", "selection_draws=32"]),
    ("dpg", "kl-dpg", ["budgets=3000,6000", "training_budget=6000", "responses_per_update=32",
                       "selection_draws=32", "warmup_updates=2", "learning_rate=3e-3"]),
    ("dpg", "gdc", ["budgets=6000", "training_budget=6000", "responses_per_update=64",
                    "loss_divisor=16", "fit_check_interval=2", "fit_check_draws=96",
                    "selection_draws=32", "warmup_updates=2", "learning_rate=3e-3"]),
]  # fmt: skip


@pytest.fixture(scope="session")
def analysis_inputs(
    tiny_model: Any,
    tiny_tokenizer: Any,
    make_tiny_original_model: Callable[..., Any],
    make_tiny_dataset: Callable[..., pathlib.Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, pathlib.Path]:
    """The model directory and the dataset of the small runs."""
    root = tmp_path_factory.mktemp("analysis_inputs")
    tiny_model.save_pretrained(root / "model")
    tiny_tokenizer.save_pretrained(root / "model")
    dataset = make_tiny_dataset(make_tiny_original_model(max_response_tokens=50), draws=1500)
    return {"model": root / "model", "data": dataset}


@pytest.fixture(scope="session")
def finished_root(
    analysis_inputs: dict[str, pathlib.Path], tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """An output root with the records of the small runs of `RUNS`, each through its command."""
    out = tmp_path_factory.mktemp("analysis_runs")
    for family, method, settings in RUNS:
        arguments = [family, "run", "--task", TASK_ID, "--out", str(out), "--device", "cpu"]
        arguments += ["--data", str(analysis_inputs["data"])]
        arguments += ["--model", str(analysis_inputs["model"]), "--evaluation-draws", "64"]
        if family == "dpg":
            arguments += ["--method", method]
        for setting in settings:
            arguments += ["--set", setting]
        result = CliRunner().invoke(cli.app, arguments)
        assert result.exit_code == 0, result.output
    return out


@pytest.fixture(scope="session")
def finished(finished_root: pathlib.Path) -> list[records.Record]:
    """The records of `finished_root`."""
    return records.find([finished_root])


@pytest.fixture(scope="session")
def hybrid_copy(tiny_hybrid_model: Any) -> Callable[[], Any]:
    """A factory of new copies of the tiny model with the 24 layers of the paper's model."""
    return lambda: copy.deepcopy(tiny_hybrid_model)


@pytest.fixture(scope="session")
def hybrid_model_dir(
    tiny_hybrid_model: Any, tiny_tokenizer: Any, tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """A directory with the tiny hybrid model and its tokenizer, as `--model` reads it."""
    directory = tmp_path_factory.mktemp("hybrid_model")
    tiny_hybrid_model.save_pretrained(directory)
    tiny_tokenizer.save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_timing(
    hybrid_copy: Callable[[], Any], tiny_task: Any
) -> tuple[gpu_times.Session, list[Any]]:
    """Every operation, timed once on the tiny hybrid model on the CPU: the session, and the
    copies of the model that the timing loaded, in order."""
    loaded: list[Any] = []

    def load() -> Any:
        loaded.append(hybrid_copy())
        return loaded[-1]

    session = time_operations.measure(
        load,
        tiny_task.prompt_token_ids,
        response_tokens=RESPONSE_TOKENS,
        stop_token_ids=(2, 0),
        repetitions=1,
        warmup=0,
        conditions={"device": "cpu"},
    )
    return session, loaded


@pytest.fixture(scope="session")
def tiny_session(tiny_timing: tuple[gpu_times.Session, list[Any]]) -> gpu_times.Session:
    return tiny_timing[0]
