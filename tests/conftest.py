"""Shared fixtures: the tiny model, the synthetic task of that model written to disk in the
dataset's layout, and a small dataset that needs no model (see `support`)."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from support import datasets
from support import tiny_model as tiny_model_support

from posteriorbench import data


@pytest.fixture(scope="session")
def tiny_model() -> Any:
    return tiny_model_support.build_original_model()


@pytest.fixture(scope="session")
def synthetic(tiny_model: Any) -> datasets.Synthetic:
    """A frequent behavior (p(S) well above 10%): the response contains the letter "a"."""
    return datasets.make_synthetic(tiny_model)


@pytest.fixture(scope="session")
def common_task(synthetic: datasets.Synthetic) -> data.Task:
    return synthetic.task


@pytest.fixture(scope="session")
def common_evaluation(synthetic: datasets.Synthetic) -> data.EvaluationData:
    return synthetic.evaluation


@pytest.fixture(scope="session")
def tiny_dataset(
    synthetic: datasets.Synthetic, tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    root = tmp_path_factory.mktemp("tiny_dataset")
    datasets.write_synthetic_dataset(root, synthetic)
    return root


@pytest.fixture()
def small_dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    return datasets.build_small_dataset(tmp_path / "ds")
