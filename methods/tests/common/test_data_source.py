"""`data_source.reading`: a dataset on the Hub whose files are not in the local cache while the
Hub cannot be reached is an error that a command reports as one line."""

from __future__ import annotations

import pathlib
from typing import Any

import huggingface_hub
import huggingface_hub.errors
import pytest
from posteriorbench import data
from typer.testing import CliRunner

from posteriorbench_methods import cli
from posteriorbench_methods.analysis import cli as analysis_cli
from posteriorbench_methods.common import data_source, run_context

RUNNER = CliRunner()


def test_the_error_names_the_dataset() -> None:
    with pytest.raises(data_source.UnreachableDataError, match="'my/dataset' is not an existing"):
        with data_source.reading("my/dataset"):
            raise huggingface_hub.errors.LocalEntryNotFoundError("not in the cache")
    with pytest.raises(data_source.UnreachableDataError, match=data.HUB_DATASET):
        with data_source.reading(None):
            raise huggingface_hub.errors.LocalEntryNotFoundError("not in the cache")
    # Another error is not changed, and a block without one runs.
    with pytest.raises(FileNotFoundError, match="another"):
        with data_source.reading("my/dataset"):
            raise FileNotFoundError("another")
    with data_source.reading("my/dataset"):
        pass


@pytest.mark.parametrize(
    "arguments",
    [
        ["sft", "run", "--task", "wc_1853/goblin_mention", "--out", "OUT"],
        ["gfn", "reselect", "--task", "wc_1853/goblin_mention", "--out", "OUT"],
        ["prompt", "write-candidates", "--task", "wc_1853/goblin_mention", "--pool-file",
         "OUT/pool.parquet", "--candidates", "4", "--writer-url", "http://localhost:1/v1",
         "--writer-model", "writer"],
        ["steering", "build-vectors", "--behavior", "goblin_mention", "--prompts", "PROMPTS",
         "--method-data", "OUT/vectors", "--judge-url", "http://localhost:1/v1", "--judge-model",
         "judge"],
        ["time-operations", "--out", "OUT/times.json"],
        ["generalization", "evaluate", "--task", "gemma-4-31b-it/wc_0670/em_dash", "--sampler",
         "fallback"],
    ],
    ids=lambda arguments: " ".join(arguments[:2]),
)  # fmt: skip
def test_a_dataset_that_cannot_be_reached_is_reported_as_one_line(
    arguments: list[str], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--data` names a dataset on the Hub, the local cache does not hold it and the Hub cannot
    be reached: every command that reads the dataset says so in one line, before it loads a
    model."""

    def not_in_the_cache(*_: Any, **__: Any) -> str:
        raise huggingface_hub.errors.LocalEntryNotFoundError("not in the cache; no connection")

    def no_model(*_: Any, **__: Any) -> Any:
        raise AssertionError("the model is not loaded")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", not_in_the_cache)
    monkeypatch.setattr(run_context, "load_original_model", no_model)
    monkeypatch.setattr(run_context, "load_task_original_model", no_model)
    monkeypatch.setattr(analysis_cli.original_model.OriginalModel, "from_pretrained", no_model)
    prompts = tmp_path / "goblin_mention.json"
    prompts.write_text(
        '{"behavior": "goblin_mention", "name": "goblin", "instruction_pairs": '
        '[{"positive": "mention a goblin", "negative": "mention no goblin"}], '
        '"questions": ["how do i make cheese"]}'
    )
    out = tmp_path / "out"
    out.mkdir()
    if arguments[:2] == ["gfn", "reselect"]:
        # `reselect` first asks for the training run that it reads.
        monkeypatch.setattr(
            "posteriorbench_methods.common.reselection.require_training_run", lambda *_: None
        )
    replaced = [
        part.replace("OUT", str(out)).replace("PROMPTS", str(prompts)) for part in arguments
    ]
    result = RUNNER.invoke(cli.app, [*replaced, "--data", "absent/dataset"])
    assert result.exit_code == 1, result.output
    assert result.output.startswith("Error: 'absent/dataset' is not an existing directory")
    assert "--data" in result.output and result.output.count("\n") == 1
