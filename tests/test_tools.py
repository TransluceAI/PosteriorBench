"""The scripts in `tools/`: the hit-file builder, the label check and adding a behavior."""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any
from unittest import mock

import add_behavior
import build_hits_index
import check_labels
import pyarrow.parquet as pq
import pytest
from support import datasets
from support.datasets import SMALL_DUPLICATE_ROWS, SMALL_TASK_ID
from typer.testing import CliRunner

import posteriorbench
from posteriorbench import benchmark, data, ledger, model_samplers, original_model

RUNNER = CliRunner()


def test_builder_writes_the_hit_file(small_dataset: pathlib.Path) -> None:
    path = small_dataset / data.hits_file("wc_0001", "goblin_mention")
    table = pq.read_table(path)
    assert table.schema.remove_metadata().equals(data.HITS_SCHEMA)
    rows = table.column("row").to_pylist()
    # The 600 hits are the even rows; the 250 references (rows 0..498) are removed.
    assert rows == list(range(500, 1200, 2))
    duplicate_rows = [r for r, d in zip(rows, table.column("reference_duplicate").to_pylist()) if d]
    assert duplicate_rows == list(SMALL_DUPLICATE_ROWS)
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    assert metadata == {
        "task_id": SMALL_TASK_ID,
        "hit_count": "600",
        "references_removed": "250",
        "reference_duplicates": "2",
        "reference_duplicate_fraction": repr(2 / 350),
        "references_with_duplicate": "2",
        "builder": f"posteriorbench {posteriorbench.__version__}",
    }
    # The unscored pair of the prompt gets no hit file.
    assert not (small_dataset / data.hits_file("wc_0001", "profanity")).exists()


def test_builder_refuses_a_wrong_hit_count(small_dataset: pathlib.Path) -> None:
    records = datasets.small_task_records()
    records[0]["hit_count"] += 2
    other = small_dataset.parent / "other"
    datasets.write_tasks(other, records)
    datasets.write_evaluation(other, [datasets.small_evaluation_data()])
    with pytest.raises(build_hits_index.HitsIndexError, match="task table says"):
        build_hits_index.build_and_write_hits_index(small_dataset, other, "wc_0001", other)


def test_builder_command(small_dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "rebuilt"
    arguments = ["--corpus", str(small_dataset), "--data", str(small_dataset)]
    arguments += ["--prompt-id", "wc_0001", "--out", str(out)]
    outcome = RUNNER.invoke(_app(build_hits_index.main), arguments)
    assert outcome.exit_code == 0, outcome.output
    (report,) = json.loads(outcome.output)
    assert report["task_id"] == SMALL_TASK_ID and report["rows"] == 350
    assert report["reference_duplicates"] == 2
    relative = data.hits_file("wc_0001", "goblin_mention")
    assert (out / relative).read_bytes() == (small_dataset / relative).read_bytes()


def _app(function: Any) -> Any:
    import typer

    app = typer.Typer()
    app.command()(function)
    return app


def test_check_labels(small_dataset: pathlib.Path, tiny_model: Any) -> None:
    by_prompt = check_labels.check_labels(str(small_dataset), prompt_id="wc_0001")
    assert by_prompt["label_mismatches"] == {"goblin_mention": 0, "profanity": 0}
    assert by_prompt["rows"] == 1200
    by_task = check_labels.check_labels(str(small_dataset), task_id=SMALL_TASK_ID, max_rows=10)
    assert by_task["label_mismatches"] == {"goblin_mention": 0} and by_task["rows"] == 10
    with pytest.raises(ValueError):
        check_labels.check_labels(str(small_dataset))
    with pytest.raises(ValueError):
        check_labels.check_labels(str(small_dataset), task_id=SMALL_TASK_ID, tokenizer_revision="x")
    transformers = pytest.importorskip("transformers")
    with mock.patch.object(
        transformers.AutoTokenizer, "from_pretrained", return_value=tiny_model.tokenizer
    ) as load:
        pinned = check_labels.check_labels(
            str(small_dataset), task_id=SMALL_TASK_ID, tokenizer=original_model.DEFAULT_MODEL
        )
        check_labels.check_labels(
            str(small_dataset), task_id=SMALL_TASK_ID, tokenizer="org/other-tokenizer"
        )
    revisions = [call.kwargs["revision"] for call in load.call_args_list]
    assert revisions == [original_model.MODEL_REVISION, None]
    assert pinned["tokenizer_revision"] == original_model.MODEL_REVISION
    outcome = RUNNER.invoke(
        _app(check_labels.main), ["--data", str(small_dataset), "--prompt-id", "wc_0001"]
    )
    assert outcome.exit_code == 0 and json.loads(outcome.output)["rows"] == 1200


def test_add_behavior(tiny_dataset: pathlib.Path, tiny_model: Any, tmp_path: pathlib.Path) -> None:
    """A new rule on the synthetic prompt gives a dataset directory that the package evaluates
    samplers on: its references are the first hits in corpus order and are not in its hit file."""
    out = tmp_path / "new_pair"
    summary = add_behavior.add_behavior(
        tiny_dataset, out, tiny_model, prompt_id="wc_1853", behavior="letter_b", rule='"b" in y',
        fallback_text="b", references=20,
    )  # fmt: skip
    bench = benchmark.Benchmark(out)
    (task,) = bench.tasks("all")
    assert task.task_id == summary["task_id"] == "wc_1853/letter_b" and task.scored
    assert (task.num_responses, task.hit_count) == (3000, summary["hit_count"])
    rule = '"b" in y'
    corpus_rows = list(data.iter_corpus(tiny_dataset, "wc_1853", columns=("row", "text")))
    hit_rows = [r["row"] for r in corpus_rows if "b" in r["text"]]
    evaluation = data.load_evaluation_data(out, task.task_id)
    assert [r.row for r in evaluation.references] == hit_rows[:20] and task.rule == rule
    spend = ledger.Ledger()
    access = bench.corpus(task, spend, exclude_reference_duplicates=False)
    assert [hit.row for hit in access.take_first(access.available)] == hit_rows[20:]
    candidate = model_samplers.original_model_sampler(tiny_model, task)
    result = bench.evaluate(task, candidate, tiny_model, ledger=spend, seed=0, replicate=0)
    assert result.task_id == task.task_id and math.isfinite(result.capped_forward_kl)
    assert bench.fallback_result(task).fallback
    with pytest.raises(ValueError, match="exists"):
        add_behavior.add_behavior(
            tiny_dataset, out, tiny_model, prompt_id="wc_1853", behavior="x", rule=rule,
            fallback_text="b",
        )  # fmt: skip
    with pytest.raises(add_behavior.NewBehaviorError, match="not a hit"):
        add_behavior.add_behavior(
            tiny_dataset, tmp_path / "a", tiny_model, prompt_id="wc_1853", behavior="x",
            rule=rule, fallback_text="c", references=20,
        )  # fmt: skip
    with pytest.raises(add_behavior.NewBehaviorError, match="needs at least 40"):
        add_behavior.add_behavior(
            tiny_dataset, tmp_path / "b", tiny_model, prompt_id="wc_1853", behavior="x",
            rule='"qqqq" in y', fallback_text="qqqq", references=20,
        )  # fmt: skip
