"""Loading the dataset layout: the task table (`Task`), the evaluation file (`EvaluationData`), the
corpus, and the same files read from a Hub dataset repository."""

from __future__ import annotations

import dataclasses
import math
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from support import datasets, hub_stub

from posteriorbench import data

HELD_OUT_COLUMNS = {
    "references",
    "reference_logp",
    "fallback_text",
    "fallback_token_ids",
    "fallback_logp",
}
SMALL_TASK_ID = datasets.SMALL_TASK_ID


def test_load_tasks(small_dataset: pathlib.Path) -> None:
    tasks = data.load_tasks(small_dataset)
    assert [t.task_id for t in tasks] == [SMALL_TASK_ID, "wc_0001/profanity"]
    task = data.get_task(tasks, SMALL_TASK_ID)
    assert task.scored and task.benchmark_task
    assert (task.hit_count, task.num_responses) == (600, 1200) and task.probability == 0.5
    assert task.log_event_probability == pytest.approx(math.log(0.5))
    unscored = data.get_task(tasks, "wc_0001/profanity")
    assert not unscored.scored
    with pytest.raises(data.DataError, match="not a scored pair"):
        unscored.require_scored()
    with pytest.raises(data.DataError, match="no hits"):
        unscored.log_event_probability
    with pytest.raises(data.DataError):
        data.get_task(tasks, "wc_9999/em_dash")
    assert data.load_tasks(small_dataset / data.TASKS_FILE) == tasks  # a single tasks file


def test_a_method_sees_no_held_out_data(small_dataset: pathlib.Path) -> None:
    """Neither the task table nor a `Task` holds a reference, the fallback response or a stored
    log p; they are in the evaluation file only."""
    assert not set(data.TASKS_SCHEMA.names) & HELD_OUT_COLUMNS
    assert set(data.EVALUATION_SCHEMA.names) == HELD_OUT_COLUMNS | {"task_id"}
    assert data.TASKS_SCHEMA.names == [field.name for field in dataclasses.fields(data.Task)]
    task = data.load_tasks(small_dataset)[0]
    for name in HELD_OUT_COLUMNS:
        assert not hasattr(task, name)


def test_bundled_tables_have_the_package_schemas() -> None:
    """The two tables shipped inside the package have the columns and types the package states,
    and the task table's redundant column agrees with the counts."""
    tasks = pq.read_table(data.BUNDLED_SOURCE / data.TASKS_FILE)
    assert tasks.schema.equals(data.TASKS_SCHEMA)
    evaluation = pq.read_schema(data.BUNDLED_SOURCE / data.EVALUATION_FILE)
    assert evaluation.equals(data.EVALUATION_SCHEMA)
    for record in tasks.to_pylist():
        assert record["probability"] == record["hit_count"] / record["num_responses"]


def test_load_evaluation_data(small_dataset: pathlib.Path) -> None:
    evaluation = data.load_evaluation_data(small_dataset, SMALL_TASK_ID)
    assert evaluation == datasets.small_evaluation_data()
    assert [r.row for r in evaluation.references] == list(range(0, 500, 2))
    assert evaluation.fallback_token_ids == (7, 8, 2) and evaluation.fallback_text == "goblin"
    with pytest.raises(data.DataError, match="only scored pairs"):
        data.load_evaluation_data(small_dataset, "wc_0001/profanity")  # not scored
    with pytest.raises(data.DataError):
        data.load_evaluation_data(small_dataset, "wc_9999/em_dash")


def test_tasks_and_evaluation_data_come_from_separate_files(small_dataset: pathlib.Path) -> None:
    """The task table alone gives the tasks, and the evaluation file alone the held-out data."""
    tasks = data.load_tasks(small_dataset)
    evaluation = data.load_evaluation_data(small_dataset, SMALL_TASK_ID)
    (small_dataset / data.EVALUATION_FILE).rename(small_dataset / "evaluation.parquet")
    assert data.load_tasks(small_dataset) == tasks  # construction does not need the evaluation file
    with pytest.raises(data.DataError, match="evaluation"):
        data.load_evaluation_data(small_dataset, SMALL_TASK_ID)
    (small_dataset / "evaluation.parquet").rename(small_dataset / data.EVALUATION_FILE)
    (small_dataset / data.TASKS_FILE).unlink()
    assert data.load_evaluation_data(small_dataset, SMALL_TASK_ID) == evaluation


def test_rejects_inconsistent_evaluation_data(small_dataset: pathlib.Path) -> None:
    evaluation = datasets.small_evaluation_data()
    short = dataclasses.replace(evaluation, reference_logp=evaluation.reference_logp[:-1])
    datasets.write_evaluation(small_dataset, [short])
    with pytest.raises(data.DataError, match="249 reference log p for 250 references"):
        data.load_evaluation_data(small_dataset, SMALL_TASK_ID)
    datasets.write_evaluation(small_dataset, [dataclasses.replace(evaluation, references=())])
    with pytest.raises(data.DataError, match="no references"):
        data.load_evaluation_data(small_dataset, SMALL_TASK_ID)
    positive = dataclasses.replace(evaluation, fallback_logp=0.5)
    datasets.write_evaluation(small_dataset, [positive])
    with pytest.raises(data.DataError, match="fallback_logp"):
        data.load_evaluation_data(small_dataset, SMALL_TASK_ID)
    datasets.write_evaluation(small_dataset, [evaluation, evaluation])
    with pytest.raises(data.DataError, match="2 rows"):
        data.load_evaluation_data(small_dataset, SMALL_TASK_ID)


def test_rejects_duplicate_task_ids(tmp_path: pathlib.Path) -> None:
    datasets.write_tasks(tmp_path, datasets.small_task_records()[:1] * 2)
    with pytest.raises(data.DataError, match="duplicate"):
        data.load_tasks(tmp_path)


def test_rejects_a_file_with_other_columns(small_dataset: pathlib.Path) -> None:
    """A table of another dataset version is refused with the missing columns and the version
    this package reads."""
    table = pa.Table.from_pylist(datasets.small_task_records(), schema=data.TASKS_SCHEMA)
    pq.write_table(table.drop_columns(["num_responses"]), small_dataset / data.TASKS_FILE)
    with pytest.raises(data.DataError, match=r"no column \['num_responses'\].*v1\.0\.0"):
        data.load_tasks(small_dataset)
    pq.write_table(table, small_dataset / data.EVALUATION_FILE)
    with pytest.raises(data.DataError, match="no column"):
        data.load_evaluation_data(small_dataset, SMALL_TASK_ID)


def test_iter_corpus(small_dataset: pathlib.Path) -> None:
    rows = list(data.iter_corpus(small_dataset, "wc_0001", columns=("row",)))
    assert [r["row"] for r in rows] == list(range(1200))
    with pytest.raises(data.DataError):
        list(data.iter_corpus(small_dataset, "wc_0002"))


def test_hub_layout(small_dataset: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hub repository has exactly the local layout: reading it through `hf_hub_download`
    (here served from the local directory) gives the same objects as reading it directly. Every
    file, corpus files included, is read at the dataset version this package reads unless a
    revision is given."""
    requested = hub_stub.install_offline_hub(monkeypatch, small_dataset)
    hub = data.HUB_DATASET
    assert data.load_tasks(hub) == data.load_tasks(small_dataset)
    assert data.load_evaluation_data(hub, SMALL_TASK_ID) == data.load_evaluation_data(
        small_dataset, SMALL_TASK_ID
    )
    hub_rows = list(data.iter_corpus(hub, "wc_0001", columns=("row",)))
    assert [r["row"] for r in hub_rows] == list(range(1200))
    assert {file for file, _ in requested} == {
        data.TASKS_FILE,
        data.EVALUATION_FILE,
        "<list>",
        "corpus/wc_0001/part-00000.parquet",
        "corpus/wc_0001/part-00001.parquet",
    }
    assert {revision for _, revision in requested} == {data.DATASET_REVISION}
    requested.clear()
    assert data.load_tasks(hub, revision="other") == data.load_tasks(small_dataset)
    list(data.iter_corpus(hub, "wc_0001", columns=("row",), revision="other"))
    assert {revision for _, revision in requested} == {"other"}
    (small_dataset / data.EVALUATION_FILE).unlink()
    with pytest.raises(data.DataError):
        data.load_evaluation_data(hub, SMALL_TASK_ID)


def test_local_sources(small_dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A revision applies to Hub sources only, and a path that does not exist is reported as
    such instead of being looked up on the Hub, for the corpus as for the tables."""
    with pytest.raises(data.DataError, match="revision"):
        data.load_tasks(small_dataset, revision="v1.0.0")
    for source in (str(tmp_path / "no-such-dir"), "./no-such-dir", "~/no-such-dir", "a/b/c"):
        with pytest.raises(data.DataError, match="no such directory"):
            data.load_tasks(source)
        with pytest.raises(data.DataError, match="no such directory"):
            list(data.iter_corpus(source, "wc_0001"))
    assert data.is_local_source(tmp_path) and data.is_local_source("./x")
    assert not data.is_local_source(data.HUB_DATASET)
