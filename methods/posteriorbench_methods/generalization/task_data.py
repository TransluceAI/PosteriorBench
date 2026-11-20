"""The data of the six tasks: a directory, or a Hub dataset repository, with one directory per task.

    <task>/evaluation/evaluation.parquet         held-out: the references and the fallback response,
                                                 with their stored log p
    <task>/hits/<prompt_id>/<behavior>.parquet   the hits without the references, which a method
                                                 may take at 1/p̂(S) inferences per hit
    <task>/labels/labels.parquet                 LLM-judged tasks, held-out: the stored label of
                                                 every corpus response that the rubric judge called
                                                 a match, the references included

`<task>` is the task id with "__" for "/". The first two files have the package's schemas, so a
task's directory is a data source of the package. A `row` is the response's index in the corpus
that p̂(S) was estimated from.
"""

from __future__ import annotations

import os
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
from posteriorbench import corpus, data, ledger

from posteriorbench_methods.common import data_source, errors
from posteriorbench_methods.generalization import judge, tasks

LABELS_FILE = "labels/labels.parquet"
# The revision of a Hub repository that is read unless one is given.
DEFAULT_REVISION = "main"
# One row per corpus response that the rubric judge called a match, in corpus order.
LABELS_SCHEMA = pa.schema(
    [
        ("row", pa.int32()),
        ("text", pa.string()),
        ("judge_match", pa.bool_()),
        ("judge_explanation", pa.string()),
        ("review_verdict", pa.string()),
        ("review_reasons", pa.list_(pa.string())),
        ("review_explanation", pa.string()),
        ("hit", pa.bool_()),
    ]
)


class TaskDataError(errors.MethodsError):
    """The data of the six tasks is not given, or a file of it is missing or does not fit its
    task."""


def task_dir(task: tasks.GeneralizationTask) -> str:
    """The name of a task's directory in the data."""
    return tasks.task_dir_name(task.task_id)


class TaskData:
    """The data of the six tasks at a local directory or a Hub dataset id.

    `revision` applies to a Hub id only. Creating a `TaskData` reads nothing; a file of a Hub
    source is downloaded into the Hugging Face cache on first use."""

    def __init__(self, source: str | os.PathLike[str], *, revision: str | None = None) -> None:
        self.source = source
        self.local = data.is_local_source(source)
        if self.local and revision is not None:
            raise ValueError("revision applies to a Hub dataset id, not to a local directory")
        self.revision = None if self.local else (revision or DEFAULT_REVISION)
        self._commit: str | None = None

    def describe(self) -> dict[str, str | None]:
        """The source, the revision and the Hub commit of the files read so far."""
        return {"source": str(self.source), "revision": self.revision, "commit": self._commit}

    def task_source(self, task: tasks.GeneralizationTask, relative_path: str) -> pathlib.Path:
        """The local directory of the task's data, after making sure that it holds
        `relative_path` (a download, for a Hub source). Raises TaskDataError if the file is not
        there."""
        try:
            with data_source.reading(self.source):
                path = data.resolve_file(
                    self.source, f"{task_dir(task)}/{relative_path}", revision=self.revision
                )
        except data.DataError as error:
            raise TaskDataError(
                f"{task.task_id}: no {relative_path} in the data at {self.source} ({error})"
            ) from error
        if not self.local:
            self._commit = data.hub_commit(path)
        return path.parents[len(pathlib.PurePosixPath(relative_path).parts) - 1]

    def evaluation_data(self, task: tasks.GeneralizationTask) -> data.EvaluationData:
        """The task's held-out evaluation data; for evaluation only."""
        source = self.task_source(task, data.EVALUATION_FILE)
        evaluation = data.load_evaluation_data(source, task.task_id)
        if len(evaluation.references) != task.references:
            raise TaskDataError(
                f"{task.task_id}: {len(evaluation.references)} references in the data, the task "
                f"has {task.references}"
            )
        return evaluation

    def corpus(
        self,
        task: tasks.GeneralizationTask,
        spend: ledger.Ledger,
        *,
        exclude_reference_duplicates: bool = True,
    ) -> corpus.CorpusAccess:
        """The package's access to the task's hits, which charges each hit it hands out
        1/p̂(S) inferences to `spend`."""
        source = self.task_source(task, data.hits_file(task.prompt_id, task.behavior))
        return corpus.CorpusAccess(
            source, task, spend, exclude_reference_duplicates=exclude_reference_duplicates
        )

    def stored_labels(self, task: tasks.GeneralizationTask) -> list[judge.Label]:
        """The labels that a task with a rubric ships. They include the references, so they are
        for the judge only."""
        source = self.task_source(task, LABELS_FILE)
        table = pq.read_table(source / LABELS_FILE)
        missing = [name for name in LABELS_SCHEMA.names if name not in table.column_names]
        if missing:
            raise TaskDataError(
                f"{task.task_id}: the labels file has no column {', '.join(missing)}"
            )
        return [
            judge.Label(
                task.task_id,
                str(record["text"]),
                judge_match=bool(record["judge_match"]),
                judge_explanation=str(record["judge_explanation"] or ""),
                review_verdict=record["review_verdict"],
                review_reasons=tuple(record["review_reasons"] or ()),
                review_explanation=str(record["review_explanation"] or ""),
                review_answered=None if record["review_verdict"] is None else True,
            )
            for record in table.to_pylist()
        ]
