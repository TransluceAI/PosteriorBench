"""`GeneralizationTasks`: the six tasks with their data and evaluation, behind the methods of the
package's `Benchmark` with the same arguments, so that code written for a benchmark task works on
these tasks. The evaluation methods read a task's held-out data themselves and label draws with
the task's judge.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Unpack

from posteriorbench import corpus, data, protocol, results, sampler
from posteriorbench.ledger import Ledger, LedgerTotals

from posteriorbench_methods.generalization import evaluation
from posteriorbench_methods.generalization import judge as judges
from posteriorbench_methods.generalization import models, task_data, tasks


def load_original_model(
    task: tasks.GeneralizationTask,
    *,
    name: str | None = None,
    revision: str | None = None,
    dtype: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    generation_batch_size: int | None = None,
) -> models.TaskOriginalModel:
    """The task's original model with its response limit; the arguments are those of
    `models.load_original_model`. Raises `models.ModelLoadError` unless the model renders the
    task's prompt to `task.prompt_token_ids`."""
    model = models.load_original_model(
        task.model,
        max_response_tokens=task.max_response_tokens,
        name=name,
        revision=revision,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        generation_batch_size=generation_batch_size,
    )
    models.check_prompt(model, task.prompt, task.prompt_token_ids)
    return model


class GeneralizationTasks:
    """The six tasks, their hits and their evaluation. Creating the object reads no data.

    source, revision: a local directory or a Hub dataset id (`task_data.TaskData`); None lists
        the tasks only.
    judge_model, review_model: needed by the tasks with a rubric only
        (`judge.rubric_judge_model`, `judge.review_model`).
    labels: the file that keeps labels; None keeps them in memory for the life of this object.
    concurrency: how many responses are judged at the same time."""

    def __init__(
        self,
        source: str | os.PathLike[str] | None = None,
        *,
        revision: str | None = None,
        judge_model: judges.Completer | None = None,
        review_model: judges.Completer | None = None,
        labels: str | os.PathLike[str] | None = None,
        concurrency: int = 32,
    ) -> None:
        self.source = source
        self._data = None if source is None else task_data.TaskData(source, revision=revision)
        self.judge_model = judge_model
        self.review_model = review_model
        self.labels_path = labels
        self.concurrency = concurrency
        self._labels: judges.LabelStore | None = None
        self._judges: dict[str, judges.Judge] = {}
        self._evaluation: dict[str, data.EvaluationData] = {}

    def tasks(self) -> list[tasks.GeneralizationTask]:
        """The six tasks, in the order of the paper's table."""
        return list(tasks.all_tasks())

    def task(self, task_id: str) -> tasks.GeneralizationTask:
        """The task with this id; `tasks.UnknownTaskError` if there is none."""
        return tasks.get_task(task_id)

    def task_data(self) -> task_data.TaskData:
        """The tasks' data. Raises TaskDataError if no source was given."""
        if self._data is None:
            raise task_data.TaskDataError(
                "the data of the generalization tasks was not given: give its directory or its "
                "Hub dataset id (--data, or the first argument of GeneralizationTasks)"
            )
        return self._data

    def require_data_of(self, task: tasks.GeneralizationTask) -> None:
        """Raise TaskDataError unless the data holds the task's evaluation file (downloading it
        from a Hub source). A run calls this before it loads the task's model."""
        self.task_data().task_source(task, data.EVALUATION_FILE)

    def describe_data(self) -> dict[str, Any]:
        """Source, revision and commit of the tasks' data, recorded in every result."""
        return dict(self.task_data().describe())

    def _stamp(self, result: results.EvaluationResult) -> results.EvaluationResult:
        return dataclasses.replace(result, dataset=self.describe_data())

    def corpus(
        self,
        task: tasks.GeneralizationTask,
        spend: Ledger,
        *,
        exclude_reference_duplicates: bool = True,
    ) -> corpus.CorpusAccess:
        """The package's `CorpusAccess` to the task's hits, which excludes the references and
        charges 1/p̂(S) inferences to `spend` for each distinct hit it hands out."""
        return self.task_data().corpus(
            task, spend, exclude_reference_duplicates=exclude_reference_duplicates
        )

    def judge(self, task: tasks.GeneralizationTask) -> judges.Judge:
        """The task's judge (`judge.rule_or_rubric`). A rubric judge starts from the labels that
        the task's data ships and from those of the labels file."""
        if task.task_id not in self._judges:
            if task.rubric is not None:
                if self._labels is None:
                    self._labels = judges.LabelStore(self.labels_path)
                if self.judge_model is None or self.review_model is None:
                    judges.rule_or_rubric(task)  # raises: names what the task needs
                self._labels.adopt(self.task_data().stored_labels(task))
            self._judges[task.task_id] = judges.rule_or_rubric(
                task,
                judge_model=self.judge_model,
                review_model=self.review_model,
                labels=self._labels,
                concurrency=self.concurrency,
            )
        return self._judges[task.task_id]

    def _evaluation_data(self, task: tasks.GeneralizationTask) -> data.EvaluationData:
        if task.task_id not in self._evaluation:
            self._evaluation[task.task_id] = self.task_data().evaluation_data(task)
        return self._evaluation[task.task_id]

    def evaluate(
        self,
        task: tasks.GeneralizationTask,
        candidate: sampler.Sampler,
        scorer: protocol.Scorer,
        *,
        reference_logp: results.ReferenceLogp = "shipped",
        **options: Unpack[results.EvaluationOptions],
    ) -> results.EvaluationResult:
        """Evaluate a sampler built for `task` (`evaluation.evaluate`). `scorer` is the task's
        original model; the keyword arguments are those of the package's `Benchmark.evaluate`."""
        return self._stamp(
            evaluation.evaluate(
                task,
                self._evaluation_data(task),
                candidate,
                scorer,
                self.judge(task),
                reference_logp=reference_logp,
                **options,
            )
        )

    def evaluate_both(
        self,
        task: tasks.GeneralizationTask,
        candidate: sampler.Sampler,
        scorer: protocol.Scorer,
        **options: Unpack[results.EvaluationOptions],
    ) -> dict[results.ReferenceLogp, results.EvaluationResult]:
        """`evaluate` with the shipped and with the recomputed log p of the references, from one
        set of draws and labels."""
        both = evaluation.evaluate_both(
            task, self._evaluation_data(task), candidate, scorer, self.judge(task), **options
        )
        return {choice: self._stamp(result) for choice, result in both.items()}

    def no_sampler_result(
        self,
        task: tasks.GeneralizationTask,
        *,
        budget: float | None,
        ledger: Ledger | LedgerTotals | None = None,
        seed: int,
        replicate: int,
        settings: results.EvaluationSettings = results.EvaluationSettings(),
        method: str | None = None,
        setting: str | None = None,
    ) -> results.EvaluationResult:
        """The package's `protocol.no_sampler_result` on this task's data."""
        return self._stamp(
            protocol.no_sampler_result(
                task,
                self._evaluation_data(task),
                budget=budget,
                ledger=ledger,
                seed=seed,
                replicate=replicate,
                settings=settings,
                method=method,
                setting=setting,
            )
        )

    def fallback_result(
        self,
        task: tasks.GeneralizationTask,
        *,
        seed: int = 0,
        replicate: int = 0,
        settings: results.EvaluationSettings = results.EvaluationSettings(),
    ) -> results.EvaluationResult:
        """The package's `protocol.fallback_result` on this task's data; calls no model."""
        return self._stamp(
            protocol.fallback_result(
                task, self._evaluation_data(task), settings=settings, seed=seed, replicate=replicate
            )
        )
