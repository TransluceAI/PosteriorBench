"""`Benchmark`: the entry point for methods and for evaluation.

A method builds its sampler from what the construction side gives it, and the evaluation side
scores the sampler:

- construction: `tasks()` / `task()` (the prompt, the behavior and its rule, p̂(S), read from the
  task table) and `corpus()` (the task's corpus hits, read from its hit file, each charged
  1/p̂(S) inferences to the method's ledger);
- evaluation: `evaluate`, `evaluate_both`, `no_sampler_result`, `fallback_result`. They load the
  task's held-out `data.EvaluationData` (references, fallback response and their stored log p)
  from the evaluation file themselves; no public method returns it.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Literal, Unpack

from posteriorbench import corpus, data, protocol, results, sampler
from posteriorbench.ledger import Ledger, LedgerTotals

TaskSubset = Literal["benchmark", "scored", "all"]
TASK_SUBSETS: tuple[TaskSubset, ...] = ("benchmark", "scored", "all")


def _require_task(task: object) -> None:
    """Raise TypeError unless `task` is a `Task`; a task id is the likely slip."""
    if not isinstance(task, data.Task):
        hint = f"; get the task with bench.task({task!r})" if isinstance(task, str) else ""
        raise TypeError(f"task must be a posteriorbench.Task, got {type(task).__name__}{hint}")


def _require_scorer(scorer: object) -> None:
    """Raise TypeError unless `scorer` has the methods of `protocol.Scorer`; a transformers model
    is the likely slip."""
    if not all(hasattr(scorer, name) for name in ("logprobs", "decode", "describe")):
        raise TypeError(
            "the third argument is the original model, a pb.OriginalModel "
            f"(pb.OriginalModel.from_pretrained()); got {type(scorer).__name__}"
        )


class Benchmark:
    """The tasks, corpus hits and evaluation of the benchmark.

    `Benchmark()` reads the task table and the evaluation file bundled with the package, so listing
    tasks and evaluating a sampler need no download; only `corpus()` needs the hit files, which it
    reads from `hits_source`: a local directory that holds `hits/<prompt_id>/<behavior>.parquet`, or
    a Hub dataset id (default: the Hub dataset `data.HUB_DATASET`).

    `Benchmark(source)` reads everything from one data source instead: a local directory with the
    dataset's layout or a Hub dataset id (see `data`); `hits_source` then defaults to `source`.
    `revision` applies to the Hub ids among the two sources (default `data.DATASET_REVISION`).

    Creating a `Benchmark` reads no data; the task table is read on first use and kept, and each
    task's evaluation data is read once when it is first evaluated. Raises ValueError if
    `revision` is given and neither source is a Hub id."""

    def __init__(
        self,
        source: str | os.PathLike[str] | None = None,
        *,
        revision: str | None = None,
        hits_source: str | os.PathLike[str] | None = None,
    ) -> None:
        self.source = source
        self.revision = revision
        if hits_source is not None:
            self.hits_source = hits_source
        else:
            self.hits_source = source if source is not None else data.HUB_DATASET
        hits_on_hub = not data.is_local_source(self.hits_source)
        if revision is not None and source is None and not hits_on_hub:
            raise ValueError("revision applies to Hub dataset ids; no source here is one")
        self._tables_source: str | os.PathLike[str] = (
            source if source is not None else data.BUNDLED_SOURCE
        )
        self._tables_revision = revision if source is not None else None
        self._hits_revision = revision if hits_on_hub else None
        self._tasks: list[data.Task] | None = None
        self._evaluation: dict[str, data.EvaluationData] = {}
        self._described: dict[str, Any] | None = None

    def describe_data(self) -> dict[str, Any]:
        """Where this benchmark's task table and evaluation file come from, recorded in every
        result: source, revision and commit. For a given `source` this is `data.describe_source`
        (a Hub id without a revision reads `data.DATASET_REVISION`). For the bundled tables the
        source is "bundled" and the revision is `data.DATASET_REVISION`, the dataset version they
        belong to."""
        if self._described is None:
            if self.source is None:
                self._described = {
                    "source": "bundled",
                    "revision": data.DATASET_REVISION,
                    "commit": None,
                }
            else:
                self._described = data.describe_source(self.source, revision=self.revision)
        return self._described

    def _stamp(self, result: results.EvaluationResult) -> results.EvaluationResult:
        return dataclasses.replace(result, dataset=dict(self.describe_data()))

    def _all_tasks(self) -> list[data.Task]:
        if self._tasks is None:
            self._tasks = data.load_tasks(self._tables_source, revision=self._tables_revision)
        return self._tasks

    def tasks(self, subset: TaskSubset = "benchmark") -> list[data.Task]:
        """The prompt-behavior pairs (`Task`s) of one subset, in table order: "benchmark" (the
        default), the benchmark's 21 tasks; "scored", the 108 scored pairs (at least 500 hits),
        which include the 21; "all", all 140 pairs. Raises ValueError for any other subset."""
        if subset not in TASK_SUBSETS:
            raise ValueError(f"subset must be one of {TASK_SUBSETS}, got {subset!r}")
        if subset == "all":
            return list(self._all_tasks())
        scored = [task for task in self._all_tasks() if task.scored]
        if subset == "scored":
            return scored
        return [task for task in scored if task.benchmark_task]

    def task(self, task_id: str) -> data.Task:
        """The task with this id (`<prompt_id>/<behavior>`); DataError if there is none."""
        return data.get_task(self._all_tasks(), task_id)

    def corpus(
        self, task: data.Task, ledger: Ledger, *, exclude_reference_duplicates: bool = True
    ) -> corpus.CorpusAccess:
        """Construction-time access to the task's corpus hits (`corpus.CorpusAccess`). It reads
        the task's hit file from `hits_source`, never the `corpus/` directory, and charges each
        distinct hit it hands out 1/p̂(S) inferences to `ledger`. The hit file does not hold the
        task's references; by default hits identical to a reference are not handed out either.
        Each call returns a new access that charges on its own, so keep one per task and
        ledger."""
        _require_task(task)
        return corpus.CorpusAccess(
            self.hits_source,
            task,
            ledger,
            exclude_reference_duplicates=exclude_reference_duplicates,
            revision=self._hits_revision,
        )

    def _evaluation_data(self, task: data.Task) -> data.EvaluationData:
        """The task's held-out evaluation data (read once, then cached). Evaluation only. Raises
        DataError if the task is not a scored pair."""
        _require_task(task)
        task.require_scored()
        if task.task_id not in self._evaluation:
            self._evaluation[task.task_id] = data.load_evaluation_data(
                self._tables_source, task.task_id, revision=self._tables_revision
            )
        return self._evaluation[task.task_id]

    def evaluate(
        self,
        task: data.Task,
        candidate: sampler.Sampler,
        scorer: protocol.Scorer,
        *,
        reference_logp: results.ReferenceLogp = "shipped",
        **options: Unpack[results.EvaluationOptions],
    ) -> results.EvaluationResult:
        """Evaluate a sampler built for `task` (`protocol.evaluate`).

        `scorer` is the original model. The keyword arguments are those of
        `results.EvaluationOptions`: `ledger` (what the method spent to build the sampler) and
        `seed` are required; `replicate` (default: `seed`), `settings`, `method`, `setting` and
        `keep_details` are optional. `reference_logp` is "shipped" (the default: the forward KL
        uses the log p of the references stored with the dataset) or "recomputed" (the scorer
        computes it)."""
        _require_task(task)
        _require_scorer(scorer)
        return self._stamp(
            protocol.evaluate(
                task,
                self._evaluation_data(task),
                candidate,
                scorer,
                reference_logp=reference_logp,
                **options,
            )
        )

    def evaluate_both(
        self,
        task: data.Task,
        candidate: sampler.Sampler,
        scorer: protocol.Scorer,
        **options: Unpack[results.EvaluationOptions],
    ) -> dict[results.ReferenceLogp, results.EvaluationResult]:
        """`evaluate` with the shipped and with the recomputed log p of the references, from one
        set of draws (`protocol.evaluate_both`): {"shipped": ..., "recomputed": ...}."""
        _require_task(task)
        _require_scorer(scorer)
        both = protocol.evaluate_both(
            task, self._evaluation_data(task), candidate, scorer, **options
        )
        return {choice: self._stamp(result) for choice, result in both.items()}

    def no_sampler_result(
        self,
        task: data.Task,
        *,
        budget: float | None,
        ledger: Ledger | LedgerTotals | None = None,
        seed: int,
        replicate: int,
        settings: results.EvaluationSettings = results.EvaluationSettings(),
        method: str | None = None,
        setting: str | None = None,
    ) -> results.EvaluationResult:
        """The result of a run that produced no sampler: the fallback's scores, at cost `budget`
        or, with `budget=None`, at the ledger's spend (`protocol.no_sampler_result`). No model
        call."""
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
        task: data.Task,
        *,
        seed: int = 0,
        replicate: int = 0,
        settings: results.EvaluationSettings = results.EvaluationSettings(),
    ) -> results.EvaluationResult:
        """The fallback's scores on `task`: what a sampler that fails the threshold receives
        (`protocol.fallback_result`). No model call."""
        return self._stamp(
            protocol.fallback_result(
                task, self._evaluation_data(task), settings=settings, seed=seed, replicate=replicate
            )
        )
