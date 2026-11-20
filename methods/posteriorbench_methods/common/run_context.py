"""What a family's `run` receives: the run options, and the context opened from them.

`open_context` builds a `RunContext` from `RunOptions`. A family's entry point is

    run(context: RunContext, settings: Settings) -> list[run_record.RunSummary]

A task of the generalization section has its own original model, response limit and judge, so a
method reads these from the context, never from the benchmark: `original`, `label`,
`scoring_batch` and `generation_batch` (README, "Tasks beyond the benchmark").
"""

from __future__ import annotations

import dataclasses
import fractions
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Literal, Protocol, TypeVar, Unpack

import posteriorbench
from posteriorbench import benchmark, corpus, data, ledger, original_model, protocol
from posteriorbench import results as pb_results
from posteriorbench import sampler

from posteriorbench_methods.common import data_source, errors, rank_table, run_record, seeds
from posteriorbench_methods.common import submission as submission_rule
from posteriorbench_methods.common import tracking
from posteriorbench_methods.generalization import judge as judges
from posteriorbench_methods.generalization import models as generalization_models
from posteriorbench_methods.generalization import task_set
from posteriorbench_methods.generalization import tasks as generalization_tasks

SettingsT = TypeVar("SettingsT")
KeepCheckpoints = Literal["all", "testable", "returned"]
KEEP_CHECKPOINTS: tuple[KeepCheckpoints, ...] = ("all", "testable", "returned")
EVALUATION_DRAWS = posteriorbench.EvaluationSettings().draws  # the benchmark's 1,024
BENCHMARK_THRESHOLD = 0.10  # the benchmark's hit-rate threshold


class OptionsError(ValueError):
    """A run option outside its range."""


class TaskOptionsError(errors.MethodsError):
    """Run options that do not fit the task, such as options of the generalization section given
    for a task of the package."""


class Evaluator(Protocol):
    """What evaluates the sampler of a record: the package's `Benchmark`, or
    `generalization.task_set.GeneralizationTasks`. `task` is a task of the one that is called."""

    def evaluate(
        self,
        task: Any,
        candidate: sampler.Sampler,
        scorer: protocol.Scorer,
        /,
        *,
        reference_logp: pb_results.ReferenceLogp = "shipped",
        **options: Unpack[pb_results.EvaluationOptions],
    ) -> posteriorbench.EvaluationResult:
        """The evaluation of `candidate` with one choice of the log p of the references."""
        ...

    def evaluate_both(
        self,
        task: Any,
        candidate: sampler.Sampler,
        scorer: protocol.Scorer,
        /,
        **options: Unpack[pb_results.EvaluationOptions],
    ) -> dict[pb_results.ReferenceLogp, posteriorbench.EvaluationResult]:
        """The evaluation of `candidate` with the shipped and with the recomputed log p of the
        references, from one set of draws."""
        ...

    def no_sampler_result(
        self,
        task: Any,
        /,
        *,
        budget: float | None,
        ledger: ledger.Ledger | ledger.LedgerTotals | None = None,
        seed: int,
        replicate: int,
        settings: posteriorbench.EvaluationSettings = ...,
        method: str | None = None,
        setting: str | None = None,
    ) -> posteriorbench.EvaluationResult:
        """The result of a run without a sampler: the fallback's scores at a cost."""
        ...


class TaskSet(Evaluator, Protocol):
    """Where a run's task comes from: the package's `Benchmark`, or
    `generalization.task_set.GeneralizationTasks`. `corpus` charges the hits it hands out."""

    def corpus(self, task: Any, spend: ledger.Ledger, /) -> corpus.CorpusAccess: ...


@dataclasses.dataclass(frozen=True)
class BatchLimits:
    """Caps on the responses in one forward pass, for a task whose model or response limit needs
    smaller batches than the family's: `scoring` when scoring given responses, `generation` when
    generating."""

    scoring: int
    generation: int

    def __post_init__(self) -> None:
        if self.scoring < 1 or self.generation < 1:
            raise OptionsError("a forward pass holds at least one response")


@dataclasses.dataclass(frozen=True)
class RunOptions:
    """The options of every family's `run` command; `cli_options` gives each its help text.

    replicate: r >= 0. It enters the seed of every random stream that the replicates of a task do
        not share, and is the seed of the evaluation.
    data, model: None is the package's dataset and the benchmark's model at its commit, or for a
        task of the generalization section the task's original model; such a task needs its data
        in `data`.
    keep_checkpoints: the checkpoint files kept after a run: "all"; "testable", those a threshold
        test can reach in some ranking, and the submitted ones; "returned", the submitted ones.
    evaluation_draws: the benchmark's number is 1,024; another number gives no benchmark result.
    judge_*, review_*: the rubric judge and the model that reviews its matches, for a task that a
        language model judges: an OpenAI chat-completions URL, a model name, and the environment
        variable of the key.
    labels: the file that keeps the judges' labels, so that a response text is judged once; None
        keeps them in memory for the run.
    batch_size, generation_batch_size: for a task of the generalization section, caps on one
        scoring and one generating forward pass; None takes the task model's
        (`generalization.models.default_batch_sizes`). The draws of a seed depend on the
        second."""

    task_id: str
    out: pathlib.Path
    replicate: int = 0
    data: str | None = None
    data_revision: str | None = None
    model: str | None = None
    model_revision: str | None = None
    device: str | None = None
    compose: bool = True
    keep_checkpoints: KeepCheckpoints = "testable"
    evaluation_draws: int = EVALUATION_DRAWS
    wandb_project: str | None = None
    wandb_group: str | None = None
    method_data: pathlib.Path | None = None
    judge_url: str | None = None
    judge_model: str | None = None
    judge_key_env: str | None = None
    review_url: str | None = None
    review_model: str | None = None
    review_key_env: str | None = None
    labels: pathlib.Path | None = None
    judge_concurrency: int = 32
    batch_size: int | None = None
    generation_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.replicate < 0:
            raise OptionsError(f"replicates are non-negative, got {self.replicate}")
        if self.keep_checkpoints not in KEEP_CHECKPOINTS:
            raise OptionsError(f"keep_checkpoints must be one of {KEEP_CHECKPOINTS}")
        if self.evaluation_draws < 1:
            raise OptionsError("evaluation_draws must be positive")
        if self.judge_concurrency < 1:
            raise OptionsError("judge_concurrency must be positive")
        for size in (self.batch_size, self.generation_batch_size):
            if size is not None and size < 1:
                raise OptionsError("a forward pass holds at least one response")

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict (paths as strings)."""
        return {
            field.name: str(value) if isinstance(value, pathlib.Path) else value
            for field in dataclasses.fields(self)
            for value in (getattr(self, field.name),)
        }

    def evaluation_settings(self) -> posteriorbench.EvaluationSettings:
        """The benchmark's evaluation settings with `evaluation_draws` draws."""
        return posteriorbench.EvaluationSettings(draws=self.evaluation_draws)

    def generalization_options(self) -> list[str]:
        """The options read only by a task of the generalization section that differ from their
        defaults."""
        defaults = {field.name: field.default for field in dataclasses.fields(self)}
        names = (
            "judge_url",
            "judge_model",
            "judge_key_env",
            "review_url",
            "review_model",
            "review_key_env",
            "labels",
            "judge_concurrency",
            "batch_size",
            "generation_batch_size",
        )
        return [name for name in names if getattr(self, name) != defaults[name]]


def resolve_device(device: str | None) -> str:
    """`device`, or "cuda" if a GPU is available and "cpu" otherwise."""
    if device is not None:
        return device
    import torch  # torch loads only when a run starts

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclasses.dataclass
class RunContext:
    """One run of a family on one task and replicate.

    method: the method that was asked for; a run may write the records of several ("sft",
        "replay" and "sft+composition", for example).
    original: the task's original model (the benchmark's in float32 for a task of the package):
        p for the method, and the scorer of the evaluation.
    load_model_copy: a new copy of the original model, for a method that trains.
    judge: None for a task with a rule.
    batch_limits: None for a task of the package.
    threshold: the hit-rate threshold of the run, 0.10 or 0.01 (`at_threshold`)."""

    family: str
    method: str
    options: RunOptions
    bench: TaskSet
    task: data.Task
    original: original_model.OriginalModel
    load_model_copy: Callable[[], original_model.OriginalModel]
    tracker: tracking.Tracker
    judge: judges.Judge | None = None
    batch_limits: BatchLimits | None = None
    threshold: float = BENCHMARK_THRESHOLD

    @property
    def replicate(self) -> int:
        return self.options.replicate

    def label(self, responses: Sequence[Sequence[int]]) -> list[bool]:
        """Whether each response is a hit: by the task's rule, or for a task that a language model
        judges by the two judge models, which are asked about every text without a label yet.
        Raises `judge.JudgeError` if a judge model cannot be asked."""
        if self.judge is None:
            return protocol.label(self.task, self.original, responses)
        return self.judge.label([self.original.decode(response) for response in responses])

    def scoring_batch(self, size: int) -> int:
        """`size`, the family's own scoring batch, capped by the task's limit."""
        return size if self.batch_limits is None else min(size, self.batch_limits.scoring)

    def generation_batch(self, size: int) -> int:
        """`size`, the family's own generation batch, capped by the task's limit."""
        return size if self.batch_limits is None else min(size, self.batch_limits.generation)

    def at_threshold(self, threshold: float) -> RunContext:
        """This run at the hit-rate threshold `threshold`, 0.10 (the benchmark's) or 0.01. Its
        records go to `records_root` and are evaluated at that threshold."""
        if threshold not in rank_table.THRESHOLDS:
            raise OptionsError(
                f"the hit-rate threshold is one of {rank_table.THRESHOLDS}, got {threshold}"
            )
        return dataclasses.replace(self, threshold=threshold)

    def rule_at_threshold(self, rule: submission_rule.Rule) -> submission_rule.Rule:
        """`rule`, a method's settings at the benchmark's threshold, at the threshold of this run
        (`submission.at_one_percent` at 1%)."""
        if self.threshold == BENCHMARK_THRESHOLD:
            return rule
        return submission_rule.at_one_percent(rule)

    @property
    def records_root(self) -> pathlib.Path:
        """Where the records of this run go: the output root, or at the 1% threshold
        `<out>/reselected/threshold-1pct/`."""
        if self.threshold == BENCHMARK_THRESHOLD:
            return self.options.out
        return run_record.one_percent_root(self.options.out)

    def _evaluation_settings(
        self, threshold: fractions.Fraction | None = None
    ) -> posteriorbench.EvaluationSettings:
        """The run's evaluation settings, at `threshold` if given, else at the run's threshold."""
        evaluation = self.options.evaluation_settings()
        if threshold is None and self.threshold != BENCHMARK_THRESHOLD:
            threshold = fractions.Fraction(str(self.threshold))
        if threshold is None:
            return evaluation
        return dataclasses.replace(evaluation, threshold=threshold)

    def stream_seed(
        self, owner: str, stream: str, *index: str | int, shared_by_replicates: bool = False
    ) -> int:
        """`seeds.stream_seed` for this run's task and replicate. With `shared_by_replicates` the
        stream is the same in every replicate."""
        replicate = None if shared_by_replicates else self.replicate
        return seeds.stream_seed(owner, self.task.task_id, replicate, stream, *index)

    def run_dir(self, method: str, setting: str) -> pathlib.Path:
        """The record directory of `method` at `setting` for this task and replicate."""
        return run_record.run_dir(
            self.records_root, method, setting, self.task.task_id, self.replicate
        )

    def training_run_dir(self, method: str, *, budget: int | None = None) -> pathlib.Path:
        """The directory of the training run of `method` for this task and replicate
        (`run_record.training_run_dir`)."""
        return run_record.training_run_dir(
            self.options.out, method, self.task.task_id, self.replicate, budget=budget
        )

    def cache_dir(self) -> pathlib.Path:
        """The directory of the work that this family shares between the replicates of the task."""
        return run_record.cache_dir(self.options.out, self.family, self.task.task_id)

    def _construction_and_config(
        self,
        *,
        method: str,
        setting: str,
        settings: Any,
        spend: ledger.Ledger | ledger.LedgerTotals | None,
        ranking: Sequence[submission_rule.RankedCandidate] | None,
        submission: submission_rule.Submission | None,
        status: str,
        submitted: Mapping[str, Any] | None,
        details: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """construction.json and config.json of one record. Off the benchmark's threshold the
        details begin with `threshold`."""
        if self.threshold != BENCHMARK_THRESHOLD:
            details = {"threshold": self.threshold, **(details or {})}
        test = None if submission is None else submission.submitted
        identified: dict[str, Any] | None = None
        if test is not None or submitted is not None:
            identified = {**({} if test is None else test.to_json()), **(submitted or {})}
        construction = run_record.construction_record(
            method=method,
            setting=setting,
            task_id=self.task.task_id,
            replicate=self.replicate,
            status=status,
            spend=spend,
            submitted=identified,
            ranking=None if ranking is None else [entry.to_json() for entry in ranking],
            submission=None if submission is None else submission.to_json(),
            details=details,
        )
        config = run_record.config_record(
            family=self.family,
            method=method,
            setting=setting,
            task_id=self.task.task_id,
            replicate=self.replicate,
            settings=settings,
            options=self.options.to_json(),
            scorer=self.original,
        )
        return construction, config

    def _write(
        self,
        out: pathlib.Path,
        directory: pathlib.Path,
        results: dict[str, posteriorbench.EvaluationResult],
        construction: Mapping[str, Any],
        config: Mapping[str, Any],
        draws: Mapping[str, Sequence[Mapping[str, Any]]],
        hit_scores: run_record.HitScores | None = None,
    ) -> run_record.Recorded:
        """Write the record in `directory`, append its line to `<out>/summary.jsonl` and log the
        event `record`."""
        run_record.write_record(
            directory,
            results=results,
            construction=construction,
            config=config,
            draws=draws,
            hit_scores=hit_scores,
        )
        summary = run_record.summarize(results["shipped"], out=out, directory=directory)
        run_record.append_summary(out, summary)
        self.tracker.event("record", status=construction["status"], **summary.to_json())
        return run_record.Recorded(results, directory, construction, summary, hit_scores)

    def evaluate_and_record(
        self,
        candidate: sampler.Sampler,
        *,
        method: str,
        setting: str,
        settings: Any,
        spend: ledger.Ledger | ledger.LedgerTotals,
        ranking: Sequence[submission_rule.RankedCandidate] | None = None,
        submission: submission_rule.Submission | None = None,
        status: str | None = None,
        submitted: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        draws: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        threshold: fractions.Fraction | None = None,
        out: pathlib.Path | None = None,
    ) -> run_record.Recorded:
        """Evaluate `candidate`, the sampler that `method` returns at `setting`, and write its
        record (`run_record`).

        spend: what the method spent to build the sampler.
        ranking, submission: present if the method chose its sampler by the submission rule.
        status: by default the submission's, or "no_selection" without one.
        submitted: what identifies the submitted sampler for the method (a checkpoint and its
            update, a prompt, a setting), recorded with the test of the submitted candidate.
        details: the method's own entries of construction.json.
        draws: rows by file name (draws/<name>.jsonl), as `run_record.write_draws` takes them.
        threshold, out: the evaluation threshold and the root, where they are not the run's."""
        if status is None:
            status = "no_selection" if submission is None else submission.status
        evaluation = self._evaluation_settings(threshold)
        construction, config = self._construction_and_config(
            method=method,
            setting=setting,
            settings=settings,
            spend=spend,
            ranking=ranking,
            submission=submission,
            status=status,
            submitted=submitted,
            details=details,
        )
        root = self.records_root if out is None else out
        both = self.bench.evaluate_both(
            self.task,
            candidate,
            self.original,
            ledger=spend,
            seed=self.replicate,
            replicate=self.replicate,
            settings=evaluation,
            method=method,
            setting=setting,
            keep_details=True,
        )
        results = {str(reference_logp): result for reference_logp, result in both.items()}
        directory = run_record.run_dir(root, method, setting, self.task.task_id, self.replicate)
        all_draws = {
            **(draws or {}),
            run_record.EVALUATION_DRAWS_NAME: run_record.evaluation_draw_rows(results["shipped"]),
        }
        hit_scores = run_record.score_every_hit(
            self.task, candidate, self.original, results["shipped"]
        )
        return self._write(root, directory, results, construction, config, all_draws, hit_scores)

    def record_no_sampler(
        self,
        *,
        method: str,
        setting: str,
        settings: Any,
        budget: float | None,
        spend: ledger.Ledger | ledger.LedgerTotals | None,
        ranking: Sequence[submission_rule.RankedCandidate] | None = None,
        submission: submission_rule.Submission | None = None,
        status: str | None = None,
        submitted: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        draws: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> run_record.Recorded:
        """Write the record of a run that produced no sampler: the fallback's scores at a cost of
        `budget` (a training budget B), or with budget=None of `spend`; `spend` may be None when
        `budget` is given. `status` is by default the submission's, or "no_sampler" without one.
        The other arguments are those of `evaluate_and_record`."""
        if status is None:
            status = "no_sampler" if submission is None else submission.status
        construction, config = self._construction_and_config(
            method=method,
            setting=setting,
            settings=settings,
            spend=spend,
            ranking=ranking,
            submission=submission,
            status=status,
            submitted=submitted,
            details=details,
        )
        result = self.bench.no_sampler_result(
            self.task,
            budget=budget,
            ledger=spend,
            seed=self.replicate,
            replicate=self.replicate,
            settings=self._evaluation_settings(),
            method=method,
            setting=setting,
        )
        root = self.records_root
        directory = run_record.run_dir(root, method, setting, self.task.task_id, self.replicate)
        return self._write(root, directory, {"shipped": result}, construction, config, draws or {})

    def record_reused(
        self,
        results: Mapping[str, posteriorbench.EvaluationResult],
        *,
        method: str,
        setting: str,
        settings: Any,
        spend: ledger.Ledger | ledger.LedgerTotals | None,
        ranking: Sequence[submission_rule.RankedCandidate] | None = None,
        submission: submission_rule.Submission | None = None,
        status: str | None = None,
        submitted: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        draws: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        hit_scores: run_record.HitScores | None = None,
    ) -> run_record.Recorded:
        """Write the record of `method` at `setting` from the evaluation of another record (its
        `results` and `hit_scores`), at the cost of `spend` (`run_record.relabel`). `status` is
        needed without a submission. The other arguments are those of `evaluate_and_record`."""
        if status is None:
            if submission is None:
                raise run_record.RecordError(
                    "a record that reuses an evaluation needs a submission or a status"
                )
            status = submission.status
        construction, config = self._construction_and_config(
            method=method,
            setting=setting,
            settings=settings,
            spend=spend,
            ranking=ranking,
            submission=submission,
            status=status,
            submitted=submitted,
            details=details,
        )
        relabeled = run_record.relabel(results, method=method, setting=setting, spend=spend)
        shipped = relabeled["shipped"]
        all_draws = dict(draws or {})
        if shipped.details is not None:
            all_draws[run_record.EVALUATION_DRAWS_NAME] = run_record.evaluation_draw_rows(shipped)
        root = self.records_root
        directory = run_record.run_dir(root, method, setting, shipped.task_id, shipped.replicate)
        return self._write(root, directory, relabeled, construction, config, all_draws, hit_scores)


@dataclasses.dataclass(frozen=True)
class MethodEntry(Generic[SettingsT]):
    """One method that a family's command group can run.

    settings: the method's settings with the paper's values (an instance of a frozen dataclass).
    original_batch_size: the responses in one forward pass of the original model
        (`OriginalModel.batch_size`). The rounding of a pass depends on its batch, so this number
        is part of what determines a method's draws.
    check: check(options, settings) raises an `errors.MethodsError` for options and settings the
        method cannot run with; it is called before the task and the model are loaded.
    reselect: for a method that trains, its rules to choose a checkpoint again (`reselection`),
        each a function with the signature of `run`.
    generalization_tasks: whether the method runs on the tasks of the generalization section."""

    settings: SettingsT
    run: Callable[[RunContext, SettingsT], list[run_record.RunSummary]]
    original_batch_size: int = 16
    check: Callable[[RunOptions, SettingsT], None] | None = None
    reselect: Mapping[str, Callable[[RunContext, SettingsT], list[run_record.RunSummary]]] = (
        dataclasses.field(default_factory=lambda: {})
    )
    generalization_tasks: bool = False


def load_original_model(
    options: RunOptions, device: str, batch_size: int
) -> original_model.OriginalModel:
    """The original model of a task of the package, in float32, from `options` (the benchmark's
    model unless one is given)."""
    return original_model.OriginalModel.from_pretrained(
        original_model.DEFAULT_MODEL if options.model is None else options.model,
        revision=options.model_revision,
        dtype="float32",
        device=device,
        batch_size=batch_size,
    )


def load_task_original_model(
    task: generalization_tasks.GeneralizationTask, options: RunOptions, device: str
) -> generalization_models.TaskOriginalModel:
    """The original model of a task of the generalization section, with the task's response
    limit and in the dtype of its model (`task_set.load_original_model`), from the location and
    with the batch sizes of `options` where they are given."""
    return task_set.load_original_model(
        task,
        name=options.model,
        revision=options.model_revision,
        device=device,
        batch_size=options.batch_size,
        generation_batch_size=options.generation_batch_size,
    )


def judge_models(options: RunOptions) -> tuple[judges.ChatModel | None, judges.ChatModel | None]:
    """The rubric judge and the reviewing model of `options`, with the request settings of the
    paper's runs; None for one whose address or name is not given."""
    rubric_judge = review = None
    if options.judge_url is not None and options.judge_model is not None:
        rubric_judge = judges.rubric_judge_model(
            options.judge_url, options.judge_model, options.judge_key_env
        )
    if options.review_url is not None and options.review_model is not None:
        review = judges.review_model(
            options.review_url, options.review_model, options.review_key_env
        )
    return rubric_judge, review


def _open_tracker(
    family: str, method: str, options: RunOptions, settings: Any, task_id: str
) -> tracking.Tracker:
    run_name = f"{method}-{run_record.task_dir_name(task_id)}-rep{options.replicate}"
    return tracking.open_tracker(
        project=options.wandb_project,
        group=options.wandb_group,
        run_name=run_name,
        config={"options": options.to_json(), "settings": run_record.settings_json(settings)},
        labels={
            "family": family,
            "method": method,
            "task": task_id,
            "replicate": options.replicate,
        },
    )


def open_context(
    family: str, method: str, entry: MethodEntry[Any], options: RunOptions, settings: Any
) -> RunContext:
    """Open a run: the task, its original model on the device (downloaded on first use) and the
    tracker, which starts a wandb run if `options.wandb_project` is set. A task id
    `<model>/<prompt_id>/<behavior>` names a task of the generalization section
    (`open_generalization_context`); any other, a task of the package. A task or options that
    cannot be run raise their error before the model is loaded."""
    if generalization_tasks.has_task_id_form(options.task_id):
        if not entry.generalization_tasks:
            raise TaskOptionsError(
                f"{method} runs on the tasks of the package; the methods that run on a task of "
                "the generalization section are those of the families prompt, steering and sft"
            )
        return open_generalization_context(family, method, options, settings)
    bench = benchmark.Benchmark(options.data, revision=options.data_revision)
    with data_source.reading(options.data):
        task = bench.task(options.task_id)
    task.require_scored()
    given = ", ".join(f"--{name.replace('_', '-')}" for name in options.generalization_options())
    if given:
        raise TaskOptionsError(
            f"{task.task_id} is a task of the package, and these options are read only for a "
            f"task of the generalization section: {given}"
        )
    device = resolve_device(options.device)
    # The record of a run names the location that its model was loaded from.
    if options.model is None:
        options = dataclasses.replace(options, model=original_model.DEFAULT_MODEL)

    def load_model_copy() -> original_model.OriginalModel:
        return load_original_model(options, device, entry.original_batch_size)

    original = load_model_copy()
    return RunContext(
        family=family,
        method=method,
        options=options,
        bench=bench,
        task=task,
        original=original,
        load_model_copy=load_model_copy,
        tracker=_open_tracker(family, method, options, settings, task.task_id),
    )


def open_generalization_context(
    family: str, method: str, options: RunOptions, settings: Any
) -> RunContext:
    """Open a run on one of the six tasks of the generalization section: the task and its data,
    its original model with its response limit and batch limits, and for a task that a language
    model judges its judge. An unknown task, missing data or a missing judge model raise their
    error before the model is loaded."""
    rubric_judge, review = judge_models(options)
    tasks = task_set.GeneralizationTasks(
        options.data,
        revision=options.data_revision,
        judge_model=rubric_judge,
        review_model=review,
        labels=options.labels,
        concurrency=options.judge_concurrency,
    )
    task = tasks.task(options.task_id)
    tasks.task_data()  # raises if the data of the tasks was not given
    judge = tasks.judge(task) if task.rubric is not None else None
    tasks.require_data_of(task)
    device = resolve_device(options.device)

    given = options  # before `model` is set below: an unnamed model loads at the task's commit

    def load_model_copy() -> generalization_models.TaskOriginalModel:
        return load_task_original_model(task, given, device)

    original = load_model_copy()
    generation = original.generation_batch_size
    # The record of a run names the location that its model was loaded from.
    if options.model is None:
        options = dataclasses.replace(options, model=task.model.name)
    return RunContext(
        family=family,
        method=method,
        options=options,
        bench=tasks,
        task=task,
        original=original,
        load_model_copy=load_model_copy,
        tracker=_open_tracker(family, method, options, settings, task.task_id),
        judge=judge,
        batch_limits=BatchLimits(
            scoring=int(original.batch_size),
            generation=int(original.batch_size if generation is None else generation),
        ),
    )


def run_method(
    family: str, method: str, entry: MethodEntry[Any], options: RunOptions, settings: Any
) -> list[run_record.RunSummary]:
    """Check the options and settings (`MethodEntry.check`), open the context, run the method
    and close the tracker."""
    if entry.check is not None:
        entry.check(options, settings)
    context = open_context(family, method, entry, options, settings)
    try:
        return entry.run(context, settings)
    finally:
        context.tracker.finish()
