"""Write the candidate prompts of a task with a writer LLM (README, "Write the candidates").

Stage 1 explains each behavior once (kept in the explanations file), stage 2 asks for
scenarios, stage 3 writes one system prompt per scenario; `writer_messages` holds the messages.
Nothing is asked twice, so a call can write fewer candidates than it asked for. The writer's
calls are not inferences and are not charged.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import math
import os
import pathlib
from collections.abc import Callable, Collection, Sequence
from typing import Any, TypeVar

from posteriorbench import data

from posteriorbench_methods.common import errors, tracking
from posteriorbench_methods.prompt import pool, writer, writer_messages

ItemT = TypeVar("ItemT")


class WritingError(errors.MethodsError, RuntimeError):
    """The writer gave nothing to write from, or the explanations file is malformed."""


@dataclasses.dataclass(frozen=True)
class Settings:
    """How the candidates are written; the defaults are those of the paper's candidates.

    `explanation_tokens`, `scenario_tokens` and `system_prompt_tokens` cap the answers of stages
    1, 2 and 3. `timeout` is in seconds per attempt."""

    scenarios_per_call: int = 50
    temperature: float = 1.0
    explanation_tokens: int = 4000
    scenario_tokens: int = 32768
    system_prompt_tokens: int = 4000
    concurrency: int = 64
    attempts: int = 40
    timeout: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "scenarios_per_call",
            "explanation_tokens",
            "scenario_tokens",
            "system_prompt_tokens",
            "concurrency",
            "attempts",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.temperature < 0:
            raise ValueError(f"temperature must not be negative, got {self.temperature}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What one call of `write_candidates` gave.

    `written` is in scenario order. `scenarios` counts the scenarios kept, each of which had one
    call of stage 3; `repeated` counts system prompts dropped as duplicates."""

    written: tuple[pool.Written, ...]
    scenario_calls: int
    scenarios: int
    without_system_prompt: int
    repeated: int


@dataclasses.dataclass(frozen=True)
class Report:
    """What `add_to_pool_file` did for one task.

    `candidates` is the count after the call. `batch` is None if nothing was added, `outcome`
    None if the task lacked no candidate."""

    task_id: str
    candidates_before: int
    candidates: int
    asked: int
    batch: int | None
    explanation_calls: int
    outcome: Outcome | None

    @property
    def writer_calls(self) -> int:
        """The writer's calls for this task, stages 1 to 3."""
        if self.outcome is None:
            return self.explanation_calls
        return self.explanation_calls + self.outcome.scenario_calls + self.outcome.scenarios

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict: the fields, with the outcome's counts in place of its candidates."""
        outcome = self.outcome
        return {
            "task_id": self.task_id,
            "candidates_before": self.candidates_before,
            "candidates": self.candidates,
            "asked": self.asked,
            "batch": self.batch,
            "explanation_calls": self.explanation_calls,
            "scenario_calls": 0 if outcome is None else outcome.scenario_calls,
            "system_prompt_calls": 0 if outcome is None else outcome.scenarios,
            "without_system_prompt": 0 if outcome is None else outcome.without_system_prompt,
            "repeated": 0 if outcome is None else outcome.repeated,
            "writer_calls": self.writer_calls,
        }


def explain(
    llm: writer.Writer, behavior: writer_messages.Behavior, settings: Settings
) -> tuple[writer_messages.Explanation, str]:
    """Stage 1: the explanation of `behavior` and the raw answer it was parsed from. Raises
    WritingError if the writer gives no answer."""
    answer = llm.complete(
        writer_messages.explanation_messages(behavior), max_tokens=settings.explanation_tokens
    )
    if answer is None:
        raise WritingError(f"the writer gave no explanation of the behavior {behavior.name}")
    return writer_messages.parse_explanation(answer), answer


def write_candidates(
    llm: writer.Writer,
    behavior: writer_messages.Behavior,
    explanation: writer_messages.Explanation,
    user_prompt: str,
    *,
    count: int,
    existing: Collection[str],
    settings: Settings,
    model: writer_messages.OriginalModel = writer_messages.BENCHMARK_MODEL,
) -> Outcome:
    """Stages 2 and 3, asked once: up to `count` new system prompts for `user_prompt`.

    ceil(count / scenarios_per_call) calls of stage 2; of their scenarios, at most
    `scenarios_per_call` per call and the first `count` overall get one call of stage 3 each. A
    system prompt already in `existing` or written for an earlier scenario is dropped. Nothing is
    asked again, so fewer than `count` may come back. `model` is how the messages name the
    original model. Raises WritingError if no call of stage 2 answers."""
    if count < 1:
        return Outcome((), 0, 0, 0, 0)
    per_call = settings.scenarios_per_call
    calls = math.ceil(count / per_call)
    asking = writer_messages.scenario_messages(
        behavior, explanation, user_prompt, per_call, model=model
    )
    answers = _concurrently(
        lambda _: llm.complete(asking, max_tokens=settings.scenario_tokens),
        range(calls),
        settings.concurrency,
    )
    if all(answer is None for answer in answers):
        raise WritingError(
            f"none of the {calls} calls for scenarios got an answer from the writer; the "
            "warnings above give the reasons"
        )
    scenarios: list[str] = []
    for answer in answers:
        if answer is not None:
            scenarios += writer_messages.parse_scenarios(answer)[:per_call]
    scenarios = scenarios[:count]

    def system_prompt(scenario: str) -> str:
        answer = llm.complete(
            writer_messages.system_prompt_messages(
                behavior, explanation, user_prompt, scenario, model=model
            ),
            max_tokens=settings.system_prompt_tokens,
        )
        return "" if answer is None else writer_messages.parse_system_prompt(answer)

    texts = _concurrently(system_prompt, scenarios, settings.concurrency)
    seen = set(existing)
    written: list[pool.Written] = []
    without_system_prompt = repeated = 0
    for scenario, text in zip(scenarios, texts, strict=True):
        if not text:
            without_system_prompt += 1
        elif text in seen:
            repeated += 1
        else:
            seen.add(text)
            written.append(pool.Written(text, scenario))
    return Outcome(tuple(written), calls, len(scenarios), without_system_prompt, repeated)


def _concurrently(
    function: Callable[[ItemT], str | None], items: Sequence[ItemT], concurrency: int
) -> list[str | None]:
    """`function` of each item, in order, with at most `concurrency` calls open. If one raises,
    the calls not yet started are cancelled and the error propagates."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(function, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def explanations_file(pool_file: pathlib.Path) -> pathlib.Path:
    """The default explanations file of `pool_file`."""
    return pool_file.with_suffix(".explanations.json")


def read_explanations(path: pathlib.Path) -> dict[str, writer_messages.Explanation]:
    """The explanations in `path` by behavior label; empty if the file does not exist. Raises
    WritingError for a malformed file."""
    return {
        label: writer_messages.Explanation(entry["understanding"], entry["motivation"])
        for label, entry in _explanation_entries(path).items()
    }


def _explanation_entries(path: pathlib.Path) -> dict[str, dict[str, str]]:
    """The stored entries: behavior label -> understanding, motivation, writer_model, answer."""
    if not path.exists():
        return {}
    try:
        loaded: object = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise WritingError(f"{path} is not a file of explanations: {error}") from error
    if not isinstance(loaded, dict):
        raise WritingError(f"{path} is not a file of explanations: it holds no JSON object")
    entries: dict[str, dict[str, str]] = {}
    for label, entry in loaded.items():  # pyright: ignore[reportUnknownVariableType]
        if not (
            isinstance(label, str)
            and isinstance(entry, dict)
            and isinstance(entry.get("understanding"), str)
            and isinstance(entry.get("motivation"), str)
        ):
            raise WritingError(
                f"{path} is not a file of explanations: the entry {label!r} has no "
                "understanding and motivation"
            )
        entries[label] = entry  # pyright: ignore[reportUnknownVariableType]
    return entries


def save_explanation(
    path: pathlib.Path,
    label: str,
    explanation: writer_messages.Explanation,
    *,
    writer_model: str,
    answer: str,
) -> None:
    """Add the explanation of behavior `label` to `path`, with the writer's model and whole
    answer. Rewrites `path` atomically and creates its directory if needed."""
    entries = _explanation_entries(path)
    entries[label] = {
        "understanding": explanation.understanding,
        "motivation": explanation.motivation,
        "writer_model": writer_model,
        "answer": answer,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n")
    os.replace(partial, path)


def add_to_pool_file(
    pool_file: pathlib.Path,
    task: data.Task,
    llm: writer.Writer,
    *,
    candidates: int,
    new_batch: bool,
    settings: Settings,
    explanations: pathlib.Path | None = None,
    tracker: tracking.Tracker | None = None,
) -> Report:
    """Ask the writer for the candidates `task` lacks to have `candidates` in `pool_file`, and
    add them (`pool.add_candidates` says what `new_batch` does).

    The behavior's explanation is read from `explanations` (default
    `explanations_file(pool_file)`); if it is missing, the writer writes one and it is saved
    before any candidate is written. A task that lacks no candidate costs no call. Side effects:
    calls the writer, writes `pool_file` and the explanations file, logs to `tracker`."""
    tracker = tracking.Tracker() if tracker is None else tracker
    explanations = explanations_file(pool_file) if explanations is None else explanations
    existing = pool.existing_candidates(pool_file, task.task_id)
    asked = candidates - len(existing)
    if asked <= 0:
        return Report(task.task_id, len(existing), len(existing), 0, None, 0, None)
    behavior = writer_messages.behavior_of_task(task)
    model = writer_messages.original_model_of_task(task)
    explanation = read_explanations(explanations).get(task.behavior)
    explanation_calls = 0
    if explanation is None:
        explanation, answer = explain(llm, behavior, settings)
        save_explanation(
            explanations, task.behavior, explanation, writer_model=llm.model, answer=answer
        )
        explanation_calls = 1
    tracker.event("explanation", behavior=task.behavior, written=bool(explanation_calls))
    outcome = write_candidates(
        llm,
        behavior,
        explanation,
        task.prompt,
        count=asked,
        existing=[candidate.system_prompt for candidate in existing],
        settings=settings,
        model=model,
    )
    tracker.event(
        "scenarios", task=task.task_id, calls=outcome.scenario_calls, scenarios=outcome.scenarios
    )
    added = pool.add_candidates(pool_file, task.task_id, outcome.written, new_batch=new_batch)
    report = Report(
        task_id=task.task_id,
        candidates_before=len(existing),
        candidates=len(existing) + len(added),
        asked=asked,
        batch=added[0].batch if added else None,
        explanation_calls=explanation_calls,
        outcome=outcome,
    )
    tracker.event("candidates", **report.to_json())
    return report
