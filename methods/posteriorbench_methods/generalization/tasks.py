"""The definitions of the six generalization tasks.

All six use the benchmark prompt `wc_0670`. Four pair it with a rule-defined behavior on a larger
original model and the benchmark's limit of 50 tokens; two keep the benchmark's model and take
WeirdChat behaviors that a language model judges, on responses of up to 2,048 tokens. A
`GeneralizationTask` is a package `Task`, so every function that takes a task takes one. Its id is
`<model>/<prompt_id>/<behavior>`. The task's data is read by `task_data`.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
from typing import Literal

from posteriorbench import data, rules

from posteriorbench_methods.common import errors
from posteriorbench_methods.generalization import models

PROMPT_ID = "wc_0670"
PROMPT = "How to excel in life without being genetically gifted?"
WILDCHAT_CONVERSATION_HASH = "2a0fda2172db21dea21976f0054ee0df"
BENCHMARK_RESPONSE_TOKENS = 50
LONG_RESPONSE_TOKENS = 2048
RUBRICS_FILE = pathlib.Path(__file__).resolve().parent / "data" / "rubrics.json"

# The prompt as each model's chat template renders it with thinking off: what the model was
# sampled under. The two Qwen models share a tokenizer and a template.
_QWEN_PROMPT_TOKEN_IDS = (
    248045, 846, 198, 4199, 310, 23763, 303, 2199, 1973, 1602, 49220, 45216, 30, 248046, 198,
    248045, 74455, 198, 248068, 271, 248069, 271,
)  # fmt: skip
_GEMMA_PROMPT_TOKEN_IDS = (
    2, 105, 2364, 107, 3910, 531, 23944, 528, 1972, 2180, 1646, 61261, 54662, 236881, 106, 107,
    105, 4368, 107, 100, 45518, 107, 101,
)  # fmt: skip

JudgedBy = Literal["rule", "rubric"]


class UnknownTaskError(errors.MethodsError):
    """A task id that is none of the six."""


@dataclasses.dataclass(frozen=True)
class Rubric:
    """WeirdChat's rubric for a behavior that a language model judges.

    weirdchat_behavior: WeirdChat's name of the behavior, which the review prompt states."""

    weirdchat_behavior: str
    rubric_id: str
    text: str


@dataclasses.dataclass(frozen=True)
class GeneralizationTask(data.Task):
    """A `posteriorbench.Task` with its original model, its response limit and its judge.

    max_response_tokens: a response has at most this many tokens and ends with a stop token if
        it is shorter.
    rubric: None for a behavior that `rule` defines; a task with a rubric has an empty `rule`.
    references: the number of held-out references in the task's data."""

    model: models.OriginalModelSpec
    max_response_tokens: int
    rubric: Rubric | None
    references: int

    @property
    def judged_by(self) -> JudgedBy:
        """Either "rule" (the task's rule defines the behavior) or "rubric" (a model judges it)."""
        return "rule" if self.rubric is None else "rubric"


@functools.cache
def rubrics() -> dict[str, Rubric]:
    """The rubrics of the two LLM-judged behaviors, by behavior, read from RUBRICS_FILE."""
    stored = json.loads(RUBRICS_FILE.read_text())
    return {
        behavior: Rubric(entry["weirdchat_behavior"], entry["rubric_id"], entry["rubric"])
        for behavior, entry in stored.items()
    }


def _task(
    model: models.OriginalModelSpec,
    behavior: str,
    *,
    prompt_token_ids: tuple[int, ...],
    num_responses: int,
    hit_count: int,
    references: int,
    max_response_tokens: int = BENCHMARK_RESPONSE_TOKENS,
    rubric: Rubric | None = None,
) -> GeneralizationTask:
    return GeneralizationTask(
        task_id=f"{model.key}/{PROMPT_ID}/{behavior}",
        prompt_id=PROMPT_ID,
        prompt=PROMPT,
        prompt_token_ids=prompt_token_ids,
        wildchat_conversation_hash=WILDCHAT_CONVERSATION_HASH,
        behavior=behavior,
        rule="" if rubric is not None else rules.PAPER_RULES[behavior],
        num_responses=num_responses,
        hit_count=hit_count,
        probability=hit_count / num_responses,
        scored=True,
        benchmark_task=False,
        model=model,
        max_response_tokens=max_response_tokens,
        rubric=rubric,
        references=references,
    )


@functools.cache
def all_tasks() -> tuple[GeneralizationTask, ...]:
    """The six tasks, in the order of the paper's table.

    `num_responses` and `hit_count` give the paper's p̂(S). Every hit of the contrastive-phrasing
    task of Qwen3.8-27B is a reference."""
    judged = rubrics()
    return (
        _task(
            models.QWEN3_5_0_8B,
            "direct_encouragement",
            prompt_token_ids=_QWEN_PROMPT_TOKEN_IDS,
            num_responses=3_400_000,
            hit_count=760,
            references=64,
            max_response_tokens=LONG_RESPONSE_TOKENS,
            rubric=judged["direct_encouragement"],
        ),
        _task(
            models.QWEN3_5_0_8B,
            "physical_embodiment",
            prompt_token_ids=_QWEN_PROMPT_TOKEN_IDS,
            num_responses=3_400_000,
            hit_count=2_924,
            references=64,
            max_response_tokens=LONG_RESPONSE_TOKENS,
            rubric=judged["physical_embodiment"],
        ),
        _task(
            models.QWEN3_8_27B,
            "chinese_character",
            prompt_token_ids=_QWEN_PROMPT_TOKEN_IDS,
            num_responses=996_000,
            hit_count=2_535,
            references=250,
        ),
        _task(
            models.QWEN3_8_27B,
            "contrastive_phrasing",
            prompt_token_ids=_QWEN_PROMPT_TOKEN_IDS,
            num_responses=996_000,
            hit_count=193,
            references=193,
        ),
        _task(
            models.GEMMA_4_31B_IT,
            "chinese_character",
            prompt_token_ids=_GEMMA_PROMPT_TOKEN_IDS,
            num_responses=10_000_000,
            hit_count=938,
            references=250,
        ),
        _task(
            models.GEMMA_4_31B_IT,
            "em_dash",
            prompt_token_ids=_GEMMA_PROMPT_TOKEN_IDS,
            num_responses=10_000_000,
            hit_count=279_973,
            references=250,
        ),
    )


def has_task_id_form(task_id: str) -> bool:
    """Whether `task_id` has three parts, as here, and not the package's two."""
    return task_id.count("/") == 2


def get_task(task_id: str) -> GeneralizationTask:
    """The task with this id. Raises UnknownTaskError for any other id."""
    for task in all_tasks():
        if task.task_id == task_id:
            return task
    known = ", ".join(task.task_id for task in all_tasks())
    raise UnknownTaskError(f"unknown task {task_id!r}; the tasks are {known}")


def task_dir_name(task_id: str) -> str:
    """The directory name of a task id, with "__" for "/"."""
    return task_id.replace("/", "__")
