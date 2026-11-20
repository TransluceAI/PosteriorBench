"""Fixtures of the generalization tests: tasks of the shared tiny model with the fields of a
`GeneralizationTask`, their data written in the layout of `task_data`, and stand-ins for the two
judge models.

`toy_tasks` holds two tasks on one corpus of the tiny model, with made-up harmless behaviors. One
is defined by a rule (the response contains "a"). The other has a rubric and is judged by the
stand-in judge models: the stand-in rubric judge calls a response a match if it contains "a", and
the stand-in review rejects a match that contains "b". No test reaches a network.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from posteriorbench import data, original_model

from posteriorbench_methods.generalization import judge, models, task_data, tasks

CORPUS_DRAWS = 1500
REFERENCES = 40
RULE = '"a" in y'
TINY_SPEC = models.OriginalModelSpec(
    key="tiny",
    paper_name="Tiny",
    name="tiny",
    revision=None,
    model_class="Qwen2ForCausalLM",
    stop_token_ids=(2, 0),  # <|im_end|>, <|endoftext|> of the tiny tokenizer
    thinking_argument=True,
    dtype="float32",
)
TOY_RUBRIC = tasks.Rubric("mentions-the-letter-a", "toy-rubric-v1", "The reply contains an a.")


def write_evaluation_file(directory: pathlib.Path, evaluation: data.EvaluationData) -> None:
    """Write `evaluation` as `<directory>/evaluation/evaluation.parquet`."""
    row = {
        "task_id": evaluation.task_id,
        "references": [
            {"row": r.row, "token_ids": list(r.token_ids), "text": r.text}
            for r in evaluation.references
        ],
        "reference_logp": list(evaluation.reference_logp),
        "fallback_text": evaluation.fallback_text,
        "fallback_token_ids": list(evaluation.fallback_token_ids),
        "fallback_logp": evaluation.fallback_logp,
    }
    path = directory / data.EVALUATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=data.EVALUATION_SCHEMA), path)


def write_hit_file(
    directory: pathlib.Path,
    task: tasks.GeneralizationTask,
    *,
    rows: Sequence[int],
    token_ids: Sequence[Sequence[int]],
    references: Sequence[data.Reference],
) -> int:
    """Write the task's hit file under `directory`: `rows` and `token_ids` in corpus order, the
    task's hits without its `references`. Returns the number of hits whose token ids equal a
    reference's. Raises `task_data.TaskDataError` unless hits and references together are
    `hit_count` and no hit is a reference row."""
    if len(rows) != len(token_ids):
        raise ValueError("one token-id list per hit row")
    reference_rows = {reference.row for reference in references}
    if len(rows) + len(references) != task.hit_count or reference_rows & set(rows):
        raise task_data.TaskDataError(
            f"{task.task_id}: {len(rows)} hits and {len(references)} references, the task has "
            f"{task.hit_count} hits in all"
        )
    reference_ids = {tuple(reference.token_ids) for reference in references}
    duplicates = [tuple(int(t) for t in ids) in reference_ids for ids in token_ids]
    table = pa.table(
        {
            "row": pa.array(rows, type=pa.int32()),
            "token_ids": pa.array([list(ids) for ids in token_ids], type=pa.list_(pa.int32())),
            "reference_duplicate": pa.array(duplicates, type=pa.bool_()),
        },
        schema=data.HITS_SCHEMA,
    ).replace_schema_metadata(
        {
            "task_id": task.task_id,
            "hit_count": str(task.hit_count),
            "references_removed": str(len(references)),
            "reference_duplicates": str(sum(duplicates)),
        }
    )
    path = directory / data.hits_file(task.prompt_id, task.behavior)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", row_group_size=100_000)
    return sum(duplicates)


def write_labels_file(directory: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write `<directory>/labels/labels.parquet` from rows with the keys of
    `task_data.LABELS_SCHEMA`."""
    path = directory / task_data.LABELS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=task_data.LABELS_SCHEMA), path)


class StandInJudgeModel:
    """A rubric judge that calls a response a match if it contains "a". It reads the response
    from the transcript in its prompt, as a judge model would, and keeps every prompt."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.answers = answers  # fixed answers, one per call, in place of the rule above

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.answers is not None:
            return self.answers[len(self.prompts) - 1]
        transcript = prompt.split("<transcript>\n", 1)[1]  # after the template's own example
        found = re.search(r"role: assistant\|>\n(.*?)\n</\|T0B1", transcript, re.DOTALL)
        assert found is not None, "the prompt holds no transcript"
        match = "a" in found.group(1)
        return f'```json\n{json.dumps({"match": match, "explanation": "it has an a"})}\n```'

    def describe(self) -> dict[str, Any]:
        return {"model": "stand-in-judge"}


class StandInReviewModel:
    """A reviewing model that rejects a response that contains "b"."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.answers = answers

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.answers is not None:
            return self.answers[len(self.prompts) - 1]
        response = prompt.split("ASSISTANT:\n", 1)[1].split("\n\nWeirdChat is", 1)[0]
        if "b" in response:
            return json.dumps(
                {"verdict": "false_positive", "fp_reasons": ["other"], "explanation": "a b"}
            )
        return json.dumps({"verdict": "genuine", "fp_reasons": [], "explanation": "fine"})

    def describe(self) -> dict[str, Any]:
        return {"model": "stand-in-review"}


def stand_in_hit(text: str) -> bool:
    """The label that the two stand-in models give together."""
    return "a" in text and "b" not in text


@dataclasses.dataclass(frozen=True)
class ToyTasks:
    """Two tasks on one corpus of the tiny model, and their data directory.

    `responses[i]` is corpus row i and `texts[i]` its text. `by_rule` is the task with a rule,
    `by_rubric` the task with a rubric; `evaluation[task_id]` is each task's held-out data."""

    root: pathlib.Path
    by_rule: tasks.GeneralizationTask
    by_rubric: tasks.GeneralizationTask
    evaluation: dict[str, data.EvaluationData]
    responses: tuple[tuple[int, ...], ...]
    texts: tuple[str, ...]


def make_task(
    scorer: original_model.OriginalModel,
    behavior: str,
    labels: list[bool],
    *,
    rubric: tasks.Rubric | None,
    spec: models.OriginalModelSpec = TINY_SPEC,
) -> tasks.GeneralizationTask:
    """A task of the tiny model whose hits are the rows with a true label."""
    hits = sum(labels)
    return tasks.GeneralizationTask(
        task_id=f"{spec.key}/wc_0001/{behavior}",
        prompt_id="wc_0001",
        prompt="hi",
        prompt_token_ids=tuple(scorer.render_prompt("hi")),
        wildchat_conversation_hash="0" * 32,
        behavior=behavior,
        rule="" if rubric is not None else RULE,
        num_responses=len(labels),
        hit_count=hits,
        probability=hits / len(labels),
        scored=True,
        benchmark_task=False,
        model=spec,
        max_response_tokens=int(scorer.max_response_tokens),
        rubric=rubric,
        references=REFERENCES,
    )


def write_task(
    root: pathlib.Path,
    task: tasks.GeneralizationTask,
    scorer: original_model.OriginalModel,
    responses: list[tuple[int, ...]],
    texts: list[str],
    labels: list[bool],
) -> data.EvaluationData:
    """Write the task's data under `root`: the first REFERENCES hits are the references, the
    next hit is the fallback response, and the stored log p are the model's own."""
    hit_rows = [row for row, hit in enumerate(labels) if hit]
    reference_rows, rest = hit_rows[:REFERENCES], hit_rows[REFERENCES:]
    fallback = responses[rest[0]]
    prompt = task.prompt_token_ids
    references = tuple(data.Reference(row, responses[row], texts[row]) for row in reference_rows)
    (fallback_logp,) = scorer.logprobs(prompt, [fallback])
    evaluation = data.EvaluationData(
        task_id=task.task_id,
        references=references,
        reference_logp=tuple(scorer.logprobs(prompt, [responses[r] for r in reference_rows])),
        fallback_text=scorer.decode(fallback),
        fallback_token_ids=fallback,
        fallback_logp=fallback_logp,
    )
    directory = root / task_data.task_dir(task)
    write_evaluation_file(directory, evaluation)
    write_hit_file(
        directory, task, rows=rest, token_ids=[responses[r] for r in rest], references=references
    )
    return evaluation


@pytest.fixture(scope="session")
def toy_tasks(
    tiny_original_model: original_model.OriginalModel, tmp_path_factory: pytest.TempPathFactory
) -> ToyTasks:
    scorer = tiny_original_model
    root = tmp_path_factory.mktemp("generalization_data")
    prompt = scorer.render_prompt("hi")
    responses = [tuple(r) for r in scorer.generate(prompt, CORPUS_DRAWS, seed=3)]
    texts = [scorer.decode(r) for r in responses]
    rule_labels = ["a" in text for text in texts]
    rubric_labels = [stand_in_hit(text) for text in texts]
    by_rule = make_task(scorer, "letter_a", rule_labels, rubric=None)
    by_rubric = make_task(scorer, "letter_a_judged", rubric_labels, rubric=TOY_RUBRIC)
    evaluation = {
        by_rule.task_id: write_task(root, by_rule, scorer, responses, texts, rule_labels),
        by_rubric.task_id: write_task(root, by_rubric, scorer, responses, texts, rubric_labels),
    }
    # The stored labels of the task with a rubric: every corpus response that the rubric judge
    # called a match, with the review's verdict.
    stored = [
        {
            "row": row,
            "text": text,
            "judge_match": True,
            "judge_explanation": "it has an a",
            "review_verdict": "genuine" if stand_in_hit(text) else "false_positive",
            "review_reasons": [] if stand_in_hit(text) else ["other"],
            "review_explanation": "",
            "hit": stand_in_hit(text),
        }
        for row, text in enumerate(texts)
        if "a" in text
    ]
    write_labels_file(root / task_data.task_dir(by_rubric), stored)
    return ToyTasks(root, by_rule, by_rubric, evaluation, tuple(responses), tuple(texts))


@pytest.fixture()
def stand_in_judge(toy_tasks: ToyTasks) -> judge.RubricJudge:
    """A rubric judge of the toy task with the two stand-in models and an empty label store."""
    return judge.RubricJudge(
        toy_tasks.by_rubric, StandInJudgeModel(), StandInReviewModel(), sleep=lambda _: None
    )


@dataclasses.dataclass(frozen=True)
class Helpers:
    """What the tests of this folder use besides fixtures (a test module cannot import this
    file): the stand-in models, the label they give, the tiny model's spec and the task makers."""

    judge_model: type[StandInJudgeModel] = StandInJudgeModel
    review_model: type[StandInReviewModel] = StandInReviewModel
    hit: Any = staticmethod(stand_in_hit)
    spec: models.OriginalModelSpec = TINY_SPEC
    rubric: tasks.Rubric = TOY_RUBRIC
    make_task: Any = staticmethod(make_task)
    write_task: Any = staticmethod(write_task)
    references: int = REFERENCES


@pytest.fixture(scope="session")
def helpers() -> Helpers:
    return Helpers()
