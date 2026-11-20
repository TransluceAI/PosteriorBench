"""Synthetic datasets in the dataset's layout.

- The synthetic task (`make_synthetic`, `write_synthetic_dataset`): a task of the tiny model whose
  corpus is a sample of the model and whose references are hits of that sample. It has the task id
  that the documentation uses in its examples, so the documented code runs against it, and a rule
  that is frequent on the tiny model. Its evaluation file holds the model's own log p(a) and, as
  the shipped log p of the references, the model's values shifted by SHIPPED_OFFSET, so tests can
  tell shipped from recomputed values.
- The small dataset (`build_small_dataset`): 1,200 corpus rows written by hand, with two hits that
  repeat references; it needs no model.

Both are written with the package's schemas, and their hit files are built from their corpus by
`tools/build_hits_index.py`.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Any

import build_hits_index
import pyarrow as pa
import pyarrow.parquet as pq

from posteriorbench import data, rules

DOCUMENTED_TASK_ID = "wc_1853/goblin_mention"
SYNTHETIC_RULE = '"a" in y'
SHIPPED_OFFSET = -0.01

# The small dataset: prompt wc_0001; even rows mention a goblin (600 hits); the first 250 hits
# (rows 0, 2, ..., 498) are the references; rows 1100 and 1102 repeat the token ids of references
# 0 and 1.
SMALL_TASK_ID = "wc_0001/goblin_mention"
SMALL_ROWS = 1200
SMALL_HIT_COUNT = 600
SMALL_REFERENCES = 250
SMALL_DUPLICATE_ROWS = (1100, 1102)


@dataclasses.dataclass(frozen=True)
class Synthetic:
    """A synthetic task of the tiny model: the construction view, the evaluation data and the
    corpus it came from (`responses[i]` is corpus row i)."""

    task: data.Task
    evaluation: data.EvaluationData
    responses: tuple[tuple[int, ...], ...]
    labels: tuple[bool, ...]
    texts: tuple[str, ...]


def make_synthetic(
    model: Any,
    rule: str = SYNTHETIC_RULE,
    *,
    task_id: str = DOCUMENTED_TASK_ID,
    seed: int = 0,
    draws: int = 3000,
    references: int = 250,
) -> Synthetic:
    """A task on the tiny model: its corpus is `draws` responses of the model, p̂(S) their hit
    fraction, the references its first hits and the fallback response the next hit."""
    prompt_id, behavior = task_id.split("/")
    prompt = model.render_prompt("hi")
    responses = [tuple(r) for r in model.generate(prompt, draws, seed=seed)]
    predicate = rules.parse_rule(rule)
    texts = [model.decode(r) for r in responses]
    labels = [predicate(text) for text in texts]
    hit_rows = [i for i, is_hit in enumerate(labels) if is_hit]
    assert len(hit_rows) >= references + 1, "the synthetic behavior is too rare for this fixture"
    reference_rows = hit_rows[:references]
    fallback = responses[hit_rows[references]]
    task = data.Task(
        task_id=task_id,
        prompt_id=prompt_id,
        prompt="hi",
        prompt_token_ids=tuple(prompt),
        wildchat_conversation_hash="0" * 32,
        behavior=behavior,
        rule=rule,
        num_responses=draws,
        hit_count=len(hit_rows),
        probability=len(hit_rows) / draws,
        scored=True,
        benchmark_task=True,
    )
    reference_ids = [responses[row] for row in reference_rows]
    recomputed = model.logprobs(prompt, reference_ids)
    (fallback_logp,) = model.logprobs(prompt, [fallback])
    evaluation = data.EvaluationData(
        task_id=task_id,
        references=tuple(data.Reference(row, responses[row], texts[row]) for row in reference_rows),
        reference_logp=tuple(value + SHIPPED_OFFSET for value in recomputed),
        fallback_text=model.decode(fallback),
        fallback_token_ids=fallback,
        fallback_logp=fallback_logp,
    )
    return Synthetic(task, evaluation, tuple(responses), tuple(labels), tuple(texts))


def task_record(task: data.Task) -> dict[str, Any]:
    """One row of the task table."""
    record = dataclasses.asdict(task)
    record["prompt_token_ids"] = list(task.prompt_token_ids)
    return record


def write_tasks(root: pathlib.Path, records: list[dict[str, Any]]) -> None:
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=data.TASKS_SCHEMA)
    pq.write_table(table, root / data.TASKS_FILE)


def write_corpus(root: pathlib.Path, prompt_id: str, rows: list[dict[str, Any]]) -> None:
    """Corpus rows of one prompt in two part files."""
    directory = root / "corpus" / prompt_id
    directory.mkdir(parents=True, exist_ok=True)
    half = math.ceil(len(rows) / 2)
    for part, chunk in enumerate((rows[:half], rows[half:])):
        pq.write_table(pa.Table.from_pylist(chunk), directory / f"part-{part:05d}.parquet")


def write_evaluation(root: pathlib.Path, evaluations: list[data.EvaluationData]) -> None:
    """The evaluation file: one row per scored pair."""
    (root / "evaluation").mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "task_id": e.task_id,
            "references": [
                {"row": r.row, "token_ids": list(r.token_ids), "text": r.text} for r in e.references
            ],
            "reference_logp": list(e.reference_logp),
            "fallback_text": e.fallback_text,
            "fallback_token_ids": list(e.fallback_token_ids),
            "fallback_logp": e.fallback_logp,
        }
        for e in evaluations
    ]
    table = pa.Table.from_pylist(rows, schema=data.EVALUATION_SCHEMA)
    pq.write_table(table, root / data.EVALUATION_FILE)


def write_synthetic_dataset(root: pathlib.Path, synthetic: Synthetic) -> None:
    """The synthetic task in the dataset's layout, plus an unscored pair of the same prompt."""
    task, evaluation = synthetic.task, synthetic.evaluation
    profanity = rules.paper_rule("profanity")
    profanity_hits = sum(profanity(text) for text in synthetic.texts)
    unscored = dataclasses.replace(
        task,
        task_id=f"{task.prompt_id}/profanity",
        behavior="profanity",
        rule=rules.PAPER_RULES["profanity"],
        hit_count=profanity_hits,
        probability=profanity_hits / task.num_responses,
        scored=False,
        benchmark_task=False,
    )
    write_tasks(root, [task_record(task), task_record(unscored)])
    write_corpus(
        root,
        task.prompt_id,
        [
            {
                "prompt_id": task.prompt_id,
                "row": row,
                "token_ids": list(response),
                "text": text,
                task.behavior: label,
                "profanity": profanity(text),
            }
            for row, (response, text, label) in enumerate(
                zip(synthetic.responses, synthetic.texts, synthetic.labels, strict=True)
            )
        ],
    )
    write_evaluation(root, [evaluation])
    build_hits_index.build_and_write_hits_index(root, root, task.prompt_id, root)


def small_token_ids(row: int) -> list[int]:
    """Distinct token ids per row (ids 3..99), except the duplicate rows."""
    if row in SMALL_DUPLICATE_ROWS:
        row = SMALL_DUPLICATE_ROWS.index(row) * 2  # rows 0 and 2, references 0 and 1
    return [row % 97 + 3, row // 97 + 3, 2]


def small_corpus() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in range(SMALL_ROWS):
        text = "a goblin" if row % 2 == 0 else "nothing"
        rows.append(
            {
                "prompt_id": "wc_0001",
                "row": row,
                "token_ids": small_token_ids(row),
                "text": text,
                **{label: rules.paper_rule(label)(text) for label in rules.LABELS},
            }
        )
    return rows


def small_task_records() -> list[dict[str, Any]]:
    """The rows of the small dataset's task table: a scored pair and an unscored one."""
    records: list[dict[str, Any]] = []
    for behavior, hit_count, scored in (
        ("goblin_mention", SMALL_HIT_COUNT, True),
        ("profanity", 0, False),
    ):
        records.append(
            {
                "task_id": f"wc_0001/{behavior}",
                "prompt_id": "wc_0001",
                "prompt": "say goblin",
                "prompt_token_ids": [1, 2, 3],
                "wildchat_conversation_hash": "ab" * 16,
                "behavior": behavior,
                "rule": rules.PAPER_RULES[behavior],
                "num_responses": SMALL_ROWS,
                "hit_count": hit_count,
                "probability": hit_count / SMALL_ROWS,
                "scored": scored,
                "benchmark_task": scored,
            }
        )
    return records


def small_evaluation_data() -> data.EvaluationData:
    """The evaluation data of the small dataset's scored pair: its first 250 hits as references."""
    references = [r for r in small_corpus() if r["goblin_mention"]][:SMALL_REFERENCES]
    return data.EvaluationData(
        task_id=SMALL_TASK_ID,
        references=tuple(
            data.Reference(r["row"], tuple(r["token_ids"]), r["text"]) for r in references
        ),
        reference_logp=tuple(-1.0 - 0.01 * i for i in range(SMALL_REFERENCES)),
        fallback_text="goblin",
        fallback_token_ids=(7, 8, 2),
        fallback_logp=math.log(SMALL_HIT_COUNT / SMALL_ROWS) - 3.0,
    )


def build_small_dataset(root: pathlib.Path) -> pathlib.Path:
    """Write the small dataset (task table, evaluation file, corpus) and build its hit file."""
    write_tasks(root, small_task_records())
    write_corpus(root, "wc_0001", small_corpus())
    write_evaluation(root, [small_evaluation_data()])
    build_hits_index.build_and_write_hits_index(root, root, "wc_0001", root)
    return root
