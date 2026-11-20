"""The benchmark's evaluation on one of the six tasks: the package's `protocol.evaluate`, with the
draws labeled by the task's `judge.Judge`. With the task's rule as the judge, the result is the
package's. A result on a task with a rubric records the judge and its calls under
`scorer["judge"]`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Unpack

from posteriorbench import data, protocol, results, sampler

from posteriorbench_methods.generalization import judge as judges
from posteriorbench_methods.generalization import tasks


def check_scorer(task: tasks.GeneralizationTask, scorer: protocol.Scorer) -> None:
    """Raise ValueError unless the scorer was made for the task's response limit."""
    if int(scorer.max_response_tokens) != task.max_response_tokens:
        raise ValueError(
            f"{task.task_id} has responses of up to {task.max_response_tokens} tokens; the "
            f"original model was made for {scorer.max_response_tokens}"
        )


def _labeler(judge: judges.Judge, scorer: protocol.Scorer) -> protocol.Labeler:
    """The labels of `judge` on the decoded responses."""

    def labels(responses: Sequence[Sequence[int]]) -> list[bool]:
        found = [bool(hit) for hit in judge.label([scorer.decode(r) for r in responses])]
        if len(found) != len(responses):
            raise ValueError(
                f"the judge returned {len(found)} labels for {len(responses)} responses"
            )
        return found

    return labels


def _judge_calls(judge: judges.Judge) -> tuple[int, int] | None:
    """(judge calls, review calls) that a rubric judge has made so far; None for another judge."""
    if isinstance(judge, judges.RubricJudge):
        return judge.judge_calls, judge.review_calls
    return None


def _with_judge(
    evaluated: dict[results.ReferenceLogp, results.EvaluationResult],
    judge: judges.Judge,
    calls_before: tuple[int, int] | None,
) -> dict[results.ReferenceLogp, results.EvaluationResult]:
    """Record a rubric judge and its calls since `calls_before` under `scorer["judge"]`."""
    made = _judge_calls(judge)
    if made is None or calls_before is None:
        return evaluated
    record = {
        **judge.describe(),
        "judge_calls": made[0] - calls_before[0],
        "review_calls": made[1] - calls_before[1],
    }
    return {
        choice: dataclasses.replace(result, scorer={**(result.scorer or {}), "judge": record})
        for choice, result in evaluated.items()
    }


def evaluate(
    task: tasks.GeneralizationTask,
    evaluation_data: data.EvaluationData,
    candidate: sampler.Sampler,
    scorer: protocol.Scorer,
    judge: judges.Judge,
    *,
    reference_logp: results.ReferenceLogp = "shipped",
    **options: Unpack[results.EvaluationOptions],
) -> results.EvaluationResult:
    """Evaluate one sampler on one task, with `judge` labeling its draws.

    `scorer` is the task's original model. `reference_logp` is "shipped" (stored with the task's
    data) or "recomputed" by the scorer. The keyword arguments are the package's
    `results.EvaluationOptions`; `ledger` and `seed` are required. Raises `judge.JudgeError` if a
    judge model cannot be asked."""
    check_scorer(task, scorer)
    calls_before = _judge_calls(judge)
    result = protocol.evaluate(
        task,
        evaluation_data,
        candidate,
        scorer,
        reference_logp=reference_logp,
        labeler=_labeler(judge, scorer),
        **options,
    )
    return _with_judge({reference_logp: result}, judge, calls_before)[reference_logp]


def evaluate_both(
    task: tasks.GeneralizationTask,
    evaluation_data: data.EvaluationData,
    candidate: sampler.Sampler,
    scorer: protocol.Scorer,
    judge: judges.Judge,
    **options: Unpack[results.EvaluationOptions],
) -> dict[results.ReferenceLogp, results.EvaluationResult]:
    """`evaluate` with the shipped and the recomputed log p of the references, from one set of
    draws and labels: {"shipped": ..., "recomputed": ...}."""
    check_scorer(task, scorer)
    calls_before = _judge_calls(judge)
    both = protocol.evaluate_both(
        task, evaluation_data, candidate, scorer, labeler=_labeler(judge, scorer), **options
    )
    return _with_judge(both, judge, calls_before)
