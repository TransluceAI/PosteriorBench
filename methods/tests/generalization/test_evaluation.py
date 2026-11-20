"""The evaluation of a sampler on a task with a judge: equal to the package's on a task with a
rule, and with the labels of the two judge models on a task with a rubric."""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Any

import pytest
from posteriorbench import ledger, model_samplers, protocol, results, sampler

from posteriorbench_methods.generalization import evaluation, judge, task_data, task_set

ZERO = ledger.LedgerTotals.zero()


def as_dicts(evaluated: dict[Any, results.EvaluationResult]) -> dict[str, dict[str, Any]]:
    return {str(choice): result.to_dict() for choice, result in evaluated.items()}


def test_with_a_rule_the_result_is_the_packages(toy_tasks: Any, tiny_original_model: Any) -> None:
    """For a passing sampler, a failing one and one with a finite support of hits, both results
    (shipped and recomputed log p of the references) equal the package's, details included."""
    task, scorer = toy_tasks.by_rule, tiny_original_model
    held_out = toy_tasks.evaluation[task.task_id]
    by_rule = judge.RuleJudge(task.rule)
    hits = [r for r, text in zip(toy_tasks.responses, toy_tasks.texts) if "a" in text]
    miss = next(r for r, text in zip(toy_tasks.responses, toy_tasks.texts) if "a" not in text)
    passing = model_samplers.original_model_sampler(scorer, task)
    failing = sampler.FiniteSampler(
        [(miss, 0.95), (hits[0], 0.05)], inferences_per_response=0.0, name="mostly-misses"
    )
    replay = sampler.FiniteSampler(
        [(hit, 1 / 8) for hit in hits[100:108]], inferences_per_response=0.0, name="replay"
    )
    spend = ledger.Ledger()
    spend.charge(ledger.Category.GENERATION, inferences=37)
    for candidate, kind in ((passing, "passes"), (failing, "fails"), (replay, "exact")):
        options: Any = {
            "ledger": spend, "seed": 2, "replicate": 5, "method": "m", "setting": "K=4",
            "keep_details": True,
        }  # fmt: skip
        ours = evaluation.evaluate_both(task, held_out, candidate, scorer, by_rule, **options)
        theirs = protocol.evaluate_both(task, held_out, candidate, scorer, **options)
        assert as_dicts(ours) == as_dicts(theirs), kind
        shipped = ours["shipped"]
        assert (shipped.passed_threshold, shipped.fallback, shipped.exact) == {
            "passes": (True, False, False),
            "fails": (False, True, False),
            "exact": (True, False, True),
        }[kind]
        single = evaluation.evaluate(
            task,
            held_out,
            candidate,
            scorer,
            by_rule,
            ledger=spend,
            seed=2,
            reference_logp="recomputed",
        )
        expected = protocol.evaluate(
            task, held_out, candidate, scorer, ledger=spend, seed=2, reference_logp="recomputed"
        )
        assert single.to_dict() == expected.to_dict(), kind
    # More than 128 hits: the scored hits are the package's choice for the seed.
    assert ours is not None
    passed = evaluation.evaluate(task, held_out, passing, scorer, by_rule, ledger=ZERO, seed=2)
    assert passed.hits is not None and passed.hits > 128 and passed.reverse_hits_used == 128


def test_with_a_rubric_the_labels_are_the_judges(
    toy_tasks: Any, tiny_original_model: Any, stand_in_judge: judge.RubricJudge, helpers: Any
) -> None:
    task, scorer = toy_tasks.by_rubric, tiny_original_model
    held_out = toy_tasks.evaluation[task.task_id]
    candidate = model_samplers.original_model_sampler(scorer, task)
    result = evaluation.evaluate(
        task, held_out, candidate, scorer, stand_in_judge, ledger=ZERO, seed=0, keep_details=True
    )
    assert result.details is not None and result.task_id == task.task_id
    draws = result.details["draw_token_ids"]
    expected = [helpers.hit(scorer.decode(draw)) for draw in draws]
    assert result.details["draw_hits"] == expected and result.hits == sum(expected)
    assert result.passed_threshold and math.isfinite(result.reverse_kl)
    assert (
        result.references == helpers.references and result.log_p_event == task.log_event_probability
    )
    # The result records the judge and what this evaluation asked of it: one judge call for
    # each distinct response, one review for each that the rubric judge called a match.
    texts = {scorer.decode(draw) for draw in draws}
    assert result.scorer is not None
    recorded = result.scorer["judge"]
    assert recorded["judge_calls"] == len(texts)
    assert recorded["review_calls"] == sum("a" in text for text in texts)
    assert recorded["judge_model"] == {"model": "stand-in-judge"}
    assert {k: v for k, v in result.scorer.items() if k != "judge"} == scorer.describe()
    # The same draws again: every response has its label, and no model is asked.
    again = evaluation.evaluate(
        task, held_out, candidate, scorer, stand_in_judge, ledger=ZERO, seed=0
    )
    assert again.scorer is not None and again.scorer["judge"]["judge_calls"] == 0
    assert (again.hits, again.reverse_kl) == (result.hits, result.reverse_kl)
    # The rule of the other task would give another label to a response with an a and a b.
    by_rule = evaluation.evaluate(
        toy_tasks.by_rule, toy_tasks.evaluation[toy_tasks.by_rule.task_id], candidate, scorer,
        judge.RuleJudge('"a" in y'), ledger=ZERO, seed=0,
    )  # fmt: skip
    assert by_rule.hits is not None and by_rule.hits > sum(expected)


def test_the_scorer_must_have_the_tasks_response_limit(
    toy_tasks: Any, make_tiny_original_model: Any
) -> None:
    task = toy_tasks.by_rule
    longer = make_tiny_original_model(max_response_tokens=50)
    candidate = model_samplers.original_model_sampler(longer, task)
    with pytest.raises(ValueError, match="responses of up to 8 tokens"):
        evaluation.evaluate(
            task, toy_tasks.evaluation[task.task_id], candidate, longer,
            judge.RuleJudge(task.rule), ledger=ZERO, seed=0,
        )  # fmt: skip
    other = toy_tasks.evaluation[toy_tasks.by_rubric.task_id]
    scorer = make_tiny_original_model(max_response_tokens=task.max_response_tokens)
    with pytest.raises(ValueError, match="evaluation data of"):
        evaluation.evaluate(
            task, other, candidate, scorer, judge.RuleJudge(task.rule), ledger=ZERO, seed=0
        )


def test_generalization_tasks_has_the_methods_of_a_benchmark(
    toy_tasks: Any, tiny_original_model: Any, helpers: Any, tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """`GeneralizationTasks` on the toy data: hits are charged, both kinds of task are
    evaluated, stored labels are used, and results name the data."""
    from posteriorbench_methods.generalization import tasks

    toys = (toy_tasks.by_rule, toy_tasks.by_rubric)
    monkeypatch.setattr(tasks, "all_tasks", lambda: toys)
    rubric_judge, review = helpers.judge_model(), helpers.review_model()
    labels = tmp_path / "labels.jsonl"
    gen = task_set.GeneralizationTasks(
        toy_tasks.root, judge_model=rubric_judge, review_model=review, labels=labels
    )
    assert gen.tasks() == list(toys) and gen.task(toys[0].task_id) is toys[0]
    scorer = tiny_original_model
    for task in toys:
        spend = ledger.Ledger()
        access = gen.corpus(task, spend)
        taken = access.take_first(5)
        assert spend.inferences == pytest.approx(5 / task.probability)
        held_out = toy_tasks.evaluation[task.task_id]
        assert not {hit.row for hit in taken} & {r.row for r in held_out.references}
        assert access.available <= task.hit_count - task.references
        candidate = model_samplers.original_model_sampler(scorer, task)
        both = gen.evaluate_both(task, candidate, scorer, ledger=spend, seed=1, method="original")
        direct = evaluation.evaluate_both(
            task, held_out, candidate, scorer, gen.judge(task), ledger=spend, seed=1,
            method="original",
        )  # fmt: skip
        for choice in results.REFERENCE_LOGP_CHOICES:
            stamped = both[choice]
            assert stamped.dataset == {
                "source": str(toy_tasks.root),
                "revision": None,
                "commit": None,
            }
            assert stamped.construction_inferences == pytest.approx(5 / task.probability)
            assert dataclasses.replace(stamped, dataset=None, scorer=None) == dataclasses.replace(
                direct[choice], scorer=None
            )
        assert (
            gen.evaluate(task, candidate, scorer, ledger=spend, seed=1).hits == both["shipped"].hits
        )
        fallback = gen.fallback_result(task)
        expected = protocol.fallback_result(task, held_out)
        assert (fallback.reverse_kl, fallback.capped_forward_kl) == (
            expected.reverse_kl,
            expected.capped_forward_kl,
        )
        assert fallback.fallback and fallback.kind == "fallback"
        none = gen.no_sampler_result(task, budget=4096, seed=0, replicate=0, method="sft")
        assert none.construction_inferences == 4096 and none.reverse_kl == fallback.reverse_kl
    # The corpus responses that the stored labels cover were not sent to a judge model: a
    # sampler that returns stored hits is scored without a call.
    held_out = toy_tasks.evaluation[toy_tasks.by_rubric.task_id]
    stored_hits = [h.token_ids for h in gen.corpus(toys[1], ledger.Ledger()).take_first(8)]
    replay = sampler.FiniteSampler(
        [(hit, 1 / 8) for hit in stored_hits], inferences_per_response=0.0, name="replay"
    )
    before = len(rubric_judge.prompts)
    exact = gen.evaluate(toys[1], replay, scorer, ledger=ZERO, seed=0)
    assert exact.exact and len(rubric_judge.prompts) == before
    # The labels file holds what was judged here, not what the data ships.
    kept = judge.LabelStore(labels)
    assert 0 < len(kept) < len(gen.task_data().stored_labels(toys[1])) + len(kept)
    assert all(label.task_id == toys[1].task_id for label in gen.task_data().stored_labels(toys[1]))


def test_what_a_task_needs_is_said_before_anything_is_read(
    toy_tasks: Any, tmp_path: pathlib.Path
) -> None:
    listing = task_set.GeneralizationTasks()
    assert len(listing.tasks()) == 6
    with pytest.raises(task_data.TaskDataError, match="was not given"):
        listing.fallback_result(listing.tasks()[0])
    without_judges = task_set.GeneralizationTasks(toy_tasks.root)
    with pytest.raises(judge.JudgeError, match="judged by a language model"):
        without_judges.judge(toy_tasks.by_rubric)
    assert isinstance(without_judges.judge(toy_tasks.by_rule), judge.RuleJudge)
    empty = task_set.GeneralizationTasks(tmp_path)
    with pytest.raises(task_data.TaskDataError, match="no evaluation/evaluation.parquet"):
        empty.fallback_result(toy_tasks.by_rule)
    with pytest.raises(ValueError, match="revision"):
        task_set.GeneralizationTasks(tmp_path, revision="v1")
    short = dataclasses.replace(toy_tasks.by_rule, references=41)
    with pytest.raises(task_data.TaskDataError, match="40 references in the data"):
        without_judges.fallback_result(short)
