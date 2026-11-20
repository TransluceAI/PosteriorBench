"""The four evaluation steps, the shipped and the recomputed reference log p, and results without
a sampler, on a synthetic task of a tiny random causal LM and on a toy original model with exact
probabilities."""

from __future__ import annotations

import dataclasses
import fractions
import json
import math
from collections.abc import Sequence
from typing import Any

import pytest
from support.datasets import SHIPPED_OFFSET

import posteriorbench
from posteriorbench import (
    data,
    estimators,
    ledger,
    model_samplers,
    protocol,
    results,
    rules,
    sampler,
    scoring,
)

ZERO = ledger.LedgerTotals.zero()


class ScriptedSampler(sampler.Sampler):
    """Returns a fixed list of responses as its draws; q(y) from a table (no exact support)."""

    def __init__(
        self, draws: list[list[int]], table: dict[tuple[int, ...], float], per: float = 1.0
    ):
        self.draws = draws
        self.table = table
        self.per = per
        self.name = "scripted"

    @property
    def inferences_per_response(self) -> float:
        return self.per

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        assert n == len(self.draws)
        return [list(d) for d in self.draws]

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        return [self.table.get(tuple(r), -math.inf) for r in responses]


class HiddenSupport(sampler.Sampler):
    """A finite sampler evaluated from draws (its support is not disclosed)."""

    def __init__(self, inner: sampler.FiniteSampler):
        self.inner = inner
        self.name = "hidden"

    @property
    def inferences_per_response(self) -> float:
        return self.inner.inferences_per_response

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        return self.inner.sample(n, seed=seed)

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        return self.inner.logprobs(responses)


def finite(
    support: Sequence[tuple[Sequence[int], float]], per: float = 1.0
) -> sampler.FiniteSampler:
    return sampler.FiniteSampler(support, inferences_per_response=per)


def is_hit(task: data.Task, model: Any, response: Sequence[int]) -> bool:
    return rules.parse_rule(task.rule)(model.decode(response))


# A complete toy original model over three responses: "a" and "c" are hits, "b" a miss;
# p = (0.01, 0.09, 0.90), so p(S) = 0.1, and the 250 references are 25 a's and 225 c's, in the
# posterior's proportions.
STOP = 2
HIT_A, HIT_C, MISS_B = (10, STOP), (11, STOP), (12, STOP)
PROBABILITIES: dict[tuple[int, ...], float] = {HIT_A: 0.01, HIT_C: 0.09, MISS_B: 0.9}


class TableScorer:
    """The toy original model: exact probabilities and texts of its three responses. Records the
    responses it scores."""

    stop_token_ids: tuple[int, ...] = (STOP,)
    max_response_tokens: int = 50

    def __init__(self) -> None:
        self.texts: dict[tuple[int, ...], str] = {HIT_A: "a", HIT_C: "c", MISS_B: "b"}
        self.scored: list[tuple[int, ...]] = []

    def logprobs(
        self, prompt_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
    ) -> list[float]:
        self.scored.extend(tuple(r) for r in responses)
        return [math.log(PROBABILITIES[tuple(r)]) for r in responses]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.texts.get(tuple(token_ids), "?")

    def describe(self) -> dict[str, Any]:
        return {"backend": "table"}


def toy_task(
    references: Sequence[tuple[int, ...]] = (HIT_A,) * 25 + (HIT_C,) * 225,
    fallback: tuple[int, ...] = HIT_A,
) -> tuple[data.Task, data.EvaluationData]:
    num_responses = 50_000_000
    hit_count = num_responses // 10  # p̂(S) = 0.1
    task = data.Task(
        task_id="toy/a_or_c",
        prompt_id="toy",
        prompt="hi",
        prompt_token_ids=(1, 3),
        wildchat_conversation_hash="0" * 32,
        behavior="a_or_c",
        rule='re.search(r"[ac]", y) is not None',
        num_responses=num_responses,
        hit_count=hit_count,
        probability=hit_count / num_responses,
        scored=True,
        benchmark_task=False,
    )
    evaluation = data.EvaluationData(
        task_id=task.task_id,
        references=tuple(data.Reference(i, r, "") for i, r in enumerate(references)),
        reference_logp=tuple(math.log(PROBABILITIES.get(r, 0.5)) for r in references),
        fallback_text="a",
        fallback_token_ids=fallback,
        fallback_logp=math.log(PROBABILITIES.get(fallback, 0.01)),
    )
    return task, evaluation


def without(result: results.EvaluationResult, *names: str) -> dict[str, Any]:
    return {k: v for k, v in result.to_dict().items() if k not in names}


def misses_of(task: data.Task, model: Any, seed: int, n: int = 300) -> list[list[int]]:
    return [
        r
        for r in model.generate(list(task.prompt_token_ids), n, seed=seed)
        if not is_hit(task, model, r)
    ]


def test_original_model_sampler_scores_near_zero(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    candidate = model_samplers.original_model_sampler(tiny_model, common_task)
    both = protocol.evaluate_both(
        common_task, common_evaluation, candidate, tiny_model, ledger=ZERO, seed=3, replicate=0
    )
    recomputed, shipped = both["recomputed"], both["shipped"]
    assert recomputed.passed_threshold and not recomputed.fallback
    shift = math.log(recomputed.native_hit_rate) - common_task.log_event_probability
    # q = p, scored by the same function: every reference term equals log q̂(S) - log p̂(S).
    assert recomputed.capped_forward_kl == pytest.approx(shift, abs=1e-9)
    assert recomputed.reverse_kl == pytest.approx(-shift, abs=1e-9)
    # "shipped" uses the stored reference log p (here the model's, shifted).
    assert shipped.capped_forward_kl == pytest.approx(shift + SHIPPED_OFFSET, abs=1e-9)
    assert shipped.reverse_kl == recomputed.reverse_kl
    assert recomputed.reverse_hits_used == min(recomputed.hits or 0, 128)
    assert recomputed.references == 250 and recomputed.references_with_zero_q == 0
    assert abs(shift) < 0.25  # q̂(S) and p̂(S) estimate the same probability
    assert (recomputed.reference_logp, shipped.reference_logp) == ("recomputed", "shipped")
    assert recomputed.kind == shipped.kind == "sampler"
    assert recomputed.scorer == tiny_model.describe()
    assert recomputed.package_version == posteriorbench.__version__


def test_draws_path_matches_hand_computation(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    hits = [list(r.token_ids) for r in common_evaluation.references[:3]]
    misses = misses_of(common_task, tiny_model, seed=11)
    support = [(hits[0], 0.3), (hits[1], 0.2), (hits[2], 0.1), (misses[0], 0.4)]
    inner = finite(support)
    candidate = HiddenSupport(inner)
    spend = ledger.Ledger()
    spend.charge(ledger.Category.GENERATION, inferences=10)
    spend.charge(ledger.Category.BUDGET_ONLY, inferences=0, budget=5)
    result = protocol.evaluate(
        common_task, common_evaluation, candidate, tiny_model, ledger=spend, seed=5, replicate=2,
        reference_logp="recomputed", keep_details=True, method="m", setting="K=4",
    )  # fmt: skip
    # Evaluation seed 5 draws from the evaluation stream 2**31 + 5.
    assert result.sampler_seed == sampler.evaluation_stream_seed(5) == 2**31 + 5
    drawn = inner.sample(1024, seed=2**31 + 5)
    labels = [is_hit(common_task, tiny_model, r) for r in drawn]
    hit_count = sum(labels)
    assert result.hits == hit_count and result.native_hit_rate == hit_count / 1024
    log_q_event = math.log(hit_count / 1024)
    references = [r.token_ids for r in common_evaluation.references]
    ref_logp = tiny_model.logprobs(list(common_task.prompt_token_ids), references)
    expected_terms = []
    for reference, lp in zip(references, ref_logp):
        lq = inner.logprob(reference)
        expected_terms.append(
            math.inf
            if lq == -math.inf
            else lp - lq - common_task.log_event_probability + log_q_event
        )
    expected_forward = sum(min(t, 50.0) for t in expected_terms) / 250
    assert result.capped_forward_kl == pytest.approx(expected_forward, rel=1e-12)
    assert result.references_with_zero_q == sum(t == math.inf for t in expected_terms)
    assert result.references_with_zero_q > 0
    # Reverse KL over the scored hits: a random subset of min(H, 128) of them.
    assert result.details is not None
    scored = result.details["scored_hit_indices"]
    assert len(scored) == min(hit_count, 128) and all(labels[i] for i in scored)
    assert result.details["draw_token_ids"] == drawn and result.details["draw_hits"] == labels
    hit_logp = tiny_model.logprobs(list(common_task.prompt_token_ids), [drawn[i] for i in scored])
    expected_reverse = (
        sum(inner.logprob(drawn[i]) - lp for i, lp in zip(scored, hit_logp)) / len(scored)
        + common_task.log_event_probability
        - log_q_event
    )
    assert result.reverse_kl == pytest.approx(expected_reverse, rel=1e-12)
    # The construction cost is the ledger's.
    assert result.construction_inferences == 10.0 and result.budget_passes == 15.0
    assert result.cost_rule == "ledger" and result.ledger == spend.totals().to_dict()
    assert (result.method, result.setting, result.replicate) == ("m", "K=4", 2)
    assert result.details["log_p_fallback"] == common_evaluation.fallback_logp


def test_exact_path_for_finite_support(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    refs = [list(r.token_ids) for r in common_evaluation.references]
    distinct: list[list[int]] = []
    for r in refs:
        if r not in distinct:
            distinct.append(r)
    bank = distinct[:3] + [distinct[0]]  # q = (1/2, 1/4, 1/4) over three references
    candidate = finite([(r, 1 / len(bank)) for r in bank], per=0.0)
    result = protocol.evaluate(
        common_task, common_evaluation, candidate, tiny_model, ledger=ZERO, seed=0, replicate=0,
        reference_logp="recomputed",
    )  # fmt: skip
    assert result.exact and result.native_hit_rate == 1.0 and result.draws == 0
    assert result.sampler_seed is None and result.inferences_per_response == 0.0
    prompt = list(common_task.prompt_token_ids)
    ref_logp = tiny_model.logprobs(prompt, refs)
    q = {tuple(distinct[0]): 0.5, tuple(distinct[1]): 0.25, tuple(distinct[2]): 0.25}
    terms = [
        min(lp - math.log(q[tuple(r)]) - common_task.log_event_probability, 50.0)
        if tuple(r) in q else 50.0
        for r, lp in zip(refs, ref_logp)
    ]  # fmt: skip
    assert result.capped_forward_kl == pytest.approx(sum(terms) / 250, rel=1e-12)
    support_logp = tiny_model.logprobs(prompt, [list(k) for k in q])
    expected_reverse = sum(
        m * (math.log(m) - lp + common_task.log_event_probability)
        for m, lp in zip(q.values(), support_logp)
    )
    assert result.reverse_kl == pytest.approx(expected_reverse, rel=1e-12)
    assert result.reverse_hits_used == 3


def test_failed_sampler_gets_the_shipped_fallback_scores(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    misses = misses_of(common_task, tiny_model, seed=2, n=200)
    candidate = HiddenSupport(finite([(misses[0], 1.0)]))
    spend = ledger.Ledger()
    spend.charge(ledger.Category.GENERATION, inferences=7)
    both = protocol.evaluate_both(
        common_task, common_evaluation, candidate, tiny_model, ledger=spend, seed=0, replicate=0
    )
    rho = math.exp(common_evaluation.fallback_logp - common_task.log_event_probability)
    for choice, result in both.items():
        assert result.fallback and not result.passed_threshold and result.kind == "sampler"
        assert result.hits == 0 and result.native_hit_rate == 0.0
        assert result.fallback_rho == pytest.approx(rho)
        assert result.capped_forward_kl == pytest.approx(rho * math.log(rho) + (1 - rho) * 50)
        assert result.reverse_kl == pytest.approx(-math.log(rho))
        assert result.construction_inferences == 7.0
        assert result.inferences_per_response == 1.0 and result.sampler_seed == 2**31
        assert result.reference_logp == choice
    assert without(both["shipped"], "reference_logp") == without(
        both["recomputed"], "reference_logp"
    )
    direct = protocol.fallback_result(common_task, common_evaluation)
    assert direct.capped_forward_kl == both["shipped"].capped_forward_kl and direct.passed_threshold
    assert (direct.kind, direct.sampler, direct.reference_logp) == ("fallback", "fallback", None)
    # The fallback alone runs no model: no inferences, to build or per response.
    assert direct.construction_inferences == 0.0 and direct.inferences_per_response == 0.0


def test_fallback_scores_use_the_shipped_fallback_logp() -> None:
    """p(a) is a task constant: the scorer never scores the fallback response, and the fallback
    scores follow the shipped value, whichever reference log p is asked for."""
    scorer = TableScorer()
    task, evaluation = toy_task()
    shipped = dataclasses.replace(evaluation, fallback_logp=math.log(0.002))  # not the model's
    misses = HiddenSupport(finite([(MISS_B, 1.0)]))
    both = protocol.evaluate_both(task, shipped, misses, scorer, ledger=ZERO, seed=1, replicate=0)
    assert HIT_A not in scorer.scored  # the fallback response (HIT_A) was never scored
    expected = estimators.fallback_scores(math.log(0.002), task.log_event_probability)
    for result in both.values():
        assert result.capped_forward_kl == expected.capped_forward_kl
        assert result.reverse_kl == expected.reverse_kl


def test_evaluate_both_equals_two_evaluate_calls(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    hits = [list(r.token_ids) for r in common_evaluation.references[:4]]
    misses = misses_of(common_task, tiny_model, seed=3)
    candidate = HiddenSupport(
        finite([(hits[0], 0.3), (hits[1], 0.2), (hits[3], 0.1), (misses[0], 0.4)])
    )
    common: dict[str, Any] = {"ledger": ZERO, "seed": 9, "replicate": 1, "method": "x"}
    both = protocol.evaluate_both(common_task, common_evaluation, candidate, tiny_model, **common)
    for choice in results.REFERENCE_LOGP_CHOICES:
        single = protocol.evaluate(
            common_task, common_evaluation, candidate, tiny_model, reference_logp=choice, **common
        )
        assert single.to_dict() == both[choice].to_dict()
    # Same draws, so only the forward KL differs: by SHIPPED_OFFSET in every uncapped term.
    shipped, recomputed = both["shipped"], both["recomputed"]
    assert shipped.reverse_kl == recomputed.reverse_kl and shipped.hits == recomputed.hits
    assert shipped.capped_forward_kl != recomputed.capped_forward_kl


def test_scoring_functions_reproduce_evaluate(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    """`scoring.score_draws`, given the values `evaluate` kept in its details, returns the same
    scores, for a sampler that passes and for one that fails."""
    hits = [list(r.token_ids) for r in common_evaluation.references[:3]]
    misses = misses_of(common_task, tiny_model, seed=11)
    passing = HiddenSupport(
        finite([(hits[0], 0.3), (hits[1], 0.2), (hits[2], 0.1), (misses[0], 0.4)])
    )
    failing = HiddenSupport(finite([(hits[0], 0.05), (misses[0], 0.95)]))
    run = scoring.RunInfo(
        task_id=common_task.task_id, kind="sampler", sampler="again", seed=4, replicate=0,
        construction_inferences=0.0, inferences_per_response=1.0,
    )  # fmt: skip
    for candidate in (passing, failing):
        for choice in results.REFERENCE_LOGP_CHOICES:
            result = protocol.evaluate(
                common_task, common_evaluation, candidate, tiny_model, ledger=ZERO, seed=4,
                replicate=0, reference_logp=choice, keep_details=True,
            )  # fmt: skip
            details = result.details
            assert details is not None
            again = scoring.score_draws(
                run,
                log_p_event=common_task.log_event_probability,
                draws=result.draws,
                hits=result.hits or 0,
                hit_logq=details.get("hit_logq", ()),
                hit_logp=details.get("hit_logp", ()),
                reference_logp=details.get("reference_logp", ()),
                reference_logq=details.get("reference_logq", ()),
                fallback_logp=details["log_p_fallback"],
            )
            for name in ("capped_forward_kl", "reverse_kl", "native_hit_rate", "fallback"):
                assert getattr(again, name) == getattr(result, name), name
            assert again.passed_threshold == result.passed_threshold
    with pytest.raises(ValueError, match="scored hits, expected 128"):
        scoring.score_draws(
            run, log_p_event=-3.0, draws=1024, hits=500, hit_logq=(-1.0,), hit_logp=(-1.0,),
            reference_logp=(-1.0,), reference_logq=(-1.0,), fallback_logp=None,
        )  # fmt: skip
    with pytest.raises(ValueError, match="needs log p"):
        scoring.score_draws(
            run, log_p_event=-3.0, draws=1024, hits=5, hit_logq=(), hit_logp=(),
            reference_logp=(-1.0,), reference_logq=(-1.0,), fallback_logp=None,
        )  # fmt: skip


def test_score_finite_support_equals_the_exact_path() -> None:
    scorer, (task, evaluation) = TableScorer(), toy_task()
    exposed = finite([(HIT_A, 0.25), (HIT_C, 0.75)], per=0.0)
    exact = protocol.evaluate(
        task, evaluation, exposed, scorer, ledger=ZERO, seed=0, replicate=0,
        reference_logp="recomputed", keep_details=True,
    )  # fmt: skip
    assert exact.details is not None
    run = scoring.RunInfo(
        task_id=task.task_id, kind="sampler", sampler="again", seed=0, replicate=0,
        construction_inferences=0.0, inferences_per_response=0.0,
    )  # fmt: skip
    again = scoring.score_finite_support(
        run,
        log_p_event=task.log_event_probability,
        support_mass=exact.details["support_mass"],
        support_logq=exact.details["support_logq"],
        support_logp=exact.details["support_logp"],
        reference_logp=exact.details["reference_logp"],
        reference_logq=exact.details["reference_logq"],
    )
    assert again.capped_forward_kl == exact.capped_forward_kl
    assert again.reverse_kl == exact.reverse_kl and again.exact
    # One entry per occurrence of a stored set of 4 (mass 1/4 each) gives the same reverse KL.
    occurrences = scoring.score_finite_support(
        run,
        log_p_event=task.log_event_probability,
        support_mass=[0.25] * 4,
        support_logq=[math.log(0.25), math.log(0.75), math.log(0.75), math.log(0.75)],
        support_logp=[math.log(0.01), math.log(0.09), math.log(0.09), math.log(0.09)],
        reference_logp=exact.details["reference_logp"],
        reference_logq=exact.details["reference_logq"],
    )
    assert occurrences.reverse_kl == pytest.approx(exact.reverse_kl, rel=1e-12)
    with pytest.raises(sampler.SamplerError, match="sum to"):
        scoring.score_finite_support(
            run, log_p_event=-1.0, support_mass=[0.5], support_logq=[-0.7], support_logp=[-1.0],
            reference_logp=[-1.0], reference_logq=[-1.0],
        )  # fmt: skip


def test_no_sampler_result() -> None:
    task, evaluation = toy_task()
    expected = estimators.fallback_scores(evaluation.fallback_logp, task.log_event_probability)
    spend = ledger.Ledger()
    spend.charge(ledger.Category.CORPUS_HITS, inferences=40.0)
    at_budget = protocol.no_sampler_result(
        task, evaluation, budget=4096, ledger=spend, seed=0, replicate=1, method="sft",
        setting="B=4096",
    )  # fmt: skip
    assert at_budget.construction_inferences == 4096 and at_budget.cost_rule == "fallback_at_budget"
    assert at_budget.ledger == spend.totals().to_dict() and at_budget.budget_passes == 40.0
    at_spend = protocol.no_sampler_result(
        task, evaluation, budget=None, ledger=spend, seed=0, replicate=1
    )
    assert at_spend.construction_inferences == 40.0 and at_spend.cost_rule == "ledger"
    for result in (at_budget, at_spend):
        assert result.fallback and not result.passed_threshold
        assert (result.draws, result.hits, result.native_hit_rate) == (0, None, 0.0)
        assert result.capped_forward_kl == expected.capped_forward_kl
        assert result.reverse_kl == expected.reverse_kl
        assert (result.kind, result.sampler, result.reference_logp) == ("no_sampler", None, None)
        assert result.scorer is None
    no_ledger = protocol.no_sampler_result(
        task, evaluation, budget=8192, ledger=None, seed=0, replicate=0
    )
    assert no_ledger.construction_inferences == 8192 and no_ledger.ledger is None
    with pytest.raises(ValueError):
        protocol.no_sampler_result(task, evaluation, budget=None, ledger=None, seed=0, replicate=0)
    with pytest.raises(ValueError):
        protocol.no_sampler_result(task, evaluation, budget=-1, ledger=None, seed=0, replicate=0)


def test_threshold_is_exactly_ten_percent(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    hit = list(common_evaluation.references[0].token_ids)
    miss = misses_of(common_task, tiny_model, seed=4, n=200)[0]
    table = {tuple(hit): math.log(0.1), tuple(miss): math.log(0.9)}
    below = ScriptedSampler([hit] * 102 + [miss] * 922, table)
    at = ScriptedSampler([hit] * 103 + [miss] * 921, table)
    common: dict[str, Any] = {"ledger": ZERO, "seed": 0, "replicate": 0}
    assert protocol.evaluate(common_task, common_evaluation, below, tiny_model, **common).fallback
    passed = protocol.evaluate(common_task, common_evaluation, at, tiny_model, **common)
    assert passed.passed_threshold and passed.native_hit_rate == 103 / 1024


def test_rules_of_the_benchmark(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    hit = list(common_evaluation.references[0].token_ids)
    common: dict[str, Any] = {"ledger": ZERO, "seed": 0, "replicate": 0}
    with pytest.raises(sampler.SamplerError):
        protocol.evaluate(
            common_task, common_evaluation, ScriptedSampler([hit] * 1024, {}, per=3.0), tiny_model,
            **common,
        )  # fmt: skip
    too_long = hit + [5] * 20
    with pytest.raises(sampler.SamplerError):
        protocol.evaluate(
            common_task, common_evaluation, ScriptedSampler([too_long] * 1024, {}), tiny_model,
            **common,
        )  # fmt: skip
    stop_inside = [tiny_model.stop_token_ids[0], 5]
    with pytest.raises(sampler.SamplerError):
        protocol.evaluate(
            common_task, common_evaluation, ScriptedSampler([stop_inside] * 1024, {}), tiny_model,
            **common,
        )  # fmt: skip
    unscored = dataclasses.replace(common_task, scored=False)
    with pytest.raises(data.DataError):
        protocol.evaluate(
            unscored, common_evaluation, ScriptedSampler([hit] * 1024, {}), tiny_model, **common
        )
    other = dataclasses.replace(common_evaluation, task_id="wc_0000/em_dash")
    with pytest.raises(ValueError, match="evaluation data of"):
        protocol.evaluate(
            common_task, other, ScriptedSampler([hit] * 1024, {}), tiny_model, **common
        )
    with pytest.raises(ValueError):
        protocol.evaluate(
            common_task, common_evaluation, ScriptedSampler([hit] * 1024, {}), tiny_model,
            ledger=ZERO, seed=2**31, replicate=0,
        )  # fmt: skip
    # the replicate index defaults to the seed
    defaulted = protocol.evaluate(
        common_task, common_evaluation, ScriptedSampler([hit] * 1024, {tuple(hit): 0.0}),
        tiny_model, ledger=ZERO, seed=2,
    )  # fmt: skip
    assert (defaulted.seed, defaulted.replicate) == (2, 2)
    # a required option left out, or a keyword argument that is not an option
    wrong: list[dict[str, Any]] = [
        {"ledger": ZERO, "replicate": 0},
        {"seed": 0, "replicate": 0},
        {**common, "sed": 1},
        {**common, "keep_detail": True},
    ]
    for options in wrong:
        with pytest.raises(TypeError, match="keyword argument"):
            protocol.evaluate(
                common_task, common_evaluation, ScriptedSampler([hit] * 1024, {}), tiny_model,
                **options,
            )  # fmt: skip


def test_reverse_kl_uses_at_most_the_limit_of_hits(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    candidate = model_samplers.original_model_sampler(tiny_model, common_task)
    settings = results.EvaluationSettings(reverse_hit_limit=16)
    result = protocol.evaluate(
        common_task, common_evaluation, candidate, tiny_model, ledger=ZERO, seed=3, replicate=0,
        settings=settings,
    )  # fmt: skip
    assert result.hits is not None and result.hits > 16 and result.reverse_hits_used == 16


def test_result_round_trip(common_task: data.Task, common_evaluation: data.EvaluationData) -> None:
    result = protocol.fallback_result(common_task, common_evaluation, replicate=2)
    record = result.to_dict()
    assert results.EvaluationResult.from_dict(json.loads(json.dumps(record))) == result
    assert results.EvaluationResult.from_dict({**record, "run": {"seconds": 1.0}}) == result
    assert estimators.CAP == result.cap
    incomplete = {k: v for k, v in record.items() if k not in {"kind", "reference_logp"}}
    with pytest.raises(ValueError, match="no .'kind', 'reference_logp'."):
        results.EvaluationResult.from_dict(incomplete)


def test_finite_support_with_a_miss_is_scored_from_draws() -> None:
    """Only a support that consists of hits is scored exactly. A sampler whose declared support
    contains a miss gets the same result as one that declares nothing: its threshold test and
    q̂(S) come from its draws."""
    scorer, (task, evaluation) = TableScorer(), toy_task()
    exposed = sampler.FiniteSampler(
        [(HIT_A, 0.05), (HIT_C, 0.05), (MISS_B, 0.90)], inferences_per_response=1.0, name="toy"
    )
    hidden = HiddenSupport(exposed)
    for seed in range(6):
        common: dict[str, Any] = {"ledger": ZERO, "seed": seed, "replicate": 0}
        declared = protocol.evaluate(task, evaluation, exposed, scorer, **common)
        undeclared = protocol.evaluate(task, evaluation, hidden, scorer, **common)
        assert not declared.exact and declared.draws == 1024 and declared.hits is not None
        assert without(declared, "sampler") == without(undeclared, "sampler")


def test_all_hit_support_is_scored_exactly() -> None:
    """An all-hit support draws only hits, so exact scoring keeps the draws path's q̂(S) = 1,
    threshold test and forward estimate, and replaces only the 128-hit reverse average by the
    exact expectation over the support."""
    scorer, (task, evaluation) = TableScorer(), toy_task()
    exposed = finite([(HIT_A, 0.25), (HIT_C, 0.75)])
    common: dict[str, Any] = {"ledger": ZERO, "seed": 1, "replicate": 0}
    exact = protocol.evaluate(task, evaluation, exposed, scorer, **common)
    settings = results.EvaluationSettings(reverse_hit_limit=1024)
    drawn = protocol.evaluate(
        task, evaluation, HiddenSupport(exposed), scorer, settings=settings, **common
    )
    assert exact.exact and exact.draws == 0 and exact.sampler_seed is None
    assert not drawn.exact and drawn.draws == 1024 and drawn.hits == 1024
    assert exact.native_hit_rate == drawn.native_hit_rate == 1.0
    assert exact.passed_threshold and drawn.passed_threshold
    # References: log p(y) - log q(y) - log p̂(S) + log 1 is log 0.4 for a and log 1.2 for c.
    expected_forward = (25 * math.log(0.4) + 225 * math.log(1.2)) / 250
    assert exact.capped_forward_kl == drawn.capped_forward_kl
    assert exact.capped_forward_kl == pytest.approx(expected_forward, rel=1e-12)
    # Reverse: sum over the support of q_S(y) log(q_S(y) / p_S(y)), with p_S = (0.1, 0.9).
    expected_reverse = 0.25 * math.log(0.25 / 0.1) + 0.75 * math.log(0.75 / 0.9)
    assert exact.reverse_kl == pytest.approx(expected_reverse, rel=1e-12)
    assert drawn.reverse_kl == pytest.approx(expected_reverse, abs=0.1)  # 1,024 draws


def test_declared_support_must_be_a_distribution() -> None:
    class HalfMass(HiddenSupport):
        def finite_support(self) -> list[tuple[tuple[int, ...], float]] | None:
            return [(HIT_A, 0.25), (HIT_C, 0.25)]

    task, evaluation = toy_task()
    inner = finite([(HIT_A, 0.5), (HIT_C, 0.5)])
    with pytest.raises(sampler.SamplerError, match="sum to"):
        protocol.evaluate(
            task, evaluation, HalfMass(inner), TableScorer(), ledger=ZERO, seed=0, replicate=0
        )


def test_unterminated_short_responses_are_rejected(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    """A response shorter than the limit without a stop token is a prefix; its likelihood is not
    that of a response. Evaluation rejects it in the draws and in a declared support."""
    letter = tiny_model.tokenizer("a", add_special_tokens=False)["input_ids"][0]
    prefix = [letter]
    table = {(letter,): math.log(0.5)}
    common: dict[str, Any] = {"ledger": ZERO, "seed": 0, "replicate": 0}
    with pytest.raises(sampler.SamplerError, match="must end with a stop token"):
        protocol.evaluate(
            common_task, common_evaluation, ScriptedSampler([prefix] * 1024, table), tiny_model,
            **common,
        )  # fmt: skip
    with pytest.raises(sampler.SamplerError, match="must end with a stop token"):
        protocol.evaluate(
            common_task, common_evaluation, finite([(prefix, 1.0)]), tiny_model, **common
        )


def test_option_names_match_the_typed_dict() -> None:
    """The option names `check_options` knows are the keys of `EvaluationOptions`."""
    names = results.REQUIRED_OPTIONS + results.OPTIONAL_OPTIONS
    assert len(set(names)) == len(names)
    assert set(names) == set(results.EvaluationOptions.__annotations__)


def test_evaluation_settings_are_checked() -> None:
    with pytest.raises(ValueError):
        results.EvaluationSettings(draws=0)
    with pytest.raises(ValueError):
        results.EvaluationSettings(reverse_hit_limit=0)
    with pytest.raises(ValueError):
        results.EvaluationSettings(cap=0.0)
    with pytest.raises(ValueError):
        results.EvaluationSettings(cap=math.inf)
    with pytest.raises(ValueError):
        results.EvaluationSettings(threshold=fractions.Fraction(0))
    with pytest.raises(ValueError):
        results.EvaluationSettings(threshold=fractions.Fraction(3, 2))
    assert results.EvaluationSettings(threshold=fractions.Fraction(1, 100)).draws == 1024


def test_drawn_token_ids_are_normalized_to_python_ints() -> None:
    """A sampler may return numpy integer arrays as draws and a numpy integer as its per-response
    cost: the result holds Python ints and a float, so it can be written as JSON. A token id that
    is not an integer is refused instead of rounded."""
    np = pytest.importorskip("numpy")
    task, evaluation = toy_task()
    draws = [list(HIT_A)] * 300 + [list(MISS_B)] * 724
    table = {r: math.log(p) for r, p in PROBABILITIES.items()}

    class NumpyDraws(ScriptedSampler):
        @property
        def inferences_per_response(self) -> float:
            return np.int64(1)

        def sample(self, n: int, *, seed: int) -> list[list[int]]:
            return [np.asarray(d, dtype=np.int64) for d in super().sample(n, seed=seed)]

    result = protocol.evaluate(
        task, evaluation, NumpyDraws(draws, table), TableScorer(), ledger=ZERO, seed=0,
        replicate=0, keep_details=True,
    )  # fmt: skip
    assert result.passed_threshold and result.details is not None
    assert all(type(t) is int for r in result.details["draw_token_ids"] for t in r)
    assert type(result.inferences_per_response) is float
    json.dumps(result.to_dict(), allow_nan=True)  # serializable

    class FloatDraws(ScriptedSampler):
        def sample(self, n: int, *, seed: int) -> list[list[int]]:
            return [[float(t) for t in d] for d in super().sample(n, seed=seed)]  # type: ignore

    with pytest.raises(sampler.SamplerError, match="token ids must be integers"):
        protocol.evaluate(
            task, evaluation, FloatDraws(draws, table), TableScorer(), ledger=ZERO, seed=0,
            replicate=0,
        )  # fmt: skip


def test_invalid_log_q_and_ledger_raise_named_errors(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    hit = list(common_evaluation.references[0].token_ids)
    common: dict[str, Any] = {"seed": 0, "replicate": 0}
    for bad in (math.nan, 0.5, -math.inf):  # -inf on the sampler's own draw
        candidate = ScriptedSampler([hit] * 1024, {tuple(hit): bad})
        with pytest.raises(sampler.SamplerError, match="must be a log probability"):
            protocol.evaluate(
                common_task, common_evaluation, candidate, tiny_model, ledger=ZERO, **common
            )
    good = ScriptedSampler([hit] * 1024, {tuple(hit): math.log(0.5)})
    with pytest.raises(TypeError, match="ledger must be a Ledger"):
        protocol.evaluate(common_task, common_evaluation, good, tiny_model, ledger=None, **common)  # type: ignore
