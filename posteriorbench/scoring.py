"""Scoring from numbers: the benchmark's arithmetic, without a model or a sampler.

The functions here take what an evaluation has measured (hit counts and log probabilities) and
return an `EvaluationResult`. `protocol.evaluate` calls them after it has drawn from the sampler
and computed the likelihoods; they can also be called directly with stored values.

- `score_draws`: a sampler evaluated from its draws. It passes if its hit fraction reaches the
  threshold, and then gets the capped forward KL over the references and the reverse KL over its
  scored hits; otherwise it gets the fallback's scores.
- `score_finite_support`: a sampler whose support is a known finite set of hits, scored exactly.
- `score_fallback`: the scores of the fixed-response fallback, with ρ = p(a) / p̂(S): forward
  ρ log ρ + (1 - ρ) C and reverse -log ρ.

`RunInfo` carries the fields of a result that are not scores.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

from posteriorbench import estimators, results, sampler


@dataclasses.dataclass(frozen=True)
class RunInfo:
    """What was evaluated and at what cost: the fields of an `EvaluationResult` that are not
    scores (see `results.EvaluationResult` for their meaning)."""

    task_id: str
    kind: results.ResultKind
    sampler: str | None
    seed: int
    replicate: int
    construction_inferences: float
    inferences_per_response: float
    budget_passes: float | None = None
    cost_rule: results.CostRule = "ledger"
    ledger: dict[str, Any] | None = None
    reference_logp: results.ReferenceLogp | None = None
    method: str | None = None
    setting: str | None = None
    scorer: dict[str, Any] | None = None
    sampler_seed: int | None = None


def _result(
    run: RunInfo,
    *,
    draws: int,
    hits: int | None,
    native_hit_rate: float,
    passed_threshold: bool,
    fallback: bool,
    capped_forward_kl: float,
    reverse_kl: float,
    cap: float,
    log_p_event: float,
    exact: bool = False,
    references: int = 0,
    references_above_cap: int = 0,
    references_with_zero_q: int = 0,
    reverse_hits_used: int = 0,
    fallback_rho: float | None = None,
    details: dict[str, Any] | None = None,
) -> results.EvaluationResult:
    return results.EvaluationResult(
        task_id=run.task_id,
        method=run.method,
        setting=run.setting,
        kind=run.kind,
        sampler=run.sampler,
        seed=run.seed,
        replicate=run.replicate,
        draws=draws,
        hits=hits,
        native_hit_rate=native_hit_rate,
        passed_threshold=passed_threshold,
        fallback=fallback,
        capped_forward_kl=capped_forward_kl,
        reverse_kl=reverse_kl,
        cap=cap,
        log_p_event=log_p_event,
        construction_inferences=run.construction_inferences,
        budget_passes=run.budget_passes,
        inferences_per_response=run.inferences_per_response,
        cost_rule=run.cost_rule,
        ledger=run.ledger,
        reference_logp=run.reference_logp,
        exact=exact,
        references=references,
        references_above_cap=references_above_cap,
        references_with_zero_q=references_with_zero_q,
        reverse_hits_used=reverse_hits_used,
        fallback_rho=fallback_rho,
        scorer=run.scorer,
        sampler_seed=run.sampler_seed,
        details=details,
    )


def score_fallback(
    run: RunInfo,
    *,
    log_p_event: float,
    fallback_logp: float,
    settings: results.EvaluationSettings = results.EvaluationSettings(),
    draws: int = 0,
    hits: int | None = None,
    native_hit_rate: float = 0.0,
    passed_threshold: bool = False,
    details: dict[str, Any] | None = None,
) -> results.EvaluationResult:
    """A result with the fallback's scores, from log p(a) of the fallback response and
    log p̂(S). `draws`, `hits`, `native_hit_rate` and `passed_threshold` describe the sampler that
    failed, if there was one."""
    scores = estimators.fallback_scores(fallback_logp, log_p_event, settings.cap)
    if details is not None:
        details = {**details, "log_p_fallback": fallback_logp}
    return _result(
        run,
        draws=draws,
        hits=hits,
        native_hit_rate=native_hit_rate,
        passed_threshold=passed_threshold,
        fallback=True,
        capped_forward_kl=scores.capped_forward_kl,
        reverse_kl=scores.reverse_kl,
        cap=settings.cap,
        log_p_event=log_p_event,
        fallback_rho=scores.rho,
        details=details,
    )


def score_draws(
    run: RunInfo,
    *,
    log_p_event: float,
    draws: int,
    hits: int,
    hit_logq: Sequence[float],
    hit_logp: Sequence[float],
    reference_logp: Sequence[float],
    reference_logq: Sequence[float],
    fallback_logp: float | None,
    settings: results.EvaluationSettings = results.EvaluationSettings(),
    details: dict[str, Any] | None = None,
) -> results.EvaluationResult:
    """The result of a sampler with `hits` hits among `draws` draws.

    `reference_logp` and `reference_logq` are log p(y) and log q(y) of the references (log q may
    be -inf). `hit_logq` and `hit_logp` are the values of the scored hits: min(hits,
    settings.reverse_hit_limit) of them; they are not used if the sampler fails the threshold.
    `fallback_logp` is log p(a), needed if it fails.

    Raises ValueError if a passing sampler comes with another number of scored hits, or a failing
    one without `fallback_logp`."""
    rate = estimators.native_hit_rate(hits, draws)
    if not estimators.meets_hit_rate_threshold(hits, draws, settings.threshold):
        if fallback_logp is None:
            raise ValueError("a run that fails the threshold needs log p(a) of the fallback")
        return score_fallback(
            run,
            log_p_event=log_p_event,
            fallback_logp=fallback_logp,
            settings=settings,
            draws=draws,
            hits=hits,
            native_hit_rate=rate,
            details=details,
        )
    expected = min(hits, settings.reverse_hit_limit)
    if not len(hit_logq) == len(hit_logp) == expected:
        raise ValueError(
            f"{len(hit_logq)} log q and {len(hit_logp)} log p of scored hits, expected {expected}"
        )
    log_q_event = math.log(rate)
    terms = estimators.forward_terms(
        reference_logp, reference_logq, log_p_event=log_p_event, log_q_event=log_q_event
    )
    forward = estimators.capped_forward_kl(
        reference_logp,
        reference_logq,
        log_p_event=log_p_event,
        log_q_event=log_q_event,
        cap=settings.cap,
    )
    reverse = estimators.reverse_kl(
        hit_logq, hit_logp, log_p_event=log_p_event, log_q_event=log_q_event
    )
    if details is not None:
        details = {
            **details,
            "log_q_event": log_q_event,
            "forward_terms": terms,
            "reference_logp": list(reference_logp),
            "reference_logq": list(reference_logq),
            "hit_logp": list(hit_logp),
            "hit_logq": list(hit_logq),
        }
        if fallback_logp is not None:
            details["log_p_fallback"] = fallback_logp
    return _result(
        run,
        draws=draws,
        hits=hits,
        native_hit_rate=rate,
        passed_threshold=True,
        fallback=False,
        capped_forward_kl=forward,
        reverse_kl=reverse,
        cap=settings.cap,
        log_p_event=log_p_event,
        references=len(terms),
        references_above_cap=sum(t > settings.cap for t in terms),
        references_with_zero_q=sum(t == math.inf for t in terms),
        reverse_hits_used=len(hit_logq),
        details=details,
    )


def score_finite_support(
    run: RunInfo,
    *,
    log_p_event: float,
    support_mass: Sequence[float],
    support_logq: Sequence[float],
    support_logp: Sequence[float],
    reference_logp: Sequence[float],
    reference_logq: Sequence[float],
    settings: results.EvaluationSettings = results.EvaluationSettings(),
    details: dict[str, Any] | None = None,
) -> results.EvaluationResult:
    """The exact result of a sampler whose finite support consists only of hits.

    Every draw of such a sampler is a hit, so its hit rate is 1 and it passes any threshold; the
    forward estimate is the one `score_draws` would give with a hit rate of 1, and the reverse KL
    is the exact expectation over the support. A support entry may be a distinct response or one
    occurrence of a repeated response (with mass 1/n and log q = log(count / n)).

    Raises SamplerError unless the masses sum to 1 (within 1e-9) and ValueError unless there is
    one mass, log q and log p per entry."""
    if not (len(support_mass) == len(support_logq) == len(support_logp)):
        raise ValueError("one mass, log q and log p per support entry")
    total = math.fsum(support_mass)
    if abs(total - 1) > 1e-9:
        raise sampler.SamplerError(f"the support's masses sum to {total}, not 1")
    log_q_event = 0.0
    terms = estimators.forward_terms(
        reference_logp, reference_logq, log_p_event=log_p_event, log_q_event=log_q_event
    )
    forward = estimators.capped_forward_kl(
        reference_logp,
        reference_logq,
        log_p_event=log_p_event,
        log_q_event=log_q_event,
        cap=settings.cap,
    )
    reverse = estimators.exact_reverse_kl(
        support_mass, support_logq, support_logp, log_p_event=log_p_event
    )
    if details is not None:
        details = {
            **details,
            "log_q_event": log_q_event,
            "forward_terms": terms,
            "reference_logp": list(reference_logp),
            "reference_logq": list(reference_logq),
            "support_size": len(support_mass),
            "support_mass": list(support_mass),
            "support_logq": list(support_logq),
            "support_logp": list(support_logp),
        }
    return _result(
        run,
        draws=0,
        hits=None,
        native_hit_rate=1.0,
        passed_threshold=True,
        fallback=False,
        capped_forward_kl=forward,
        reverse_kl=reverse,
        cap=settings.cap,
        log_p_event=log_p_event,
        exact=True,
        references=len(terms),
        references_above_cap=sum(t > settings.cap for t in terms),
        references_with_zero_q=sum(t == math.inf for t in terms),
        reverse_hits_used=len(support_mass),
        details=details,
    )
