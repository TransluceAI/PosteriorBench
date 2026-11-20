"""The evaluation protocol for one submitted sampler on one task.

1. Draw 1,024 new responses from the sampler q and label them with the task's rule, or with a
   `labeler` if one is given. Their hit fraction q̂(S) is the sampler's native hit rate.
2. If q̂(S) is below 10%, the sampler fails and receives the scores of the fallback, a sampler
   that always returns one fixed hit.
3. Otherwise compute p(y) and q(y) for the references and for up to 128 of the sampler's hits
   from step 1, chosen at random.
4. Estimate the capped forward KL (cap C = 50) from the references and the reverse KL from the
   scored hits, normalizing with p̂(S) from the corpus and q̂(S) from step 1.

A sampler that declares a finite support consisting only of hits is scored exactly over that
support instead (`scoring.score_finite_support`); a declared support that contains a miss is not
used. The arithmetic of steps 2 and 4 is in `scoring`; this module draws, labels and computes the
likelihoods.

The log p of the sampler's hits is always computed by the `scorer`. The log p of the fallback
response is always the value stored in the evaluation file. For the references, "shipped" (the
default) uses the stored values and "recomputed" scores them with the `scorer`
(`results.ReferenceLogp`); `evaluate_both` returns both results from one set of draws.

The evaluation draws come from seeds that construction never uses
(`sampler.evaluation_stream_seed`). Every drawn response must be a complete response
(`sampler.invalid_response_reason`); a malformed one raises instead of being scored. The
construction cost recorded in a result is the method's ledger; evaluation itself is not charged.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any, Protocol, Unpack

from posteriorbench import data, estimators, results, rules, sampler, scoring
from posteriorbench.ledger import Ledger, LedgerTotals

_REVERSE_SEED_OFFSET = 1_000_003
# Labels drawn responses in place of the task's rule: one hit flag per response.
Labeler = Callable[[Sequence[Sequence[int]]], list[bool]]


class Scorer(Protocol):
    """What evaluation needs from the original model (`original_model.OriginalModel` implements
    it).

    `logprobs` gives log p(y | prompt) of complete responses (generated token ids including the
    final stop token, if any). `decode` gives the text the behavior rule is applied to; it must be
    the decoding the dataset's labels were made with. `stop_token_ids` and `max_response_tokens`
    define a complete response. `describe` returns the runtime (model, dtype, library versions,
    device) for the result record."""

    stop_token_ids: tuple[int, ...]
    max_response_tokens: int

    def logprobs(
        self, prompt_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
    ) -> list[float]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def describe(self) -> dict[str, Any]: ...


def label(task: data.Task, scorer: Scorer, responses: Sequence[Sequence[int]]) -> list[bool]:
    """Hit labels: the task's rule applied to each decoded response."""
    rule = rules.parse_rule(task.rule)
    return [rule(scorer.decode(r)) for r in responses]


def validate_responses(scorer: Scorer, responses: Sequence[Sequence[int]]) -> None:
    """Raise SamplerError unless every response is complete for this scorer
    (`sampler.invalid_response_reason`). Malformed responses are rejected, never repaired."""
    for index, response in enumerate(responses):
        reason = sampler.invalid_response_reason(
            response,
            stop_token_ids=scorer.stop_token_ids,
            max_response_tokens=scorer.max_response_tokens,
        )
        if reason is not None:
            raise sampler.SamplerError(f"response {index}: {reason}")


def _as_token_ids(response: Sequence[int], index: int) -> list[int]:
    """A drawn response as a list of Python ints (samplers may return numpy or torch integers,
    which a result record could not serialize). Raises SamplerError for a token id that is not an
    integer."""
    try:
        return [operator.index(token) for token in response]
    except TypeError as error:
        raise sampler.SamplerError(f"response {index}: token ids must be integers") from error


def _checked_count(values: Sequence[float], expected: int, what: str) -> list[float]:
    if len(values) != expected:
        raise sampler.SamplerError(f"expected {expected} log {what} values, got {len(values)}")
    return [float(v) for v in values]


def _checked_logq(values: Sequence[float], expected: int, *, own_draws: bool) -> list[float]:
    """The sampler's log q values, checked: one per response, each a log probability (<= 0, not
    NaN), and above -inf on the sampler's own draws. Raises SamplerError otherwise."""
    checked = _checked_count(values, expected, "q")
    for value in checked:
        if math.isnan(value) or value > 0 or (own_draws and value == -math.inf):
            where = "its own draws" if own_draws else "the references"
            raise sampler.SamplerError(
                f"the sampler's log q on {where} must be a log probability, got {value}"
            )
    return checked


def _totals(spend: object) -> LedgerTotals:
    """The totals of what a caller passed as `ledger`. Raises TypeError for anything else."""
    if isinstance(spend, Ledger):
        return spend.totals()
    if isinstance(spend, LedgerTotals):
        return spend
    raise TypeError(
        "ledger must be a Ledger or LedgerTotals (Ledger() for a method that spent nothing)"
    )


def _check_task(task: data.Task, evaluation_data: data.EvaluationData) -> None:
    task.require_scored()
    if evaluation_data.task_id != task.task_id:
        raise ValueError(
            f"evaluation data of {evaluation_data.task_id} passed with task {task.task_id}"
        )


def _reference_logp_values(
    choices: Sequence[results.ReferenceLogp],
    evaluation_data: data.EvaluationData,
    scorer: Scorer,
    prompt: Sequence[int],
    references: Sequence[Sequence[int]],
) -> dict[results.ReferenceLogp, list[float]]:
    """log p of the references under each requested choice; the scorer is called at most once."""
    values: dict[results.ReferenceLogp, list[float]] = {}
    for choice in choices:
        if choice == "shipped":
            values[choice] = list(evaluation_data.reference_logp)
        elif choice not in values:
            values[choice] = _checked_count(
                scorer.logprobs(prompt, references), len(references), "p"
            )
    return values


def _evaluate(
    task: data.Task,
    evaluation_data: data.EvaluationData,
    candidate: sampler.Sampler,
    scorer: Scorer,
    choices: Sequence[results.ReferenceLogp],
    options: results.EvaluationOptions,
    labeler: Labeler | None = None,
) -> dict[results.ReferenceLogp, results.EvaluationResult]:
    """The protocol, once, with one result per requested choice of the reference log p (one set
    of draws, hit scores and reference log q)."""
    for choice in choices:
        if choice not in results.REFERENCE_LOGP_CHOICES:
            raise ValueError(
                f"reference_logp must be one of {results.REFERENCE_LOGP_CHOICES}, got {choice!r}"
            )
    results.check_options(options)
    _check_task(task, evaluation_data)
    settings = options.get("settings", results.EvaluationSettings())
    keep_details = options.get("keep_details", False)
    seed = options["seed"]
    sampler.validate_inferences_per_response(candidate.inferences_per_response)
    sampler_seed = sampler.evaluation_stream_seed(seed)
    totals = _totals(options["ledger"])
    run = scoring.RunInfo(
        task_id=task.task_id,
        kind="sampler",
        sampler=candidate.name,
        seed=seed,
        replicate=options.get("replicate", seed),
        construction_inferences=totals.inferences,
        # a numpy scalar would not serialize in the record
        inferences_per_response=float(candidate.inferences_per_response),
        budget_passes=totals.budget,
        ledger=totals.to_dict(),
        method=options.get("method"),
        setting=options.get("setting"),
        scorer=scorer.describe(),
    )
    prompt = task.prompt_token_ids
    log_p_event = task.log_event_probability
    references = [r.token_ids for r in evaluation_data.references]
    labels_of: Labeler = (
        (lambda responses: label(task, scorer, responses)) if labeler is None else labeler
    )

    declared = candidate.finite_support()
    if declared is not None:
        support = sampler.merge_support(declared)
        support_responses = [response for response, _ in support]
        validate_responses(scorer, support_responses)
        if all(labels_of(support_responses)):
            log_q = {response: math.log(mass) for response, mass in support}
            support_logp = _checked_count(
                scorer.logprobs(prompt, support_responses), len(support), "p"
            )
            reference_logp = _reference_logp_values(
                choices, evaluation_data, scorer, prompt, references
            )
            return {
                choice: scoring.score_finite_support(
                    dataclasses.replace(run, reference_logp=choice),
                    log_p_event=log_p_event,
                    support_mass=[mass for _, mass in support],
                    support_logq=[log_q[r] for r in support_responses],
                    support_logp=support_logp,
                    reference_logp=reference_logp[choice],
                    reference_logq=[log_q.get(r, -math.inf) for r in references],
                    settings=settings,
                    details={} if keep_details else None,
                )
                for choice in choices
            }
        # A support with a miss is not used: the sampler is scored from its draws below.

    # Step 1: new draws and their labels.
    run = dataclasses.replace(run, sampler_seed=sampler_seed)
    drawn = candidate.sample(settings.draws, seed=sampler_seed)
    if len(drawn) != settings.draws:
        raise sampler.SamplerError(f"asked for {settings.draws} responses, got {len(drawn)}")
    responses = [_as_token_ids(response, index) for index, response in enumerate(drawn)]
    validate_responses(scorer, responses)
    labels = labels_of(responses)
    hits = sum(labels)
    details: dict[str, Any] | None = None
    if keep_details:
        details = {
            "draw_token_ids": [list(r) for r in responses],
            "draw_hits": [bool(x) for x in labels],
        }

    # Step 2: the threshold test. A failing sampler needs no likelihoods.
    hit_logq: list[float] = []
    hit_logp: list[float] = []
    reference_logq: list[float] = []
    reference_logp = {choice: list[float]() for choice in choices}
    if estimators.meets_hit_rate_threshold(hits, settings.draws, settings.threshold):
        # Step 3: likelihoods of the references and of a random subset of the hits.
        reference_logq = _checked_logq(
            candidate.logprobs(references), len(references), own_draws=False
        )
        hit_indices = [i for i, is_hit in enumerate(labels) if is_hit]
        scored = estimators.choose_reverse_hits(
            hit_indices, settings.reverse_hit_limit, seed + _REVERSE_SEED_OFFSET
        )
        scored_responses = [responses[i] for i in scored]
        hit_logq = _checked_logq(candidate.logprobs(scored_responses), len(scored), own_draws=True)
        hit_logp = _checked_count(scorer.logprobs(prompt, scored_responses), len(scored), "p")
        reference_logp = _reference_logp_values(
            choices, evaluation_data, scorer, prompt, references
        )
        if details is not None:
            details["scored_hit_indices"] = scored

    # Steps 2 and 4: the fallback's scores, or the two estimates.
    return {
        choice: scoring.score_draws(
            dataclasses.replace(run, reference_logp=choice),
            log_p_event=log_p_event,
            draws=settings.draws,
            hits=hits,
            hit_logq=hit_logq,
            hit_logp=hit_logp,
            reference_logp=reference_logp[choice],
            reference_logq=reference_logq,
            fallback_logp=evaluation_data.fallback_logp,
            settings=settings,
            details=None if details is None else dict(details),
        )
        for choice in choices
    }


def evaluate(
    task: data.Task,
    evaluation_data: data.EvaluationData,
    candidate: sampler.Sampler,
    scorer: Scorer,
    *,
    reference_logp: results.ReferenceLogp = "shipped",
    labeler: Labeler | None = None,
    **options: Unpack[results.EvaluationOptions],
) -> results.EvaluationResult:
    """Evaluate one sampler on one task with the four steps of the protocol.

    `reference_logp` picks the log p of the references ("shipped" or "recomputed"), and
    `labeler`, if given, labels the drawn responses in place of the task's rule; the other
    keyword arguments are those of `results.EvaluationOptions` (`ledger` and `seed` are
    required; `replicate` defaults to `seed`). With `keep_details` the result's `details` hold
    the draws (`draw_token_ids`, `draw_hits`), `scored_hit_indices`, `hit_logp`, `hit_logq`,
    `reference_logp` (the values used), `reference_logq`, `forward_terms`, `log_q_event` and
    `log_p_fallback`.

    Raises SamplerError if the sampler breaks the benchmark's rules (inference limit, malformed
    responses, support or log q), DataError if the task is not a scored pair, TypeError if a
    required keyword argument is missing, a keyword argument is not an option or `ledger` is not
    a ledger, and ValueError for a seed outside [0, 2**31) or evaluation data of another task."""
    return _evaluate(task, evaluation_data, candidate, scorer, (reference_logp,), options, labeler)[
        reference_logp
    ]


def evaluate_both(
    task: data.Task,
    evaluation_data: data.EvaluationData,
    candidate: sampler.Sampler,
    scorer: Scorer,
    *,
    labeler: Labeler | None = None,
    **options: Unpack[results.EvaluationOptions],
) -> dict[results.ReferenceLogp, results.EvaluationResult]:
    """`evaluate` with the shipped and with the recomputed log p of the references, from one set
    of draws: {"shipped": ..., "recomputed": ...}, each equal to the `evaluate` call with that
    choice and the same seed. The references are scored once more than in a single evaluation
    (one scorer pass per reference). Takes `labeler` and the keyword arguments of
    `results.EvaluationOptions`, and raises what `evaluate` raises."""
    return _evaluate(
        task, evaluation_data, candidate, scorer, results.REFERENCE_LOGP_CHOICES, options, labeler
    )


def no_sampler_result(
    task: data.Task,
    evaluation_data: data.EvaluationData,
    *,
    budget: float | None,
    ledger: Ledger | LedgerTotals | None,
    seed: int,
    replicate: int,
    settings: results.EvaluationSettings = results.EvaluationSettings(),
    method: str | None = None,
    setting: str | None = None,
) -> results.EvaluationResult:
    """The result of a run that produced no sampler: the fallback's scores (no model call).

    With `budget` = B, a run whose budget was too small to produce any sampler is scored as the
    fallback at a cost equal to its budget: `construction_inferences` = B and `cost_rule`
    "fallback_at_budget". With `budget` = None, a method without a budget that ended with no
    sampler is scored as the fallback at what it spent: `construction_inferences` is the ledger's
    inference count and `cost_rule` "ledger". Either way the result has `kind` "no_sampler", no
    draws, a hit rate of 0, and `ledger` records the spend (if a ledger is given).

    Raises ValueError for a negative or non-finite budget, for `budget` = None without a ledger,
    and for a seed outside [0, 2**31); DataError if the task is not a scored pair."""
    _check_task(task, evaluation_data)
    sampler.evaluation_stream_seed(seed)  # the same seed range as every evaluation
    totals = None if ledger is None else _totals(ledger)
    cost_rule: results.CostRule
    if budget is not None:
        if not (math.isfinite(budget) and budget >= 0):
            raise ValueError(f"the budget must be finite and non-negative, got {budget}")
        construction_inferences, cost_rule = float(budget), "fallback_at_budget"
    else:
        if totals is None:
            raise ValueError("a no-sampler result without a budget is charged its ledger's spend")
        construction_inferences, cost_rule = totals.inferences, "ledger"
    run = scoring.RunInfo(
        task_id=task.task_id,
        kind="no_sampler",
        sampler=None,
        seed=seed,
        replicate=replicate,
        construction_inferences=construction_inferences,
        inferences_per_response=0.0,
        budget_passes=None if totals is None else totals.budget,
        cost_rule=cost_rule,
        ledger=None if totals is None else totals.to_dict(),
        method=method,
        setting=setting,
    )
    return scoring.score_fallback(
        run,
        log_p_event=task.log_event_probability,
        fallback_logp=evaluation_data.fallback_logp,
        settings=settings,
    )


def fallback_result(
    task: data.Task,
    evaluation_data: data.EvaluationData,
    *,
    settings: results.EvaluationSettings = results.EvaluationSettings(),
    seed: int = 0,
    replicate: int = 0,
    method: str | None = None,
    setting: str | None = None,
) -> results.EvaluationResult:
    """The fallback reported as a sampler of its own (`kind` "fallback"): the scores a failed
    sampler receives on this task. It runs no model, so it costs nothing to build and nothing per
    response, and its scores come from the stored log p(a). Raises DataError if the task is not a
    scored pair and ValueError for a seed outside [0, 2**31)."""
    _check_task(task, evaluation_data)
    sampler.evaluation_stream_seed(seed)
    totals = LedgerTotals.zero()
    run = scoring.RunInfo(
        task_id=task.task_id,
        kind="fallback",
        sampler="fallback",
        seed=seed,
        replicate=replicate,
        construction_inferences=0.0,
        inferences_per_response=0.0,
        budget_passes=0.0,
        ledger=totals.to_dict(),
        method=method,
        setting=setting,
    )
    return scoring.score_fallback(
        run,
        log_p_event=task.log_event_probability,
        fallback_logp=evaluation_data.fallback_logp,
        settings=settings,
        native_hit_rate=1.0,
        passed_threshold=True,
    )
