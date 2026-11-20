"""Prompt selection (`select`, `select_by_ce`) and prompt mixtures (`mix`) in a pool.

A pool is the first K candidates of the replicate's order; each candidate has m selection draws.
Methods that test the same candidate read the same threshold-test draws, and each pays for the
draws its own tests read.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from posteriorbench import ledger

from posteriorbench_methods.common import charges, rank_table, run_context, run_record, submission
from posteriorbench_methods.prompt import (
    ce,
    cost,
    draws,
    methods,
    mixture,
    pool,
    samplers,
    selection_signals,
    shared,
)

# The candidates of a pool are tested in rank order until one passes or the ranking ends. Ties in
# the ranking go to the candidate that comes first in the pool.
RULE = submission.Rule(max_tests=None, ties="order")


def run_rule(session: shared.Session) -> submission.Rule:
    """The submission rule of a run: `RULE` at the hit-rate threshold of the run."""
    return session.context.rule_at_threshold(RULE)


@dataclasses.dataclass(frozen=True)
class Member:
    """A candidate in a pool, with its sampler and selection draws."""

    candidate: pool.Candidate
    spec: samplers.Spec
    selection: draws.LabeledDraws
    statistics: selection_signals.Statistics

    @property
    def candidate_id(self) -> str:
        """The candidate's id in a ranking: its index in the pool file."""
        return str(self.candidate.index)

    def to_json(self) -> dict[str, Any]:
        """`candidate_id` (its ranking id), `pool_candidate_id` (its id in the pool file) and
        `system_prompt`."""
        return {
            "candidate_id": self.candidate_id,
            "pool_candidate_id": self.candidate.candidate_id,
            "system_prompt": self.candidate.system_prompt,
        }


def ordered_candidates(context: run_context.RunContext, largest: int) -> list[pool.Candidate]:
    """The first `largest` candidates in the replicate's order; a pool of size K is the first K.
    Raises pool.PoolError if the task has fewer."""
    path = pool.pool_file(context.options.method_data)
    candidates = pool.load_candidates(path, context.task.task_id)
    if largest > len(candidates):
        raise pool.PoolError(
            f"a pool of {largest} candidates was asked for, and {path} has {len(candidates)} "
            f"candidates for {context.task.task_id}"
        )

    def seed_of_batch(batch: int) -> int:
        # Batch 0 has the seed of a file without batches: where its candidates are the first of
        # the file, their order is that of a file which holds only them.
        index = () if batch == 0 else (batch,)
        return context.stream_seed(draws.OWNER, "pool-order", *index)

    order = pool.batched_order([candidate.batch for candidate in candidates], seed_of_batch)
    return [candidates[index] for index in order[:largest]]


def draw_selection(session: shared.Session, candidates: Sequence[pool.Candidate]) -> list[Member]:
    """m selection draws of each candidate, with q and p on the hits."""
    context = session.context
    members: list[Member] = []
    for candidate in candidates:
        prompt = context.original.render_prompt(context.task.prompt, system=candidate.system_prompt)
        spec = samplers.Spec((tuple(prompt),))
        drawn = session.draws.selection(candidate, spec, session.settings.selection_draws)
        members.append(Member(candidate, spec, drawn, selection_signals.statistics(drawn.values)))
    if members:
        context.tracker.event("selection_draws", new_candidates=len(members))
    return members


def rank(
    members: Sequence[Member], scores: Sequence[float | None], rule: submission.Rule = RULE
) -> list[submission.RankedCandidate]:
    """Rank the pool under `rule` by `scores` (None where the signal gives none). A member's
    order is its pool position; members with the same `sampler_id` are ranked once."""
    candidates = [
        submission.Candidate(
            candidate_id=member.candidate_id,
            order=position,
            sampler_id=member.spec.sampler_id,
            selection_draws=member.statistics.draws,
            selection_hits=member.statistics.hits,
            score=score,
        )
        for position, (member, score) in enumerate(zip(members, scores, strict=True))
    ]
    return submission.rank_candidates(candidates, rule)


def run_tests(
    session: shared.Session,
    members: Sequence[Member],
    ranking: Sequence[submission.RankedCandidate],
    rule: submission.Rule,
) -> tuple[submission.Submission, list[dict[str, Any]]]:
    """Run the threshold tests of `ranking`; return the submission and the rows the tests read,
    each labeled with its test's rank and candidate."""
    spec = {member.candidate_id: member.spec for member in members}

    def test(entry: submission.RankedCandidate, threshold_test: rank_table.ThresholdTest) -> int:
        rows = session.draws.threshold_test(spec[entry.candidate_id], threshold_test.draws)
        return draws.count_hits(rows)

    result = submission.submit(ranking, test, rule)
    rows = [
        {"rank": outcome.rank, "candidate_id": outcome.candidate_id, **row}
        for outcome in result.tests
        for row in session.draws.threshold_test(spec[outcome.candidate_id], outcome.draws)
    ]
    return result, rows


def selection_rows(members: Sequence[Member]) -> list[dict[str, Any]]:
    """The rows of the selection draws of a pool."""
    return [
        row for member in members for row in member.selection.rows(candidate_id=member.candidate_id)
    ]


def pool_hits(members: Sequence[Member]) -> int:
    """The hits among the selection draws of a pool."""
    return sum(member.statistics.hits for member in members)


def select(
    session: shared.Session, members: Sequence[Member], method: methods.Selection
) -> shared.Returned:
    """Prompt selection: score by the method's signal, test in rank order, submit the first
    that passes or else the best-observed one."""
    if method.signal == "ce":
        return select_by_ce(session, members)
    scores = selection_signals.scores(
        method.signal,
        [member.statistics for member in members],
        signal_hit_fraction=run_rule(session).signal_hit_fraction,
    )
    reads_p = method.signal in selection_signals.READ_P_ON_HITS
    spend = ledger.Ledger()
    charges.apply(
        spend,
        cost.selection_draws(
            len(members),
            session.settings.selection_draws,
            p_on_hits=pool_hits(members) if reads_p else 0,
        ),
    )
    return rank_and_submit(session, members, method, scores, spend)


def select_by_ce(session: shared.Session, members: Sequence[Member]) -> shared.Returned:
    """Prompt selection with the CE method (`ce`): candidates above the rule's signal hit
    fraction are scored on a bank of the pool's selection draws. If there are candidates to score
    and no hit in the bank, the pool cannot be ranked and the record has no sampler."""
    method = methods.Selection("ce")
    m, hit_fraction = session.settings.selection_draws, run_rule(session).signal_hit_fraction
    seed = session.context.stream_seed(draws.OWNER, "ce-bank", len(members))
    bank = ce.build_bank(
        ce.choose_bank(len(members), m, session.settings.ce_bank_size, seed),
        token_ids=[member.selection.token_ids for member in members],
        hit=[member.selection.values.hit.tolist() for member in members],
        logp=[member.selection.values.logp.tolist() for member in members],
        logq=[member.selection.values.logq.tolist() for member in members],
    )
    scored = [
        i for i, member in enumerate(members) if member.statistics.hit_fraction > hit_fraction
    ]
    pairs = bank.cross_scored_pairs(scored)
    spend = ledger.Ledger()
    charges.apply(spend, cost.selection_draws(len(members), m, p_on_hits=0))
    charges.apply(spend, cost.ce_bank(len(bank.hits), pairs))
    details: dict[str, Any] = {
        "bank": {
            "draws": [[members[candidate].candidate_id, draw] for candidate, draw in bank.draws],
            "hits": len(bank.hits),
            "distinct_hit_responses": len(bank.responses),
            "scored_candidates": [members[candidate].candidate_id for candidate in scored],
            "cross_scored_pairs": pairs,
        }
    }
    if scored and not bank.hits:
        setting = run_record.setting_label(pool_size=len(members))
        recorded = session.context.record_no_sampler(
            method=method.name,
            setting=setting,
            settings=session.settings,
            budget=None,
            spend=spend,
            details={"pool_size": len(members), "signal": method.signal, **details},
            draws={"selection": selection_rows(members)},
        )
        return shared.Returned(method.name, setting, None, "no_sampler", spend, recorded)
    scores: list[float | None] = [None] * len(members)
    if scored:
        original = session.context.original
        candidate_logq = {
            candidate: original.logprobs(members[candidate].spec.prompts[0], bank.responses)
            for candidate in scored
        }
        for candidate, score in zip(scored, ce.scores(bank, scored, candidate_logq), strict=True):
            scores[candidate] = score
    return rank_and_submit(session, members, method, scores, spend, details=details)


def rank_and_submit(
    session: shared.Session,
    members: Sequence[Member],
    method: methods.Selection,
    scores: Sequence[float | None],
    spend: ledger.Ledger,
    *,
    details: dict[str, Any] | None = None,
) -> shared.Returned:
    """Rank by `scores`, run the threshold tests and charge them to `spend`, then evaluate and
    record the submitted candidate. `details` are added to the record's details."""
    setting = run_record.setting_label(pool_size=len(members))
    rule = run_rule(session)
    ranking = rank(members, scores, rule)
    result, test_rows = run_tests(session, members, ranking, rule)
    charges.apply(spend, charges.threshold_test(result.test_draws, note="threshold tests"))
    assert result.submitted is not None, "a pool has at least one candidate to test"
    member_of = {member.candidate_id: member for member in members}
    submitted = member_of[result.submitted.candidate_id]
    recorded = session.context.evaluate_and_record(
        session.factory.build(submitted.spec, name=method.name),
        method=method.name,
        setting=setting,
        settings=session.settings,
        spend=spend,
        ranking=ranking,
        submission=result,
        submitted=submitted.to_json(),
        details={"pool_size": len(members), "signal": method.signal, **(details or {})},
        draws={"selection": selection_rows(members), "threshold_tests": test_rows},
    )
    return shared.Returned(method.name, setting, submitted.spec, result.status, spend, recorded)


def mix(
    session: shared.Session, members: Sequence[Member], method: methods.Mixture
) -> shared.Returned:
    """A prompt mixture: test in EB-JC rank order until `method.components` have passed, and mix
    those. If none passes, the best-observed candidate is submitted alone."""
    setting = run_record.setting_label(pool_size=len(members))
    statistics = [member.statistics for member in members]
    rule = dataclasses.replace(run_rule(session), passes_needed=method.components)
    scores = selection_signals.ebjc_scores(statistics, signal_hit_fraction=rule.signal_hit_fraction)
    ranking = rank(members, scores, rule)
    result, test_rows = run_tests(session, members, ranking, rule)
    spend = ledger.Ledger()
    charges.apply(
        spend,
        cost.selection_draws(
            len(members), session.settings.selection_draws, p_on_hits=pool_hits(members)
        ),
    )
    charges.apply(spend, charges.threshold_test(result.test_draws, note="threshold tests"))
    assert result.submitted is not None, "a pool has at least one candidate to test"
    member_of = {member.candidate_id: member for member in members}
    components = [member_of[test.candidate_id] for test in result.passes or (result.submitted,)]
    shrinkage = mixture.fit_shrinkage(
        [s for s in statistics if s.hit_fraction > rule.signal_hit_fraction]
    )
    weights = mixture.weights(method.weights, [c.statistics for c in components], shrinkage)
    spec = samplers.Spec(tuple(c.spec.prompts[0] for c in components), tuple(weights))
    recorded = session.context.evaluate_and_record(
        session.factory.build(spec, name=method.name),
        method=method.name,
        setting=setting,
        settings=session.settings,
        spend=spend,
        ranking=ranking,
        submission=result,
        submitted={
            "components": [
                {**component.to_json(), "weight": weight}
                for component, weight in zip(components, weights, strict=True)
            ]
        },
        details={
            "pool_size": len(members),
            "components": method.components,
            "weights": method.weights,
            "shrinkage": None if shrinkage is None else dataclasses.asdict(shrinkage),
        },
        draws={"selection": selection_rows(members), "threshold_tests": test_rows},
    )
    return shared.Returned(method.name, setting, spec, result.status, spend, recorded)
