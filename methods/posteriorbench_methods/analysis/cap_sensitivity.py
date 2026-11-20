"""Sensitivity to the cap C of the forward KL: the scores of finished runs at other caps.

The capped forward KL of a sampler that passed the threshold is the mean over the references of
min(t, C), with t = log(p_S(y) / q_S(y)). A record keeps the terms t (`forward_terms` in
evaluation_details.json), so another C needs no model. A sampler scored as the fallback has
rho log rho + (1 - rho) C, with rho from its result. Points are averaged as `posteriorbench
aggregate` averages. The scores use the shipped log p of the references, as result.json does.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

from posteriorbench import aggregate
from posteriorbench import results as pb_results

from posteriorbench_methods.analysis import records

CAPS: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0, 200.0)
# A record's own score, computed again from its terms, agrees with the stored one to this.
TOLERANCE = 1e-9
SCORES = ("capped_forward_kl", "reverse_kl", "native_hit_rate", "construction_inferences")


def passed(record: records.Record) -> bool:
    """Whether the record's sampler passed the threshold, so that its scores are its own."""
    return not record.result["fallback"]


def forward_terms(record: records.Record) -> list[float]:
    """The terms of the references of a sampler that passed; +inf where q(y) = 0."""
    return [records.number(term) for term in record.evaluation_details["forward_terms"]]


def capped_forward_kl(record: records.Record, cap: float) -> float:
    """The capped forward KL of the record's run at `cap`. Raises records.RecordsError if the
    record's terms do not give its stored score at its own cap."""
    stored = records.number(record.result["capped_forward_kl"])
    stored_cap = records.number(record.result["cap"])
    if not passed(record):
        rho = records.number(record.result["fallback_rho"])
        return stored + (1 - rho) * (cap - stored_cap)
    terms = forward_terms(record)

    def score(limit: float) -> float:
        return math.fsum(min(term, limit) for term in terms) / len(terms)

    if abs(score(stored_cap) - stored) > TOLERANCE:
        raise records.RecordsError(
            f"{record.directory}: the stored terms give a capped forward KL of "
            f"{score(stored_cap)}, and the result has {stored}"
        )
    return score(cap)


def references_at_cap(record: records.Record, cap: float) -> tuple[int, int] | None:
    """(references with a term at or above `cap`, references) of a sampler that passed and was
    scored from its draws; None for any other record, such as Replay's exact score, whose terms
    are all infinite."""
    if not passed(record) or record.result["exact"]:
        return None
    terms = forward_terms(record)
    return sum(term >= cap for term in terms), len(terms)


def _result_at(record: records.Record, cap: float) -> pb_results.EvaluationResult:
    """The record's result with the capped forward KL at `cap`."""
    stored = {**record.result, **{name: records.number(record.result[name]) for name in SCORES}}
    result = pb_results.EvaluationResult.from_dict(stored)
    return dataclasses.replace(result, capped_forward_kl=capped_forward_kl(record, cap), cap=cap)


def _share(found: Sequence[records.Record], cap: float) -> dict[str, Any]:
    counts = [counted for counted in (references_at_cap(r, cap) for r in found) if counted]
    at_cap = sum(count for count, _ in counts)
    scored = sum(total for _, total in counts)
    return {
        "samplers": len(counts),
        "references": scored,
        "references_at_cap": at_cap,
        "share": at_cap / scored if scored else None,
    }


def analyze(
    found: Sequence[records.Record],
    caps: Sequence[float] = CAPS,
    *,
    require_equal_replicates: bool = True,
    with_runs: bool = False,
) -> dict[str, Any]:
    """The analysis of `found` at each of `caps`, as a JSON-ready dict:

    share_at_cap: for each cap, over all the records and for each method, the samplers that
        passed, their references, and how many of these are at or above the cap.
    points: for each cap and each point, the summary of `posteriorbench.aggregate.aggregate`.
    runs (with `with_runs`): for each record, its capped forward KL at each cap.

    Raises records.RecordsError if the records of a point cannot be averaged."""
    methods = list(dict.fromkeys(record.method for record in found))
    analysis: dict[str, Any] = {"caps": list(caps), "share_at_cap": [], "points": []}
    for cap in caps:
        analysis["share_at_cap"].append(
            {
                "cap": cap,
                **_share(found, cap),
                "by_method": {
                    method: _share([r for r in found if r.method == method], cap)
                    for method in methods
                },
            }
        )
        at_cap = [_result_at(record, cap) for record in found]
        for (method, setting), group in aggregate.group_results(at_cap).items():
            try:
                summary = aggregate.aggregate(
                    group, require_equal_replicates=require_equal_replicates
                )
            except ValueError as error:
                raise records.RecordsError(f"{method} at {setting}: {error}") from error
            analysis["points"].append(
                {"cap": cap, "method": method, "setting": setting, **summary.to_dict()}
            )
    if with_runs:
        analysis["runs"] = [
            {
                **record.label,
                "passed_threshold": passed(record),
                "capped_forward_kl": {str(cap): capped_forward_kl(record, cap) for cap in caps},
            }
            for record in found
        ]
    return analysis
