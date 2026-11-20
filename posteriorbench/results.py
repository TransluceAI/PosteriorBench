"""The settings of an evaluation and the result it returns.

`EvaluationSettings` holds the benchmark's settings. `EvaluationResult` is the record of one
evaluation: the scores, the construction cost, and what is needed to interpret and reproduce them
(seeds, the scorer's runtime, the data source, the package version). `EvaluationOptions` lists the
keyword arguments that `Benchmark.evaluate` and `Benchmark.evaluate_both` share.
"""

from __future__ import annotations

import dataclasses
import fractions
import math
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

from posteriorbench import _version, estimators
from posteriorbench.ledger import Ledger, LedgerTotals

EVALUATION_DRAWS = 1024
REVERSE_HIT_LIMIT = 128

# Where the log p of the references comes from: "shipped" uses the values stored in the evaluation
# file, "recomputed" scores the references with the evaluation's own scorer.
ReferenceLogp = Literal["shipped", "recomputed"]
REFERENCE_LOGP_CHOICES: tuple[ReferenceLogp, ...] = ("shipped", "recomputed")
# What was evaluated: a submitted sampler, a run that produced no sampler, or the fallback itself.
ResultKind = Literal["sampler", "no_sampler", "fallback"]
# How the construction cost was set: from the ledger, or at the budget of a run that produced no
# sampler.
CostRule = Literal["ledger", "fallback_at_budget"]


@dataclasses.dataclass(frozen=True)
class EvaluationSettings:
    """Benchmark settings; the defaults are the benchmark's.

    `draws` responses are drawn from the sampler; it fails if fewer than `threshold` of them are
    hits; the reverse KL averages over at most `reverse_hit_limit` of its hits; `cap` is the cap C
    of the forward KL. Raises ValueError on construction unless `draws` > 0,
    `reverse_hit_limit` > 0, `cap` is finite and > 0 and `threshold` lies in (0, 1]."""

    draws: int = EVALUATION_DRAWS
    reverse_hit_limit: int = REVERSE_HIT_LIMIT
    cap: float = estimators.CAP
    threshold: fractions.Fraction = estimators.HIT_RATE_THRESHOLD

    def __post_init__(self) -> None:
        if self.draws <= 0:
            raise ValueError(f"draws must be positive, got {self.draws}")
        if self.reverse_hit_limit <= 0:
            raise ValueError(f"reverse_hit_limit must be positive, got {self.reverse_hit_limit}")
        if not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError(f"the cap must be finite and positive, got {self.cap}")
        if not 0 < self.threshold <= 1:
            raise ValueError(f"the hit-rate threshold must lie in (0, 1], got {self.threshold}")


class EvaluationOptions(TypedDict):
    """The keyword arguments shared by `evaluate` and `evaluate_both`.

    `ledger` is what the method spent to build the sampler (a `Ledger` or its totals; an empty
    `Ledger()` for a method that spent nothing). `seed` (0 <= seed < 2**31) seeds the evaluation
    draws and the choice of scored hits; a sampler draws the same responses again only on the same
    device, library versions and batch size. `replicate` is the replicate index recorded in the
    result; it defaults to `seed`, so `seed=r` evaluates replicate r. `settings` defaults to the
    benchmark's. `method` and `setting` are labels recorded in the result, by
    which `aggregate` groups results. With `keep_details` the result also holds the draws and the
    per-response values the scores were computed from."""

    ledger: Ledger | LedgerTotals
    seed: int
    replicate: NotRequired[int]
    settings: NotRequired[EvaluationSettings]
    method: NotRequired[str | None]
    setting: NotRequired[str | None]
    keep_details: NotRequired[bool]


REQUIRED_OPTIONS: tuple[str, ...] = ("ledger", "seed")
OPTIONAL_OPTIONS: tuple[str, ...] = ("replicate", "settings", "method", "setting", "keep_details")


def check_options(options: Mapping[str, object]) -> None:
    """Raise TypeError if `options` holds a keyword argument that `EvaluationOptions` does not
    have, or lacks a required one."""
    known = REQUIRED_OPTIONS + OPTIONAL_OPTIONS
    unknown = sorted(set(options) - set(known))
    if unknown:
        raise TypeError(f"unexpected keyword argument(s) {unknown}; the options are {list(known)}")
    missing = [name for name in REQUIRED_OPTIONS if name not in options]
    if missing:
        raise TypeError(f"missing keyword argument(s) {missing}")


@dataclasses.dataclass
class EvaluationResult:
    """The scores of one sampler on one task, with its cost.

    Scores. `native_hit_rate` is the hit fraction of the evaluation draws and `passed_threshold`
    whether it reached the threshold. `capped_forward_kl` and `reverse_kl` are the sampler's own
    scores if it passed, else the fallback's (`fallback` is then true and `fallback_rho` holds
    p(a) / p̂(S)). Lower is better for both. `exact` is true for a sampler scored exactly over an
    all-hit finite support. `references`, `references_above_cap`, `references_with_zero_q` and
    `reverse_hits_used` count the responses behind the two estimates.

    Cost. `construction_inferences` is the benchmark's cost axis: the ledger's inference count
    (`cost_rule` "ledger"), or the budget of a run that produced no sampler (`cost_rule`
    "fallback_at_budget"). `budget_passes` is the ledger's budget count, `ledger` its totals
    (`LedgerTotals.to_dict`), and `inferences_per_response` what the sampler spends per response.

    What was evaluated. `kind` is "sampler" for a submitted sampler, "no_sampler" for a run that
    produced none, and "fallback" for the fallback reported on its own. `sampler` is the sampler's
    name (None for "no_sampler"); `method` and `setting` are the caller's labels. `reference_logp`
    says which log p of the references the forward KL used, "shipped" or "recomputed"; it is None
    when no sampler was evaluated. `seed` is the evaluation seed and `sampler_seed` the seed the
    draws were made with (None when no draws were made). `scorer` is the scorer's runtime
    (`Scorer.describe`), `dataset` the data source (source, revision, commit) when the result came
    from a `Benchmark`, `package_version` the version that computed it. `details` holds the draws
    and per-response values if they were kept."""

    task_id: str
    method: str | None
    setting: str | None
    kind: ResultKind
    sampler: str | None
    seed: int
    replicate: int
    draws: int
    hits: int | None
    native_hit_rate: float
    passed_threshold: bool
    fallback: bool
    capped_forward_kl: float
    reverse_kl: float
    cap: float
    log_p_event: float
    construction_inferences: float
    budget_passes: float | None
    inferences_per_response: float
    cost_rule: CostRule
    ledger: dict[str, Any] | None
    reference_logp: ReferenceLogp | None
    exact: bool = False
    references: int = 0
    references_above_cap: int = 0
    references_with_zero_q: int = 0
    reverse_hits_used: int = 0
    fallback_rho: float | None = None
    scorer: dict[str, Any] | None = None
    sampler_seed: int | None = None
    package_version: str = _version.__version__
    dataset: dict[str, Any] | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable copy (floats may be ±inf in `details`; write with allow_nan)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationResult:
        """The inverse of `to_dict`. Keys that are not fields of a result are ignored (the
        command-line tool adds a "run" block). Raises ValueError if a field without a default is
        missing."""
        fields = {f.name: f for f in dataclasses.fields(cls)}
        missing = [
            name
            for name, field in fields.items()
            if name not in value and field.default is dataclasses.MISSING
        ]
        if missing:
            raise ValueError(
                f"not a result record of posteriorbench {_version.__version__}: no {missing}"
            )
        return cls(**{name: value[name] for name in fields if name in value})
