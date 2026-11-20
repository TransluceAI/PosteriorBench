"""What a run did, by kind of operation, read from its record's ledger and details.

The compute of a run depends on more than its inference count: on which model generated a
response (the original model or the method's sampler) and on the backward passes of gradient
steps, which the count leaves out. Passes that the count leaves out because a method need not
make them (log q of a sampler's own draws, p where the method does not use it) are left out here
too. The README's section "What the compute of a run is" tabulates where a record has each number.
"""

from __future__ import annotations

import dataclasses
import math

from posteriorbench import ledger

from posteriorbench_methods.analysis import flops, records
from posteriorbench_methods.common import errors

Category = ledger.Category
COMPOSITION_SUFFIX = "+composition"
# What each method trains. The methods of the prompt family, which train nothing, are told by
# their names (`trains`).
TRAINS: dict[str, flops.Trains] = {
    "sft": "last_layers",
    "replay": "nothing",
    "gflownet": "adapters",
    "ctl": "twist",
    "kl-dpg": "all_weights",
    "js-dpg": "all_weights",
    "gdc": "all_weights",
}
TRAINING_FREE_PREFIXES = ("prompt-", "mixture-")


class NotCoveredError(errors.MethodsError, ValueError):
    """A record of a method whose compute this analysis has no rule for."""


def trains(method: str) -> flops.Trains | None:
    """What `method` trains; None for a method that the analysis does not cover: a composed
    method, whose responses run two models for part of their tokens, Steering, whose vectors are
    built from longer responses, and Instruction, which costs nothing."""
    if method in TRAINS:
        return TRAINS[method]
    if method.startswith(TRAINING_FREE_PREFIXES) and not method.endswith(COMPOSITION_SUFFIX):
        return "nothing"
    return None


@dataclasses.dataclass(frozen=True)
class Operations:
    """What one run did, in responses.

    scored: counted log-probabilities, p(y) or q(y) of a response.
    gradient_responses: responses with a backward pass, which the count leaves out; for CTL,
        those on which the twist network was updated.
    updates: optimizer steps. 0 for CTL, whose update is timed as a whole.
    update_batch: SFT only; the last update may have fewer responses."""

    draws_from_p: float = 0.0
    sampler_draws: float = 0.0
    scored: float = 0.0
    training_forwards: float = 0.0
    gradient_responses: float = 0.0
    updates: float = 0.0
    update_batch: int | None = None

    @property
    def counted(self) -> float:
        """The inferences that these operations count."""
        return math.fsum(
            [self.draws_from_p, self.sampler_draws, self.scored, self.training_forwards]
        )


def _selection_draws(record: records.Record) -> float:
    """The selection draws of the candidates that the run chose from."""
    return float(sum(int(row["selection_draws"]) for row in record.details["candidates"]))


def _last_update(record: records.Record) -> float:
    """The number of updates after which the last candidate was saved."""
    candidates = record.details["candidates"]
    return float(candidates[-1]["update"]) if candidates else 0.0


def of_record(record: records.Record) -> Operations:
    """The operations of the run that wrote `record`. They add up to the run's inference count,
    except for a run that produced no sampler and is costed at its budget.

    Raises NotCoveredError for a method without a rule (`trains`) and for a ledger with charges
    that are not passes of the model over a response."""
    kind = trains(record.method)
    if kind is None:
        raise NotCoveredError(f"no rule for the compute of {record.method!r}")
    totals = record.ledger_totals
    if totals is None:
        # A run whose budget was too small for a sampler. SFT and Replay had made their draws by
        # then, none of them a hit; the other methods had done nothing.
        division = record.details.get("division")
        return Operations(draws_from_p=0.0 if division is None else float(division["draws"]))

    def inferences(category: ledger.Category) -> float:
        return totals.by_category[category.value][0]

    if inferences(Category.CORPUS_HITS) or inferences(Category.SHARED):
        raise NotCoveredError(
            f"{record.directory}: the ledger has corpus hits or shared construction, which are "
            "not passes of the model over a response"
        )
    generated = inferences(Category.GENERATION) + inferences(Category.THRESHOLD_TEST)
    scored = inferences(Category.P_SCORING) + inferences(Category.Q_SCORING)
    forwards = inferences(Category.TRAINING_FORWARD)
    assert math.isclose(
        math.fsum([generated, scored, forwards]), totals.inferences
    ), "generation, log-probabilities and forward passes are all that a ledger counts"
    if kind == "nothing":
        return Operations(draws_from_p=generated, scored=scored)
    if kind == "last_layers":
        division = record.details["division"]
        draws = float(division["draws"])
        assert forwards == division["training_examples"], "SFT trains on what its division says"
        return Operations(
            draws_from_p=draws,
            sampler_draws=generated - draws,
            scored=scored,
            training_forwards=forwards,
            gradient_responses=forwards,
            updates=float(math.ceil(forwards / division["batch"])),
            update_batch=int(division["batch"]),
        )
    if kind == "adapters":
        # A fresh training response is generated and trained on; a replayed one is trained on.
        training_draws = inferences(Category.GENERATION) - _selection_draws(record)
        gradient_responses, updates = training_draws + forwards, _last_update(record)
    elif kind == "twist":
        gradient_responses = inferences(Category.GENERATION) - _selection_draws(record)
        updates = 0.0
    elif record.method == "gdc":
        # Every update of GDC takes a step of Adam, on its hits or on a zero gradient.
        gradient_responses, updates = forwards, _last_update(record)
    else:
        gradient_responses = forwards
        updates = forwards / int(record.settings["responses_per_update"])
    return Operations(
        sampler_draws=generated,
        scored=scored,
        training_forwards=forwards,
        gradient_responses=gradient_responses,
        updates=updates,
    )
