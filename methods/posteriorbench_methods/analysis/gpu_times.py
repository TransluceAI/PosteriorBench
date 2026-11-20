"""The GPU time of a run, priced from measured times of its operations.

A times file holds sessions of measurements. A measurement is the time of one call of an
operation by one model (`trains`: "nothing" is the original model) on `batch` responses of the
benchmark's length:

    generation                   generates `batch` responses
    forward                      a teacher-forced pass, without gradients
    training_forward             the forward pass of a gradient step
    training_forward_backward    that pass and its backward pass
    optimizer_step               one step of the optimizer (`batch` is 0)
    update                       one whole update: forward, backward, step

`run_seconds` prices each operation at the batch size at which the method runs it: the original
model's draws at 500 and counted log-probabilities at 32 a pass, a sampler's draws as in
`SAMPLER`; SFT and CTL by whole updates; the other methods by training_forward per counted
forward pass, training_forward_backward minus training_forward per response of a gradient step,
and optimizer_step per update. One inference is priced at `Session.unit`. A method is priced from
one session, the last that has every measurement it needs.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from posteriorbench_methods.analysis import flops, operations
from posteriorbench_methods.common import errors, run_record
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import twisted_model
from posteriorbench_methods.dpg import samplers as dpg_samplers
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.sft import samplers as sft_samplers
from posteriorbench_methods.sft import settings as sft_settings

Operation = Literal[
    "generation",
    "forward",
    "training_forward",
    "training_forward_backward",
    "optimizer_step",
    "update",
]
Key = tuple[Operation, flops.Trains, int]  # operation, the model that runs it, batch
PAPER_TIMES = pathlib.Path(__file__).resolve().parent / "data" / "h100_times.json"

P_DRAW_BATCH = 500  # responses that the original model generates in one forward pass
SCORING_BATCH = 32  # responses in one pass that computes a counted log-probability
# The sampler of each method that trains: the model that is timed for its draws, and the number
# of responses that it generates in one forward pass when it draws for training. A sampler whose
# weights alone differ from the original model's is timed as the original model.
SAMPLER: dict[flops.Trains, tuple[flops.Trains, int]] = {
    "last_layers": ("nothing", sft_samplers.GENERATION_PART),
    "adapters": ("adapters", gfn_settings.Settings().generation_batch_size),
    "twist": ("twist", twisted_model.GENERATION_BATCH),
    "all_weights": ("nothing", dpg_samplers.GENERATION_PART),
}
# The responses in one forward pass of a gradient step.
TRAINING_BATCH: dict[flops.Trains, int] = {
    "adapters": gfn_settings.Settings().forward_batch_size,
    "all_weights": dpg_samplers.SCORING_BATCH,
}
# The batch sizes of an update of SFT: a run with k hits trains in batches of min(k, 32).
SFT_UPDATE_BATCHES = tuple(range(1, sft_settings.Settings().batch_size + 1))
TWIST_UPDATE_RESPONSES = ctl_settings.Settings().responses_per_update

UNIT: Key = ("generation", "nothing", P_DRAW_BATCH)
SCORING: Key = ("forward", "nothing", SCORING_BATCH)


class TimesError(errors.MethodsError, ValueError):
    """A times file that cannot be read, or that lacks a measurement that a method needs."""


def needed(kind: flops.Trains) -> list[Key]:
    """The measurements that pricing a run of a method that trains `kind` reads."""
    keys: list[Key] = [UNIT, SCORING]
    if kind == "nothing":
        return keys
    model, batch = SAMPLER[kind]
    keys.append(("generation", model, batch))
    if kind == "last_layers":
        return keys + [("update", kind, batch) for batch in SFT_UPDATE_BATCHES]
    if kind == "twist":
        return keys + [("update", kind, TWIST_UPDATE_RESPONSES)]
    return keys + [
        ("training_forward", kind, TRAINING_BATCH[kind]),
        ("training_forward_backward", kind, TRAINING_BATCH[kind]),
        ("optimizer_step", kind, 0),
    ]


def every_key() -> list[Key]:
    """Every measurement that some method needs, each once, in a fixed order."""
    kinds: tuple[flops.Trains, ...] = ("nothing", "all_weights", "last_layers", "adapters", "twist")
    return list(dict.fromkeys(key for kind in kinds for key in needed(kind)))


@dataclasses.dataclass(frozen=True)
class Measurement:
    """One timed call. `seconds` is the median over `repetitions` calls, and `batch` the number
    of responses of a call (0 for an optimizer step)."""

    operation: Operation
    trains: flops.Trains
    batch: int
    seconds: float
    repetitions: int

    @property
    def key(self) -> Key:
        return (self.operation, self.trains, self.batch)


@dataclasses.dataclass(frozen=True)
class Session:
    """Measurements made in one go, and the conditions under which they were made (the GPU, the
    library versions, the lengths of the prompt and of the responses)."""

    conditions: Mapping[str, Any]
    measurements: tuple[Measurement, ...]

    def seconds(self, key: Key) -> float:
        """The time of one call of `key`. Raises TimesError if the session did not measure it."""
        for measurement in self.measurements:
            if measurement.key == key:
                return measurement.seconds
        raise TimesError(f"no measurement of {key[0]} by {key[1]!r} at batch {key[2]}")

    def per_response(self, key: Key) -> float:
        """The time of `key` for each response of its batch."""
        return self.seconds(key) / key[2]

    def has(self, keys: Sequence[Key]) -> bool:
        measured = {measurement.key for measurement in self.measurements}
        return all(key in measured for key in keys)

    @property
    def unit(self) -> float:
        """What one inference is priced at: one response generated by the original model, 500
        at a time."""
        return self.per_response(UNIT)

    def to_json(self) -> dict[str, Any]:
        return {
            "conditions": dict(self.conditions),
            "measurements": [dataclasses.asdict(measurement) for measurement in self.measurements],
        }


@dataclasses.dataclass(frozen=True)
class Times:
    """The sessions of a times file, in its order."""

    sessions: tuple[Session, ...]

    def session_for(self, kind: flops.Trains) -> Session:
        """The session that prices a method that trains `kind`: the last one that has every
        measurement the method needs. Raises TimesError if none has."""
        for session in reversed(self.sessions):
            if session.has(needed(kind)):
                return session
        raise TimesError(
            f"no session of the times file has every measurement for a method that trains {kind!r}"
        )


def write(path: pathlib.Path, sessions: Sequence[Session]) -> None:
    """Write a times file."""
    run_record.write_json(path, {"sessions": [session.to_json() for session in sessions]})


def read(path: pathlib.Path = PAPER_TIMES) -> Times:
    """The times of a file; by default the paper's measurements. Raises TimesError for a file
    that is missing or is not a times file."""
    if not path.exists():
        raise TimesError(f"{path} does not exist")
    try:
        sessions = tuple(
            Session(
                conditions=dict(session["conditions"]),
                measurements=tuple(Measurement(**entry) for entry in session["measurements"]),
            )
            for session in run_record.read_json(path)["sessions"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TimesError(f"{path} is not a times file: {error!r}") from error
    return Times(sessions)


def run_seconds(kind: flops.Trains, done: operations.Operations, session: Session) -> float:
    """The GPU seconds of a run of a method that trains `kind` (module docstring). Raises
    TimesError if `session` lacks a measurement that the run needs."""
    seconds = done.draws_from_p * session.unit + done.scored * session.per_response(SCORING)
    if kind == "nothing":
        return seconds
    model, batch = SAMPLER[kind]
    seconds += done.sampler_draws * session.per_response(("generation", model, batch))
    if kind == "last_layers":
        if done.update_batch is None:  # a run that did not train
            return seconds
        full, rest = divmod(round(done.gradient_responses), done.update_batch)
        seconds += full * session.seconds(("update", kind, done.update_batch))
        return seconds + (session.seconds(("update", kind, rest)) if rest else 0.0)
    if kind == "twist":
        update = session.per_response(("update", kind, TWIST_UPDATE_RESPONSES))
        return seconds + done.gradient_responses * update
    forward = session.per_response(("training_forward", kind, TRAINING_BATCH[kind]))
    both = session.per_response(("training_forward_backward", kind, TRAINING_BATCH[kind]))
    return (
        seconds
        + done.training_forwards * forward
        + done.gradient_responses * (both - forward)
        + done.updates * session.seconds(("optimizer_step", kind, 0))
    )
