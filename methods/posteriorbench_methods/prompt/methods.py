"""The method names of the prompt family (README table): `prompt-<signal>`, `mixture-eb<s>`,
`mixture-uniform<s>` and `instruction`, each optionally followed by `+composition`."""

from __future__ import annotations

import dataclasses
import re
import typing
from typing import Literal

from posteriorbench_methods.common import errors

Signal = Literal["ebjc", "jc", "hitrate", "iwae", "ce"]
SIGNALS: tuple[Signal, ...] = typing.get_args(Signal)
Weights = Literal["eb", "uniform"]
COMPOSITION = "+composition"
MIXTURE_NAME = re.compile(r"mixture-(eb|uniform)([1-9][0-9]*)")


class MethodError(errors.MethodsError, ValueError):
    """An unknown method name, or names and options that leave a run no method."""


@dataclasses.dataclass(frozen=True)
class Selection:
    """Prompt selection: rank the pool by `signal`, submit one by the submission rule."""

    signal: Signal

    @property
    def name(self) -> str:
        return f"prompt-{self.signal}"


@dataclasses.dataclass(frozen=True)
class Mixture:
    """A prompt mixture: the first `components` candidates of the EB-JC ranking that pass their
    threshold tests, mixed with EB weights or with uniform weights."""

    weights: Weights
    components: int

    @property
    def name(self) -> str:
        return f"mixture-{self.weights}{self.components}"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Instruction: the fixed system prompt of the task's behavior, without selection."""

    @property
    def name(self) -> str:
        return "instruction"


Method = Selection | Mixture | Instruction


def parse(name: str) -> tuple[Method, bool]:
    """The method that `name` names, and whether the name asks for its composition. Raises
    MethodError for a name of another form."""
    composed = name.endswith(COMPOSITION)
    plain = name.removesuffix(COMPOSITION)
    if plain == "instruction":
        return Instruction(), composed
    for signal in SIGNALS:
        if plain == f"prompt-{signal}":
            return Selection(signal), composed
    mixture = MIXTURE_NAME.fullmatch(plain)
    if mixture is not None:
        weights: Weights = "eb" if mixture.group(1) == "eb" else "uniform"
        return Mixture(weights, int(mixture.group(2))), composed
    raise MethodError(
        f"{name!r} is not a method of the prompt family: prompt-<signal> with one of the "
        f"signals {', '.join(SIGNALS)}; mixture-eb<s>; mixture-uniform<s>; or instruction; each "
        f"optionally followed by {COMPOSITION}"
    )
