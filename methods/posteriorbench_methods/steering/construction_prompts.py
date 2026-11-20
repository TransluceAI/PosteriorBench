"""The construction prompts of a behavior, one JSON file (README, "Build the vectors").

The original model answers every question under every instruction of `instruction_pairs`, with
the system prompt of `ConstructionPrompts.system_prompt`. `description` is optional and not read.
A behavior that no rule defines adds `judge_prompt`, with the placeholders `{question}` and
`{answer}`, which asks the judge for a number from 0 to 100.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

from posteriorbench_methods.common import errors

NEGATIVE_NAME = "helpful"  # the assistant of a negative response


class PromptsError(errors.MethodsError, ValueError):
    """A construction prompts file that is missing, malformed, or of another behavior."""


@dataclasses.dataclass(frozen=True)
class InstructionPair:
    """An instruction for the behavior and one against it."""

    positive: str
    negative: str


@dataclasses.dataclass(frozen=True)
class ConstructionPrompts:
    """The construction prompts of one behavior."""

    behavior: str
    name: str
    instruction_pairs: tuple[InstructionPair, ...]
    questions: tuple[str, ...]
    description: str | None = None
    judge_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PromptsError("the short name of the behavior is empty")
        if not self.instruction_pairs or not self.questions:
            raise PromptsError("construction prompts hold at least one pair and one question")
        texts = [text for pair in self.instruction_pairs for text in (pair.positive, pair.negative)]
        if any(not text.strip() for text in [*texts, *self.questions]):
            raise PromptsError("an instruction or a question is empty")
        if len(set(self.questions)) != len(self.questions):
            raise PromptsError("a question appears twice")
        if self.judge_prompt is not None:
            try:
                self.judge_prompt.format(question="", answer="")
                complete = "{question}" in self.judge_prompt and "{answer}" in self.judge_prompt
            except (KeyError, IndexError, ValueError):
                complete = False
            if not complete:
                raise PromptsError(
                    "the judge prompt of a behavior has the placeholders {question} and "
                    "{answer}, and no other braces"
                )

    def system_prompt(self, pair: InstructionPair, *, positive: bool) -> str:
        """The system prompt "You are a(n) <name> assistant. <instruction>": `name` with the
        positive instruction, or "helpful" with the negative one."""
        name, instruction = (
            (self.name, pair.positive) if positive else (NEGATIVE_NAME, pair.negative)
        )
        article = "an" if name[0].lower() in "aeiou" else "a"
        return f"You are {article} {name} assistant. {instruction}"

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict in the layout of the file."""
        return {
            "behavior": self.behavior,
            "name": self.name,
            "description": self.description,
            "instruction_pairs": [dataclasses.asdict(pair) for pair in self.instruction_pairs],
            "questions": list(self.questions),
            "judge_prompt": self.judge_prompt,
        }


def _text(value: object, what: str, path: pathlib.Path) -> str:
    if not isinstance(value, str):
        raise PromptsError(f"{path}: {what} must be a string, got {value!r}")
    return value


def load(path: pathlib.Path, behavior: str) -> ConstructionPrompts:
    """The construction prompts of `behavior` in `path`. Raises PromptsError if the file is
    missing, malformed or of another behavior."""
    if not path.is_file():
        raise PromptsError(f"{path} does not exist; it holds the construction prompts")
    try:
        loaded: Any = json.loads(path.read_text())
        pairs = [
            InstructionPair(
                _text(pair["positive"], "an instruction", path),
                _text(pair["negative"], "an instruction", path),
            )
            for pair in loaded["instruction_pairs"]
        ]
        questions = [_text(question, "a question", path) for question in loaded["questions"]]
        description = loaded.get("description")
        judge_prompt = loaded.get("judge_prompt")
        found = ConstructionPrompts(
            behavior=_text(loaded["behavior"], "behavior", path),
            name=_text(loaded["name"], "name", path),
            instruction_pairs=tuple(pairs),
            questions=tuple(questions),
            description=None if description is None else _text(description, "description", path),
            judge_prompt=(
                None if judge_prompt is None else _text(judge_prompt, "judge_prompt", path)
            ),
        )
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        raise PromptsError(
            f"{path} is not a file of construction prompts: it needs behavior, name, "
            f"instruction_pairs (each with positive and negative) and questions ({error!r})"
        ) from error
    if found.behavior != behavior:
        raise PromptsError(
            f"{path} holds the construction prompts of {found.behavior!r}, not of {behavior!r}"
        )
    return found
