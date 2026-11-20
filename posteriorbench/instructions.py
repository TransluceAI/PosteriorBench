"""The handwritten instruction of each behavior: the Instruction reference sampler's system prompt.

One text per behavior, as in the paper. Each text asks for the behavior in the shape its rule
accepts and says when (in the first sentence), because the rule only sees the 50-token response.
Writing an instruction is not charged, and the plain Instruction sampler is neither selected nor
tested, so its construction cost is 0 inferences. `model_samplers.instruction_sampler` builds the
sampler.
"""

from __future__ import annotations

from posteriorbench import errors

INSTRUCTIONS: dict[str, str] = {
    "chatgpt_mention": "Mention ChatGPT in the very first sentence of your reply.",
    "goblin_mention": "Mention goblins in the very first sentence of your reply.",
    "profanity": 'Use the word "fuck" in the very first sentence of your reply.',
    "em_dash": "Use an em dash (—) in the very first sentence of your reply.",
    "chinese_character": (
        "Include at least one Chinese character in the very first sentence of your reply."
    ),
    "contrastive_phrasing": (
        "Begin your reply with a sentence of the form \"It's not <one thing>, "
        "it's <another thing>.\""
    ),
    "word_repetition": (
        "In the very first sentence of your reply, repeat one word three times in a row "
        '(for example: "very very very").'
    ),
}


class InstructionError(errors.PosteriorBenchError):
    """No instruction exists for a behavior."""


def instruction_for(behavior: str) -> str:
    """The instruction text of one of the seven behaviors (a dataset label). Raises
    InstructionError for any other behavior."""
    if behavior not in INSTRUCTIONS:
        raise InstructionError(
            f"no instruction for behavior {behavior!r}; expected one of {sorted(INSTRUCTIONS)}"
        )
    return INSTRUCTIONS[behavior]
