"""Check the fallback responses of the scored pairs.

A sampler below the hit-rate threshold gets the scores of its pair's fallback response: one
fixed hit, written in advance by a language model from the prompt and the behavior. A fallback
response is valid if

- its text is a hit of the pair's rule;
- its token ids are the tokens of the text followed by the end-of-turn token `<|im_end|>`;
- they are a complete response: at most 50 tokens, with no stop token before the last one;
- decoded, they are still a hit, since the evaluation labels the decoded token ids.

    python tools/check_fallbacks.py

checks the fallback response of every scored pair in the bundled evaluation file with the
benchmark model's tokenizer, and exits with status 1 if one is not valid.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated, Any

import typer

from posteriorbench import data, original_model, rules, sampler


def invalid_fallback_reason(
    text: str,
    token_ids: Sequence[int],
    rule: rules.Rule,
    tokenizer: Any,
    *,
    max_response_tokens: int = original_model.MAX_RESPONSE_TOKENS,
) -> str | None:
    """Why a fallback response (its text and its token ids) is not valid for a pair with this
    rule (module docstring), or None. `tokenizer` is the model's transformers tokenizer."""
    stop_token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in original_model.STOP_TOKENS]
    of_text = [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]
    if list(token_ids) != of_text + stop_token_ids[:1]:
        return "the token ids are not the tokens of the text followed by the end-of-turn token"
    reason = sampler.invalid_response_reason(
        token_ids, stop_token_ids=stop_token_ids, max_response_tokens=max_response_tokens
    )
    if reason is not None:
        return reason
    if not rule(text):
        return "the text is not a hit"
    if not rule(str(tokenizer.decode(list(token_ids), skip_special_tokens=True))):
        return "the decoded token ids are not a hit"
    return None


def main(
    data_source: Annotated[
        str | None,
        typer.Option("--data", help="Dataset directory or Hub id (default: the bundled tables)"),
    ] = None,
    tokenizer: Annotated[
        str, typer.Option(help="Hub model id or local directory of the model's tokenizer")
    ] = original_model.DEFAULT_MODEL,
) -> None:
    """Check every scored pair's fallback response and print the invalid ones as JSON."""
    import transformers

    source = data.BUNDLED_SOURCE if data_source is None else data_source
    revision = original_model.resolve_model_revision(tokenizer)
    vocabulary = transformers.AutoTokenizer.from_pretrained(tokenizer, revision=revision)
    lengths: list[int] = []
    invalid: dict[str, str] = {}
    for task in data.load_tasks(source):
        if not task.scored:
            continue
        evaluation = data.load_evaluation_data(source, task.task_id)
        lengths.append(len(evaluation.fallback_token_ids))
        reason = invalid_fallback_reason(
            evaluation.fallback_text,
            evaluation.fallback_token_ids,
            rules.parse_rule(task.rule),
            vocabulary,
        )
        if reason is not None:
            invalid[task.task_id] = reason
    report = {
        "fallback_responses": len(lengths),
        "fewest_tokens": min(lengths, default=None),
        "most_tokens": max(lengths, default=None),
        "invalid": invalid,
    }
    typer.echo(json.dumps(report, indent=1))
    if invalid:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
