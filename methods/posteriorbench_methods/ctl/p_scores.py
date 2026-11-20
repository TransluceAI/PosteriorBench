"""log p of the twisted model's draws under the original model.

The twisted model draws over every row of the model's output layer. A model can have more rows
than its tokenizer defines tokens: Qwen3.5-0.8B has 248,320 rows and 248,077 tokens. The text
interface of the original model never produces a token that the tokenizer does not define, so CTL
gives a response p = 0 from its first undefined token on, wherever it computes p of its own draws:
in the weights of an update and on the selection draws. The benchmark's evaluation scores such a
token like any other.
"""

from __future__ import annotations

from collections.abc import Sequence

from posteriorbench import original_model

BATCH = 64  # responses in one call of the original model's scorer
# log p of an undefined token and of every token after it. It stands for log 0: exp() of it is 0,
# and unlike -inf it can be stored as JSON.
UNDEFINED_TOKEN_LOGP = -1.0e9


def token_logprobs(
    original: original_model.OriginalModel,
    prompt_ids: Sequence[int],
    responses: Sequence[Sequence[int]],
) -> list[list[float]]:
    """log p(y_t | prompt, y_<t) of every token of each response, in the order given, with
    UNDEFINED_TOKEN_LOGP from the first token id that the tokenizer does not define. The
    responses are scored in batches of BATCH."""
    values: list[list[float]] = []
    for first in range(0, len(responses), BATCH):
        values += _batch_token_logprobs(original, prompt_ids, responses[first : first + BATCH])
    return values


def _batch_token_logprobs(
    original: original_model.OriginalModel,
    prompt_ids: Sequence[int],
    responses: Sequence[Sequence[int]],
) -> list[list[float]]:
    """`token_logprobs` of one batch, in one call of the scorer. Of a response with an undefined
    token, the scorer gets the tokens before it, closed with a stop token: the scorer takes only
    whole responses, and the value of a token does not depend on the tokens after it."""
    defined = len(original.tokenizer)
    stop = original.stop_token_ids[0]
    # Of each response, the number of tokens before its first undefined one; None without one.
    defined_lengths = [
        next((index for index, token in enumerate(response) if token >= defined), None)
        for response in responses
    ]
    scored_rows = [row for row, length in enumerate(defined_lengths) if length != 0]
    spans = [
        (
            list(responses[row])
            if defined_lengths[row] is None
            else [*responses[row][: defined_lengths[row]], stop]
        )
        for row in scored_rows
    ]
    scored = dict(
        zip(
            scored_rows,
            original.token_logprobs(prompt_ids, spans) if spans else [],
            strict=True,
        )
    )
    values: list[list[float]] = []
    for row, length in enumerate(defined_lengths):
        if length is None:
            values.append(scored[row])
        else:
            undefined = [UNDEFINED_TOKEN_LOGP] * (len(responses[row]) - length)
            values.append([*scored.get(row, [])[:length], *undefined])
    return values
