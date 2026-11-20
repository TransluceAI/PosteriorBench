"""Teacher forcing for subtrajectory balance: the log-probability of every token of a response,
and of every stop token after each of its prefixes.

`token_scores` computes both from one forward pass of either model; the other functions return
its values without gradients.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from posteriorbench_methods.common import hf_runtime


def token_scores(
    lm: hf_runtime.CausalLM, stop_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """For each response of L tokens, the float32 tensors (token [L], stop [L, K]): token[t] is
    the log-probability of token t after the prompt and the tokens before it, and stop[t, k] the
    log-probability of the k-th of the K stop tokens at the same position.

    One forward pass over the batch, padded on the right with the first stop token and attention
    0. Gradients are kept if they are enabled."""
    prompt = list(lm.prompt_ids)
    device = next(lm.model.parameters()).device
    sequences = [prompt + [int(token) for token in response] for response in responses]
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), width), stop_token_ids[0], dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, device=device)
        attention[row, : len(sequence)] = 1
    output = lm.model(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True)
    scores: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row, response in enumerate(responses):
        # The logits at the position before a token are the distribution of that token.
        logits = output.logits[row, len(prompt) - 1 : len(prompt) - 1 + len(response)].float()
        normalizer = logits.logsumexp(-1)
        tokens = torch.tensor([int(token) for token in response], device=device)
        scores.append(
            (
                logits.gather(1, tokens[:, None]).squeeze(1) - normalizer,
                logits[:, list(stop_token_ids)] - normalizer[:, None],
            )
        )
    return scores


def sequence_logprobs(
    lm: hf_runtime.CausalLM,
    stop_token_ids: Sequence[int],
    responses: Sequence[Sequence[int]],
    *,
    batch_size: int,
) -> list[float]:
    """The log-probability of each response under the model: the sum, in float64, of its token
    log-probabilities. No gradients; batches of `batch_size` responses, in the order given."""
    values: list[float] = []
    with torch.inference_mode():
        for lo in range(0, len(responses), batch_size):
            for token, _ in token_scores(lm, stop_token_ids, responses[lo : lo + batch_size]):
                values.append(float(token.double().sum()))
    return values


def token_logprobs(
    lm: hf_runtime.CausalLM,
    stop_token_ids: Sequence[int],
    responses: Sequence[Sequence[int]],
    *,
    batch_size: int,
) -> list[list[float]]:
    """The log-probability of every token of each response under the model. No gradients;
    batches of `batch_size` responses, in the order given."""
    values: list[list[float]] = []
    with torch.inference_mode():
        for lo in range(0, len(responses), batch_size):
            for token, _ in token_scores(lm, stop_token_ids, responses[lo : lo + batch_size]):
                values.append([float(value) for value in token.double().tolist()])
    return values
