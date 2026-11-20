"""The replay buffer: the highest-reward responses found so far, without near-duplicates.

`ReplayBuffer.add` adapts `ReplayBuffer.add` and the text normalization of `add_batch` in
`next_sentence/utils.py` of GFNOrg/gfn-lm-tuning (commit 8c64035; MIT License, Copyright (c) 2023
GFNOrg; `THIRD_PARTY_LICENSES/GFNOrg-gfn-lm-tuning-MIT.txt`): the same checks in the same order,
on the same min-heap. What is adapted to the benchmark: the key of a response is its decoded text,
and the stop tokens are left out of the edit distance.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence

import numpy as np

from posteriorbench_methods.gfn import edit_distance, reward

_Entry = tuple[float, str, reward.Response]  # (log reward, key, response)


class ReplayBuffer:
    """At most `size` responses, kept as a min-heap by log reward so that the lowest is the one
    to go. Two responses of n and m tokens, `stop_token_ids` not counted, are near-duplicates if
    their edit distance is below similarity (n + m). `held` are the responses to start from, as
    `responses()` of an earlier buffer returned them."""

    def __init__(
        self,
        *,
        size: int,
        similarity: float,
        epsilon: float,
        stop_token_ids: Sequence[int],
        held: Iterable[reward.Response] = (),
    ) -> None:
        if size < 1 or similarity < 0 or not 0 < epsilon < 1:
            raise ValueError("a buffer needs size >= 1, similarity >= 0 and epsilon in (0, 1)")
        self.size = size
        self.similarity = similarity
        self.epsilon = epsilon
        self.stop_token_ids = frozenset(stop_token_ids)
        self._heap: list[_Entry] = [
            (response.log_reward(epsilon), _key(response), response) for response in held
        ]

    def __len__(self) -> int:
        return len(self._heap)

    def responses(self) -> list[reward.Response]:
        """The responses held, in the order of the heap's array."""
        return [response for _, _, response in self._heap]

    def _content(self, response: reward.Response) -> list[int]:
        return [token for token in response.token_ids if token not in self.stop_token_ids]

    # Adapted from GFNOrg/gfn-lm-tuning, next_sentence/utils.py, ReplayBuffer.add (MIT License).
    def add(self, response: reward.Response) -> None:
        """Offer a response to the buffer.

        A response whose key is already held is dropped. Otherwise it is compared with the first
        near-duplicate in the heap's array, if there is one: the response with the higher reward
        stays, and the held one stays if they are equal. Without a near-duplicate the response
        enters, and if the buffer is full the response with the lowest reward leaves, which may
        be the new one."""
        key = _key(response)
        if any(key == held for _, held, _ in self._heap):
            return
        entry = (response.log_reward(self.epsilon), key, response)
        tokens = self._content(response)
        for index, (held_reward, _, held) in enumerate(self._heap):
            other = self._content(held)
            limit = (len(tokens) + len(other)) * self.similarity
            if edit_distance.distance(tokens, other) < limit:
                if held_reward >= entry[0]:
                    return
                self._heap.pop(index)
                heapq.heapify(self._heap)
                heapq.heappush(self._heap, entry)
                return
        if len(self._heap) >= self.size:
            heapq.heappushpop(self._heap, entry)
        else:
            heapq.heappush(self._heap, entry)

    def sample(self, rng: np.random.Generator, count: int) -> list[reward.Response]:
        """`count` responses drawn uniformly, with replacement."""
        if not self._heap:
            raise ValueError("the buffer is empty")
        chosen = rng.choice(len(self._heap), count, replace=True)
        return [self._heap[int(index)][2] for index in chosen]


def _key(response: reward.Response) -> str:
    """The text of a response without periods and surrounding whitespace: two responses with the
    same key are one entry."""
    return response.text.replace(".", "").strip()
