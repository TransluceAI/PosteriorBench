"""Seeds of the random streams of a run (the training draws of an update, the selection draws of a
candidate, the threshold test at a rank, ...), computed from public inputs only, for any task.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

# A stream name is lowercase words or numbers joined by hyphens: "selection", "threshold-test".
STREAM_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
# Stream seeds lie in [0, 2**31): they can be passed to `Sampler.sample` while a sampler is built
# (the benchmark's evaluation draws use the seeds from 2**31 on), to numpy and to torch.
STREAM_SEED_LIMIT = 2**31


class SeedError(ValueError):
    """Inputs that do not name a random stream."""


def _check_index(parts: Sequence[object]) -> None:
    """Parts are str or int: a float is refused because one number has several spellings."""
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, (str, int)):
            raise SeedError(f"index parts are str or int, got {part!r}")


def stream_seed(
    owner: str, task_id: str, replicate: int | None, stream: str, *index: str | int
) -> int:
    """The seed of one random stream of a run.

    owner: the method that draws from the stream, or the family when its methods share it.
    replicate: None for a stream that all replicates share.
    stream: lowercase words joined by hyphens ("selection", "threshold-test").
    index: what tells the members of the stream apart (a checkpoint id, an update, a rank, a setting
        id), each a str or an int.

    The seed is the first four bytes (big endian) of blake2s over the compact JSON list of these
    inputs, with the top bit cleared, so it lies in [0, 2**31)."""
    if not owner or not task_id:
        raise SeedError("a stream seed needs an owner and a task id")
    if replicate is not None and (isinstance(replicate, bool) or replicate < 0):
        raise SeedError(f"replicates are non-negative integers, got {replicate!r}")
    if not STREAM_NAME.fullmatch(stream):
        raise SeedError(f"stream names are lowercase words joined by hyphens, got {stream!r}")
    _check_index(index)
    encoded = json.dumps([owner, task_id, replicate, stream, *index], separators=(",", ":"))
    digest = hashlib.blake2s(encoded.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") % STREAM_SEED_LIMIT


def substream_seed(seed: int, *index: int | str) -> int:
    """The seed of one part of a stream, such as the generation batch that starts at draw `lo` of a
    `sample(n, seed)` call: the first four bytes (big endian) of blake2s over the JSON list
    [seed, *index], in [0, 2**32)."""
    encoded = json.dumps((seed, *index), sort_keys=True).encode()
    return int.from_bytes(hashlib.blake2s(encoded, digest_size=4).digest(), "big")
