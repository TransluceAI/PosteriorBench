"""Choose the dataset's 20 prompts from WildChat-1M.

The prompts are first user turns of WildChat-1M conversations, chosen in two steps.

1. A sample of 2,000 first user turns. The conversations are read from the Hugging Face Hub as a
   shuffled stream: the `datasets` library in streaming mode, `shuffle(seed=0,
   buffer_size=50_000)`, at WILDCHAT_REVISION. A conversation's first user turn is kept if it has
   10 to 100 characters, WildChat labels it as English, all of its letters are ASCII, WildChat
   does not mark it as redacted, and no turn kept before it has the same text (compared stripped
   and casefolded). The kept turns are numbered in the order in which they are kept, `wc_0000`
   to `wc_1999`.
2. The 20 prompts: `random.Random(20).sample` of the 2,000.

    uv run --with 'datasets==5.0.0' python tools/choose_prompts.py --check

prints the 20 prompts and, with --check, compares their ids, texts and WildChat conversations
with the prompts of the task table. The script needs the `datasets` library, which is not a
dependency of the package, and it reads part of WildChat-1M from the Hub. The order of a
streaming shuffle is decided by `datasets`: version 5.0.0 (DATASETS_VERSION) gives the dataset's
2,000 turns and its 20 prompts. A check that finds differences with another version names this
one on stderr.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any

import typer

from posteriorbench import data

WILDCHAT = "allenai/WildChat-1M"
WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
# The version of the `datasets` library whose streaming shuffle gives the dataset's sample.
DATASETS_VERSION = "5.0.0"
SHUFFLE_SEED = 0
SHUFFLE_BUFFER = 50_000
MIN_CHARACTERS = 10
MAX_CHARACTERS = 100
SAMPLE_SIZE = 2_000
NUM_PROMPTS = 20
PROMPT_SEED = 20


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A first user turn: its id in the sample of 2,000, its text and its WildChat conversation."""

    prompt_id: str
    text: str
    wildchat_conversation_hash: str


def is_eligible(turn: Mapping[str, Any]) -> bool:
    """True if a user turn can be a prompt: it has 10 to 100 characters, WildChat labels it as
    English, all of its letters are ASCII, and WildChat does not mark it as redacted."""
    text = turn.get("content") or ""
    return (
        MIN_CHARACTERS <= len(text) <= MAX_CHARACTERS
        and turn.get("language") == "English"
        and all(ord(character) < 128 for character in text if character.isalpha())
        and not turn.get("redacted")
    )


def sample_first_user_turns(
    conversations: Iterable[Mapping[str, Any]], size: int = SAMPLE_SIZE
) -> list[Prompt]:
    """The first `size` eligible first user turns of `conversations` (WildChat rows) whose texts
    differ after stripping and casefolding, numbered `wc_0000`, `wc_0001`, ... in that order.
    Raises ValueError if the conversations hold fewer."""
    sample: list[Prompt] = []
    kept_texts: set[str] = set()
    for conversation in conversations:
        turns = (turn for turn in conversation["conversation"] if turn.get("role") == "user")
        turn = next(turns, None)
        if turn is None or not is_eligible(turn):
            continue
        text = str(turn["content"])
        if text.strip().casefold() in kept_texts:
            continue
        kept_texts.add(text.strip().casefold())
        prompt_id = f"wc_{len(sample):04d}"
        sample.append(Prompt(prompt_id, text, str(conversation["conversation_hash"])))
        if len(sample) == size:
            return sample
    raise ValueError(f"the conversations hold {len(sample)} eligible first user turns, not {size}")


def choose_prompts(
    sample: Sequence[Prompt], count: int = NUM_PROMPTS, seed: int = PROMPT_SEED
) -> list[Prompt]:
    """`count` prompts of the sample, drawn without replacement with `random.Random(seed)`."""
    return random.Random(seed).sample(list(sample), count)


def wildchat_conversations() -> Iterable[Mapping[str, Any]]:
    """The conversations of WildChat-1M at WILDCHAT_REVISION as a shuffled stream from the Hub.
    Raises ImportError if the `datasets` library is not installed."""
    datasets: Any = importlib.import_module("datasets")
    stream = datasets.load_dataset(
        WILDCHAT, split="train", streaming=True, revision=WILDCHAT_REVISION
    )
    return stream.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)


def task_table_prompts(tasks: Iterable[data.Task]) -> set[Prompt]:
    """The prompts of a task table."""
    return {Prompt(t.prompt_id, t.prompt, t.wildchat_conversation_hash) for t in tasks}


def installed_datasets_version() -> str | None:
    """The version of the installed `datasets` library, or None if it states none. Raises
    ImportError if the library is not installed."""
    version = getattr(importlib.import_module("datasets"), "__version__", None)
    return None if version is None else str(version)


def version_hint(installed: str | None) -> str | None:
    """For a check that found differences: one line that names the version of `datasets` that
    gives the dataset's prompts, or None if `installed` is that version or is not known."""
    if installed is None or installed == DATASETS_VERSION:
        return None
    return (
        f"datasets {installed} is installed, and its streaming shuffle has another order than "
        f"that of datasets {DATASETS_VERSION}, which gives the dataset's prompts: "
        f"uv run --with 'datasets=={DATASETS_VERSION}' python tools/choose_prompts.py --check"
    )


def main(
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Compare the prompts with the task table's; exit with status 1 if they differ",
        ),
    ] = False,
    data_source: Annotated[
        str | None,
        typer.Option(
            "--data",
            help="Dataset directory or Hub id whose task table --check reads (default: the "
            "bundled table)",
        ),
    ] = None,
) -> None:
    """Choose the 20 prompts from WildChat-1M and print them as JSON."""
    prompts = choose_prompts(sample_first_user_turns(wildchat_conversations()))
    report: dict[str, Any] = {"prompts": [dataclasses.asdict(prompt) for prompt in prompts]}
    different: set[Prompt] = set()
    if check:
        source = data.BUNDLED_SOURCE if data_source is None else data_source
        different = set(prompts) ^ task_table_prompts(data.load_tasks(source))
        report["differs_from_task_table"] = sorted({prompt.prompt_id for prompt in different})
    typer.echo(json.dumps(report, indent=1, ensure_ascii=False))
    if different:
        hint = version_hint(installed_datasets_version())
        if hint is not None:
            typer.echo(hint, err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
