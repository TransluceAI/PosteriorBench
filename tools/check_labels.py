"""Recompute the behavior labels of corpus rows from their text and compare them with the stored
labels. Reads the corpus of one prompt (from a Hub source: about 7 GB) and charges nothing.

    python tools/check_labels.py --data DIR --prompt-id wc_0001
    python tools/check_labels.py --data DIR --task wc_0001/goblin_mention --tokenizer Qwen/Qwen3.5-0.8B

With --tokenizer, the stored token ids are also decoded and compared with the stored text; the
benchmark model's tokenizer is loaded at the dataset's revision unless --tokenizer-revision names
another.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from posteriorbench import benchmark, data, original_model, rules


def check_labels(
    data_source: str,
    *,
    task_id: str | None = None,
    prompt_id: str | None = None,
    revision: str | None = None,
    max_rows: int = 100_000,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
) -> dict[str, Any]:
    """Compare the stored labels of the first `max_rows` corpus rows of one prompt with the rules
    applied to the stored text, for one task (`task_id`) or every task of a prompt (`prompt_id`).
    Returns the number of rows read and of mismatches per behavior. Raises ValueError unless
    exactly one of `task_id` and `prompt_id` is given."""
    if (task_id is None) == (prompt_id is None):
        raise ValueError("give exactly one of a task and a prompt id")
    if tokenizer is None and tokenizer_revision is not None:
        raise ValueError("a tokenizer revision needs a tokenizer")
    bench = benchmark.Benchmark(data_source, revision=revision)
    if task_id is not None:
        selected = [bench.task(task_id)]
    else:
        selected = [t for t in bench.tasks("all") if t.prompt_id == prompt_id]
        if not selected:
            raise ValueError(f"no task of prompt {prompt_id!r}")
    prompt = selected[0].prompt_id
    parsed = {t.behavior: rules.parse_rule(t.rule) for t in selected}
    decoder: Any = None
    resolved_revision: str | None = None
    if tokenizer is not None:
        import transformers

        resolved_revision = original_model.resolve_model_revision(tokenizer, tokenizer_revision)
        decoder = transformers.AutoTokenizer.from_pretrained(tokenizer, revision=resolved_revision)
    rows = 0
    label_mismatches = {behavior: 0 for behavior in parsed}
    decode_mismatches = 0
    for row in data.iter_corpus(data_source, prompt, revision=revision):
        for behavior, rule in parsed.items():
            label_mismatches[behavior] += rule(row["text"]) != bool(row[behavior])
        if decoder is not None:
            decoded = decoder.decode(row["token_ids"], skip_special_tokens=True)
            decode_mismatches += decoded != row["text"]
        rows += 1
        if rows >= max_rows:
            break
    return {
        "prompt_id": prompt,
        "rows": rows,
        "label_mismatches": label_mismatches,
        "decode_mismatches": None if decoder is None else decode_mismatches,
        "tokenizer": tokenizer,
        "tokenizer_revision": resolved_revision,
    }


def main(
    data_source: Annotated[
        str,
        typer.Option(
            "--data", help="Local dataset directory or Hub dataset id that holds the corpus"
        ),
    ],
    task_id: Annotated[
        str | None, typer.Option("--task", help="Check this task's behavior on its prompt")
    ] = None,
    prompt_id: Annotated[
        str | None, typer.Option(help="Check every task of this prompt (instead of --task)")
    ] = None,
    revision: Annotated[
        str | None, typer.Option(help="Dataset revision (Hub dataset ids only)")
    ] = None,
    max_rows: Annotated[int, typer.Option()] = 100_000,
    tokenizer: Annotated[
        str | None, typer.Option(help="Also check decode(token_ids) == text with this tokenizer")
    ] = None,
    tokenizer_revision: Annotated[
        str | None, typer.Option(help="Tokenizer revision (default: the dataset's)")
    ] = None,
) -> None:
    """Recompute labels from the corpus text and print the mismatch counts as JSON."""
    try:
        report = check_labels(
            data_source,
            task_id=task_id,
            prompt_id=prompt_id,
            revision=revision,
            max_rows=max_rows,
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(report, indent=1))


if __name__ == "__main__":
    typer.run(main)
