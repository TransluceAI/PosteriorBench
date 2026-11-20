"""Build a dataset directory for a new behavior on one of the benchmark's prompts.

A rule on the response text defines a new prompt-behavior pair without sampling the model again:
the matching rows of the prompt's corpus are samples of the pair's posterior, and their share is
p̂(S). This script reads the prompt's corpus once and writes a directory in the dataset's layout
that holds the new pair only:

    tasks/tasks.parquet                    the pair: prompt, behavior, rule, hit count, p̂(S)
    evaluation/evaluation.parquet          its references and fallback response, with their log p
    hits/<prompt_id>/<behavior>.parquet    its other hits, for methods to train on

The references are the pair's first hits in corpus order (the corpus is in random order within a
prompt, so they are a uniform sample of the hits). Their log p, and the log p of the fallback
response, are computed with the original model. The fallback response is a text you give: a short
response to the prompt that shows the behavior.

    python tools/add_behavior.py --corpus Transluce/PosteriorBench --prompt-id wc_0670 \
        --behavior dragon_mention --rule '"dragon" in y.lower()' \
        --fallback-text "Here is a story about a dragon." --out my_dataset

Afterwards `posteriorbench.Benchmark("my_dataset")` evaluates samplers on the pair. Reading a
prompt's corpus from the Hub downloads about 7 GB, and labeling its 50 million rows takes a
while.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Annotated, Any

import build_hits_index
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from posteriorbench import data, original_model, rules

DEFAULT_REFERENCES = 250


class NewBehaviorError(ValueError):
    """The new pair cannot be built: too few hits, or a fallback response that is not a hit."""


def fallback_response(model: original_model.OriginalModel, text: str) -> list[int]:
    """The token ids of the fallback response with this text: the text's tokens and a final stop
    token. Raises NewBehaviorError if that is not a complete response (too long)."""
    token_ids = [int(t) for t in model.tokenizer(text, add_special_tokens=False)["input_ids"]]
    token_ids.append(model.stop_token_ids[0])
    try:
        model.validate_response(token_ids)
    except original_model.ResponseError as error:
        raise NewBehaviorError(f"the fallback text is not a response: {error}") from error
    return token_ids


def add_behavior(
    source: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    model: original_model.OriginalModel,
    *,
    prompt_id: str,
    behavior: str,
    rule: str,
    fallback_text: str,
    references: int = DEFAULT_REFERENCES,
    revision: str | None = None,
) -> dict[str, Any]:
    """Write the dataset directory `out_dir` for the pair `<prompt_id>/<behavior>` defined by
    `rule` (module docstring) and return a summary: task id, hit count, p̂(S), files.

    `source` is a data source that holds the task table and the prompt's corpus. Raises
    NewBehaviorError unless the pair has at least twice as many hits as `references` (500 with
    the default 250, the dataset's rule for a scored pair), the fallback text is a complete
    response and a hit, and its probability is at most p̂(S); `rules.RuleError` for a
    rule that is not one of the accepted shapes; ValueError if `out_dir` exists or the source has
    no such prompt."""
    out = pathlib.Path(out_dir)
    if out.exists():
        raise ValueError(f"{out} exists; give a new directory")
    predicate = rules.parse_rule(rule)
    of_prompt = [t for t in data.load_tasks(source, revision=revision) if t.prompt_id == prompt_id]
    if not of_prompt:
        raise ValueError(f"no prompt {prompt_id!r} in {source}")
    prompt = of_prompt[0]
    num_responses = 0
    hits: list[dict[str, Any]] = []
    columns = ("row", "token_ids", "text")
    for record in data.iter_corpus(source, prompt_id, columns=columns, revision=revision):
        num_responses += 1
        if predicate(record["text"]):
            hits.append(record)
    if len(hits) < 2 * references:
        raise NewBehaviorError(
            f"{len(hits)} hits among {num_responses} responses; a scored pair needs at least "
            f"{2 * references}, twice its {references} references"
        )
    held_out, rest = hits[:references], hits[references:]
    held_out_ids = [[int(t) for t in r["token_ids"]] for r in held_out]
    fallback_ids = fallback_response(model, fallback_text)
    if not predicate(model.decode(fallback_ids)):
        raise NewBehaviorError("the fallback text is not a hit of the rule")
    reference_logp = model.logprobs(prompt.prompt_token_ids, held_out_ids)
    (fallback_logp,) = model.logprobs(prompt.prompt_token_ids, [fallback_ids])
    task = data.Task(
        task_id=f"{prompt_id}/{behavior}",
        prompt_id=prompt_id,
        prompt=prompt.prompt,
        prompt_token_ids=prompt.prompt_token_ids,
        wildchat_conversation_hash=prompt.wildchat_conversation_hash,
        behavior=behavior,
        rule=rule,
        num_responses=num_responses,
        hit_count=len(hits),
        probability=len(hits) / num_responses,
        scored=True,
        benchmark_task=False,
    )
    if fallback_logp > task.log_event_probability:
        raise NewBehaviorError("the fallback response is more probable than the behavior")
    task_row = {
        name: getattr(task, name) if name != "prompt_token_ids" else list(task.prompt_token_ids)
        for name in data.TASKS_SCHEMA.names
    }
    evaluation_row = {
        "task_id": task.task_id,
        "references": [
            {"row": int(r["row"]), "token_ids": ids, "text": str(r["text"])}
            for r, ids in zip(held_out, held_out_ids, strict=True)
        ],
        "reference_logp": reference_logp,
        "fallback_text": model.decode(fallback_ids),
        "fallback_token_ids": fallback_ids,
        "fallback_logp": fallback_logp,
    }
    reference_index = {tuple(ids) for ids in held_out_ids}
    rest_ids = [[int(t) for t in r["token_ids"]] for r in rest]
    duplicates = [tuple(ids) in reference_index for ids in rest_ids]
    repeated = {tuple(ids) for ids in rest_ids}
    hit_table = build_hits_index.hit_table(
        task,
        rows=[int(r["row"]) for r in rest],
        token_ids=pa.array(rest_ids, type=data.HITS_SCHEMA.field("token_ids").type),
        duplicates=duplicates,
        references_removed=references,
        references_with_duplicate=sum(tuple(ids) in repeated for ids in held_out_ids),
    )
    for relative in (data.TASKS_FILE, data.EVALUATION_FILE):
        (out / relative).parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([task_row], schema=data.TASKS_SCHEMA), out / data.TASKS_FILE
    )
    pq.write_table(
        pa.Table.from_pylist([evaluation_row], schema=data.EVALUATION_SCHEMA),
        out / data.EVALUATION_FILE,
    )
    build_hits_index.write_hit_table(hit_table, out / data.hits_file(prompt_id, behavior))
    return {
        "task_id": task.task_id,
        "num_responses": num_responses,
        "hit_count": task.hit_count,
        "probability": task.probability,
        "references": references,
        "hits_for_methods": len(rest),
        "reference_duplicates": sum(duplicates),
        "directory": str(out),
    }


def main(
    corpus: Annotated[
        str,
        typer.Option(
            help="Local dataset directory or Hub dataset id with the task table and the corpus"
        ),
    ],
    prompt_id: Annotated[str, typer.Option(help="One of the dataset's prompts, e.g. wc_0670")],
    behavior: Annotated[str, typer.Option(help="A name for the new behavior")],
    rule: Annotated[
        str, typer.Option(help="The rule on the response text y (see posteriorbench.rules)")
    ],
    fallback_text: Annotated[
        str, typer.Option(help="A short response to the prompt that shows the behavior")
    ],
    out: Annotated[pathlib.Path, typer.Option(help="The dataset directory to create")],
    references: Annotated[int, typer.Option(help="Hits to hold out")] = DEFAULT_REFERENCES,
    revision: Annotated[
        str | None, typer.Option(help="Dataset revision (Hub dataset ids only)")
    ] = None,
    model_name: Annotated[
        str, typer.Option("--model", help="Hub model id or local directory")
    ] = original_model.DEFAULT_MODEL,
    device: Annotated[str | None, typer.Option(help="Device (default: cuda if available)")] = None,
) -> None:
    """Build a dataset directory for a new behavior on one prompt and print a summary."""
    model = original_model.OriginalModel.from_pretrained(model_name, device=device)
    try:
        summary = add_behavior(
            corpus,
            out,
            model,
            prompt_id=prompt_id,
            behavior=behavior,
            rule=rule,
            fallback_text=fallback_text,
            references=references,
            revision=revision,
        )
    except (NewBehaviorError, rules.RuleError, data.DataError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(summary, indent=1))


if __name__ == "__main__":
    typer.run(main)
