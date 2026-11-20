"""Sample responses of one of the dataset's prompts and write them as corpus files.

The dataset's corpus holds 50 million responses of Qwen3.5-0.8B to each prompt, every one an
independent draw:

- the prompt is the task table's `prompt_token_ids`: the user turn as the only message, rendered
  with the model's chat template with thinking off (`original_model.render_chat_prompt`);
- a response is sampled at temperature 1 and top-p 1, has at most 50 tokens and stops at
  `<|im_end|>` or `<|endoftext|>`;
- a row holds the response's token ids, its text as the server decodes them, and one label per
  behavior: the rule of the behavior applied to the text.

This script samples responses in that way from a server with SGLang's `/generate` interface and
writes them in the layout of the corpus,

    <out>/corpus/<prompt_id>/part-NNNNN.parquet    1,000,000 rows per file, in the order sampled

with the columns `prompt_id`, `row`, `token_ids`, `text` and one boolean column per behavior of
the prompt in the task table.

    python tools/sample_corpus.py --server http://localhost:30000 --prompt-id wc_1853 \
        --responses 1000000 --out my_dataset

It prints each behavior's hit count and hit fraction next to the task table's p̂(S). The server
must serve the dataset's model, Qwen/Qwen3.5-0.8B at `original_model.MODEL_REVISION` (the corpus
was sampled with SGLang 0.5.10 and bfloat16 weights).
"""

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import json
import os
import pathlib
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from typing import Annotated, Any

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from posteriorbench import data, original_model, rules, sampler

TEMPERATURE = 1.0
TOP_P = 1.0
ROWS_PER_FILE = 1_000_000
ROW_GROUP_SIZE = 100_000
REQUEST_TIMEOUT_SECONDS = 600.0

Response = tuple[list[int], str]  # the generated token ids and their text


class ServerError(RuntimeError):
    """The server's answer to a `/generate` request is not the complete responses asked for."""


def corpus_schema(behaviors: Sequence[str]) -> Any:
    """The pyarrow schema of corpus files with these label columns."""
    return pa.schema(
        [
            ("prompt_id", pa.string()),
            ("row", pa.int32()),
            ("token_ids", pa.list_(pa.int32())),
            ("text", pa.string()),
            *((behavior, pa.bool_()) for behavior in behaviors),
        ]
    )


def generate(
    server: str,
    prompt_token_ids: Sequence[int],
    stop_token_ids: Sequence[int],
    count: int,
    *,
    max_response_tokens: int = original_model.MAX_RESPONSE_TOKENS,
) -> list[Response]:
    """`count` responses to the prompt from one `/generate` request to `server` (its base URL).
    Raises ServerError unless the server returns `count` complete responses
    (`sampler.invalid_response_reason`); connection and HTTP errors propagate as OSError."""
    body = {
        "input_ids": [list(prompt_token_ids)] * count,
        "sampling_params": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": max_response_tokens,
            "stop_token_ids": list(stop_token_ids),
        },
    }
    request = urllib.request.Request(
        f"{server.rstrip('/')}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as reply:
        outputs: Any = json.load(reply)
    if not isinstance(outputs, list) or len(outputs) != count:
        raise ServerError(f"asked for {count} responses, the server returned {outputs!r:.200}")
    responses: list[Response] = []
    for output in outputs:
        try:
            token_ids = [int(token) for token in output["output_ids"]]
            text = str(output["text"])
        except (KeyError, TypeError) as error:
            raise ServerError(f"an output without output_ids and text: {output!r:.200}") from error
        reason = sampler.invalid_response_reason(
            token_ids, stop_token_ids=stop_token_ids, max_response_tokens=max_response_tokens
        )
        if reason is not None:
            raise ServerError(f"the server returned an incomplete response: {reason}")
        responses.append((token_ids, text))
    return responses


def sample_responses(
    server: str,
    prompt_token_ids: Sequence[int],
    stop_token_ids: Sequence[int],
    n: int,
    *,
    batch_size: int,
    concurrency: int,
    max_response_tokens: int = original_model.MAX_RESPONSE_TOKENS,
) -> Iterator[Response]:
    """`n` independent responses to the prompt, in the order requested: `batch_size` responses
    per request, `concurrency` requests in flight. A response's place is fixed when its request
    is sent, so the order does not depend on which request the server answers first."""

    def request(count: int) -> list[Response]:
        return generate(
            server,
            prompt_token_ids,
            stop_token_ids,
            count,
            max_response_tokens=max_response_tokens,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        in_flight: collections.deque[concurrent.futures.Future[list[Response]]]
        in_flight = collections.deque()
        for start in range(0, n, batch_size):
            if len(in_flight) == concurrency:
                yield from in_flight.popleft().result()
            in_flight.append(executor.submit(request, min(batch_size, n - start)))
        while in_flight:
            yield from in_flight.popleft().result()


def write_corpus_files(
    rows: Iterable[dict[str, Any]],
    directory: pathlib.Path,
    schema: Any,
    *,
    rows_per_file: int = ROWS_PER_FILE,
    row_group_size: int = ROW_GROUP_SIZE,
) -> None:
    """Create `directory` and write `rows` to `part-00000.parquet`, `part-00001.parquet`, ...:
    `rows_per_file` rows per file, in order, in row groups of `row_group_size` rows (zstd).
    Raises ValueError unless a file holds a whole number of row groups, and FileExistsError if
    the directory exists."""
    if rows_per_file % row_group_size != 0:
        raise ValueError("a file holds a whole number of row groups")
    directory.mkdir(parents=True)
    remaining = iter(rows)
    written = 0
    writer: Any = None
    try:
        while group := list(itertools.islice(remaining, row_group_size)):
            if written % rows_per_file == 0:
                if writer is not None:
                    writer.close()
                path = directory / f"part-{written // rows_per_file:05d}.parquet"
                writer = pq.ParquetWriter(path, schema, compression="zstd")
            writer.write_table(pa.Table.from_pylist(group, schema=schema))
            written += len(group)
    finally:
        if writer is not None:
            writer.close()


def sample_and_write_corpus(
    server: str,
    tasks: Sequence[data.Task],
    out_dir: str | os.PathLike[str],
    n: int,
    *,
    stop_token_ids: Sequence[int],
    batch_size: int = 32,
    concurrency: int = 8,
    max_response_tokens: int = original_model.MAX_RESPONSE_TOKENS,
    rows_per_file: int = ROWS_PER_FILE,
    row_group_size: int = ROW_GROUP_SIZE,
) -> dict[str, int]:
    """Sample `n` responses to the prompt of `tasks` (the pairs of one prompt) from `server`,
    label each with the rule of every task, and write them to
    `<out_dir>/corpus/<prompt_id>/` (module docstring). Returns the hit count of each behavior.

    Raises ValueError unless the tasks share one prompt, FileExistsError if the prompt already
    has a corpus directory, and ServerError for an answer that is not complete responses."""
    prompts = {(task.prompt_id, task.prompt_token_ids) for task in tasks}
    if len(prompts) != 1:
        raise ValueError(f"expected the pairs of one prompt, got {len(prompts)} prompts")
    ((prompt_id, prompt_token_ids),) = prompts
    predicates = {task.behavior: rules.parse_rule(task.rule) for task in tasks}
    hits = dict.fromkeys(predicates, 0)

    def rows() -> Iterator[dict[str, Any]]:
        responses = sample_responses(
            server,
            prompt_token_ids,
            stop_token_ids,
            n,
            batch_size=batch_size,
            concurrency=concurrency,
            max_response_tokens=max_response_tokens,
        )
        for row, (token_ids, text) in enumerate(responses):
            labels = {behavior: predicate(text) for behavior, predicate in predicates.items()}
            for behavior, is_hit in labels.items():
                hits[behavior] += is_hit
            stored = {"prompt_id": prompt_id, "row": row, "token_ids": token_ids, "text": text}
            yield stored | labels

    write_corpus_files(
        rows(),
        pathlib.Path(out_dir) / "corpus" / prompt_id,
        corpus_schema(list(predicates)),
        rows_per_file=rows_per_file,
        row_group_size=row_group_size,
    )
    return hits


def main(
    server: Annotated[
        str, typer.Option(help="Base URL of a server with SGLang's /generate interface")
    ],
    prompt_id: Annotated[str, typer.Option(help="One of the dataset's prompts, e.g. wc_1853")],
    responses: Annotated[int, typer.Option(min=1, help="Number of responses to sample")],
    out: Annotated[
        pathlib.Path, typer.Option(help="Dataset directory; corpus/<prompt_id>/ is created in it")
    ],
    data_source: Annotated[
        str | None,
        typer.Option("--data", help="Dataset directory or Hub id (default: the bundled table)"),
    ] = None,
    tokenizer: Annotated[
        str, typer.Option(help="Hub model id or local directory of the model's tokenizer")
    ] = original_model.DEFAULT_MODEL,
    batch_size: Annotated[int, typer.Option(min=1, help="Responses per request")] = 32,
    concurrency: Annotated[int, typer.Option(min=1, help="Requests in flight")] = 8,
) -> None:
    """Sample responses of one prompt from a server, label them and write corpus files."""
    import transformers

    tasks = data.load_tasks(data.BUNDLED_SOURCE if data_source is None else data_source)
    of_prompt = [task for task in tasks if task.prompt_id == prompt_id]
    if not of_prompt:
        raise typer.BadParameter(f"no prompt {prompt_id!r} in the task table")
    revision = original_model.resolve_model_revision(tokenizer)
    vocabulary = transformers.AutoTokenizer.from_pretrained(tokenizer, revision=revision)
    stop_token_ids = [int(vocabulary.convert_tokens_to_ids(t)) for t in original_model.STOP_TOKENS]
    try:
        hits = sample_and_write_corpus(
            server,
            of_prompt,
            out,
            responses,
            stop_token_ids=stop_token_ids,
            batch_size=batch_size,
            concurrency=concurrency,
        )
    except FileExistsError as error:
        raise typer.BadParameter(f"{error.filename} exists; give another directory") from error
    summary = {
        "prompt_id": prompt_id,
        "responses": responses,
        "directory": str(out / "corpus" / prompt_id),
        "behaviors": {
            task.behavior: {
                "hits": hits[task.behavior],
                "hit_fraction": hits[task.behavior] / responses,
                "task_table_probability": task.probability,
            }
            for task in of_prompt
        },
    }
    typer.echo(json.dumps(summary, indent=1))


if __name__ == "__main__":
    typer.run(main)
