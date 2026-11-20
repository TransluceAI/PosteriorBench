"""Loading the benchmark dataset.

The dataset has four parts:

    tasks/tasks.parquet                    the task table: what a method may see
    evaluation/evaluation.parquet          held-out data of the scored pairs; evaluation only
    hits/<prompt_id>/<behavior>.parquet    a scored pair's hits without its references
    corpus/<prompt_id>/part-*.parquet      every response (146 GB); for adding behaviors

The task table and the evaluation file are bundled with the package (`BUNDLED_SOURCE`), so listing
tasks and evaluating a sampler need no download. Any other data source is a local directory with
this layout or a Hugging Face Hub dataset id (such as `Transluce/PosteriorBench`). Hub files are
downloaded one at a time, on first use, into the Hugging Face cache
(`huggingface_hub.hf_hub_download`), at the dataset version `DATASET_REVISION` unless a revision is
given; a local copy of the Hub repository is a valid local source. A single tasks parquet file also
works as a source for `load_tasks`.

The files keep apart what a method may see and what only evaluation reads. `load_tasks` reads the
task table and returns `Task`s: the prompt, the behavior and its rule, and the corpus estimate
p̂(S). `load_evaluation_data` reads the evaluation file and returns a scored pair's
`EvaluationData`: its references with their stored log p, and its fallback response with its
stored log p. Only evaluation code calls it. Corpus hits for construction come from
`corpus.CorpusAccess`, which reads the pair's hit file and charges the hits to the method's ledger.
`iter_corpus` streams a prompt's whole corpus, for defining new behaviors.

`TASKS_SCHEMA`, `EVALUATION_SCHEMA` and `HITS_SCHEMA` are the columns and types of the three kinds
of files the package reads.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
from collections.abc import Iterator, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from posteriorbench import errors

TASKS_FILE = "tasks/tasks.parquet"
EVALUATION_FILE = "evaluation/evaluation.parquet"
HUB_REPO_TYPE = "dataset"
HUB_DATASET = "Transluce/PosteriorBench"
# The dataset version this package version reads: the tag a Hub source is read at unless a
# revision is given, and the version of the two bundled tables.
DATASET_REVISION = "v1.0.0"
# The task table and the evaluation file of that dataset version, shipped inside the package. They
# are part of the dataset and carry its license, not the package's (dataset/README.md).
BUNDLED_SOURCE = pathlib.Path(__file__).resolve().parent / "dataset"

TASKS_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("prompt_id", pa.string()),
        ("prompt", pa.string()),
        ("prompt_token_ids", pa.list_(pa.int32())),
        ("wildchat_conversation_hash", pa.string()),
        ("behavior", pa.string()),
        ("rule", pa.string()),
        ("num_responses", pa.int64()),
        ("hit_count", pa.int64()),
        ("probability", pa.float64()),
        ("scored", pa.bool_()),
        ("benchmark_task", pa.bool_()),
    ]
)
EVALUATION_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        (
            "references",
            pa.list_(
                pa.struct(
                    [
                        ("row", pa.int32()),
                        ("token_ids", pa.list_(pa.int32())),
                        ("text", pa.string()),
                    ]
                )
            ),
        ),
        ("reference_logp", pa.list_(pa.float64())),
        ("fallback_text", pa.string()),
        ("fallback_token_ids", pa.list_(pa.int32())),
        ("fallback_logp", pa.float64()),
    ]
)
# One row per hit, in corpus order (the pair's references are not in the file).
HITS_SCHEMA = pa.schema(
    [
        ("row", pa.int32()),
        ("token_ids", pa.list_(pa.int32())),
        ("reference_duplicate", pa.bool_()),
    ]
)


class DataError(errors.PosteriorBenchError):
    """The dataset does not have the expected layout or contents."""


@dataclasses.dataclass(frozen=True)
class Task:
    """What a method may see: one prompt-behavior pair (one row of the task table).

    `hit_count` of the prompt's `num_responses` corpus responses have the behavior; their share
    `probability` is p̂(S), the corpus estimate the evaluation normalizes with. `scored` is true
    for the scored pairs (at least 500 hits; only they have references and a fallback response,
    which are in the evaluation file). `benchmark_task` is true for the benchmark's tasks, which
    are scored pairs."""

    task_id: str
    prompt_id: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    wildchat_conversation_hash: str
    behavior: str
    rule: str
    num_responses: int
    hit_count: int
    probability: float
    scored: bool
    benchmark_task: bool

    @property
    def log_event_probability(self) -> float:
        """log p̂(S) = log(hit_count / num_responses), the corpus estimate used for normalizing.
        Raises DataError for a pair without hits."""
        if self.hit_count <= 0:
            raise DataError(f"{self.task_id}: no hits, log p̂(S) is undefined")
        return math.log(self.hit_count) - math.log(self.num_responses)

    def require_scored(self) -> None:
        """Raise DataError unless this pair is scored, so that it can be evaluated."""
        if not self.scored:
            raise DataError(
                f"{self.task_id} is not a scored pair (only scored pairs have references and a "
                "fallback response)"
            )


@dataclasses.dataclass(frozen=True)
class Reference:
    """One held-out posterior sample: its corpus row, generated token ids and decoded text."""

    row: int
    token_ids: tuple[int, ...]
    text: str


@dataclasses.dataclass(frozen=True)
class EvaluationData:
    """Held-out data of one scored pair (its row of the evaluation file), read only by evaluation;
    construction code never loads it.

    `references` are the pair's references and `reference_logp` their stored log p under the
    original model, in the same order. `fallback_text` and `fallback_token_ids` are the fallback
    response a, and `fallback_logp` is its stored log p(a). The stored values were computed with
    another runtime than this package's (docs/the_package_and_the_paper.md); evaluation uses
    `fallback_logp` always and `reference_logp` unless asked to recompute it."""

    task_id: str
    references: tuple[Reference, ...]
    reference_logp: tuple[float, ...]
    fallback_text: str
    fallback_token_ids: tuple[int, ...]
    fallback_logp: float


def hits_file(prompt_id: str, behavior: str) -> str:
    """The relative path of a scored pair's hit file inside a data source."""
    return f"hits/{prompt_id}/{behavior}.parquet"


def hub_commit(path: pathlib.Path) -> str | None:
    """The commit a file downloaded from the Hub belongs to (the `snapshots/<commit>/` directory of the
    Hugging Face cache), or None for a file outside the cache."""
    parts = path.parts
    if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts) - 1:
        return parts[parts.index("snapshots") + 1]
    return None


def is_local_source(source: str | os.PathLike[str]) -> bool:
    """True if `source` names a local path rather than a Hub dataset id: a path object, a string that
    exists, or one that starts with "/", "." or "~" or has more than one "/" (a Hub id is "name" or
    "namespace/name")."""
    if not isinstance(source, str):
        return True
    return (
        pathlib.Path(source).expanduser().exists()
        or source.startswith(("/", ".", "~"))
        or source.count("/") > 1
    )


def _local_root(source: str | os.PathLike[str], revision: str | None) -> pathlib.Path | None:
    """The local path of a local source ("~" expanded), or None for a Hub dataset id. Raises
    DataError if a local source does not exist or is given a revision."""
    if not is_local_source(source):
        return None
    path = pathlib.Path(source).expanduser()
    if not path.exists():
        raise DataError(f"{path}: no such directory")
    if revision is not None:
        raise DataError(f"{path} is a local source; a revision applies to Hub dataset ids")
    return path


def _hub_revision(revision: str | None) -> str:
    return revision if revision is not None else DATASET_REVISION


def _hub_file(repo_id: str, filename: str, revision: str) -> pathlib.Path:
    """Download one file of a Hub dataset repository into the Hugging Face cache (once) and
    return its local path. Raises DataError if the repository, the revision or the file does not
    exist; network and access errors propagate (they are OSErrors)."""
    import huggingface_hub
    import huggingface_hub.errors

    try:
        local = huggingface_hub.hf_hub_download(
            repo_id, filename, repo_type=HUB_REPO_TYPE, revision=revision
        )
    except huggingface_hub.errors.RemoteEntryNotFoundError as error:
        raise DataError(f"{repo_id}: no {filename}") from error
    except huggingface_hub.errors.RevisionNotFoundError as error:
        raise DataError(
            f"{repo_id}: no revision {revision} (this package reads dataset version "
            f"{DATASET_REVISION})"
        ) from error
    except huggingface_hub.errors.RepositoryNotFoundError as error:
        raise DataError(
            f"{repo_id!r} is not an existing directory, and the Hub has no dataset of that name "
            "that this machine can read. If the dataset is private, log in with `hf auth login`; "
            "to read a local copy, pass its directory (Benchmark(hits_source=...))."
        ) from error
    return pathlib.Path(str(local))


def resolve_file(
    source: str | os.PathLike[str], relative_path: str, *, revision: str | None = None
) -> pathlib.Path:
    """The local path of one dataset file: a join under a local directory, or a Hub download.

    A local source that is a single file is accepted only as the tasks file. `revision` applies to
    Hub sources only; a Hub source without one is read at DATASET_REVISION. Raises DataError if
    the file or the local directory does not exist."""
    root = _local_root(source, revision)
    if root is None:
        return _hub_file(str(source), relative_path, _hub_revision(revision))
    if root.is_file():
        if relative_path == TASKS_FILE:
            return root
        raise DataError(f"{root} is a single tasks file; {relative_path} needs a directory")
    candidate = root / relative_path
    if not candidate.is_file():
        raise DataError(f"{root}: no {relative_path}")
    return candidate


def describe_source(
    source: str | os.PathLike[str], *, revision: str | None = None
) -> dict[str, str | None]:
    """The data source of results: the source as given, the revision read (DATASET_REVISION for a Hub
    id given none; None for a local directory) and, for the Hub, the commit it resolved to."""
    if is_local_source(source):
        return {"source": str(source), "revision": None, "commit": None}
    return {
        "source": str(source),
        "revision": _hub_revision(revision),
        "commit": hub_commit(resolve_file(source, TASKS_FILE, revision=revision)),
    }


def _require_columns(present: Sequence[str], schema: Any, file: str) -> None:
    """Raise DataError unless `file` has every column of `schema`."""
    missing = [name for name in schema.names if name not in present]
    if missing:
        raise DataError(
            f"{file} has no column {missing}; this package reads dataset version {DATASET_REVISION}"
        )


def _task_from_record(record: dict[str, Any]) -> Task:
    """The `Task` of one row of the task table."""
    return Task(
        task_id=str(record["task_id"]),
        prompt_id=str(record["prompt_id"]),
        prompt=str(record["prompt"]),
        prompt_token_ids=tuple(int(t) for t in record["prompt_token_ids"] or ()),
        wildchat_conversation_hash=str(record["wildchat_conversation_hash"]),
        behavior=str(record["behavior"]),
        rule=str(record["rule"]),
        num_responses=int(record["num_responses"]),
        hit_count=int(record["hit_count"]),
        probability=float(record["probability"]),
        scored=bool(record["scored"]),
        benchmark_task=bool(record["benchmark_task"]),
    )


def load_tasks(source: str | os.PathLike[str], *, revision: str | None = None) -> list[Task]:
    """All prompt-behavior pairs of a data source, read from its task table alone. Raises DataError
    if the table lacks a column of TASKS_SCHEMA or repeats a task id."""
    table = pq.read_table(resolve_file(source, TASKS_FILE, revision=revision))
    _require_columns(table.column_names, TASKS_SCHEMA, TASKS_FILE)
    records = table.to_pylist()
    if len({str(r["task_id"]) for r in records}) != len(records):
        raise DataError("duplicate task ids")
    return [_task_from_record(r) for r in records]


def get_task(tasks: Sequence[Task], task_id: str) -> Task:
    """The task with this id (`<prompt_id>/<behavior>`). Raises DataError if there is none."""
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise DataError(f"unknown task {task_id!r}")


def _log_probabilities(values: Any, what: str, task_id: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    for value in result:
        if not (math.isfinite(value) and value <= 0):
            raise DataError(f"{task_id}: {what} must be finite log probabilities, got {value}")
    return result


def load_evaluation_data(
    source: str | os.PathLike[str], task_id: str, *, revision: str | None = None
) -> EvaluationData:
    """The held-out evaluation data of one scored pair, read from the evaluation file alone: its
    references with their stored log p, and its fallback response with its stored log p(a).

    For evaluation code only; construction code never calls this. Raises DataError if the file has
    no row for `task_id` (only scored pairs have one), or if the row does not hold references, one
    finite log p for each of them, and a fallback response with a finite log p."""
    path = resolve_file(source, EVALUATION_FILE, revision=revision)
    _require_columns(pq.read_schema(path).names, EVALUATION_SCHEMA, EVALUATION_FILE)
    rows = pq.read_table(path, filters=[("task_id", "=", task_id)]).to_pylist()
    if len(rows) != 1:
        raise DataError(
            f"{task_id}: {len(rows)} rows in {EVALUATION_FILE}, expected 1 (only scored pairs "
            "have evaluation data)"
        )
    (record,) = rows
    references = tuple(
        Reference(int(r["row"]), tuple(int(t) for t in r["token_ids"]), str(r["text"]))
        for r in record["references"] or ()
    )
    if not references:
        raise DataError(f"{task_id}: no references")
    if record["fallback_text"] is None or record["fallback_token_ids"] is None:
        raise DataError(f"{task_id}: no fallback response")
    reference_logp = _log_probabilities(record["reference_logp"] or (), "reference_logp", task_id)
    if len(reference_logp) != len(references):
        raise DataError(
            f"{task_id}: {len(reference_logp)} reference log p for {len(references)} references"
        )
    (fallback_logp,) = _log_probabilities([record["fallback_logp"]], "fallback_logp", task_id)
    return EvaluationData(
        task_id=task_id,
        references=references,
        reference_logp=reference_logp,
        fallback_text=str(record["fallback_text"]),
        fallback_token_ids=tuple(int(t) for t in record["fallback_token_ids"]),
        fallback_logp=fallback_logp,
    )


def corpus_files(
    source: str | os.PathLike[str], prompt_id: str, *, revision: str | None = None
) -> list[pathlib.Path]:
    """The local paths of one prompt's corpus files, in file order; from a Hub source they are
    downloaded first (about 7 GB per prompt), at DATASET_REVISION unless `revision` is given.
    Raises DataError if the source has no corpus files for the prompt."""
    root = _local_root(source, revision)
    if root is not None:
        files = sorted((root / "corpus" / prompt_id).glob("*.parquet"))
        if not files:
            raise DataError(f"{root}: no corpus files for {prompt_id}")
        return files
    import huggingface_hub

    read = _hub_revision(revision)
    prefix = f"corpus/{prompt_id}/"
    names = sorted(
        name
        for name in huggingface_hub.list_repo_files(
            str(source), repo_type=HUB_REPO_TYPE, revision=read
        )
        if name.startswith(prefix) and name.endswith(".parquet")
    )
    if not names:
        raise DataError(f"{source}: no corpus files for {prompt_id}")
    return [_hub_file(str(source), name, read) for name in names]


def iter_corpus(
    source: str | os.PathLike[str],
    prompt_id: str,
    columns: Sequence[str] | None = None,
    *,
    revision: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream one prompt's corpus rows (all columns unless `columns` is given), in file order.
    Rows are in random order within a prompt, so any prefix is a uniform sample.

    For defining new behaviors. The corpus holds every response, the references of every pair
    included, and reading it charges nothing: a method that is scored on the benchmark builds its
    sampler from `corpus.CorpusAccess` instead. From the Hub this downloads the prompt's whole
    corpus (`corpus_files`)."""
    for file in corpus_files(source, prompt_id, revision=revision):
        handle = pq.ParquetFile(file)
        for batch in handle.iter_batches(columns=None if columns is None else list(columns)):
            yield from batch.to_pylist()
