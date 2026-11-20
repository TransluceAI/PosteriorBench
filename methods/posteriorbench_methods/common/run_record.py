"""Run records: where a run writes its files, and what they hold (README, "Records").

    <out>/<method>/<setting>/<task id with "/" -> "__">/rep<r>/   one record (`write_record`)
        result.json, result_recomputed.json, evaluation_details.json, evaluation_hits.json,
        construction.json, config.json, draws/<name>.jsonl
    <out>/training_runs/<method>/<task>/rep<r>/   a training run (`training_run`)
    <out>/cache/<family>/<task>/                  work that the replicates of a task share
    <out>/reselected/<rule>/                      records of another rule, in the same layout
    <out>/summary.jsonl                           one line per record written

Every file is strict JSON: a non-finite float is written as the string "inf", "-inf" or "nan".
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import posteriorbench
from posteriorbench import data, ledger, protocol
from posteriorbench import results as pb_results
from posteriorbench import sampler

import posteriorbench_methods

RESULT_FILE = "result.json"
RESULT_RECOMPUTED_FILE = "result_recomputed.json"
EVALUATION_DETAILS_FILE = "evaluation_details.json"
EVALUATION_HITS_FILE = "evaluation_hits.json"
CONSTRUCTION_FILE = "construction.json"
CONFIG_FILE = "config.json"
DRAWS_DIR = "draws"
EVALUATION_DRAWS_NAME = "evaluation"
SUMMARY_FILE = "summary.jsonl"
TRAINING_RUNS_DIR = "training_runs"
CACHE_DIR = "cache"
RESELECTED_DIR = "reselected"
ONE_PERCENT_RULE = "threshold-1pct"
# One set of evaluation draws gives two results (`Benchmark.evaluate_both`): with the log p of the
# references shipped in the dataset (the benchmark's default) and recomputed by the original model.
REFERENCE_LOGP = pb_results.REFERENCE_LOGP_CHOICES  # ("shipped", "recomputed")
# Keys of a result's `details` that go to draws/evaluation.jsonl, not to evaluation_details.json.
DRAW_DETAIL_KEYS = ("draw_token_ids", "draw_hits")
# The draws files of the submission rule and of composition, and the keys that each of their rows
# has in every family; a family may add its own. `rank` is the rank of the threshold test that drew
# the row, `seed` the seed of the call that drew it and `draw_index` its place in that call or test.
# `logq` and `logp` are None where the method did not compute them (log p of a miss, for example).
DRAW_KEYS: dict[str, tuple[str, ...]] = {
    "selection": ("candidate_id", "draw_index", "token_ids", "hit", "logq", "logp", "seed"),
    "threshold_tests": ("candidate_id", "rank", "draw_index", "token_ids", "hit", "seed"),
    "composition_selection": ("setting", "draw_index", "token_ids", "hit", "logq", "logp", "seed"),
    "composition_threshold_tests": ("setting", "rank", "draw_index", "token_ids", "hit", "seed"),
}


class RecordError(ValueError):
    """A record that would break the layout, such as a draw row without token ids."""


def task_dir_name(task_id: str) -> str:
    """The directory name of a task id: "wc_1853/goblin_mention" -> "wc_1853__goblin_mention"."""
    return task_id.replace("/", "__")


def run_dir(
    out: pathlib.Path, method: str, setting: str, task_id: str, replicate: int
) -> pathlib.Path:
    """<out>/<method>/<setting>/<task id with "/" -> "__">/rep<r>/"""
    return out / method / setting / task_dir_name(task_id) / f"rep{replicate}"


def training_run_dir(
    out: pathlib.Path, method: str, task_id: str, replicate: int, *, budget: int | None = None
) -> pathlib.Path:
    """<out>/training_runs/<method>/<task>/rep<r>/, the training run of a method
    (`training_run`), or its subdirectory B=<budget>/ for a method that trains separately at each
    budget (SFT)."""
    directory = out / TRAINING_RUNS_DIR / method / task_dir_name(task_id) / f"rep{replicate}"
    return directory if budget is None else directory / setting_label(budget=budget)


def cache_dir(out: pathlib.Path, family: str, task_id: str) -> pathlib.Path:
    """<out>/cache/<family>/<task>/: work that the replicates of one task share (for example the
    selection draws of Steering)."""
    return out / CACHE_DIR / family / task_dir_name(task_id)


def one_percent_root(out: pathlib.Path) -> pathlib.Path:
    """<out>/reselected/threshold-1pct/, the output root of the records at the 1% threshold, so
    that they do not replace those at 10%."""
    return out / RESELECTED_DIR / ONE_PERCENT_RULE


def setting_label(
    *, budget: int | None = None, update_cap: int | None = None, pool_size: int | None = None
) -> str:
    """The setting in a record's path and in `EvaluationResult.setting`: "B=<budget>" (a training
    budget, in passes), "u=<update cap>", "K=<pool size>", or "none" for a method without a
    setting. Give at most one argument."""
    given = [
        (prefix, value)
        for prefix, value in (("B", budget), ("u", update_cap), ("K", pool_size))
        if value is not None
    ]
    if len(given) > 1:
        raise RecordError(f"one setting at a time, got {given}")
    if not given:
        return "none"
    prefix, value = given[0]
    if isinstance(value, bool) or value < 1:
        raise RecordError(f"a setting value is a positive integer, got {value!r}")
    return f"{prefix}={value}"


def jsonable(value: Any) -> Any:
    """A strict-JSON copy of `value`: non-finite floats become "inf" / "-inf" / "nan", numpy
    scalars and arrays become Python numbers and lists, tuples become lists, dict keys strings."""
    if not isinstance(value, (str, bytes, dict, list, tuple)) and hasattr(value, "tolist"):
        value = value.tolist()  # numpy arrays and scalars, torch tensors
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        items: dict[Any, Any] = value  # pyright: ignore[reportUnknownVariableType]
        return {str(key): jsonable(item) for key, item in items.items()}
    if isinstance(value, (list, tuple)):
        elements: Sequence[Any] = value  # pyright: ignore[reportUnknownVariableType]
        return [jsonable(item) for item in elements]
    return value


def _write_atomically(path: pathlib.Path, lines: Iterable[str]) -> None:
    """Write `lines` to a temporary file next to `path`, then rename it to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.writelines(lines)
    os.replace(temporary, path)


def write_json(path: pathlib.Path, value: Any) -> None:
    """Write `value` as strict JSON (indent 1, keys in insertion order), atomically. Creates the
    parent directories."""
    text = json.dumps(jsonable(value), indent=1, ensure_ascii=False, allow_nan=False)
    _write_atomically(path, [text + "\n"])


def write_jsonl(path: pathlib.Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write one strict-JSON object per line, atomically, one row at a time."""
    _write_atomically(
        path,
        (
            json.dumps(jsonable(dict(row)), ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
    )


def read_json(path: pathlib.Path) -> Any:
    """The value of a JSON file."""
    return json.loads(path.read_text())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Rows of a JSON-lines file. Splits on "\\n" only: generated texts may contain U+2028,
    U+2029 or U+0085, which str.splitlines would treat as line breaks."""
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write_draws(directory: pathlib.Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write draws/<name>.jsonl of a record directory. Every row needs "token_ids" (text alone
    cannot be scored again); a row of a file named in `DRAW_KEYS` needs those keys, which are
    written first."""
    keys = DRAW_KEYS.get(name, ())
    ordered: list[dict[str, Any]] = []
    for row in rows:
        if "token_ids" not in row:
            raise RecordError(f"draws/{name}: every stored draw carries its token ids")
        missing = [key for key in keys if key not in row]
        if missing:
            raise RecordError(f"draws/{name}: a row has no {missing}; its keys are {keys}")
        ordered.append({**{key: row[key] for key in keys}, **row})
    write_jsonl(directory / DRAWS_DIR / f"{name}.jsonl", ordered)


def ledger_record(spend: ledger.Ledger | ledger.LedgerTotals) -> dict[str, Any]:
    """The "ledger" entry of construction.json: the totals, the charges made to this ledger (in
    order) and the totals it was resumed from (None if it started empty). A `LedgerTotals` has no
    list of charges: its charges are []."""
    if isinstance(spend, ledger.LedgerTotals):
        return {"totals": spend.to_dict(), "charges": [], "resumed_from": None}
    resumed = spend.resumed_from
    return {
        "totals": spend.totals().to_dict(),
        "charges": [
            {
                "category": charge.category.value,
                "inferences": charge.inferences,
                "budget": charge.budget,
                "note": charge.note,
            }
            for charge in spend.charges()
        ],
        "resumed_from": None if resumed is None else resumed.to_dict(),
    }


def construction_record(
    *,
    method: str,
    setting: str,
    task_id: str,
    replicate: int,
    status: str,
    spend: ledger.Ledger | ledger.LedgerTotals | None,
    submitted: Mapping[str, Any] | None = None,
    ranking: Sequence[Mapping[str, Any]] | None = None,
    submission: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """construction.json. `status` is the submission's ("passed", "best_observed",
    "no_candidate") or why there was none ("no_sampler", "not_composed", "no_selection",
    "own_rule"); `ledger` is None for a run that kept no ledger."""
    return {
        "method": method,
        "setting": setting,
        "task_id": task_id,
        "replicate": replicate,
        "status": status,
        "submitted": None if submitted is None else dict(submitted),
        "ranking": None if ranking is None else [dict(entry) for entry in ranking],
        "submission": None if submission is None else dict(submission),
        "ledger": None if spend is None else ledger_record(spend),
        "details": {} if details is None else dict(details),
    }


def settings_json(settings: Any) -> dict[str, Any]:
    """A dataclass of settings (or a mapping) as a JSON-ready dict."""
    if dataclasses.is_dataclass(settings) and not isinstance(settings, type):
        return jsonable(dataclasses.asdict(settings))
    if isinstance(settings, Mapping):
        mapping: Mapping[Any, Any] = settings  # pyright: ignore[reportUnknownVariableType]
        return jsonable(dict(mapping))
    raise RecordError(f"settings are a dataclass instance or a mapping, got {type(settings)}")


def versions() -> dict[str, str]:
    """The versions of the benchmark package and of the methods package."""
    return {
        "posteriorbench": posteriorbench.__version__,
        "posteriorbench_methods": posteriorbench_methods.__version__,
    }


def config_record(
    *,
    family: str,
    method: str,
    setting: str,
    task_id: str,
    replicate: int,
    settings: Any,
    options: Mapping[str, Any],
    scorer: protocol.Scorer | None,
) -> dict[str, Any]:
    """config.json; `scorer` is `Scorer.describe()` of the original model, None without one."""
    return {
        "family": family,
        "method": method,
        "setting": setting,
        "task_id": task_id,
        "replicate": replicate,
        "settings": settings_json(settings),
        "options": jsonable(dict(options)),
        "versions": versions(),
        "scorer": None if scorer is None else scorer.describe(),
    }


def evaluation_draw_rows(result: posteriorbench.EvaluationResult) -> list[dict[str, Any]]:
    """The evaluation draws of a result made with keep_details=True, one row per draw:
    draw_index, token_ids, hit, and for a hit that the reverse KL is averaged over, logq, logp and
    scored_for_reverse_kl = True."""
    details = result.details
    if details is None:
        raise RecordError("the result was evaluated without keep_details")
    token_ids = details.get("draw_token_ids") or []
    hits = details.get("draw_hits") or []
    scored = [int(i) for i in details.get("scored_hit_indices") or []]
    hit_logq = details.get("hit_logq") or []
    hit_logp = details.get("hit_logp") or []
    if not len(scored) == len(hit_logq) == len(hit_logp):
        raise RecordError("scored hits, their log q and their log p differ in number")
    by_index = {
        index: (float(q), float(p)) for index, q, p in zip(scored, hit_logq, hit_logp, strict=True)
    }
    rows: list[dict[str, Any]] = []
    for index, (ids, hit) in enumerate(zip(token_ids, hits, strict=True)):
        row: dict[str, Any] = {"draw_index": index, "token_ids": list(ids), "hit": bool(hit)}
        if index in by_index:
            row["logq"], row["logp"] = by_index[index]
            row["scored_for_reverse_kl"] = True
        rows.append(row)
    return rows


@dataclasses.dataclass(frozen=True)
class HitScores:
    """log q and log p of every hit among the evaluation draws of a sampler, by increasing
    `draw_index`: evaluation_hits.json. The benchmark's evaluation computes them only for the hits
    that its reverse KL is averaged over, at most 128, and for none of the hits of a sampler below
    the threshold; an importance-sampling estimate of p(S) needs them all."""

    draw_index: tuple[int, ...]
    logq: tuple[float, ...]
    logp: tuple[float, ...]

    def to_json(self) -> dict[str, list[Any]]:
        return {name: list(values) for name, values in dataclasses.asdict(self).items()}

    @classmethod
    def from_json(cls, value: Mapping[str, Sequence[Any]]) -> HitScores:
        """The inverse of `to_json`, for a file read with `read_json` (a non-finite value is a
        string there)."""
        return cls(
            draw_index=tuple(int(index) for index in value["draw_index"]),
            logq=tuple(float(q) for q in value["logq"]),
            logp=tuple(float(p) for p in value["logp"]),
        )


def score_every_hit(
    task: data.Task,
    candidate: sampler.Sampler,
    scorer: protocol.Scorer,
    result: posteriorbench.EvaluationResult,
) -> HitScores:
    """log q and log p of every hit among the evaluation draws of `result`, an evaluation of
    `candidate` made with keep_details=True. The hits that the evaluation scored keep its values;
    the others are scored here by the sampler and the original model. Nothing is drawn, and
    evaluation is not construction cost, so nothing is charged."""
    details = result.details
    if details is None:
        raise RecordError("the result was evaluated without keep_details")
    token_ids = details.get("draw_token_ids") or []
    hits = [index for index, hit in enumerate(details.get("draw_hits") or []) if hit]
    scored: dict[int, tuple[float, float]] = {
        int(index): (float(q), float(p))
        for index, q, p in zip(
            details.get("scored_hit_indices") or [],
            details.get("hit_logq") or [],
            details.get("hit_logp") or [],
            strict=True,
        )
    }
    others = [index for index in hits if index not in scored]
    if others:
        responses = [token_ids[index] for index in others]
        logq = candidate.logprobs(responses)
        logp = scorer.logprobs(task.prompt_token_ids, responses)
        if not len(logq) == len(logp) == len(others):
            raise RecordError(
                f"{len(logq)} log q and {len(logp)} log p for {len(others)} hits of the evaluation"
            )
        scored.update(
            (index, (float(q), float(p))) for index, q, p in zip(others, logq, logp, strict=True)
        )
    return HitScores(
        draw_index=tuple(hits),
        logq=tuple(scored[index][0] for index in hits),
        logp=tuple(scored[index][1] for index in hits),
    )


def _result_json(result: posteriorbench.EvaluationResult) -> dict[str, Any]:
    value = result.to_dict()
    value.pop("details", None)
    return value


def write_record(
    directory: pathlib.Path,
    *,
    results: Mapping[str, posteriorbench.EvaluationResult],
    construction: Mapping[str, Any],
    config: Mapping[str, Any],
    draws: Mapping[str, Sequence[Mapping[str, Any]]],
    hit_scores: HitScores | None = None,
) -> None:
    """Write one record directory. `results` maps "shipped" (required) and "recomputed" (absent
    for a run without a sampler) to result.json and result_recomputed.json; their details, without
    the draws, go to evaluation_details.json, and the draws come in `draws`. `hit_scores` go to
    evaluation_hits.json. result.json is written last, so a record that has one is complete."""
    if "shipped" not in results:
        raise RecordError("a record needs the result with the shipped reference log p")
    unknown = sorted(set(results) - set(REFERENCE_LOGP))
    if unknown:
        raise RecordError(f"results are keyed by {REFERENCE_LOGP}, got {unknown}")
    for name, rows in draws.items():
        write_draws(directory, name, rows)
    write_json(directory / CONSTRUCTION_FILE, construction)
    write_json(directory / CONFIG_FILE, config)
    details = {
        reference_logp: {k: v for k, v in result.details.items() if k not in DRAW_DETAIL_KEYS}
        for reference_logp, result in results.items()
        if result.details is not None
    }
    if details:
        write_json(directory / EVALUATION_DETAILS_FILE, details)
    if hit_scores is not None:
        write_json(directory / EVALUATION_HITS_FILE, hit_scores.to_json())
    if "recomputed" in results:
        write_json(directory / RESULT_RECOMPUTED_FILE, _result_json(results["recomputed"]))
    write_json(directory / RESULT_FILE, _result_json(results["shipped"]))


@dataclasses.dataclass(frozen=True)
class RunSummary:
    """One written record, as a line of <out>/summary.jsonl and of the command's output.
    `record` is the record directory, relative to the output root."""

    method: str
    setting: str
    task_id: str
    replicate: int
    kind: str
    fallback: bool
    native_hit_rate: float
    capped_forward_kl: float
    reverse_kl: float
    construction_inferences: float
    budget_passes: float | None
    record: str

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict (non-finite floats as strings)."""
        return jsonable(dataclasses.asdict(self))


def summarize(
    result: posteriorbench.EvaluationResult, *, out: pathlib.Path, directory: pathlib.Path
) -> RunSummary:
    """The summary line of the record of `result` written at `directory` under `out`."""
    try:
        record = str(directory.relative_to(out))
    except ValueError:
        record = str(directory)
    assert result.method is not None and result.setting is not None, "records carry labels"
    return RunSummary(
        method=result.method,
        setting=result.setting,
        task_id=result.task_id,
        replicate=result.replicate,
        kind=result.kind,
        fallback=result.fallback,
        native_hit_rate=result.native_hit_rate,
        capped_forward_kl=result.capped_forward_kl,
        reverse_kl=result.reverse_kl,
        construction_inferences=result.construction_inferences,
        budget_passes=result.budget_passes,
        record=record,
    )


def append_summary(out: pathlib.Path, summary: RunSummary) -> None:
    """Append one line to <out>/summary.jsonl, in one write call so that concurrent runs into the
    same output root do not interleave within a line."""
    out.mkdir(parents=True, exist_ok=True)
    line = json.dumps(summary.to_json(), ensure_ascii=False, allow_nan=False) + "\n"
    with (out / SUMMARY_FILE).open("a") as handle:
        handle.write(line)


@dataclasses.dataclass(frozen=True)
class Recorded:
    """What writing one record returns. `results` holds "shipped", and "recomputed" if there is
    one; `hit_scores` is None for a record without a sampler."""

    results: dict[str, posteriorbench.EvaluationResult]
    directory: pathlib.Path
    construction: Mapping[str, Any]
    summary: RunSummary
    hit_scores: HitScores | None = None

    @property
    def shipped(self) -> posteriorbench.EvaluationResult:
        """The result with the shipped reference log p, the benchmark's default."""
        return self.results["shipped"]


def relabel(
    results: Mapping[str, posteriorbench.EvaluationResult],
    *,
    method: str,
    setting: str,
    spend: ledger.Ledger | ledger.LedgerTotals | None,
) -> dict[str, posteriorbench.EvaluationResult]:
    """An evaluation recorded again for another method or setting, such as a composed method whose
    chosen setting is the plain one. A result costed by a ledger takes the cost of `spend`; the
    result of a run without a sampler costed at its budget keeps that cost, and `spend` may then
    be None."""
    totals = spend.totals() if isinstance(spend, ledger.Ledger) else spend
    relabeled: dict[str, posteriorbench.EvaluationResult] = {}
    for reference_logp, result in results.items():
        if result.cost_rule == "ledger":
            if totals is None:
                raise RecordError("an evaluation that is costed by a ledger needs one")
            relabeled[reference_logp] = dataclasses.replace(
                result,
                method=method,
                setting=setting,
                construction_inferences=totals.inferences,
                budget_passes=totals.budget,
                ledger=totals.to_dict(),
            )
        else:
            relabeled[reference_logp] = dataclasses.replace(result, method=method, setting=setting)
    return relabeled
