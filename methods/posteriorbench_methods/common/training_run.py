"""A training run on disk (README, "Training runs"): its checkpoints, their selection draws, the
candidates at each budget, which checkpoint files are kept, and how training continues.

    index.jsonl                       one `Checkpoint` per line, in the order saved
    checkpoints/<checkpoint_id>.pt    {"tensors": {name: tensor}}: the trained tensors
    selection/<checkpoint_id>.jsonl   its selection draws (token ids, hit, log q, log p)
    training/update_<NNNNNN>.jsonl    the draws of each update (the family decides what a row holds)
    settings.json                     the settings that the training was started with
    resume.pt                         what training needs to continue (`save_resume`)
    done.json                         written when training stops (`Done`)

A run that finds done.json uses the training run as it is if it holds every checkpoint the run
needs (`serves`); otherwise it removes done.json, loads resume.pt and trains on. A directory with
resume.pt and no done.json was interrupted and continues the same way. If the family's loop decides
whether to stop at the start of an update and draws each update from seeds of its own, training in
pieces, in whatever order the budgets are asked for, runs the same updates with the same seeds as
training in one go, and saves the same checkpoints wherever the device's arithmetic repeats.
resume.pt is written at every checkpoint and when training stops, and deleted when the training
run is complete.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import random
import shutil
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch

from posteriorbench_methods.common import errors, run_context, run_record, submission

INDEX = "index.jsonl"
SETTINGS = "settings.json"
RESUME = "resume.pt"
DONE = "done.json"
DoneStatus = Literal["complete", "stopped"]


class TrainingRunError(errors.MethodsError, RuntimeError):
    """A training run on disk cannot be used: it was started with other settings, or a checkpoint
    file it needs was deleted."""


class BudgetError(errors.MethodsError, ValueError):
    """A training budget cannot be closed: it does not pay for its threshold tests, the budget
    count is beyond its limit already, or the checkpoint that would close it takes it there."""


class CandidatesError(errors.MethodsError, ValueError):
    """The checkpoints of a training run do not give the candidates that were asked for: the budget
    is too small, the update cap was not saved, or the index is inconsistent."""


def checkpoint_id(update: int) -> str:
    """The id of the checkpoint saved after `update` updates: "update_000128". GDC, which saves
    two kinds of checkpoint, puts the kind before it ("pi_update_000128")."""
    return f"update_{update:06d}"


def checkpoint_path(directory: pathlib.Path, checkpoint: str) -> pathlib.Path:
    return directory / "checkpoints" / f"{checkpoint}.pt"


def selection_draws_path(directory: pathlib.Path, checkpoint: str) -> pathlib.Path:
    return directory / "selection" / f"{checkpoint}.jsonl"


def training_draws_path(directory: pathlib.Path, update: int) -> pathlib.Path:
    return directory / "training" / f"update_{update:06d}.jsonl"


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """One saved checkpoint: a line of index.jsonl.

    update: the number of updates after which it was saved.
    budget_spent, ledger: the budget count and the ledger totals when it was saved, its own
        selection draws included.
    sampler_id: the sha256 of its trained tensors (`sampler_id`). Checkpoints with the same
        weights are ranked once, and composition reads it to tell the original model.
    boundary_for: the budgets that this checkpoint closes, as the last checkpoint within each.
    forced: whether it was saved only to close a budget, off the method's schedule of checkpoints.
    extra: the family's own fields."""

    checkpoint_id: str
    update: int
    budget_spent: float
    ledger: dict[str, Any]
    sampler_id: str
    selection_draws: int
    selection_hits: int
    boundary_for: tuple[int, ...] = ()
    forced: bool = False
    extra: dict[str, Any] = dataclasses.field(default_factory=lambda: {})

    @property
    def hit_fraction(self) -> float:
        """selection_hits / selection_draws."""
        return self.selection_hits / self.selection_draws

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"boundary_for": list(self.boundary_for)}

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Checkpoint:
        return cls(
            checkpoint_id=str(row["checkpoint_id"]),
            update=int(row["update"]),
            budget_spent=float(row["budget_spent"]),
            ledger=dict(row["ledger"]),
            sampler_id=str(row["sampler_id"]),
            selection_draws=int(row["selection_draws"]),
            selection_hits=int(row["selection_hits"]),
            boundary_for=tuple(int(b) for b in row.get("boundary_for", ())),
            forced=bool(row.get("forced", False)),
            extra=dict(row.get("extra", {})),
        )


def write_index(directory: pathlib.Path, checkpoints: Sequence[Checkpoint]) -> None:
    """Rewrite index.jsonl (atomically)."""
    run_record.write_jsonl(directory / INDEX, [row.to_json() for row in checkpoints])


def read_index(directory: pathlib.Path) -> list[Checkpoint]:
    """The checkpoints of index.jsonl, in the order saved; [] without the file."""
    path = directory / INDEX
    return (
        [Checkpoint.from_json(row) for row in run_record.read_jsonl(path)] if path.exists() else []
    )


def mark_boundary(checkpoints: Sequence[Checkpoint], budget: int) -> list[Checkpoint]:
    """`checkpoints` with `budget` added to the `boundary_for` of the last one."""
    if not checkpoints:
        raise CandidatesError("a budget is closed by a checkpoint, and there is none")
    last = checkpoints[-1]
    return [
        *checkpoints[:-1],
        dataclasses.replace(last, boundary_for=(*last.boundary_for, budget)),
    ]


def sampler_id(tensors: Mapping[str, torch.Tensor]) -> str:
    """The sha256 over the tensors in the order of their names: of each, its name, dtype, shape
    (as a JSON list) and bytes. Equal weights give equal ids."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        if tensor.numel():
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_tensors(model: Any) -> dict[str, torch.Tensor]:
    """Copies, on the CPU, of the model's parameters that require a gradient, by name."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_tensors(model: Any, tensors: Mapping[str, torch.Tensor]) -> None:
    """Copy `tensors` into the model's parameters that require a gradient (changes the model).
    Raises TrainingRunError if the names differ."""
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(parameters) != set(tensors):
        raise TrainingRunError("the saved tensors do not match the model's trainable parameters")
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(tensors[name].to(value.device))


def save_checkpoint(
    directory: pathlib.Path,
    checkpoint: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    earlier: Sequence[Checkpoint] = (),
) -> str:
    """Write checkpoints/<checkpoint>.pt (atomically) and return the `sampler_id` of the tensors.
    If one of the `earlier` checkpoints has the same tensors, the file is a hard link to its file
    (a copy where links are not possible)."""
    identity = sampler_id(tensors)
    target = checkpoint_path(directory, checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    same = (
        checkpoint_path(directory, row.checkpoint_id)
        for row in earlier
        if row.sampler_id == identity
    )
    source = next((path for path in same if path.exists()), None)
    if source is not None:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
    else:
        torch.save({"tensors": dict(tensors)}, temporary)
    temporary.replace(target)
    return identity


def load_checkpoint(directory: pathlib.Path, checkpoint: str) -> dict[str, torch.Tensor]:
    """The tensors of a saved checkpoint, on the CPU. Raises TrainingRunError if an earlier run
    deleted the file (`prune_checkpoints`)."""
    path = checkpoint_path(directory, checkpoint)
    if not path.exists():
        raise TrainingRunError(
            f"{path} was deleted when an earlier run ended; run again in a new output directory, "
            "with --keep-checkpoints all if other budgets or `reselect` will use this training run"
        )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return dict(saved["tensors"])


def write_selection_draws(
    directory: pathlib.Path, checkpoint: str, rows: Iterable[Mapping[str, Any]]
) -> None:
    """Write the selection draws of a checkpoint, one row per draw."""
    run_record.write_jsonl(selection_draws_path(directory, checkpoint), list(rows))


def read_selection_draws(directory: pathlib.Path, checkpoint: str) -> list[dict[str, Any]]:
    """The selection draws of a checkpoint, in the order written."""
    return run_record.read_jsonl(selection_draws_path(directory, checkpoint))


def selection_rows(
    directory: pathlib.Path, checkpoints: Sequence[Checkpoint]
) -> list[dict[str, Any]]:
    """The selection draws of `checkpoints` as the rows of a record's draws/selection.jsonl: the
    rows of each checkpoint's file, each with the checkpoint's id as its `candidate_id`."""
    return [
        {**row, "candidate_id": checkpoint.checkpoint_id}
        for checkpoint in checkpoints
        for row in read_selection_draws(directory, checkpoint.checkpoint_id)
    ]


def candidate_limit(budget: int, reserve: int = submission.THRESHOLD_TEST_RESERVE) -> int:
    """The largest budget count at which a checkpoint is a candidate at `budget`: the budget less
    the `reserve` for its threshold tests. Raises BudgetError for a budget that does not pay for
    them."""
    if budget <= reserve:
        raise BudgetError(f"budget {budget} does not pay for the threshold tests of a run")
    return budget - reserve


def check_chronological(checkpoints: Sequence[Checkpoint]) -> None:
    """Raise CandidatesError unless `update` increases strictly and `budget_spent` never decreases
    along `checkpoints`."""
    updates = [row.update for row in checkpoints]
    spent = [row.budget_spent for row in checkpoints]
    if updates != sorted(set(updates)):
        raise CandidatesError("the updates of the checkpoints do not increase strictly")
    if spent != sorted(spent):
        raise CandidatesError("the budget spent decreases from one checkpoint to the next")


def candidates_at_budget(
    checkpoints: Sequence[Checkpoint], budget: int, *, reserve: int, require_boundary: bool
) -> list[Checkpoint]:
    """The candidates of the run at budget B: every checkpoint with budget_spent <= B - `reserve`,
    where `reserve` is what the run sets aside for its threshold tests. With `require_boundary`,
    for a budget that the training itself closed, raises CandidatesError unless the checkpoint
    that closed it is among them."""
    check_chronological(checkpoints)
    limit = candidate_limit(budget, reserve)
    candidates = [row for row in checkpoints if row.budget_spent <= limit]
    if require_boundary and sum(budget in row.boundary_for for row in candidates) != 1:
        raise CandidatesError(f"no candidate at budget {budget} is the checkpoint that closed it")
    return candidates


def candidates_at_update_cap(
    checkpoints: Sequence[Checkpoint], update_cap: int, *, allow_boundary: bool = True
) -> list[Checkpoint]:
    """The candidates of the run at an update cap: every checkpoint saved in the first `update_cap`
    updates. Raises CandidatesError unless one was saved at that update, and with
    `allow_boundary=False` if one of them closed a budget or was saved only for that."""
    check_chronological(checkpoints)
    candidates = [row for row in checkpoints if row.update <= update_cap]
    if not candidates or candidates[-1].update != update_cap:
        raise CandidatesError(f"this training run saved no checkpoint at update {update_cap}")
    if not allow_boundary and any(row.boundary_for or row.forced for row in candidates):
        raise CandidatesError(
            f"a candidate at update cap {update_cap} is a checkpoint that closed a budget"
        )
    return candidates


@dataclasses.dataclass(frozen=True)
class BudgetGroup:
    """Budgets that have the same candidates: the first `count` checkpoints of the training run,
    the last of which is `last_checkpoint` (saved after `update` updates, at `budget_spent`)."""

    last_checkpoint: str
    update: int
    budget_spent: float
    count: int
    budgets: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"budgets": list(self.budgets)}


def group_budgets(
    checkpoints: Sequence[Checkpoint], budgets: Sequence[int], *, reserve: int
) -> list[BudgetGroup]:
    """`budgets` grouped by their candidates (`candidates_at_budget` without `require_boundary`),
    from the fewest candidates to the most. A budget without a candidate is left out."""
    check_chronological(checkpoints)
    by_count: dict[int, list[int]] = {}
    for budget in sorted(budgets):
        limit = candidate_limit(budget, reserve)
        count = sum(1 for row in checkpoints if row.budget_spent <= limit)
        if count:
            by_count.setdefault(count, []).append(budget)
    out: list[BudgetGroup] = []
    for count, members in sorted(by_count.items()):
        row = checkpoints[count - 1]
        out.append(
            BudgetGroup(row.checkpoint_id, row.update, row.budget_spent, count, tuple(members))
        )
    return out


def samplers_to_keep(
    mode: run_context.KeepCheckpoints, *, testable: Iterable[str], returned: Iterable[str]
) -> set[str] | None:
    """The sampler ids of the checkpoints whose files a run keeps under `mode`
    (`run_context.RunOptions.keep_checkpoints`); None keeps all."""
    if mode not in run_context.KEEP_CHECKPOINTS:
        raise ValueError(
            f"keep_checkpoints must be one of {run_context.KEEP_CHECKPOINTS}, got {mode!r}"
        )
    if mode == "all":
        return None
    keep = set(returned)
    if mode == "testable":
        keep |= set(testable)
    return keep


def prune_checkpoints(
    directory: pathlib.Path, checkpoints: Sequence[Checkpoint], keep: set[str] | None
) -> list[str]:
    """Delete the file of every checkpoint whose sampler id is not in `keep`; None deletes
    nothing. Returns the ids of the checkpoints whose files were deleted."""
    if keep is None:
        return []
    deleted: list[str] = []
    for row in checkpoints:
        path = checkpoint_path(directory, row.checkpoint_id)
        if row.sampler_id not in keep and path.exists():
            path.unlink()
            deleted.append(row.checkpoint_id)
    return deleted


def capture_rng_state() -> dict[str, Any]:
    """The states of the global random generators of torch (CPU and CUDA), numpy and Python."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Set the global random generators to a state from `capture_rng_state`."""
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


@dataclasses.dataclass
class ResumeState:
    """What training needs to continue: the trained tensors, the state dict of the optimizer, the
    states of the global random generators, and the state of the family's loop (counters, buffers,
    the checkpoints so far)."""

    tensors: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    rng: dict[str, Any]
    loop: dict[str, Any]


def save_resume(directory: pathlib.Path, state: ResumeState) -> None:
    """Write resume.pt (atomically)."""
    path = directory / RESUME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    saved = {
        "tensors": state.tensors,
        "optimizer": state.optimizer,
        "rng": state.rng,
        "loop": state.loop,
    }
    torch.save(saved, temporary)
    temporary.replace(path)


def load_resume(directory: pathlib.Path) -> ResumeState | None:
    """The state in resume.pt; None without the file."""
    path = directory / RESUME
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return ResumeState(saved["tensors"], saved["optimizer"], saved["rng"], saved["loop"])


def delete_resume(directory: pathlib.Path) -> None:
    (directory / RESUME).unlink(missing_ok=True)


def check_settings(directory: pathlib.Path, settings: Mapping[str, Any]) -> None:
    """Write settings.json the first time; afterwards raise TrainingRunError unless `settings`
    equal what it holds. `settings` are what makes two training runs differ (the method's
    settings, the task, the replicate), not the budgets or update caps that a run asks for."""
    path = directory / SETTINGS
    current = run_record.jsonable(dict(settings))
    if not path.exists():
        run_record.write_json(path, current)
        return
    recorded = run_record.read_json(path)
    if recorded != current:
        raise TrainingRunError(
            f"{directory} holds a training run that was started with other settings; use a new "
            "output directory or delete this one"
        )


@dataclasses.dataclass(frozen=True)
class Done:
    """Where training stopped, and why.

    status: "complete" if the training run reached its end (its largest budget, or its last
        update), "stopped" if it ended earlier because the run needed no more.
    next_checkpoint_lower_bound: a lower bound on `budget_spent` of any checkpoint that training
        would still save if it continued; +inf when complete.
    extra: the family's own fields."""

    status: DoneStatus
    updates: int
    next_checkpoint_lower_bound: float
    extra: dict[str, Any] = dataclasses.field(default_factory=lambda: {})

    def to_json(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        bound = self.next_checkpoint_lower_bound
        value["next_checkpoint_lower_bound"] = None if bound == float("inf") else bound
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Done:
        bound = value["next_checkpoint_lower_bound"]
        return cls(
            status=value["status"],
            updates=int(value["updates"]),
            next_checkpoint_lower_bound=float("inf") if bound is None else float(bound),
            extra=dict(value.get("extra", {})),
        )


def write_done(directory: pathlib.Path, done: Done) -> None:
    run_record.write_json(directory / DONE, done.to_json())


def read_done(directory: pathlib.Path) -> Done | None:
    """What done.json holds; None without the file."""
    path = directory / DONE
    return Done.from_json(json.loads(path.read_text())) if path.exists() else None


def clear_done(directory: pathlib.Path) -> None:
    """Remove done.json, before training continues."""
    (directory / DONE).unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class Request:
    """What a run needs from a training run: the candidates at its update caps and at its budgets,
    where `reserve` is what a budget sets aside for its threshold tests. With `to_end` it needs the
    whole training run."""

    update_caps: tuple[int, ...] = ()
    budgets: tuple[int, ...] = ()
    reserve: int = 0
    to_end: bool = False


def candidates_are_final(
    checkpoints: Sequence[Checkpoint],
    done: Done,
    *,
    limit: float | None = None,
    update: int | None = None,
) -> bool:
    """Whether training on could not add a candidate: to those of a budget B, given `limit` =
    B minus the reserve, or to those of the update cap `update`. Give exactly one of the two."""
    if (limit is None) == (update is None):
        raise ValueError("give exactly one of limit and update")
    if done.status == "complete":
        return True
    if update is not None:
        return done.updates >= update
    assert limit is not None
    return done.next_checkpoint_lower_bound > limit or any(
        row.budget_spent > limit for row in checkpoints
    )


def serves(checkpoints: Sequence[Checkpoint], done: Done | None, request: Request) -> bool:
    """Whether a training run that stopped with `done` (None if it has no done.json) already
    holds every checkpoint that `request` needs."""
    if done is None:
        return False
    if done.status == "complete":
        return True
    if request.to_end:
        return False
    return all(
        candidates_are_final(checkpoints, done, update=u) for u in request.update_caps
    ) and all(
        candidates_are_final(checkpoints, done, limit=b - request.reserve) for b in request.budgets
    )


def stop_reached(request: Request, *, updates: int, next_checkpoint_lower_bound: float) -> bool:
    """Whether a loop, at the start of an update, may stop early for `request`: every update cap
    has been reached, and no checkpoint that it would still save could be a candidate at one of the
    budgets. A request with `to_end`, or without update caps and budgets, never stops early."""
    if request.to_end or not (request.update_caps or request.budgets):
        return False
    caps_done = all(updates >= u for u in request.update_caps)
    budgets_done = all(next_checkpoint_lower_bound > b - request.reserve for b in request.budgets)
    return caps_done and budgets_done
