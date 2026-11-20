"""Choosing the checkpoint of a finished training run again, by another rule, without training
(README, "Choosing a checkpoint again").

A family names its rules in `run_context.MethodEntry.reselect`. `threshold-1pct` is the submission
rule at the 1% threshold (`submission.at_one_percent`), with new threshold tests from random
streams of their own, evaluated at 1%. A method's own rule, such as `final-checkpoint`, submits
without a threshold test and is evaluated at the benchmark's threshold. A result costs what the
training had spent when it saved the last candidate (`spend_after`), plus any tests at 1%; these
draw more than a budget sets aside, so the cost is not held to the budget.
"""

from __future__ import annotations

import fractions
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from posteriorbench import ledger
from posteriorbench import sampler as pb_sampler

from posteriorbench_methods.common import errors, run_context, run_record
from posteriorbench_methods.common import submission as submission_rule
from posteriorbench_methods.common import training_run

THRESHOLD_1PCT = run_record.ONE_PERCENT_RULE  # "threshold-1pct"
FINAL_CHECKPOINT = "final-checkpoint"
EVALUATION_THRESHOLD_1PCT = fractions.Fraction(1, 100)
TEST_STREAM_1PCT = "threshold-test-1pct"
OWN_RULE_STATUS = "own_rule"


class NoTrainingRunError(errors.MethodsError, RuntimeError):
    """`run` has not written the training run, or has not trained it as far as the budgets and
    update caps need."""


def records_root(out: pathlib.Path, rule: str) -> pathlib.Path:
    """<out>/reselected/<rule>/, the output root of the records that `rule` chose."""
    return out / run_record.RESELECTED_DIR / rule


def _no_training_run(directory: pathlib.Path) -> NoTrainingRunError:
    return NoTrainingRunError(
        f"{directory} holds no training run; `run` writes one (use --keep-checkpoints all)"
    )


def require_training_run(options: run_context.RunOptions, method: str) -> None:
    """Raise NoTrainingRunError unless the output root of `options` holds a training run of
    `method` for their task and replicate. It needs neither the task nor the model, so a command
    checks it before loading them."""
    directory = run_record.training_run_dir(options.out, method, options.task_id, options.replicate)
    if not directory.exists():
        raise _no_training_run(directory)


def read_training_run(
    directory: pathlib.Path, settings: Mapping[str, Any], request: training_run.Request
) -> list[training_run.Checkpoint]:
    """The checkpoints of the training run that `run` left in `directory`, in the order saved,
    after checking that it was trained with `settings` (`training_run.check_settings`) and as far
    as `request` needs (NoTrainingRunError otherwise). Nothing is trained or written."""
    if not (directory / training_run.SETTINGS).exists():
        raise _no_training_run(directory)
    training_run.check_settings(directory, settings)
    checkpoints = training_run.read_index(directory)
    if not training_run.serves(checkpoints, training_run.read_done(directory), request):
        raise NoTrainingRunError(
            f"the training run in {directory} has not trained as far as these settings need; "
            "`run` with the same settings trains it"
        )
    return checkpoints


def spend_after(checkpoint: training_run.Checkpoint) -> ledger.Ledger:
    """A ledger that holds what a training run had spent when it saved `checkpoint`, without a
    budget limit."""
    return ledger.Ledger.resume(ledger.LedgerTotals.from_dict(checkpoint.ledger))


def final_checkpoint(candidates: Sequence[training_run.Checkpoint]) -> training_run.Checkpoint:
    """The rule `final-checkpoint`: the last of the candidates, the checkpoint that training
    saved last within the budget or the update cap."""
    return candidates[-1]


def _submitted(checkpoint: training_run.Checkpoint) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint.checkpoint_id, "update": checkpoint.update}


def record_threshold_1pct(
    context: run_context.RunContext,
    sampler: pb_sampler.Sampler,
    *,
    method: str,
    setting: str,
    settings: Any,
    spend: ledger.Ledger,
    ranking: Sequence[submission_rule.RankedCandidate],
    submission: submission_rule.Submission,
    checkpoint: training_run.Checkpoint,
    details: Mapping[str, Any],
    selection_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> run_record.Recorded:
    """Evaluate the checkpoint that the rule at 1% submitted, at that threshold, and write its
    record under `records_root(out, "threshold-1pct")`.

    sampler: the method's sampler, with the weights of the submitted `checkpoint`.
    spend: what the training had spent when it saved the last candidate, and the tests at 1%.
    details: the family's entries of construction.json, as in the record of `run`.
    selection_rows, test_rows: the rows of draws/selection.jsonl and draws/threshold_tests.jsonl."""
    return context.evaluate_and_record(
        sampler,
        method=method,
        setting=setting,
        settings=settings,
        spend=spend,
        ranking=ranking,
        submission=submission,
        submitted=_submitted(checkpoint),
        details={"threshold": float(EVALUATION_THRESHOLD_1PCT), **details},
        draws={"selection": selection_rows, "threshold_tests": test_rows},
        threshold=EVALUATION_THRESHOLD_1PCT,
        out=records_root(context.options.out, THRESHOLD_1PCT),
    )


def record_own_rule(
    context: run_context.RunContext,
    sampler: pb_sampler.Sampler,
    *,
    rule: str,
    method: str,
    setting: str,
    settings: Any,
    checkpoint: training_run.Checkpoint,
    candidates: Sequence[training_run.Checkpoint],
    details: Mapping[str, Any],
    draws: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> run_record.Recorded:
    """Evaluate the checkpoint that a method's own rule chose, with the benchmark's threshold,
    and write its record under `records_root(out, rule)`, with the status `own_rule`, no ranking,
    and the cost of the training when it saved the last of `candidates`.

    sampler: the method's sampler, with the weights of the chosen `checkpoint`.
    details: the family's entries of construction.json, as in the record of `run`.
    draws: the draws that the rule read, if any, as rows by file name."""
    return context.evaluate_and_record(
        sampler,
        method=method,
        setting=setting,
        settings=settings,
        spend=spend_after(candidates[-1]),
        status=OWN_RULE_STATUS,
        submitted=_submitted(checkpoint),
        details={"own_rule": rule, **details},
        draws=draws,
        out=records_root(context.options.out, rule),
    )
