"""The records that the analyses read: the directory that a run writes for one method, setting,
task and replicate (`common.run_record`).
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from posteriorbench import ledger

from posteriorbench_methods.common import errors, run_record

# The records of an output root: <out>/<method>/<setting>/<task>/rep<r>/result.json.
RECORDS_OF_A_ROOT = f"*/*/*/rep*/{run_record.RESULT_FILE}"


class RecordsError(errors.MethodsError, ValueError):
    """A path that holds no record, or a record without a file that an analysis needs."""


def number(value: Any) -> float:
    """A float of a record. A record writes a non-finite float as "inf", "-inf" or "nan"."""
    return float(value)


@dataclasses.dataclass(frozen=True)
class Record:
    """One record directory. `result` is its result.json; the other files are read on first
    use."""

    directory: pathlib.Path
    result: Mapping[str, Any]

    @property
    def method(self) -> str:
        return str(self.result["method"])

    @property
    def setting(self) -> str:
        return str(self.result["setting"])

    @property
    def task_id(self) -> str:
        return str(self.result["task_id"])

    @property
    def replicate(self) -> int:
        return int(self.result["replicate"])

    @property
    def label(self) -> dict[str, Any]:
        """What names the record: method, setting, task_id and replicate."""
        return {
            "method": self.method,
            "setting": self.setting,
            "task_id": self.task_id,
            "replicate": self.replicate,
        }

    @property
    def construction_inferences(self) -> float:
        """The run's inference count, the cost that the benchmark reports."""
        return number(self.result["construction_inferences"])

    @property
    def ledger_totals(self) -> ledger.LedgerTotals | None:
        """What the run spent, by category; None for a run that kept no ledger."""
        totals = self.result.get("ledger")
        return None if totals is None else ledger.LedgerTotals.from_dict(totals)

    def _read(self, name: str) -> Any:
        path = self.directory / name
        if not path.exists():
            raise RecordsError(f"{self.directory} has no {name}")
        return run_record.read_json(path)

    @functools.cached_property
    def construction(self) -> Mapping[str, Any]:
        """construction.json: what the method did."""
        return self._read(run_record.CONSTRUCTION_FILE)

    @functools.cached_property
    def config(self) -> Mapping[str, Any]:
        """config.json: the settings, the options, the versions."""
        return self._read(run_record.CONFIG_FILE)

    @functools.cached_property
    def evaluation_details(self) -> Mapping[str, Any]:
        """The entry of evaluation_details.json behind result.json, the one with the shipped log
        p of the references."""
        return self._read(run_record.EVALUATION_DETAILS_FILE)["shipped"]

    @property
    def details(self) -> Mapping[str, Any]:
        """The family's own entries of construction.json."""
        return self.construction["details"]

    @property
    def settings(self) -> Mapping[str, Any]:
        """The method's settings in the run."""
        return self.config["settings"]


def _read(path: pathlib.Path) -> Record:
    return Record(path.parent, run_record.read_json(path))


def find(paths: Iterable[pathlib.Path]) -> list[Record]:
    """The records that `paths` name, in order. A path is an output root, whose records are all
    read; a record directory; or a record's result.json. The records under `<out>/reselected/`
    are not those of the output root: name `<out>/reselected/<rule>` to read them. Raises
    RecordsError for a path that holds no record."""
    found: list[Record] = []
    for path in paths:
        if path.is_file():
            if path.name != run_record.RESULT_FILE:
                raise RecordsError(f"{path} is not a record's {run_record.RESULT_FILE}")
            found.append(_read(path))
        elif (path / run_record.RESULT_FILE).is_file():
            found.append(_read(path / run_record.RESULT_FILE))
        elif path.is_dir():
            results = sorted(path.glob(RECORDS_OF_A_ROOT))
            if not results:
                raise RecordsError(
                    f"{path} holds no record: an output root has "
                    f"<method>/<setting>/<task>/rep<r>/{run_record.RESULT_FILE}"
                )
            found.extend(_read(result) for result in results)
        else:
            raise RecordsError(f"{path} does not exist")
    return found


def check_one_record_per_run(found: Sequence[Record]) -> None:
    """Raise RecordsError if two of `found` have the same method, setting, task and replicate.
    The records of `run` and of a rule of `reselect` have the same labels, so they are analysed
    in separate calls."""
    seen: set[tuple[str, str, str, int]] = set()
    for record in found:
        key = (record.method, record.setting, record.task_id, record.replicate)
        if key in seen:
            raise RecordsError(
                f"{record.method} at {record.setting}: {record.task_id} has more than one record "
                f"of replicate {record.replicate}; a point holds at most one record for a task "
                "and replicate (give the records of `run` and of each rule of `reselect` in "
                "separate calls)"
            )
        seen.add(key)
