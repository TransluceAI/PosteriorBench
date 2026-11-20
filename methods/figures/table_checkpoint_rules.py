"""The submission rule against each training method's own checkpoint rule (Table 7).

    python figures/table_checkpoint_rules.py [--runs TABLE ...]

Prints, for each method and own rule, the capped forward KL and the reverse KL averaged over 12
of the 21 tasks at replicate 0, under the own rule and under the submission rule, and the runs
scored as the fallback under each. With `--runs` the cells are computed from tables of runs whose
column `rule` names the own rule of a `reselect --rule` run and is empty for a `run` run. The
paper's cells come from `data/main_comparison_runs.csv.gz` and `data/reselected_runs.csv`.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Annotated, Any

import paper_runs
import paper_style
import typer

CELLS = paper_runs.DATA_DIR / "checkpoint_rules_cells.csv"
# Each method's budget, and its own rules by their names in `reselect --rule`.
SETTINGS = {
    "gfn": "B=1048576",
    "ctl": "B=1048576",
    "gdc": "B=524288",
    "kl-dpg": "B=1048576",
    "js-dpg": "B=1048576",
}
OWN_RULES = {
    "gfn": ("highest-mean-log-reward",),
    "ctl": ("final-checkpoint",),
    "gdc": ("last-sampling-distribution", "final-policy"),
    "kl-dpg": ("final-checkpoint",),
    "js-dpg": ("final-checkpoint",),
}
REPLICATE = "0"
# The `rule` of submission-rule rows that are not runs of the main comparison: the paper's own-rule
# runs of KL-DPG and JS-DPG come from separate trainings, and their submission-rule rows with them.
SUBMISSION = "submission-rule"
TASKS = (
    "wc_0346/em_dash",
    "wc_0532/profanity",
    "wc_0670/chinese_character",
    "wc_0670/contrastive_phrasing",
    "wc_0842/chatgpt_mention",
    "wc_1380/word_repetition",
    "wc_1405/chinese_character",
    "wc_1569/em_dash",
    "wc_1569/goblin_mention",
    "wc_1786/goblin_mention",
    "wc_1852/chatgpt_mention",
    "wc_1853/profanity",
)
SCORES = ("capped_forward_kl", "reverse_kl")


def side(runs: Sequence[paper_runs.Row], method: str, rule: str) -> dict[str, Any] | None:
    """The runs of `method` under `rule` on `TASKS` at replicate 0, summarized: the two scores
    averaged over the tasks, and the runs scored as the fallback. None if `runs` hold no such
    run. `SUBMISSION` asks for the submission rule: the rows whose rule is `submission-rule`
    if the method has any, and the rows of `run` (an empty rule) otherwise."""
    mine = [
        row
        for row in runs
        if (row["method"], row["setting"]) == (method, SETTINGS[method])
        and row["replicate"] == REPLICATE
        and row["task_id"] in TASKS
    ]
    if rule == SUBMISSION and not any(row.get("rule", "") == SUBMISSION for row in mine):
        rule = ""
    mine = [row for row in mine if row.get("rule", "") == rule]
    return paper_runs.summarize(mine, SCORES) if mine else None


def cells(runs: Sequence[paper_runs.Row]) -> list[dict[str, Any]]:
    """One row per method and own rule that `runs` hold under both rules: `runs`, and under
    `own_` and `ours_` the two scores and `fallback_runs`."""
    paper_runs.require_columns(runs, paper_runs.RUN_KEYS)
    result: list[dict[str, Any]] = []
    for method, rules in OWN_RULES.items():
        ours = side(runs, method, SUBMISSION)
        for rule in rules:
            own = side(runs, method, rule)
            if own is None or ours is None:
                continue
            cell: dict[str, Any] = {"method": method, "own_rule": rule, "setting": SETTINGS[method]}
            cell["runs"] = own["runs"]
            for prefix, summary in (("own", own), ("ours", ours)):
                for score in SCORES:
                    cell[f"{prefix}_{score}"] = summary[score]
                cell[f"{prefix}_fallback_runs"] = summary["fallback_runs"]
            result.append(cell)
    if not result:
        wanted = "; ".join(
            f"{method} at {SETTINGS[method]} ({', '.join(rules)})"
            for method, rules in OWN_RULES.items()
        )
        raise paper_runs.TableError(
            "the table has no method with runs of `run` and of `reselect --rule` at replicate "
            f"{REPLICATE} of the table's 12 tasks; the methods, their settings and own rules "
            f"are {wanted}"
        )
    return result


def read_cells(path: pathlib.Path) -> list[dict[str, Any]]:
    """The table of cells, with the keys of `cells`."""
    result: list[dict[str, Any]] = []
    for row in paper_runs.read_table(path):
        cell: dict[str, Any] = {name: row[name] for name in ("method", "own_rule", "setting")}
        cell["runs"] = int(row["runs"])
        for prefix in ("own", "ours"):
            for score in SCORES:
                cell[f"{prefix}_{score}"] = paper_runs.number(row, f"{prefix}_{score}")
            cell[f"{prefix}_fallback_runs"] = int(row[f"{prefix}_fallback_runs"])
        result.append(cell)
    return result


def main(
    runs: Annotated[
        list[pathlib.Path] | None, typer.Option(help="A table of runs with a column `rule`.")
    ] = None,
) -> None:
    """Print the table."""
    if runs:
        table = cells([row for path in runs for row in paper_runs.read_table(path)])
    else:
        table = read_cells(CELLS)
    typer.echo(f"{'method':<10}{'own rule':<28}{'own':>16}{'ours':>16}{'fallback, own / ours':>24}")
    for cell in table:
        own = f"{cell['own_capped_forward_kl']:.2f} / {cell['own_reverse_kl']:.2f}"
        ours = f"{cell['ours_capped_forward_kl']:.2f} / {cell['ours_reverse_kl']:.2f}"
        fallback = f"{cell['own_fallback_runs']} / {cell['ours_fallback_runs']}"
        label = paper_style.series(cell["method"]).label
        typer.echo(f"{label:<10}{cell['own_rule']:<28}{own:>16}{ours:>16}{fallback:>24}")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
