"""`posteriorbench-methods`, the root command.

It holds one group per family (`run`, `config` and, for the families that train, `reselect`; the
options are in `common.cli_options`), the group `generalization`, the threshold test at a rank, and
the analyses of finished runs (`analysis.cli`).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Annotated

import typer

from posteriorbench_methods.analysis import cli as analysis_cli
from posteriorbench_methods.common import rank_table
from posteriorbench_methods.ctl import cli as ctl_cli
from posteriorbench_methods.dpg import cli as dpg_cli
from posteriorbench_methods.generalization import cli as generalization_cli
from posteriorbench_methods.gfn import cli as gfn_cli
from posteriorbench_methods.prompt import cli as prompt_cli
from posteriorbench_methods.sft import cli as sft_cli
from posteriorbench_methods.steering import cli as steering_cli

# The command group of each family, in the order of the help text.
FAMILIES: dict[str, typer.Typer] = {
    "sft": sft_cli.app,
    "prompt": prompt_cli.app,
    "steering": steering_cli.app,
    "gfn": gfn_cli.app,
    "ctl": ctl_cli.app,
    "dpg": dpg_cli.app,
}

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    add_completion=False,
    help="The methods of the PosteriorBench paper. `<family> run --help` lists the options of a "
    "run, which are the same in every family.",
)
for _name, _family_app in FAMILIES.items():
    app.add_typer(_family_app, name=_name)
# The tasks beyond the benchmark and their evaluation: not a method family.
app.add_typer(generalization_cli.app, name="generalization")


@app.command(
    "threshold-test",
    help="Print the threshold test of the submission rule at a rank: the new responses it draws "
    "and the hits it needs to pass.",
)
def threshold_test(
    rank: Annotated[int, typer.Option(min=1, help="The rank j of the tested candidate.")] = 1,
) -> None:
    typer.echo(json.dumps(dataclasses.asdict(rank_table.threshold_test(rank)), indent=1))


analysis_cli.register(app)


if __name__ == "__main__":
    app()
