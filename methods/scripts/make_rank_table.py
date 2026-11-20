"""Write the tables of the threshold test, `posteriorbench_methods/common/data/rank_table*.json`.

    python scripts/make_rank_table.py            # write both tables
    python scripts/make_rank_table.py --check    # exit 1 if a table in data/ differs

`posteriorbench_methods.common.rank_table` computes the tables, in about a second.
"""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from posteriorbench_methods.common import rank_table


def main(
    check: Annotated[
        bool, typer.Option(help="Compare with the tables in the directory; write nothing.")
    ] = False,
    directory: Annotated[
        pathlib.Path, typer.Option(help="Where the tables are.")
    ] = rank_table.DATA_DIR,
) -> None:
    """Compute the table of the 10% threshold (ranks 1..1,024) and of the 1% threshold (ranks
    1..16), and write them or compare them with the files."""
    differing: list[str] = []
    for name, (threshold, alternative, max_rank) in rank_table.TABLES.items():
        text = rank_table.render(rank_table.build_rank_table(threshold, alternative, max_rank))
        path = directory / name
        if check:
            if not path.exists() or path.read_text() != text:
                differing.append(name)
        else:
            path.write_text(text)
            typer.echo(f"wrote {path}")
    if differing:
        typer.echo(f"differ from the computed tables: {', '.join(differing)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
