"""The base class of the errors that a command reports as one line."""

from __future__ import annotations

import contextlib
import types

import posteriorbench
import typer


class MethodsError(Exception):
    """An error the user of a command can fix: a missing or unusable input, settings a method
    cannot run with, a training run on disk that does not fit the command."""


class OneLineErrors(contextlib.ContextDecorator):
    """Print a `MethodsError` or a `PosteriorBenchError` as `Error: <message>` and exit with
    status 1. Use as a `with` block or as a decorator of a command."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if isinstance(error, (posteriorbench.PosteriorBenchError, MethodsError)):
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error
