"""A dataset file that is neither in the local Hugging Face cache nor reachable on the Hub.

Offline, or with `HF_HUB_OFFLINE=1`, such a read raises huggingface_hub's
`LocalEntryNotFoundError`; `reading` turns it into an error that a command reports as one line.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator

from posteriorbench import data

from posteriorbench_methods.common import errors


class UnreachableDataError(errors.MethodsError):
    """A file of a dataset on the Hub is not in the local cache, and the Hub cannot be reached."""


@contextlib.contextmanager
def reading(source: str | os.PathLike[str] | None) -> Generator[None, None, None]:
    """Run a block that reads the dataset `source` (a local directory, a Hub id, or None for the
    package's dataset), raising UnreachableDataError for a file the Hub cannot serve."""
    import huggingface_hub.errors  # loaded when a dataset is read

    try:
        yield
    except huggingface_hub.errors.LocalEntryNotFoundError as error:
        name = data.HUB_DATASET if source is None else str(source)
        raise UnreachableDataError(
            f"{name!r} is not an existing directory, and its files are not in the local cache "
            "of the Hugging Face Hub, which cannot be reached (is HF_HUB_OFFLINE set?). Give "
            "the directory of a local copy of the dataset as --data, or download the dataset "
            "while online."
        ) from error
