"""A stand-in for the Hugging Face Hub that serves a dataset repository from a local directory."""

from __future__ import annotations

import importlib
import pathlib
import shutil
from typing import Any

import pytest

from posteriorbench import data


def install_offline_hub(
    monkeypatch: pytest.MonkeyPatch, local: pathlib.Path, repo_id: str = data.HUB_DATASET
) -> list[tuple[str, str | None]]:
    """Make `huggingface_hub.hf_hub_download` (and `list_repo_files`) serve `repo_id` from the
    local directory `local`, copying each requested file into a cache directory as the Hub client
    does. Returns the list of (file, revision) requested, in order."""
    import huggingface_hub
    import huggingface_hub.errors

    # The HTTP library of the installed huggingface_hub (the base class of its HTTP errors).
    http_error = huggingface_hub.errors.HfHubHTTPError.__mro__[1]
    http: Any = importlib.import_module(http_error.__module__.partition(".")[0])

    cache = local.parent / f"{local.name}_hub_cache"
    requested: list[tuple[str, str | None]] = []

    def hf_hub_download(
        repo: str, filename: str, *, repo_type: str | None = None, revision: str | None = None
    ) -> str:
        assert repo == repo_id, f"unexpected repository {repo!r}"
        assert repo_type == "dataset"
        requested.append((filename, revision))
        source = local / filename
        if not source.is_file():
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
            response = http.Response(404, request=http.Request("GET", url))
            raise huggingface_hub.errors.RemoteEntryNotFoundError(
                f"{filename} not found", response=response
            )
        target = cache / (revision or "main") / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return str(target)

    def list_repo_files(
        repo: str, *, repo_type: str | None = None, revision: str | None = None
    ) -> list[str]:
        assert repo == repo_id and repo_type == "dataset"
        requested.append(("<list>", revision))
        return sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", list_repo_files)
    return requested
