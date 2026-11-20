"""The refusal direction: the ablation, and the search that finds the direction. The pipeline of
these tests is the stand-in checkout, which has the interfaces of the real one."""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import pytest
import torch
from posteriorbench import original_model

from posteriorbench_methods.steering import refusal

HIDDEN_SIZE = 32


def _is_pipeline(module_name: str) -> bool:
    return module_name == "pipeline" or module_name.startswith("pipeline.")


@pytest.fixture(scope="module")
def hooks(lmtailrisk_checkout: pathlib.Path) -> Any:
    return refusal.import_from_checkout(lmtailrisk_checkout, "pipeline.utils.hook_utils")


def _residual_stream(model: Any, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    with torch.inference_mode():
        return model(input_ids=ids, output_hidden_states=True).hidden_states


@pytest.mark.parametrize("fixture", ["tiny_model", "tiny_hybrid_model"])
def test_the_ablation_removes_the_direction_after_every_layer(
    fixture: str, hooks: Any, request: pytest.FixtureRequest
) -> None:
    """With intensity 1 the residual stream after every layer has no component along the
    direction, in a model with self-attention and in one with linear-attention layers; with a
    smaller intensity a part of the component remains."""
    model = request.getfixturevalue(fixture)
    direction = torch.randn(HIDDEN_SIZE, generator=torch.Generator().manual_seed(0))
    unit = direction / direction.norm()
    ids = torch.tensor([[5, 9, 12, 20, 7]])
    plain = _residual_stream(model, ids)
    assert max(float((state @ unit).abs().max()) for state in plain[1:]) > 1e-2
    with refusal.ablate(model, direction, 1.0, hooks):
        ablated = _residual_stream(model, ids)
    assert max(float((state @ unit).abs().max()) for state in ablated[1:]) < 1e-4
    with refusal.ablate(model, direction, 0.8, hooks):
        partly = _residual_stream(model, ids)
    assert float((partly[1] @ unit).abs().max()) > 1e-4
    # The hooks are gone when the context closes, also after an error.
    with pytest.raises(RuntimeError, match="inside"), refusal.ablate(model, direction, 1.0, hooks):
        raise RuntimeError("inside")
    restored = _residual_stream(model, ids)
    assert all(torch.equal(a, b) for a, b in zip(plain, restored, strict=True))


def test_the_search_writes_the_direction_and_counts_its_forward_passes(
    steering_original: original_model.OriginalModel,
    lmtailrisk_checkout: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    directory = tmp_path / "steering_vectors"
    assert not refusal.has_direction(directory)
    record = refusal.find_and_save(steering_original, lmtailrisk_checkout, directory)
    direction, read = refusal.load(directory)
    assert read == record and refusal.has_direction(directory)
    assert direction.shape == (HIDDEN_SIZE,) and bool(torch.isfinite(direction).all())
    assert record["repository"] == refusal.REPOSITORY and record["commit"] == refusal.COMMIT
    # 128 and 32 instructions of each kind are drawn, and the refusal score keeps some of each.
    sampled = {name: counts["sampled"] for name, counts in record["instructions"].items()}
    assert sampled == {
        "harmful_train": 128,
        "harmless_train": 128,
        "harmful_val": 32,
        "harmless_val": 32,
    }
    kept = sum(counts["kept"] for counts in record["instructions"].values())
    assert all(0 < counts["kept"] < counts["sampled"] for counts in record["instructions"].values())
    # Every prompt that passed through the model is counted: the 3 instructions that give the
    # tokens of a refusal, the 320 that are scored, and those that the stand-in pipeline runs
    # once more, the kept ones.
    assert record["forward_passes"] == 3 + 320 + kept
    # The stand-in returns the candidate with the largest norm: a layer's input at one of the
    # three token positions that follow the instruction in the test template.
    assert record["layer"] in (0, 1) and record["position"] in (-3, -2, -1)
    candidates = torch.load(
        directory / refusal.SEARCH_FOLDER / "generate_directions" / "mean_diffs.pt"
    )
    assert candidates.shape == (3, 2, HIDDEN_SIZE)
    assert torch.equal(candidates[record["position"], record["layer"]], direction)
    # The same search gives the same direction.
    again = refusal.find(steering_original, lmtailrisk_checkout, tmp_path / "again")
    assert torch.equal(again.direction, direction)
    assert again.forward_passes == record["forward_passes"]


def test_a_search_without_a_direction_is_an_error(
    steering_original: original_model.OriginalModel,
    lmtailrisk_checkout: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select = sys.modules["pipeline.submodules.select_direction"]

    def reject(*arguments: Any) -> Any:
        raise AssertionError("All scores have been filtered out!")

    monkeypatch.setattr(select, "select_direction", reject)
    with pytest.raises(refusal.RefusalError, match="rejected every candidate direction"):
        refusal.find(steering_original, lmtailrisk_checkout, tmp_path / "search")

    def all_refused(model: Any, instructions: list[str], *arguments: Any) -> torch.Tensor:
        return torch.ones(len(instructions))

    monkeypatch.setattr(select, "get_refusal_scores", all_refused)
    with pytest.raises(refusal.RefusalError, match="kept no instruction of .'harmless_train'"):
        refusal.find(steering_original, lmtailrisk_checkout, tmp_path / "search")


def test_a_directory_or_a_checkout_without_what_is_needed_is_refused(
    lmtailrisk_checkout: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(refusal.RefusalError, match="holds no refusal direction"):
        refusal.load(tmp_path)
    with pytest.raises(refusal.RefusalError, match="is not a checkout of"):
        refusal.import_from_checkout(tmp_path, "pipeline.utils.hook_utils")
    with pytest.raises(refusal.RefusalError, match="cannot import pipeline.utils.absent"):
        refusal.import_from_checkout(lmtailrisk_checkout, "pipeline.utils.absent")
    # A checkout whose module cannot be imported leaves nothing imported under its name.
    broken = tmp_path / "broken" / "refusal_direction" / "pipeline"
    (broken / "utils").mkdir(parents=True)
    for folder in (broken, broken / "utils"):
        (folder / "__init__.py").write_text("")
    (broken / "utils" / "hook_utils.py").write_text("import a_package_that_is_not_installed\n")
    imported = {name: sys.modules.pop(name) for name in list(sys.modules) if _is_pipeline(name)}
    try:
        with pytest.raises(refusal.RefusalError, match="No module named 'a_package_that_is_not"):
            refusal.import_from_checkout(tmp_path / "broken", "pipeline.utils.hook_utils")
        assert not [name for name in sys.modules if _is_pipeline(name)]
    finally:
        sys.modules.update(imported)
    # Another checkout than the one whose modules are imported is refused, not mixed in.
    other = tmp_path / "other" / "refusal_direction" / "pipeline"
    other.mkdir(parents=True)
    with pytest.raises(refusal.RefusalError, match="already imported"):
        refusal.import_from_checkout(tmp_path / "other", "pipeline.utils.hook_utils")
    assert str(lmtailrisk_checkout / "refusal_direction") not in sys.path
