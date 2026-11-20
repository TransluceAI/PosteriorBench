"""Scoring and sampling with Hugging Face, on a tiny randomly initialized causal LM."""

from __future__ import annotations

import collections
import math
from typing import Any
from unittest import mock

import pytest

import posteriorbench
from posteriorbench import original_model, protocol, sampler

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def unbatched_logprob(model: Any, prompt: list[int], response: list[int]) -> float:
    ids = torch.tensor([prompt + response])
    with torch.no_grad():
        logits = model.model(input_ids=ids).logits[0].float()
    steps = torch.log_softmax(logits[len(prompt) - 1 : -1], dim=-1)
    return float(steps.gather(1, torch.tensor(response).unsqueeze(1)).sum())


def test_batched_scoring_equals_unbatched(tiny_model: Any) -> None:
    prompt = tiny_model.render_prompt("hello")
    # Complete responses: shorter than the 8-token cap and stopped, or cut at the cap.
    responses = [[5, 6, 7, 2], [8, 0], [9, 10, 11, 12, 13, 2], [4, 4, 4, 4, 4, 4, 4, 4], [14, 0]]
    batched = tiny_model.logprobs(prompt, responses)
    for response, value in zip(responses, batched, strict=True):
        assert value == pytest.approx(unbatched_logprob(tiny_model, prompt, response), abs=1e-4)
    per_token = tiny_model.token_logprobs(prompt, responses)
    assert [len(v) for v in per_token] == [len(r) for r in responses]
    assert all(t <= 0 for v in per_token for t in v)


def test_generation_rules_and_seeds(tiny_model: Any) -> None:
    prompt = tiny_model.render_prompt("hi")
    responses = tiny_model.generate(prompt, 60, seed=1)
    assert len(responses) == 60
    for response in responses:
        tiny_model.validate_response(response)
    stopped = [r for r in responses if r[-1] in tiny_model.stop_token_ids]
    assert all(len(r) == tiny_model.max_response_tokens for r in responses if r not in stopped)
    assert tiny_model.generate(prompt, 60, seed=1) == responses
    assert tiny_model.generate(prompt, 60, seed=2) != responses
    with pytest.raises(original_model.ResponseError):
        tiny_model.validate_response([tiny_model.stop_token_ids[0], 5])
    with pytest.raises(original_model.ResponseError):
        tiny_model.validate_response([])


def test_sampling_has_no_truncation(tiny_model: Any) -> None:
    """Temperature 1 and no top-k / top-p, even if the model ships other generation defaults."""
    one_token = original_model.OriginalModel(
        tiny_model.model, tiny_model.tokenizer, batch_size=1024, max_response_tokens=1
    )
    one_token.model.generation_config = transformers.GenerationConfig(
        do_sample=True, temperature=0.3, top_k=3, top_p=0.5
    )
    prompt = tiny_model.render_prompt("hi")
    draws = 20_000
    counts = collections.Counter(r[0] for r in one_token.generate(prompt, draws, seed=0))
    with torch.no_grad():
        logits = tiny_model.model(input_ids=torch.tensor([prompt])).logits[0, -1].float()
    probabilities = torch.softmax(logits, dim=-1).tolist()
    assert len(counts) > 20  # far more than a top-k of 3 would allow
    chi_square = sum(
        (counts.get(token, 0) - draws * p) ** 2 / (draws * p)
        for token, p in enumerate(probabilities)
    )
    degrees = len(probabilities) - 1
    assert chi_square < degrees + 5 * math.sqrt(2 * degrees)


def test_decode_skips_special_tokens(tiny_model: Any) -> None:
    ids = tiny_model.tokenizer("ab", add_special_tokens=False)["input_ids"]
    assert tiny_model.decode(ids + [1, tiny_model.stop_token_ids[0]]) == "ab"


def test_render_prompt_with_system(tiny_model: Any) -> None:
    plain = tiny_model.render_prompt("hi")
    system = tiny_model.render_prompt("hi", system="be odd")
    assert len(system) > len(plain) and system[-len(plain) :] == plain


def test_short_response_must_end_with_a_stop_token(tiny_model: Any) -> None:
    """A response shorter than the cap stopped, so its last token is a stop token; one that
    reaches the cap may end without one. A shorter sequence without a stop token is a prefix
    and is rejected at the scoring boundary instead of receiving the prefix's likelihood."""
    prompt = tiny_model.render_prompt("hi")
    stop = tiny_model.stop_token_ids[0]
    cap = tiny_model.max_response_tokens
    letter = tiny_model.tokenizer("a", add_special_tokens=False)["input_ids"][0]
    prefix = [letter]  # decodes to "a", no stop token, 1 < cap tokens
    for valid in ([letter, stop], [letter] * cap, [letter] * (cap - 1) + [stop], [stop]):
        tiny_model.validate_response(valid)
    with pytest.raises(original_model.ResponseError, match="must end with a stop token"):
        tiny_model.validate_response(prefix)
    with pytest.raises(original_model.ResponseError):
        tiny_model.logprobs(prompt, [[letter, stop], prefix])
    with pytest.raises(original_model.ResponseError):
        tiny_model.token_logprobs(prompt, [prefix])
    with pytest.raises(original_model.ResponseError):
        tiny_model.logprobs(prompt, [[letter] * (cap + 1)])
    # The same rule at the evaluator boundary.
    with pytest.raises(sampler.SamplerError, match="must end with a stop token"):
        protocol.validate_responses(tiny_model, [prefix])
    protocol.validate_responses(tiny_model, [[letter, stop], [letter] * cap])


def test_from_pretrained_pins_the_dataset_revision(tiny_model: Any) -> None:
    """The benchmark model's weights and tokenizer are loaded at MODEL_REVISION unless another
    revision is given; another model is loaded at its default branch."""
    with (
        mock.patch.object(
            transformers.AutoConfig, "from_pretrained", return_value=tiny_model.model.config
        ) as config_load,
        mock.patch.object(
            transformers.AutoModelForCausalLM, "from_pretrained", return_value=tiny_model.model
        ) as model_load,
        mock.patch.object(
            transformers.AutoTokenizer, "from_pretrained", return_value=tiny_model.tokenizer
        ) as tokenizer_load,
    ):
        original_model.OriginalModel.from_pretrained(device="cpu")
        original_model.OriginalModel.from_pretrained(revision="other-revision", device="cpu")
        original_model.OriginalModel.from_pretrained("org/other-model", device="cpu")
        original_model.OriginalModel.from_pretrained("org/other-model", revision="r1", device="cpu")
    expected = [original_model.MODEL_REVISION, "other-revision", None, "r1"]
    assert [call.kwargs["revision"] for call in config_load.call_args_list] == expected
    assert [call.kwargs["revision"] for call in model_load.call_args_list] == expected
    assert [call.kwargs["revision"] for call in tokenizer_load.call_args_list] == expected
    # float32 by default.
    assert {call.kwargs["dtype"] for call in model_load.call_args_list} == {torch.float32}
    with pytest.raises(ValueError, match="dtype"):
        original_model.OriginalModel.from_pretrained(dtype="int8", device="cpu")


def test_describe_records_the_runtime(tiny_model: Any) -> None:
    described = tiny_model.describe()
    for key in (
        "backend", "model", "revision", "model_commit", "dtype", "device", "torch", "transformers",
        "cuda", "gpu_name", "fla_importable", "causal_conv1d_importable", "posteriorbench",
    ):  # fmt: skip
        assert key in described, key
    assert described["backend"] == "hf" and described["dtype"] == "float32"
    assert described["torch"] == torch.__version__
    assert described["transformers"] == transformers.__version__
    assert described["posteriorbench"] == posteriorbench.__version__
    assert isinstance(described["fla_importable"], bool)


def test_model_commit_without_the_config_attribute(tmp_path: Any) -> None:
    """The commit of a Hub model is read from the cache path of its config.json when the installed
    transformers does not keep it on the config; a local directory has no commit."""
    resolve: Any = getattr(original_model, "_resolved_commit")
    cached = tmp_path / "models--x--y" / "snapshots" / "0123abcd" / "config.json"
    with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=str(cached)) as lookup:
        assert resolve("x/y", "main", object()) == "0123abcd"
        assert lookup.call_args.args == ("x/y", "config.json")
        assert lookup.call_args.kwargs == {"revision": "main"}
        assert resolve("x/y", "main", mock.Mock(_commit_hash="fromconfig")) == "fromconfig"
        assert resolve(str(tmp_path), None, object()) is None
    with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None):
        assert resolve("x/y", "main", object()) is None


def test_from_pretrained_accepts_a_local_directory(tiny_model: Any, tmp_path: Any) -> None:
    """A local copy of a model loads by path, with the tokenizer saved next to it."""
    tiny_model.model.save_pretrained(tmp_path / "model")
    tiny_model.tokenizer.save_pretrained(tmp_path / "model")
    loaded = original_model.OriginalModel.from_pretrained(str(tmp_path / "model"), device="cpu")
    prompt = tiny_model.render_prompt("hi")
    response = [5, 6, tiny_model.stop_token_ids[0]]
    assert loaded.logprobs(prompt, [response]) == pytest.approx(
        tiny_model.logprobs(prompt, [response]), abs=1e-6
    )
    assert loaded.describe()["model"] == str(tmp_path / "model")
    assert loaded.describe()["revision"] is None
