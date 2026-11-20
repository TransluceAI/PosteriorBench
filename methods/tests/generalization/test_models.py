"""Loading the original models, on tiny random models of the two classes: a Qwen3.5 text model
and a Gemma-4 text model, each saved as its real checkpoint is laid out (the text model under
`model.language_model`, next to a vision tower)."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import pytest
import safetensors.torch
import torch
import transformers
from posteriorbench import original_model

from posteriorbench_methods.generalization import models, task_set

RESPONSE = [5, 6, 7, 2]  # three letters and <|im_end|>
TEMPLATE_THAT_SHOWS_THE_ARGUMENT = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}{{ m['content'] }}<|im_end|>{% endfor %}"
    "<|im_start|>a{% if enable_thinking is defined %}t{% endif %}"
)


def save_checkpoint(
    directory: pathlib.Path, model: Any, tokenizer: Any, extra: dict[str, torch.Tensor]
) -> None:
    """Save `model` as a multimodal checkpoint keeps a text model: every key `model.X` as
    `model.language_model.X`, with the tensors of `extra` next to them. An output layer that is
    tied to the embedding is not stored, as in a real checkpoint."""
    state = {
        (f"model.language_model.{key[len('model.'):]}" if key.startswith("model.") else key): (
            value.clone()
        )
        for key, value in model.state_dict().items()
    }
    if model.config.tie_word_embeddings:
        del state["lm_head.weight"]
    safetensors.torch.save_file(
        {**state, **extra}, directory / "model.safetensors", {"format": "pt"}
    )
    model.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)


@pytest.fixture(scope="module")
def tiny_gemma(tiny_tokenizer: Any) -> Any:
    """A tiny Gemma-4 text model (6 blocks, hidden size 32), float32."""
    torch.manual_seed(1)
    config = transformers.Gemma4TextConfig(
        vocab_size=len(tiny_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=16,
    )
    return transformers.Gemma4ForCausalLM(config).float().eval()


def logits(model: Any, prompt: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.tensor([prompt + RESPONSE])).logits.float()


def test_qwen_text_model_loads_from_a_multimodal_checkpoint(
    tiny_hybrid_model: Any, tiny_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """The spec of Qwen3.8-27B on a tiny Qwen3.5: the text class reads the text model from
    under `model.language_model`, in bfloat16 by default, with the task's response limit."""
    extra = {"model.visual.patch.weight": torch.zeros(2, 2), "mtp.head.weight": torch.zeros(2, 2)}
    save_checkpoint(tmp_path, tiny_hybrid_model, tiny_tokenizer, extra)
    spec = dataclasses.replace(models.QWEN3_8_27B, stop_token_ids=(2, 0))
    loaded = models.load_original_model(
        spec, max_response_tokens=50, name=str(tmp_path), dtype="float32", device="cpu"
    )
    assert type(loaded.model).__name__ == "Qwen3_5ForCausalLM"
    assert isinstance(loaded, original_model.OriginalModel)
    assert (loaded.max_response_tokens, loaded.stop_token_ids) == (50, (2, 0))
    assert (loaded.batch_size, loaded.generation_batch_size) == (16, None)
    prompt = loaded.render_prompt("hi")
    assert torch.allclose(
        logits(loaded.model, prompt), logits(tiny_hybrid_model, prompt), atol=1e-6
    )
    reference = original_model.OriginalModel(
        tiny_hybrid_model, tiny_tokenizer, max_response_tokens=50
    )
    assert loaded.logprobs(prompt, [RESPONSE]) == pytest.approx(
        reference.logprobs(prompt, [RESPONSE]), abs=1e-5
    )
    described = loaded.describe()
    assert (described["model"], described["dtype"], described["max_response_tokens"]) == (
        str(tmp_path),
        "float32",
        50,
    )
    assert described["revision"] is None and described["model_commit"] is None
    half = models.load_original_model(
        spec, max_response_tokens=50, name=str(tmp_path), device="cpu"
    )
    assert next(half.model.parameters()).dtype == torch.bfloat16
    assert half.describe()["dtype"] == "bfloat16"
    # Responses of up to 2,048 tokens are scored two at a time and generated 128 at a time.
    long = models.load_original_model(
        dataclasses.replace(models.QWEN3_5_0_8B, stop_token_ids=(2, 0)),
        max_response_tokens=2048, name=str(tmp_path), device="cpu",
    )  # fmt: skip
    assert (long.batch_size, long.generation_batch_size, long.dtype) == (2, 128, "float32")
    with pytest.raises(ValueError, match="dtype"):
        models.load_original_model(spec, max_response_tokens=50, name=str(tmp_path), dtype="int8")


def test_gemma_text_model_loads_with_the_key_mapping(
    tiny_gemma: Any, tiny_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """The spec of Gemma-4-31B-it on a tiny Gemma-4: the checkpoint's keys are mapped to the
    text class, the vision tower is left out, and a checkpoint that does not fill the text
    model is refused."""
    good = tmp_path / "good"
    good.mkdir()
    vision = {"model.vision_tower.patch.weight": torch.zeros(2, 2)}
    save_checkpoint(good, tiny_gemma, tiny_tokenizer, vision)
    spec = dataclasses.replace(models.GEMMA_4_31B_IT, stop_token_ids=(2, 0))
    loaded = models.load_original_model(
        spec, max_response_tokens=50, name=str(good), dtype="float32", device="cpu"
    )
    assert type(loaded.model).__name__ == "Gemma4ForCausalLM"
    prompt = loaded.render_prompt("hi")
    assert torch.equal(logits(loaded.model, prompt), logits(tiny_gemma, prompt))
    (value,) = loaded.logprobs(prompt, [RESPONSE])
    assert value < 0
    responses = loaded.generate(prompt, 5, seed=0)
    assert len(responses) == 5 and all(1 <= len(r) <= 50 for r in responses)
    # Without the mapping nothing of the text model is found; with a key missing, or with a key
    # that neither the text model nor the vision tower explains, loading is refused.
    with pytest.raises(models.ModelLoadError, match="missing keys"):
        models.load_original_model(
            dataclasses.replace(spec, key_mapping=()), max_response_tokens=50, name=str(good),
            device="cpu",
        )  # fmt: skip
    state = safetensors.torch.load_file(good / "model.safetensors")
    for name, change in (("incomplete", "drop"), ("unexplained", "add")):
        directory = tmp_path / name
        directory.mkdir()
        changed = dict(state)
        if change == "drop":
            del changed["model.language_model.layers.0.self_attn.q_proj.weight"]
        else:
            changed["model.another_tower.weight"] = torch.zeros(1)
        safetensors.torch.save_file(changed, directory / "model.safetensors", {"format": "pt"})
        tiny_gemma.config.save_pretrained(directory)
        tiny_tokenizer.save_pretrained(directory)
        with pytest.raises(models.ModelLoadError, match="does not load as Gemma4ForCausalLM"):
            models.load_original_model(
                spec, max_response_tokens=50, name=str(directory), device="cpu"
            )


def test_rendering_and_generation_batch(tiny_model: Any, tiny_tokenizer: Any) -> None:
    """A template that takes `enable_thinking` is given it, as the package does; one that does
    not is rendered with its defaults. `generate` uses the generation batch size."""
    import copy

    tokenizer = copy.deepcopy(tiny_tokenizer)
    tokenizer.chat_template = TEMPLATE_THAT_SHOWS_THE_ARGUMENT
    options: dict[str, Any] = {"max_response_tokens": 8, "batch_size": 3}
    with_argument = models.TaskOriginalModel(tiny_model, tokenizer, **options)
    defaults = models.TaskOriginalModel(
        tiny_model, tokenizer, thinking_argument=False, generation_batch_size=64, **options
    )
    package = original_model.OriginalModel(tiny_model, tokenizer, **options)
    marker = tokenizer("t", add_special_tokens=False)["input_ids"]
    assert with_argument.render_prompt("hi") == package.render_prompt("hi")
    assert with_argument.render_prompt("hi")[-1:] == marker
    assert defaults.render_prompt("hi") == package.render_prompt("hi")[:-1]
    system = defaults.render_prompt("hi", system="be odd")
    assert (
        len(system) > len(defaults.render_prompt("hi"))
        and system[-3:] == defaults.render_prompt("hi")[-3:]
    )
    prompt = defaults.render_prompt("hi")
    # 64 responses per forward pass draw the same responses as the package does at 64.
    assert defaults.generate(prompt, 40, seed=4) == package.generate(
        prompt, 40, seed=4, batch_size=64
    )
    assert with_argument.generate(prompt, 40, seed=4) == package.generate(prompt, 40, seed=4)
    assert defaults.generate(prompt, 40, seed=4) != package.generate(prompt, 40, seed=4)
    assert defaults.describe()["generation_batch_size"] == 64
    assert defaults.logprobs(prompt, [RESPONSE]) == package.logprobs(prompt, [RESPONSE])


def test_a_task_checks_the_prompt_of_its_model(
    toy_tasks: Any, tiny_model: Any, tiny_tokenizer: Any, tmp_path: pathlib.Path, helpers: Any
) -> None:
    """`task_set.load_original_model` loads the task's model with the task's limit and refuses
    a model that renders the task's prompt to other tokens than the task's."""
    tiny_model.save_pretrained(tmp_path)
    tiny_tokenizer.save_pretrained(tmp_path)
    task = toy_tasks.by_rule
    loaded = task_set.load_original_model(task, name=str(tmp_path), device="cpu")
    assert loaded.max_response_tokens == task.max_response_tokens == 8
    assert loaded.stop_token_ids == helpers.spec.stop_token_ids
    other = dataclasses.replace(task, prompt_token_ids=task.prompt_token_ids[:-1])
    with pytest.raises(models.ModelLoadError, match="tokenizer or chat template"):
        task_set.load_original_model(other, name=str(tmp_path), device="cpu")
