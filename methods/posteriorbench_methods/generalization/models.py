"""The original models of the six tasks and their loading with transformers.

`load_original_model` returns the package's `OriginalModel`, so generation and log p are the
package's code. Each checkpoint holds a vision tower next to the text model; only the text model is
loaded, and a checkpoint that leaves one of its parameters at the random initial value is refused
(`ModelLoadError`).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from collections.abc import Sequence
from typing import Any

from posteriorbench import data, original_model

from posteriorbench_methods.common import errors

DTYPES = original_model.DTYPES  # float32, bfloat16, float16
# Responses scored in one forward pass, by response limit. A pass over 2,048-token responses
# returns logits [batch, about 2,070, vocabulary], about 2 GB per response in float32.
SCORING_BATCH_SIZE = 16
LONG_RESPONSE_SCORING_BATCH_SIZE = 2
LONG_RESPONSE_GENERATION_BATCH_SIZE = 128
LONG_RESPONSE_TOKENS = 256  # a response limit above this takes the two sizes above


class ModelLoadError(errors.MethodsError):
    """A checkpoint that does not fill the text model, or a model whose rendering of a task's
    prompt differs from the task's prompt token ids."""


@dataclasses.dataclass(frozen=True)
class OriginalModelSpec:
    """One original model.

    key: the model's name in task ids and data directories ("gemma-4-31b-it").
    name, revision: the Hub id and commit of the checkpoint; a revision of None is the default
        branch.
    model_class: the transformers class of the text model.
    key_mapping: regex (pattern, replacement) pairs, as `from_pretrained(key_mapping=...)` takes.
    ignored_key_prefixes: checkpoint keys that the text model does not use (the vision tower).
    stop_token_ids: the first also pads the rows of a batch that have ended.
    thinking_argument: whether the chat template takes `enable_thinking=False`; a template that
        does not leaves thinking off by default.
    dtype: the default dtype, that of the paper's runs."""

    key: str
    paper_name: str
    name: str
    revision: str | None
    model_class: str
    stop_token_ids: tuple[int, ...]
    thinking_argument: bool
    dtype: str
    key_mapping: tuple[tuple[str, str], ...] = ()
    ignored_key_prefixes: tuple[str, ...] = ()


# <|im_end|> and <|endoftext|> in the tokenizer that the two Qwen models share.
_QWEN_STOP_TOKEN_IDS = (248046, 248044)
_QWEN_IGNORED_KEYS = ("mtp.", "model.visual.", "model.language_model.mtp")

QWEN3_5_0_8B = OriginalModelSpec(
    key="qwen3.5-0.8b",
    paper_name="Qwen3.5-0.8B",
    name=original_model.DEFAULT_MODEL,
    revision=original_model.MODEL_REVISION,
    model_class="Qwen3_5ForCausalLM",
    stop_token_ids=_QWEN_STOP_TOKEN_IDS,
    thinking_argument=True,
    dtype="float32",
    ignored_key_prefixes=_QWEN_IGNORED_KEYS,
)
# 64 decoder blocks, hidden size 5120. The checkpoint has the layout of Qwen3.5's, and the
# Qwen3.5 text class reads it.
QWEN3_8_27B = OriginalModelSpec(
    key="qwen3.8-27b",
    paper_name="Qwen3.8-27B",
    name="Qwen/Qwen3.8-27B",
    revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_class="Qwen3_5ForCausalLM",
    stop_token_ids=_QWEN_STOP_TOKEN_IDS,
    thinking_argument=True,
    dtype="bfloat16",
    ignored_key_prefixes=_QWEN_IGNORED_KEYS,
)
# 60 decoder blocks, hidden size 5376. The checkpoint keeps the text model under
# `model.language_model`. A response ends at <eos> (1), at the end of the model's turn (106), or
# at token 50, which the checkpoint's generation settings also name as an end.
GEMMA_4_31B_IT = OriginalModelSpec(
    key="gemma-4-31b-it",
    paper_name="Gemma-4-31B-it",
    name="google/gemma-4-31B-it",
    revision="145dc2508c480a64b47242f160d286cff94a2343",
    model_class="Gemma4ForCausalLM",
    stop_token_ids=(1, 106, 50),
    thinking_argument=False,
    dtype="bfloat16",
    key_mapping=((r"^model\.language_model\.", "model."),),
    ignored_key_prefixes=("model.vision_tower.", "model.embed_vision.", "model.audio_tower."),
)

MODELS: dict[str, OriginalModelSpec] = {
    spec.key: spec for spec in (QWEN3_5_0_8B, QWEN3_8_27B, GEMMA_4_31B_IT)
}


class TaskOriginalModel(original_model.OriginalModel):
    """The package's `OriginalModel`, plus a chat template rendered with its defaults and a
    generation batch size apart from the scoring one. The likelihoods are the package's code.

    generation_batch_size: used where `generate` is not given a batch size; None uses
        `batch_size`, as the package does."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        thinking_argument: bool = True,
        generation_batch_size: int | None = None,
        **options: Any,
    ) -> None:
        super().__init__(model, tokenizer, **options)
        self.thinking_argument = thinking_argument
        self.generation_batch_size = generation_batch_size

    def render_prompt(self, user: str, system: str | None = None) -> list[int]:
        """Token ids of a chat prompt with thinking off."""
        if self.thinking_argument:
            return super().render_prompt(user, system)
        messages = [{"role": "user", "content": user}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        n: int,
        *,
        seed: int,
        temperature: float = 1.0,
        batch_size: int | None = None,
    ) -> list[list[int]]:
        """The package's `generate`; `batch_size` defaults to `generation_batch_size`."""
        return super().generate(
            prompt_token_ids,
            n,
            seed=seed,
            temperature=temperature,
            batch_size=batch_size or self.generation_batch_size,
        )

    def describe(self) -> dict[str, Any]:
        """The package's runtime record, with the generation batch size."""
        return {**super().describe(), "generation_batch_size": self.generation_batch_size}


def check_loading_info(spec: OriginalModelSpec, info: dict[str, Any]) -> None:
    """Raise ModelLoadError unless the checkpoint filled the text model: no missing, mismatched or
    failed key, and no unused key outside `spec.ignored_key_prefixes`. `info` is what
    `from_pretrained(output_loading_info=True)` returns."""
    missing = sorted(info.get("missing_keys") or [])
    mismatched = sorted(str(key) for key in info.get("mismatched_keys") or [])
    failed = [str(message) for message in info.get("error_msgs") or []]
    unused = sorted(
        key
        for key in info.get("unexpected_keys") or []
        if not str(key).startswith(spec.ignored_key_prefixes)
    )
    if missing or mismatched or failed or unused:
        raise ModelLoadError(
            f"{spec.name} does not load as {spec.model_class}: missing keys {missing[:5]}, keys "
            f"of another shape {mismatched[:5]}, unused keys {unused[:5]}, errors {failed[:2]}"
        )


def _hub_commit(name: str, revision: str | None) -> str | None:
    """The Hub commit of a just-loaded model, from its cached config; None for a directory."""
    if os.path.isdir(name):
        return None
    import huggingface_hub

    cached = huggingface_hub.try_to_load_from_cache(name, "config.json", revision=revision)
    return data.hub_commit(pathlib.Path(cached)) if isinstance(cached, str) else None


def default_batch_sizes(max_response_tokens: int) -> tuple[int, int | None]:
    """(scoring, generation) batch sizes for a response limit; a generation size of None means
    the scoring size."""
    if max_response_tokens <= LONG_RESPONSE_TOKENS:
        return SCORING_BATCH_SIZE, None
    return LONG_RESPONSE_SCORING_BATCH_SIZE, LONG_RESPONSE_GENERATION_BATCH_SIZE


def load_original_model(
    spec: OriginalModelSpec,
    *,
    max_response_tokens: int,
    name: str | None = None,
    revision: str | None = None,
    dtype: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    generation_batch_size: int | None = None,
) -> TaskOriginalModel:
    """Load the text model and tokenizer of `spec`, downloading them on first use.

    name, revision: another location of the same checkpoint, such as a local directory.
    dtype: one of DTYPES; by default `spec.dtype`.
    device: by default "cuda" if available. The weights are read on the CPU first, so the machine
        needs the model's size in memory.
    batch_size, generation_batch_size: by default `default_batch_sizes`. The responses that a
        seed draws depend on the generation batch size.

    Raises ImportError if the installed transformers has no `spec.model_class`, and
    ModelLoadError if the checkpoint does not fill the text model."""
    import torch
    import transformers

    dtype = spec.dtype if dtype is None else dtype
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    model_class: Any = getattr(transformers, spec.model_class, None)
    if model_class is None:
        raise ImportError(
            f"transformers {transformers.__version__} has no {spec.model_class}, which "
            f"{spec.paper_name} needs"
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    source = spec.name if name is None else name
    if revision is None and name is None:
        revision = spec.revision
    scoring, generation = default_batch_sizes(max_response_tokens)
    mapping: dict[str, Any] = {}
    if spec.key_mapping:
        mapping["key_mapping"] = dict(spec.key_mapping)
    model, info = model_class.from_pretrained(
        source,
        revision=revision,
        dtype=getattr(torch, dtype),
        output_loading_info=True,
        **mapping,
    )
    check_loading_info(spec, info)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, revision=revision)
    return TaskOriginalModel(
        model.to(device),
        tokenizer,
        thinking_argument=spec.thinking_argument,
        generation_batch_size=(
            generation if generation_batch_size is None else generation_batch_size
        ),
        batch_size=scoring if batch_size is None else batch_size,
        max_response_tokens=max_response_tokens,
        stop_token_ids=spec.stop_token_ids,
        name=source,
        revision=revision,
        dtype=dtype,
        model_commit=_hub_commit(source, revision),
    )


def check_prompt(model: original_model.OriginalModel, prompt: str, expected: Sequence[int]) -> None:
    """Raise ModelLoadError unless `model` renders `prompt` to `expected`, the token ids that the
    task's responses were sampled under (another tokenizer or chat template would differ)."""
    rendered = [int(token) for token in model.render_prompt(prompt)]
    if rendered != [int(token) for token in expected]:
        raise ModelLoadError(
            f"the model renders the task's prompt to {len(rendered)} tokens that differ from "
            f"the task's {len(expected)} prompt token ids: its tokenizer or chat template is "
            "not the one the task's data was made with"
        )
