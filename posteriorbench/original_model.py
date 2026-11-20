"""The original model p with Hugging Face transformers: log-likelihoods, sampling and decoding.

The benchmark's original model is Qwen3.5-0.8B (chat model, thinking off), sampled at temperature
1, top-p 1, no top-k or min-p truncation, at most 50 new tokens, stopping at `<|im_end|>` or
`<|endoftext|>`. A response's likelihood is the product over its generated token ids, including
the final stop token when there is one. A response shorter than 50 tokens ends with a stop token;
scoring anything else (such as a prefix of a response) raises ResponseError
(`sampler.invalid_response_reason`). The prompt is the dataset's `prompt_token_ids` (the user turn
rendered with the chat template, `enable_thinking=False`; it ends with an empty think block).

`OriginalModel.from_pretrained()` loads the benchmark model's weights and tokenizer at
MODEL_REVISION, the commit whose tokenizer the dataset's token ids, prompts and labels were made
with. It also accepts another revision, another model, or a local directory holding a copy of a
model. Weights are loaded in float32 by default (bfloat16 and float16 are options).
`OriginalModel.describe` records the runtime for result records.

torch and transformers are imported when a model is loaded or used, not when this module is
imported, so the rest of the package works in an environment without a working torch.
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
from collections.abc import Sequence
from typing import Any

from posteriorbench import _version, data, errors, sampler

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
# The commit of DEFAULT_MODEL that the dataset was generated with (weights and tokenizer).
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MAX_RESPONSE_TOKENS = 50
STOP_TOKENS: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>")
DTYPES: tuple[str, ...] = ("float32", "bfloat16", "float16")


def _resolved_commit(name: str, revision: str | None, config: Any) -> str | None:
    """The Hub commit the loaded files belong to, or None for a local directory.

    Some transformers releases keep the commit on the config. Where they do not, it is read from
    the Hugging Face cache path of the model's `config.json` (`snapshots/<commit>/config.json`),
    which loading the config has just put there."""
    commit = getattr(config, "_commit_hash", None)
    if commit is not None or os.path.isdir(name):
        return commit
    import huggingface_hub

    cached = huggingface_hub.try_to_load_from_cache(name, "config.json", revision=revision)
    if not isinstance(cached, str):
        return None
    return data.hub_commit(pathlib.Path(cached))


class ResponseError(errors.PosteriorBenchError):
    """A response passed to the model for scoring is not a complete response."""


def resolve_model_revision(name: str, revision: str | None = None) -> str | None:
    """The revision of model `name` to load (weights and tokenizer): an explicit `revision` wins;
    the benchmark's model (DEFAULT_MODEL) gets MODEL_REVISION; any other model its default branch
    (None)."""
    if revision is not None:
        return revision
    if name == DEFAULT_MODEL:
        return MODEL_REVISION
    return None


def render_chat_prompt(tokenizer: Any, user: str, system: str | None = None) -> list[int]:
    """Token ids of a chat prompt with thinking off. With no system message this is how the
    dataset's `prompt_token_ids` were rendered."""
    messages = ([{"role": "system", "content": system}] if system is not None else []) + [
        {"role": "user", "content": user}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


class OriginalModel:
    """Teacher-forced scoring, sampling and decoding for one causal language model.

    `model` is a transformers causal LM, `tokenizer` its tokenizer. Responses are lists of
    generated token ids; prompts are lists of prompt token ids. Scoring right-pads a batch, so a
    response's log-likelihood depends only on its own tokens up to floating-point effects of the
    batch composition (larger in bfloat16)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        batch_size: int = 16,
        max_response_tokens: int = MAX_RESPONSE_TOKENS,
        stop_token_ids: Sequence[int] | None = None,
        name: str | None = None,
        revision: str | None = None,
        dtype: str | None = None,
        model_commit: str | None = None,
    ) -> None:
        self.model = model
        self.name = name
        self.revision = revision
        self.model_commit = model_commit
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_response_tokens = max_response_tokens
        if stop_token_ids is None:
            ids = [tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS]
            unknown = getattr(tokenizer, "unk_token_id", None)
            stop_token_ids = [i for i in ids if i is not None and i != unknown]
        if not stop_token_ids:
            raise ValueError("no stop token ids")
        self.stop_token_ids: tuple[int, ...] = tuple(int(i) for i in stop_token_ids)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        name: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        dtype: str = "float32",
        device: str | None = None,
        batch_size: int = 16,
    ) -> OriginalModel:
        """Load a model and its tokenizer.

        `name` is a Hub model id or a local directory. `revision` picks the commit of the weights
        and the tokenizer (`resolve_model_revision`: MODEL_REVISION for the benchmark's model
        unless given). `dtype` is float32 (the default), bfloat16 or float16. `device` defaults to
        "cuda" if a GPU is available, else "cpu". `batch_size` is the number of responses
        generated or scored per forward pass; generated responses depend on it (the same seed
        gives other draws at another batch size). Qwen3.5 checkpoints are vision-language
        checkpoints; only their text model is loaded (`Qwen3_5ForCausalLM`).

        Raises ValueError for another dtype and ImportError if the installed transformers has no
        Qwen3.5 support."""
        import torch
        import transformers

        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        revision = resolve_model_revision(name, revision)
        config = transformers.AutoConfig.from_pretrained(name, revision=revision)
        model_class: Any = transformers.AutoModelForCausalLM
        if getattr(config, "model_type", None) == "qwen3_5":
            model_class = getattr(transformers, "Qwen3_5ForCausalLM", None)
            if model_class is None:
                raise ImportError(
                    f"transformers {transformers.__version__} has no Qwen3_5ForCausalLM; "
                    "install a release with Qwen3.5 support"
                )
        model = model_class.from_pretrained(name, revision=revision, dtype=getattr(torch, dtype))
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        return cls(
            model.to(device),
            tokenizer,
            batch_size=batch_size,
            name=name,
            revision=revision,
            dtype=dtype,
            model_commit=_resolved_commit(name, revision, config),
        )

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def describe(self) -> dict[str, Any]:
        """The runtime of this scorer, for result records: model, revision and resolved commit,
        dtype, device, torch / transformers / CUDA versions, GPU name, and whether the
        `flash-linear-attention` (`fla`) and `causal-conv1d` packages are importable (with them
        installed, Qwen3.5's linear-attention layers run other kernels)."""
        import torch
        import transformers

        device = self.device
        gpu_name = None
        if getattr(device, "type", None) == "cuda":
            gpu_name = str(torch.cuda.get_device_name(device))
        parameter = next(self.model.parameters())
        return {
            "backend": "hf",
            "model": self.name,
            "revision": self.revision,
            "model_commit": self.model_commit,
            "dtype": self.dtype if self.dtype is not None else str(parameter.dtype).split(".")[-1],
            "device": str(device),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": gpu_name,
            "fla_importable": importlib.util.find_spec("fla") is not None,
            "causal_conv1d_importable": importlib.util.find_spec("causal_conv1d") is not None,
            "max_response_tokens": self.max_response_tokens,
            "stop_token_ids": list(self.stop_token_ids),
            "batch_size": self.batch_size,
            "posteriorbench": _version.__version__,
        }

    def decode(self, token_ids: Sequence[int]) -> str:
        """The response text the labels are computed on (special tokens skipped)."""
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))

    def render_prompt(self, user: str, system: str | None = None) -> list[int]:
        """Token ids of a chat prompt with thinking off (`render_chat_prompt`)."""
        return render_chat_prompt(self.tokenizer, user, system)

    def validate_response(self, token_ids: Sequence[int]) -> None:
        """Raise ResponseError unless `token_ids` is a complete response of this model's
        generation (`sampler.invalid_response_reason`)."""
        reason = sampler.invalid_response_reason(
            token_ids,
            stop_token_ids=self.stop_token_ids,
            max_response_tokens=self.max_response_tokens,
        )
        if reason is not None:
            raise ResponseError(reason)

    def token_logprobs(
        self, prompt_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
    ) -> list[list[float]]:
        """Per-token log p(y_t | prompt, y_<t) of each response (float32 log-softmax). Raises
        ResponseError, before any model call, unless every response is complete
        (`validate_response`): the likelihood of a prefix is not that of a response."""
        import torch

        if not prompt_token_ids:
            raise ValueError("empty prompt")
        for index, response in enumerate(responses):
            try:
                self.validate_response(response)
            except ResponseError as error:
                raise ResponseError(f"response {index}: {error}") from error
        order = sorted(range(len(responses)), key=lambda i: len(responses[i]))
        result: list[list[float]] = [[] for _ in responses]
        prompt = [int(t) for t in prompt_token_ids]
        n_prompt = len(prompt)
        with torch.no_grad():
            for lo in range(0, len(order), self.batch_size):
                batch = order[lo : lo + self.batch_size]
                rows = [prompt + [int(t) for t in responses[i]] for i in batch]
                width = max(len(r) for r in rows)
                input_ids = torch.zeros((len(rows), width), dtype=torch.long)
                attention = torch.zeros((len(rows), width), dtype=torch.long)
                for j, row in enumerate(rows):
                    input_ids[j, : len(row)] = torch.tensor(row, dtype=torch.long)
                    attention[j, : len(row)] = 1
                input_ids = input_ids.to(self.device)
                attention = attention.to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention).logits
                for j, i in enumerate(batch):
                    end = len(rows[j])
                    step = torch.log_softmax(logits[j, n_prompt - 1 : end - 1].float(), dim=-1)
                    targets = input_ids[j, n_prompt:end]
                    values = step.gather(1, targets.unsqueeze(1)).squeeze(1)
                    result[i] = [float(v) for v in values.cpu().tolist()]
        return result

    def logprobs(
        self, prompt_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
    ) -> list[float]:
        """log p(y | prompt) of each complete response: the sum of its per-token
        log-probabilities, final stop token included. Raises ResponseError for anything that is
        not a complete response (`validate_response`)."""
        return [math.fsum(v) for v in self.token_logprobs(prompt_token_ids, responses)]

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        n: int,
        *,
        seed: int,
        temperature: float = 1.0,
        batch_size: int | None = None,
    ) -> list[list[int]]:
        """n responses sampled at the given temperature with no other truncation, each cut after
        its first stop token or at max_response_tokens. Seeds torch's global RNG (side effect)."""
        import torch
        import transformers

        config = transformers.GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            max_new_tokens=self.max_response_tokens,
            eos_token_id=list(self.stop_token_ids),
            pad_token_id=self.stop_token_ids[0],
        )
        # The model's own generation defaults (e.g. a shipped top-k or top-p) must not be merged in.
        self.model.generation_config = config
        size = batch_size or self.batch_size
        prompt = torch.tensor([int(t) for t in prompt_token_ids], dtype=torch.long)
        torch.manual_seed(seed)
        responses: list[list[int]] = []
        with torch.no_grad():
            while len(responses) < n:
                count = min(size, n - len(responses))
                input_ids = prompt.unsqueeze(0).repeat(count, 1).to(self.device)
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    generation_config=config,
                )
                for row in output[:, input_ids.shape[1] :].cpu().tolist():
                    responses.append(self._cut(row))
        return responses

    def _cut(self, generated: Sequence[int]) -> list[int]:
        response: list[int] = []
        for token in generated[: self.max_response_tokens]:
            response.append(int(token))
            if int(token) in self.stop_token_ids:
                break
        return response
