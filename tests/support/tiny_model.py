"""A tiny randomly initialized causal language model with a character tokenizer."""

from __future__ import annotations

import string
from typing import Any

import pytest

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]
CHARACTERS = list(string.ascii_lowercase) + [" ", ".", ",", "'", "—", "中"]
MAX_TOKENS = 8
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}{{ m['content'] }}<|im_end|>{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>a{% endif %}"
)


def build_tokenizer() -> Any:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {token: i for i, token in enumerate(SPECIALS + CHARACTERS)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex("."), behavior="isolated"
    )
    backend.decoder = tokenizers.decoders.Fuse()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def build_model(vocab_size: int, seed: int = 0) -> Any:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return transformers.Qwen2ForCausalLM(config).float().eval()


def build_original_model() -> Any:
    """The tiny model as an `OriginalModel` with responses of at most MAX_TOKENS tokens."""
    from posteriorbench import original_model

    tokenizer = build_tokenizer()
    return original_model.OriginalModel(
        build_model(len(tokenizer)),
        tokenizer,
        batch_size=7,
        max_response_tokens=MAX_TOKENS,
        name="tiny",
        dtype="float32",
    )
