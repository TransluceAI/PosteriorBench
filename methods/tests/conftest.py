"""Shared test fixtures: one tiny tokenizer, one tiny model and one tiny dataset for every family.

- The tokenizer maps each character of a small alphabet to one token, with the chat specials
  (<|endoftext|>, <|im_start|>, <|im_end|>) and a chat template; `<|im_end|>` is its eos token.
- The model is a randomly initialized 2-layer Qwen2 causal LM (hidden size 32); `build_model` with
  the same seed always gives the same weights. `build_hybrid_model` is a tiny Qwen3.5 with the
  24 decoder blocks of the paper's model (linear-attention and full-attention layers), for code
  that needs the paper's layer indices or the recurrent cache of that architecture.
- The dataset is one task of the tiny model whose behavior ("the response contains the letter a")
  is frequent: its corpus is `CORPUS_DRAWS` responses of the model, p̂(S) their hit fraction, the
  references the first 250 hits and the fallback response the next hit. It is written with the
  package's schemas and file names (task table, evaluation file, hit file) and read back with the
  package's readers.

`toy_context` is a run context without a model, for the tests of what a run records.

A family's own `conftest.py` may define fixtures with the same names; inside that family's folder
its own fixtures take precedence.
"""

from __future__ import annotations

import dataclasses
import pathlib
import string
from collections.abc import Callable, Sequence
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer.rich_utils
from posteriorbench import benchmark, data, original_model, rules

from posteriorbench_methods.common import run_context, tracking

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]
CHARACTERS = list(string.ascii_lowercase) + [" ", ".", ",", "'", "—", "中"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}{{ m['content'] }}<|im_end|>{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>a{% endif %}"
)
MAX_TOKENS = 8  # response limit of the tiny original model and dataset
TASK_ID = "wc_1853/goblin_mention"
RULE = '"a" in y'
USER_PROMPT = "hi"
CORPUS_DRAWS = 3000
REFERENCES = 250


def build_tokenizer() -> Any:
    """The character tokenizer (eos <|im_end|>, padding <|endoftext|>)."""
    import tokenizers
    import transformers

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
    """A fresh tiny Qwen2 causal LM in float32, eval mode; the same seed gives the same weights.
    Side effect: seeds torch's global generator with `seed`."""
    import torch
    import transformers

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


def build_hybrid_model(vocab_size: int, seed: int = 0, *, layers: int = 24) -> Any:
    """A fresh tiny Qwen3.5 text model (`layers` blocks, three linear-attention blocks per
    full-attention block, hidden size 32) in float32, eval mode, its output layer scaled by 6 so
    its next-token probabilities are far from uniform. Side effect: seeds torch's global
    generator."""
    import torch
    import transformers

    if layers % 4:
        raise ValueError("the hybrid model has a multiple of 4 blocks")
    torch.manual_seed(seed)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * (layers // 4),
        tie_word_embeddings=False,
        eos_token_id=SPECIALS.index("<|im_end|>"),
        pad_token_id=SPECIALS.index("<|endoftext|>"),
    )
    model = transformers.Qwen3_5ForCausalLM(config).float().eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(6)
    return model


def make_original_model(
    model: Any, tokenizer: Any, *, max_response_tokens: int = MAX_TOKENS, batch_size: int = 16
) -> original_model.OriginalModel:
    """The package's `OriginalModel` around a tiny model (float32)."""
    return original_model.OriginalModel(
        model,
        tokenizer,
        batch_size=batch_size,
        max_response_tokens=max_response_tokens,
        name="tiny",
        dtype="float32",
    )


@dataclasses.dataclass(frozen=True)
class Synthetic:
    """The synthetic task: its construction view, held-out data, and the corpus it came from
    (`responses[i]` is corpus row i, `labels[i]` its hit flag)."""

    task: data.Task
    evaluation: data.EvaluationData
    responses: tuple[tuple[int, ...], ...]
    labels: tuple[bool, ...]


def make_synthetic(
    scorer: original_model.OriginalModel,
    *,
    task_id: str = TASK_ID,
    rule: str = RULE,
    draws: int = CORPUS_DRAWS,
    seed: int = 0,
) -> Synthetic:
    """A task on `scorer`'s model: the corpus is `draws` responses to the chat prompt of
    USER_PROMPT, p̂(S) = hits / draws, the references the first REFERENCES hits, the fallback
    response the next hit; the stored log p are the model's own."""
    prompt_id, behavior = task_id.split("/")
    prompt = scorer.render_prompt(USER_PROMPT)
    responses = [tuple(r) for r in scorer.generate(prompt, draws, seed=seed)]
    predicate = rules.parse_rule(rule)
    texts = [scorer.decode(r) for r in responses]
    labels = [bool(predicate(text)) for text in texts]
    hit_rows = [i for i, hit in enumerate(labels) if hit]
    assert len(hit_rows) > REFERENCES, "the synthetic behavior is too rare for this fixture"
    reference_rows = hit_rows[:REFERENCES]
    fallback = responses[hit_rows[REFERENCES]]
    task = data.Task(
        task_id=task_id,
        prompt_id=prompt_id,
        prompt=USER_PROMPT,
        prompt_token_ids=tuple(prompt),
        wildchat_conversation_hash="0" * 32,
        behavior=behavior,
        rule=rule,
        num_responses=draws,
        hit_count=len(hit_rows),
        probability=len(hit_rows) / draws,
        scored=True,
        benchmark_task=True,
    )
    reference_logp = scorer.logprobs(prompt, [responses[r] for r in reference_rows])
    (fallback_logp,) = scorer.logprobs(prompt, [fallback])
    evaluation = data.EvaluationData(
        task_id=task_id,
        references=tuple(data.Reference(r, responses[r], texts[r]) for r in reference_rows),
        reference_logp=tuple(reference_logp),
        fallback_text=scorer.decode(fallback),
        fallback_token_ids=fallback,
        fallback_logp=fallback_logp,
    )
    return Synthetic(task, evaluation, tuple(responses), tuple(labels))


def write_dataset(root: pathlib.Path, synthetic: Synthetic) -> pathlib.Path:
    """Write the task table, the evaluation file and the task's hit file under `root` with the
    package's schemas and file names (`posteriorbench.data`); returns `root`. The hit file holds the
    corpus hits other than the references, in corpus order, with its reference-duplicate flags and
    the metadata the package's corpus reader checks."""
    task, evaluation = synthetic.task, synthetic.evaluation
    record: dict[str, Any] = dataclasses.asdict(task)
    record["prompt_token_ids"] = list(task.prompt_token_ids)
    (root / data.TASKS_FILE).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([record], schema=data.TASKS_SCHEMA), root / data.TASKS_FILE)
    held_out = {
        "task_id": evaluation.task_id,
        "references": [
            {"row": r.row, "token_ids": list(r.token_ids), "text": r.text}
            for r in evaluation.references
        ],
        "reference_logp": list(evaluation.reference_logp),
        "fallback_text": evaluation.fallback_text,
        "fallback_token_ids": list(evaluation.fallback_token_ids),
        "fallback_logp": evaluation.fallback_logp,
    }
    (root / data.EVALUATION_FILE).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([held_out], schema=data.EVALUATION_SCHEMA),
        root / data.EVALUATION_FILE,
    )
    reference_rows = {r.row for r in evaluation.references}
    reference_ids = {r.token_ids for r in evaluation.references}
    rows = [i for i, hit in enumerate(synthetic.labels) if hit and i not in reference_rows]
    ids = [synthetic.responses[i] for i in rows]
    table = pa.table(
        {
            "row": pa.array(rows, type=pa.int32()),
            "token_ids": pa.array([list(t) for t in ids], type=pa.list_(pa.int32())),
            "reference_duplicate": pa.array([t in reference_ids for t in ids], type=pa.bool_()),
        },
        schema=data.HITS_SCHEMA,
    ).replace_schema_metadata(
        {
            "task_id": task.task_id,
            "hit_count": str(task.hit_count),
            "references_removed": str(len(reference_rows)),
        }
    )
    hits_path = root / data.hits_file(task.prompt_id, task.behavior)
    hits_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, hits_path, row_group_size=97)
    return root


@pytest.fixture(scope="session", autouse=True)
def plain_help_texts() -> None:
    """Help texts are rendered without terminal styling. typer forces styling on some CI
    runners, and the escape codes split the option names that tests look for."""
    typer.rich_utils.FORCE_TERMINAL = False


@pytest.fixture(scope="session")
def tiny_tokenizer() -> Any:
    return build_tokenizer()


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer: Any) -> Any:
    """The tiny Qwen2 (seed 0). Shared by the session: do not train it; use `make_tiny_model`."""
    return build_model(len(tiny_tokenizer))


@pytest.fixture(scope="session")
def make_tiny_model(tiny_tokenizer: Any) -> Callable[..., Any]:
    """A factory of fresh tiny Qwen2 models: `make_tiny_model()` has the weights of `tiny_model`
    (e.g. a copy to train), `make_tiny_model(seed=s)` other weights (e.g. a stand-in for q)."""

    def make(seed: int = 0) -> Any:
        return build_model(len(tiny_tokenizer), seed)

    return make


@pytest.fixture(scope="session")
def tiny_original_model(tiny_model: Any, tiny_tokenizer: Any) -> original_model.OriginalModel:
    """`tiny_model` as the package's `OriginalModel` (responses of at most MAX_TOKENS tokens)."""
    return make_original_model(tiny_model, tiny_tokenizer)


@pytest.fixture(scope="session")
def tiny_hybrid_model(tiny_tokenizer: Any) -> Any:
    """The 24-block tiny Qwen3.5 (seed 0). Shared by the session: do not train it."""
    return build_hybrid_model(len(tiny_tokenizer))


@pytest.fixture(scope="session")
def tiny_synthetic(tiny_original_model: original_model.OriginalModel) -> Synthetic:
    return make_synthetic(tiny_original_model)


@pytest.fixture(scope="session")
def tiny_task(tiny_synthetic: Synthetic) -> data.Task:
    return tiny_synthetic.task


@pytest.fixture(scope="session")
def tiny_dataset(
    tiny_synthetic: Synthetic, tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """The synthetic task written as a data source (pass the path as the package's `source`)."""
    return write_dataset(tmp_path_factory.mktemp("tiny_dataset"), tiny_synthetic)


@pytest.fixture(scope="session")
def make_tiny_original_model(tiny_model: Any, tiny_tokenizer: Any) -> Callable[..., Any]:
    """A factory of `OriginalModel`s around `tiny_model` (or `model=`) with another response limit
    or batch size: `make_tiny_original_model(max_response_tokens=50)` for code that keeps the
    benchmark's 50 tokens; `lambda: make_tiny_original_model(model=make_tiny_model())` as a
    `RunContext.load_model_copy` (a fresh copy with the original's weights)."""

    def make(
        *, max_response_tokens: int = MAX_TOKENS, model: Any = None, batch_size: int = 16
    ) -> Any:
        return make_original_model(
            tiny_model if model is None else model,
            tiny_tokenizer,
            max_response_tokens=max_response_tokens,
            batch_size=batch_size,
        )

    return make


@pytest.fixture(scope="session")
def make_tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., pathlib.Path]:
    """A factory of synthetic datasets: `make_tiny_dataset(scorer, task_id=..., rule=...,
    draws=...)` samples the task from `scorer`'s model (`make_synthetic`) and writes it like
    `tiny_dataset`; returns the data source path."""

    def make(scorer: original_model.OriginalModel, **options: Any) -> pathlib.Path:
        return write_dataset(
            tmp_path_factory.mktemp("tiny_dataset"), make_synthetic(scorer, **options)
        )

    return make


class ToyScorer:
    """A scorer without a model, with the benchmark's response limit: token 1 decodes to a goblin
    mention, and every log p is -5."""

    stop_token_ids: tuple[int, ...] = (0,)
    max_response_tokens: int = 50

    def logprobs(
        self, prompt_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
    ) -> list[float]:
        return [-5.0] * len(responses)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "a goblin" if 1 in token_ids else "nothing"

    def describe(self) -> dict[str, Any]:
        return {"backend": "fake"}


@pytest.fixture
def toy_context() -> Callable[[pathlib.Path], run_context.RunContext]:
    """`toy_context(out)`: a run of a family "toy" on the package's task wc_1853/goblin_mention,
    replicate 2, with 64 evaluation draws and a `ToyScorer` as the original model. For tests of
    what a run records that need no model."""

    def make(out: pathlib.Path) -> run_context.RunContext:
        bench = benchmark.Benchmark()
        task_id = "wc_1853/goblin_mention"
        options = run_context.RunOptions(task_id=task_id, out=out, replicate=2, evaluation_draws=64)
        original = cast(Any, ToyScorer())
        return run_context.RunContext(
            family="toy",
            method="toy",
            options=options,
            bench=bench,
            task=bench.task(task_id),
            original=original,
            load_model_copy=lambda: original,
            tracker=tracking.Tracker(),
        )

    return make
