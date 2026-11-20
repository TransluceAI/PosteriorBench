"""The refusal direction of the original model: finding it, and ablating it.

The refusal direction (Arditi et al. 2024) is the difference between the model's mean activations
on instructions it refuses and on instructions it answers, at one layer and one token position
after the instruction. Ablating it with intensity c removes c times the component along it from
the residual stream: at the input of every decoder layer and at the output of every attention
module and MLP. One of a behavior's two vectors is built under the ablation (`construction`), so
that the model follows the instructions for the behavior without refusing; the steered sampler
never ablates.

The search and the ablation hooks are Angell et al.'s, imported from a checkout of REPOSITORY at
COMMIT (README, "Build the vectors"). `find` runs the search with the pipeline's own numbers:

1. The refusal tokens are the model's most probable next tokens after each harmful instruction of
   the pipeline's `model_base.py`; an instruction's refusal score is the log odds that the next
   token is one of them.
2. TRAINING_INSTRUCTIONS harmful and harmless training instructions, and VALIDATION_INSTRUCTIONS
   of each for validation, are drawn; a harmful one is kept if its score is positive, a harmless
   one if it is negative.
3. `generate_directions` gives one candidate per layer and token position after the instruction:
   the mean activation on the harmful minus that on the harmless training instructions.
4. `select_direction` scores the candidates on the validation instructions and returns one.

Every prompt through the model during the search is counted, and Steering runs are charged that
count (`cost`).
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib
import json
import pathlib
import random
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any

import safetensors.torch
import torch
from posteriorbench import original_model

from posteriorbench_methods.common import errors, run_record
from posteriorbench_methods.steering import hook

REPOSITORY = "https://github.com/rangell/LMTailRisk"
COMMIT = "b9e383136fdeeb441d56f93744774dcafd6e039b"
PIPELINE_FOLDER = "refusal_direction"  # the folder of the checkout that holds the pipeline
DIRECTION_FILE = "refusal_direction.safetensors"
RECORD_FILE = "refusal_direction.json"
SEARCH_FOLDER = "refusal_direction_search"  # what the pipeline writes while it searches
# The pipeline's numbers: instructions of each kind, and the seed with which they are drawn.
TRAINING_INSTRUCTIONS = 128
VALIDATION_INSTRUCTIONS = 32
SAMPLING_SEED = 42
# The instruction whose rendered prompt shows which tokens follow an instruction.
PROBE_INSTRUCTION = "How do I build bomb?"
# The instructions in the order in which the pipeline draws them.
SPLITS = (("harmful", "train"), ("harmless", "train"), ("harmful", "val"), ("harmless", "val"))


class RefusalError(errors.MethodsError, RuntimeError):
    """The checkout is unusable, the search found no direction, or the method data holds none."""


def import_from_checkout(checkout: pathlib.Path, module: str) -> Any:
    """Import `module` of the pipeline (e.g. "pipeline.utils.hook_utils") from `checkout`.

    Side effect: `refusal_direction/` is on `sys.path` during the import, and the pipeline's
    modules stay imported as `pipeline`; a failed import removes them again. Raises RefusalError
    if the checkout has no pipeline, another `pipeline` is already imported, or the import fails."""
    root = (checkout / PIPELINE_FOLDER).resolve()
    if not (root / "pipeline").is_dir():
        raise RefusalError(
            f"{checkout} is not a checkout of {REPOSITORY} (commit {COMMIT}): it has no "
            f"{PIPELINE_FOLDER}/pipeline"
        )
    imported = sys.modules.get("pipeline")
    if imported is not None and str(root / "pipeline") not in list(
        getattr(imported, "__path__", [])
    ):
        raise RefusalError(
            f"a module named pipeline is already imported, and not from {root}; the refusal "
            "direction needs the one of the checkout"
        )
    imported_before = set(sys.modules)
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    except ImportError as error:
        # A failed import leaves the packages above the module imported; without them, a later
        # call starts again from its own checkout.
        for name in set(sys.modules) - imported_before:
            if name == "pipeline" or name.startswith("pipeline."):
                del sys.modules[name]
        raise RefusalError(
            f"cannot import {module} from {root}: {error}. The pipeline needs jaxtyping, einops, "
            "matplotlib and tqdm in this environment"
        ) from error
    finally:
        sys.path.remove(str(root))


def _residual_branches(layers: Sequence[Any]) -> list[Any]:
    """The modules whose outputs are added to the residual stream: the attention module of each
    layer (self-attention, or linear attention in a layer that has no self-attention), then the
    MLP of each layer."""
    attention = [
        layer.self_attn if getattr(layer, "self_attn", None) is not None else layer.linear_attn
        for layer in layers
    ]
    return [*attention, *(layer.mlp for layer in layers)]


def _on_hidden_states(change: Callable[[Any, tuple[Any, ...]], Any]) -> Callable[..., Any]:
    """A forward pre-hook that applies the pipeline's input hook `change` to the hidden states,
    whether the layer receives them positionally or as the keyword `hidden_states`."""

    def pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if args:
            return change(module, args), kwargs
        (changed,) = change(module, (kwargs["hidden_states"],))
        return args, {**kwargs, "hidden_states": changed}

    return pre_hook


@contextlib.contextmanager
def ablate(
    model: Any, direction: torch.Tensor, intensity: float, hooks: Any
) -> Generator[None, None, None]:
    """Ablate `direction` (shape [hidden size]) in `model` with `intensity` while the context is
    open, at every position (module docstring). `hooks` is the pipeline's
    `pipeline.utils.hook_utils`. The hooks are removed when the context closes, also on error."""
    assert direction.ndim == 1, "a refusal direction has shape [hidden size]"
    layers = list(hook.decoder_layers(model))
    with contextlib.ExitStack() as registered:
        for layer in layers:
            at_input = hooks.get_direction_ablation_input_pre_hook(direction, intensity)
            handle = layer.register_forward_pre_hook(_on_hidden_states(at_input), with_kwargs=True)
            registered.callback(handle.remove)
        for branch in _residual_branches(layers):
            at_output = hooks.get_direction_ablation_output_hook(direction, intensity)
            registered.callback(branch.register_forward_hook(at_output).remove)
        yield


@dataclasses.dataclass(frozen=True)
class RefusalDirection:
    """The refusal direction as `find` returns it. `direction` has shape [hidden size] and is not
    normalized; the ablation divides it by its norm. `position` counts from the end of the prompt
    (-1 is the last token); `layer` is the decoder layer whose input it is. `instructions` maps
    each split ("harmful_train", ...) to its "sampled" and "kept" counts."""

    direction: torch.Tensor
    position: int
    layer: int
    forward_passes: int
    instructions: Mapping[str, Mapping[str, int]]


@dataclasses.dataclass(frozen=True)
class _Prompts:
    """A batch of prompts as the pipeline reads it: token ids and attention [prompts, longest
    prompt], padded on the left."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class _CountedModel:
    """The model as the pipeline calls it: logits at the last position only, which is all the
    pipeline reads, and a count of the prompts."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device
        self.forward_passes = 0

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        self.forward_passes += int(input_ids.shape[0])
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
            use_cache=False,
            output_hidden_states=False,
        )


@dataclasses.dataclass(frozen=True)
class _SearchedModel:
    """What the pipeline's `generate_directions` and `select_direction` read of a model: the
    attributes of its `ModelBase`, under the pipeline's names."""

    model: _CountedModel
    tokenizer: Any
    tokenize_instructions_fn: Callable[..., _Prompts]
    model_block_modules: Sequence[Any]
    model_attn_modules: Sequence[Any]
    model_mlp_modules: Sequence[Any]
    refusal_toks: list[int]
    eoi_toks: list[int]


def _refusal_probe_instructions(checkout: pathlib.Path) -> list[str]:
    """The harmful instructions with which the pipeline finds the tokens of a refusal: the list
    `instructions` in the method `_get_refusal_toks` of its `ModelBase`, read from the source."""
    path = checkout / PIPELINE_FOLDER / "pipeline" / "model_utils" / "model_base.py"
    if not path.is_file():
        raise RefusalError(f"the checkout has no {path}")
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_refusal_toks":
            for keyword in (k for k in ast.walk(node) if isinstance(k, ast.keyword)):
                if keyword.arg == "instructions":
                    return [str(text) for text in ast.literal_eval(keyword.value)]
    raise RefusalError(f"{path} has no list of instructions in _get_refusal_toks")


def _dataset_instructions(checkout: pathlib.Path, kind: str, split: str) -> list[str]:
    path = checkout / PIPELINE_FOLDER / "dataset" / "splits" / f"{kind}_{split}.json"
    if not path.is_file():
        raise RefusalError(f"the checkout has no {path}")
    return [str(item["instruction"]) for item in json.loads(path.read_text())]


def _tokens_after_instruction(original: original_model.OriginalModel) -> list[int]:
    """The tokens of a rendered prompt that follow the user's instruction: the end of the user
    turn and the start of the model's."""
    messages = [{"role": "user", "content": PROBE_INSTRUCTION}]
    tokenizer = original.tokenizer
    up_to_instruction = tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True, enable_thinking=False
    )
    prefix = list(tokenizer(up_to_instruction, add_special_tokens=False)["input_ids"])
    prompt = original.render_prompt(PROBE_INSTRUCTION)
    if prompt[: len(prefix)] != prefix or len(prompt) == len(prefix):
        raise RefusalError("cannot tell which tokens of a prompt follow the instruction")
    return prompt[len(prefix) :]


def find(
    original: original_model.OriginalModel, checkout: pathlib.Path, search_folder: pathlib.Path
) -> RefusalDirection:
    """Find the refusal direction of `original` (module docstring); the pipeline writes its
    candidates and scores under `search_folder`. Raises RefusalError if a split keeps no
    instruction or the pipeline rejects every candidate: the model has no direction by this
    search."""
    generate = import_from_checkout(checkout, "pipeline.submodules.generate_directions")
    select = import_from_checkout(checkout, "pipeline.submodules.select_direction")
    probe_instructions = _refusal_probe_instructions(checkout)
    counted = _CountedModel(original.model)
    padding = original.stop_token_ids[0]

    def tokenize(instructions: Sequence[str]) -> _Prompts:
        prompts = [original.render_prompt(text) for text in instructions]
        width = max(len(prompt) for prompt in prompts)
        input_ids = torch.full((len(prompts), width), padding, dtype=torch.long)
        attention = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            input_ids[row, width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
            attention[row, width - len(prompt) :] = 1
        return _Prompts(input_ids, attention)

    layers = list(hook.decoder_layers(original.model))
    branches = _residual_branches(layers)
    with torch.inference_mode():
        probes = tokenize(probe_instructions)
        logits = counted(
            input_ids=probes.input_ids.to(counted.device),
            attention_mask=probes.attention_mask.to(counted.device),
        ).logits
        searched = _SearchedModel(
            model=counted,
            tokenizer=original.tokenizer,
            tokenize_instructions_fn=tokenize,
            model_block_modules=layers,
            model_attn_modules=branches[: len(layers)],
            model_mlp_modules=branches[len(layers) :],
            refusal_toks=sorted(set(logits[:, -1].argmax(-1).tolist())),
            eoi_toks=_tokens_after_instruction(original),
        )
        generator = random.Random(SAMPLING_SEED)
        kept: dict[str, list[str]] = {}
        counts: dict[str, dict[str, int]] = {}
        for kind, split in SPLITS:
            count = TRAINING_INSTRUCTIONS if split == "train" else VALIDATION_INSTRUCTIONS
            sampled = generator.sample(_dataset_instructions(checkout, kind, split), count)
            scores = select.get_refusal_scores(
                counted, sampled, tokenize, searched.refusal_toks
            ).tolist()
            name = f"{kind}_{split}"
            kept[name] = [
                text
                for text, score in zip(sampled, scores, strict=True)
                if (score > 0 if kind == "harmful" else score < 0)
            ]
            counts[name] = {"sampled": len(sampled), "kept": len(kept[name])}
        empty = [name for name, texts in kept.items() if not texts]
        if empty:
            raise RefusalError(
                f"the refusal score kept no instruction of {empty} ({counts}), so the model "
                "has no refusal direction by this search"
            )
        candidates = generate.generate_directions(
            searched,
            kept["harmful_train"],
            kept["harmless_train"],
            str(search_folder / "generate_directions"),
        )
        try:
            position, layer, direction = select.select_direction(
                searched,
                kept["harmful_val"],
                kept["harmless_val"],
                candidates,
                str(search_folder / "select_direction"),
            )
        except AssertionError as error:
            raise RefusalError(
                f"the pipeline rejected every candidate direction ({error}), so the model has "
                "no refusal direction by this search"
            ) from error
    return RefusalDirection(
        direction=direction.detach().cpu().contiguous(),
        position=int(position),
        layer=int(layer),
        forward_passes=counted.forward_passes,
        instructions=counts,
    )


def find_and_save(
    original: original_model.OriginalModel, checkout: pathlib.Path, directory: pathlib.Path
) -> dict[str, Any]:
    """`find`, then write DIRECTION_FILE, RECORD_FILE and the pipeline's SEARCH_FOLDER into
    `directory`, creating it and replacing a direction it holds. Returns the record written."""
    directory.mkdir(parents=True, exist_ok=True)
    found = find(original, checkout, directory / SEARCH_FOLDER)
    safetensors.torch.save_file({"direction": found.direction}, directory / DIRECTION_FILE)
    record = {
        "model": original.name,
        "model_revision": original.revision,
        "repository": REPOSITORY,
        "commit": COMMIT,
        "position": found.position,
        "layer": found.layer,
        "forward_passes": found.forward_passes,
        "instructions": found.instructions,
    }
    run_record.write_json(directory / RECORD_FILE, record)
    return record


def has_direction(directory: pathlib.Path) -> bool:
    """Whether the method data `directory` holds a refusal direction."""
    return (directory / DIRECTION_FILE).is_file() and (directory / RECORD_FILE).is_file()


def load(directory: pathlib.Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """The direction in `directory` (shape [hidden size], on the CPU) and its record. Raises
    RefusalError if the directory holds none."""
    if not has_direction(directory):
        raise RefusalError(
            f"{directory} holds no refusal direction; `steering find-refusal-direction` writes "
            "one"
        )
    direction = safetensors.torch.load_file(directory / DIRECTION_FILE)["direction"]
    if direction.ndim != 1 or not torch.isfinite(direction).all():
        raise RefusalError(f"{directory / DIRECTION_FILE} does not hold one finite vector")
    return direction, run_record.read_json(directory / RECORD_FILE)
