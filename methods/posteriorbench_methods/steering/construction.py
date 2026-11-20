"""Build the steering vectors of a behavior: persona vectors, the mean activation on responses
that show the behavior minus that on responses that avoid it. `build` takes four steps.

1. Construction responses. The original model answers every question under every instruction
   pair: positive responses under the instruction for the behavior, drawn once under the refusal
   ablation and once without (one vector each), and negative ones under the instruction against
   it, which both vectors share.
2. Labels. The behavior's rule says whether a response shows the behavior, and the judge rates
   its coherence (`coherence`). For a behavior that no rule defines, the judge also scores the
   behavior with the behavior's judge prompt; a response without that score is kept on neither
   side.
3. Filter. A response is kept if it is coherent and shows the behavior (positive) or does not
   (negative).
4. Vectors. A response's activation at a layer is the residual stream after the layer, averaged
   over its tokens; a vector is the mean over kept positive minus kept negative responses.
   Activations of responses drawn under the ablation are read under the ablation.

The vectors go to the method data directory (`vectors`). The construction keeps its own files
under `construction/<behavior>/`: `source.json`, one file of rows per kind, question and pair in
`responses/`, `coherence/` and `behavior/`, and `summary.json`. A file that exists is not made
again, so a construction that stopped goes on where it was.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import math
import pathlib
from collections.abc import Sequence
from typing import Any

import torch
from posteriorbench import benchmark, ledger, original_model, rules

from posteriorbench_methods.common import charges, errors, hf_runtime, run_record, seeds, tracking
from posteriorbench_methods.steering import coherence, construction_prompts, cost, hook, refusal
from posteriorbench_methods.steering import vectors as steering_vectors

WITH_ABLATION, WITHOUT_ABLATION = steering_vectors.VECTORS
POSITIVE_WITH_ABLATION, POSITIVE_WITHOUT_ABLATION, NEGATIVE = steering_vectors.KEPT_RESPONSES
# The kind of positive response that each vector is built from.
POSITIVE_RESPONSES = {
    WITH_ABLATION: POSITIVE_WITH_ABLATION,
    WITHOUT_ABLATION: POSITIVE_WITHOUT_ABLATION,
}
CONSTRUCTION_FOLDER = "construction"
SOURCE_FILE = "source.json"
SUMMARY_FILE = "summary.json"
# The folders of the judge's answers, and the key of the score in a row of each.
COHERENCE = "coherence"
BEHAVIOR = "behavior"

Row = dict[str, Any]


class ConstructionError(errors.MethodsError, RuntimeError):
    """Settings the model does not fit, a behavior without one rule or judge prompt, a vector
    with ablation without a refusal direction, or a directory built with other inputs."""


@dataclasses.dataclass(frozen=True)
class Settings:
    """The settings of the construction; the defaults are the paper's.

    `ablation_intensity` is the share of the component along the refusal direction that the
    ablation removes. `ablated_top_k` limits each token of the responses drawn under the ablation
    to the most probable ones; 0, as for every other response, is the whole vocabulary. Both
    thresholds are inclusive. `layers` count from 0; None builds after every layer."""

    vectors: tuple[str, ...] = steering_vectors.VECTORS
    responses_per_prompt: int = 10
    response_limit: int = 1000
    ablation_intensity: float = 0.8
    ablated_top_k: int = 50
    coherence_threshold: float = 50.0
    behavior_threshold: float = 50.0
    layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.vectors or len(set(self.vectors)) != len(self.vectors):
            raise ValueError(f"vectors are distinct and at least one, got {self.vectors}")
        if any(vector not in steering_vectors.VECTORS for vector in self.vectors):
            raise ValueError(f"the vectors are {steering_vectors.VECTORS}, got {self.vectors}")
        if self.responses_per_prompt < 1 or self.response_limit < 1:
            raise ValueError("responses_per_prompt and response_limit are at least 1")
        if not (math.isfinite(self.ablation_intensity) and 0 < self.ablation_intensity <= 1):
            raise ValueError(f"ablation_intensity lies in (0, 1], got {self.ablation_intensity}")
        if self.ablated_top_k < 0:
            raise ValueError(f"ablated_top_k is 0 or more, got {self.ablated_top_k}")
        if self.layers is not None and (
            any(layer < 0 for layer in self.layers) or len(set(self.layers)) != len(self.layers)
        ):
            raise ValueError(f"layers are distinct and counted from 0, got {self.layers}")

    def kinds(self) -> list[str]:
        """The kinds of response these vectors need, in drawing order: each vector's positive
        responses, then the negative ones."""
        positive = [
            POSITIVE_RESPONSES[vector]
            for vector in steering_vectors.VECTORS
            if vector in self.vectors
        ]
        return [*positive, NEGATIVE]


@dataclasses.dataclass(frozen=True)
class Ablation:
    """The arguments of `refusal.ablate`: the direction, shape [hidden size], and the hooks."""

    direction: torch.Tensor
    hooks: Any


def open_ablation(
    settings: Settings, directory: pathlib.Path, checkout: pathlib.Path | None, model: str | None
) -> tuple[Ablation | None, int]:
    """The ablation (None if no vector is built with it) and the forward passes that found the
    direction in `directory` (0 if it holds none). Raises ConstructionError if the vector with
    ablation is asked for without `checkout`, or the direction is not of the model `model`."""
    if not refusal.has_direction(directory) and WITH_ABLATION not in settings.vectors:
        return None, 0
    direction, record = refusal.load(directory)
    if record.get("model") != model:
        raise ConstructionError(
            f"the refusal direction in {directory} is that of {record.get('model')!r}, not of "
            f"{model!r}"
        )
    passes = int(record["forward_passes"])
    if WITH_ABLATION not in settings.vectors:
        return None, passes
    if checkout is None:
        raise ConstructionError(
            "the vector with ablation needs the hooks of the refusal-direction pipeline: name "
            "the checkout (--lmtailrisk-checkout), or build the other vector alone (--set "
            "vectors=without_ablation)"
        )
    hooks = refusal.import_from_checkout(checkout, "pipeline.utils.hook_utils")
    return Ablation(direction, hooks), passes


def behavior_rule(bench: benchmark.Benchmark, behavior: str) -> str:
    """The rule that the benchmark's tasks of `behavior` share. Raises ConstructionError unless
    there is exactly one."""
    found = sorted({task.rule for task in bench.tasks("all") if task.behavior == behavior})
    if len(found) != 1:
        raise ConstructionError(
            f"the vectors of a behavior need its rule, and the dataset's tasks with the "
            f"behavior {behavior!r} have {len(found)} rules"
        )
    return found[0]


def response_seed(behavior: str, kind: str, question: int, pair: int) -> int:
    """The seed of one kind of response to one question and pair. It omits the response limit,
    so a smaller limit gives prefixes of the same responses."""
    return seeds.stream_seed(
        "steering", behavior, None, "construction-responses", kind, question, pair
    )


class _NextTokenLogits:
    """The next-token logits of the original model while one batch of construction responses is
    drawn (`hf_runtime.LogitSource`, with the model as q and no p)."""

    def __init__(self, source: _PromptedModel, rows: int) -> None:
        self.source = source
        self.prompt = torch.tensor(
            [list(source.prompt_ids)] * rows, dtype=torch.long, device=source.device
        )
        self.cache: Any = None

    def logits(
        self, response: torch.Tensor, attention: torch.Tensor, *, q: bool, p: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """float32 next-token logits [rows, vocab] after the prompt and `response` [rows, t]. At
        the first position the stop tokens are excluded, so a response has a token before its
        stop token; with `top_k`, all but the top_k most probable are excluded. The first call
        runs the prompt; later ones feed the last token."""
        assert q and not p, "a construction response is drawn from one model"
        first = response.shape[1] == 0
        prompt_attention = torch.ones_like(self.prompt)
        with torch.inference_mode():
            out = self.source.model(
                input_ids=self.prompt if first else response[:, -1:],
                attention_mask=(
                    prompt_attention if first else torch.cat((prompt_attention, attention), dim=1)
                ),
                past_key_values=self.cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            if out.past_key_values is None:
                raise RuntimeError("the model returned no recurrent/KV cache")
            self.cache = out.past_key_values
            logits = out.logits[:, -1].float().clone()
            if first:
                logits[:, list(self.source.stop_token_ids)] = -torch.inf
            top_k = self.source.top_k
            if top_k:
                smallest_kept = torch.topk(logits, min(top_k, logits.shape[-1]))[0][:, -1:]
                logits.masked_fill_(logits < smallest_kept, -torch.inf)
        return logits, None


@dataclasses.dataclass(frozen=True, eq=False)
class _PromptedModel:
    """The original model under one construction prompt, as `hf_runtime.sample_groups` reads a
    sampler (`hf_runtime.NextTokenModels`)."""

    model: Any
    prompt_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    top_k: int

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def start(self, rows: int) -> hf_runtime.LogitSource:
        return _NextTokenLogits(self, rows)


def draw_responses(
    original: original_model.OriginalModel,
    prompt_ids: Sequence[int],
    count: int,
    seed: int,
    *,
    response_limit: int,
    top_k: int = 0,
) -> list[list[int]]:
    """`count` responses to `prompt_ids` as token ids, one batch seeded with `seed`, at
    temperature 1. A response has at most `response_limit` tokens and ends after its first stop
    token, which is never its first token. Under a smaller limit and the same seed, the responses
    are prefixes of those under a larger one. Under an open ablation the responses are ablated."""
    stop_token_ids = tuple(int(token) for token in original.stop_token_ids)
    models = _PromptedModel(original.model, tuple(prompt_ids), stop_token_ids, top_k)
    generation = hf_runtime.Generation(hf_runtime.ResponseFormat(stop_token_ids, response_limit))
    (drawn,) = hf_runtime.sample_groups(models, [hf_runtime.DrawGroup(count, seed)], generation)
    return [draw.token_ids for draw in drawn]


def mean_activations(
    original: original_model.OriginalModel,
    prompt_ids: Sequence[int],
    response: Sequence[int],
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """The residual stream after each of `layers`, averaged over the response's tokens without a
    final stop token: float32 [hidden size] on the CPU. One forward pass; under an open ablation
    the activations are ablated."""
    visible = len(response) - int(response[-1] in original.stop_token_ids)
    assert visible > 0, "a construction response has a token before its stop token"
    ids = torch.tensor([[*prompt_ids, *response]], dtype=torch.long, device=original.device)
    # The output of each layer, [1, positions, hidden size], is kept as the tensor that the next
    # layer receives and is averaged after the pass. An ablation hook that changes a layer's
    # input in place has by then changed the kept tensor too, so what is read is the residual
    # stream as the next layer sees it.
    outputs: dict[int, torch.Tensor] = {}

    def keep_output_of(layer: int) -> Any:
        def keep(module: Any, inputs: Any, output: Any) -> None:
            outputs[layer] = output[0] if isinstance(output, tuple) else output

        return keep

    decoder_layers = hook.decoder_layers(original.model)
    with contextlib.ExitStack() as registered:
        for layer in layers:
            handle = decoder_layers[layer].register_forward_hook(keep_output_of(layer))
            registered.callback(handle.remove)
        with torch.inference_mode():
            original.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
            )
    start = len(prompt_ids)
    return {
        layer: outputs[layer][0, start : start + visible].float().mean(0).cpu() for layer in layers
    }


def is_kept(
    hit: bool | None, coherence_score: float | None, *, positive: bool, threshold: float
) -> bool:
    """The filter: coherent, and `hit` is `positive`. A response without a behavior score (`hit`
    None) is never kept."""
    return coherence.is_coherent(coherence_score, threshold) and hit is positive


@dataclasses.dataclass(frozen=True)
class Construction:
    """What the steps of one construction share. `rule` is None for a behavior labeled by the
    judge prompt of `prompts`; `ablation` is None if no vector is built with it."""

    original: original_model.OriginalModel
    prompts: construction_prompts.ConstructionPrompts
    rule: rules.Rule | None
    settings: Settings
    ablation: Ablation | None
    folder: pathlib.Path
    tracker: tracking.Tracker

    @property
    def parts(self) -> list[tuple[int, int]]:
        """Every (question, instruction pair), by index, in the order in which they are drawn."""
        return [
            (question, pair)
            for question in range(len(self.prompts.questions))
            for pair in range(len(self.prompts.instruction_pairs))
        ]

    def prompt_ids(self, kind: str, question: int, pair: int) -> list[int]:
        """The rendered chat prompt: the pair's system prompt for `kind`, then the question."""
        system = self.prompts.system_prompt(
            self.prompts.instruction_pairs[pair], positive=kind != NEGATIVE
        )
        return self.original.render_prompt(self.prompts.questions[question], system)

    def ablated(self, kind: str) -> contextlib.AbstractContextManager[None]:
        """The context for generating and reading responses of `kind`: the ablation for
        POSITIVE_WITH_ABLATION, else nothing."""
        if kind != POSITIVE_WITH_ABLATION:
            return contextlib.nullcontext()
        assert self.ablation is not None, "the vector with ablation needs the refusal direction"
        return refusal.ablate(
            self.original.model,
            self.ablation.direction,
            self.settings.ablation_intensity,
            self.ablation.hooks,
        )

    def path(self, stage: str, kind: str, question: int, pair: int) -> pathlib.Path:
        """The file of one question and pair; `stage` is "responses", "coherence" or "behavior"."""
        return self.folder / stage / kind / f"question{question:02d}_pair{pair}.jsonl"


def check_source(construction: Construction, judge_model: str, judge_prompt: str) -> None:
    """Write `source.json` (model, prompts, rule, judge, response settings) in a new folder;
    raise ConstructionError if an existing one was written with other inputs."""
    original, settings = construction.original, construction.settings
    source = {
        "model": original.name,
        "model_revision": original.revision,
        "prompts": construction.prompts.to_json(),
        "rule": None if construction.rule is None else construction.rule.expression,
        "responses_per_prompt": settings.responses_per_prompt,
        "response_limit": settings.response_limit,
        "ablation_intensity": settings.ablation_intensity,
        "ablated_top_k": settings.ablated_top_k,
        "judge_model": judge_model,
        "judge_prompt": judge_prompt,
    }
    path = construction.folder / SOURCE_FILE
    if not path.exists():
        run_record.write_json(path, source)
    elif run_record.read_json(path) != run_record.jsonable(source):
        raise ConstructionError(
            f"{construction.folder} holds the construction responses of another model, other "
            "prompts, another rule or judge, or other settings; use another directory or delete "
            "it"
        )


def draw_construction_responses(construction: Construction) -> None:
    """Step 1 and the rule's half of step 2: draw and write the responses the folder lacks,
    labeled by the rule. A token beyond the tokenizer's vocabulary makes a response a miss;
    without a rule, `hit` is null."""
    original, settings = construction.original, construction.settings
    vocabulary = len(original.tokenizer)
    for kind in settings.kinds():
        with construction.ablated(kind):
            for question, pair in construction.parts:
                path = construction.path("responses", kind, question, pair)
                if path.exists():
                    continue
                seed = response_seed(construction.prompts.behavior, kind, question, pair)
                responses = draw_responses(
                    original,
                    construction.prompt_ids(kind, question, pair),
                    settings.responses_per_prompt,
                    seed,
                    response_limit=settings.response_limit,
                    top_k=settings.ablated_top_k if kind == POSITIVE_WITH_ABLATION else 0,
                )
                rows: list[Row] = []
                for index, token_ids in enumerate(responses):
                    text = original.decode(token_ids)
                    hit = None
                    if construction.rule is not None:
                        hit = construction.rule(text) and max(token_ids) < vocabulary
                    rows.append(
                        {
                            "draw_index": index,
                            "token_ids": token_ids,
                            "hit": hit,
                            "seed": seed,
                            "text": text,
                        }
                    )
                run_record.write_jsonl(path, rows)
                construction.tracker.event(
                    "construction_responses",
                    kind=kind,
                    question=question,
                    pair=pair,
                    hits=sum(row["hit"] is True for row in rows),
                )


def judge_construction_responses(
    construction: Construction,
    judge: coherence.Judge,
    judge_prompt: str,
    *,
    concurrency: int,
    stage: str = COHERENCE,
) -> None:
    """The judge's half of step 2: ask `judge` with `judge_prompt` about every response without
    an answer yet, `concurrency` at a time, and write the answers. `stage` ("coherence" or
    "behavior") names their folder and the key of the score."""
    pending = [
        (kind, question, pair)
        for kind in construction.settings.kinds()
        for question, pair in construction.parts
        if not construction.path(stage, kind, question, pair).exists()
    ]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    try:
        asked: list[tuple[pathlib.Path, list[tuple[int, concurrent.futures.Future[Row]]]]] = []
        for kind, question, pair in pending:
            rows = run_record.read_jsonl(construction.path("responses", kind, question, pair))
            answers = [
                (
                    int(row["draw_index"]),
                    executor.submit(
                        coherence.score_response,
                        judge,
                        judge_prompt,
                        construction.prompts.questions[question],
                        str(row["text"]),
                        name=stage,
                    ),
                )
                for row in rows
            ]
            asked.append((construction.path(stage, kind, question, pair), answers))
        for done, (path, answers) in enumerate(asked, start=1):
            run_record.write_jsonl(
                path, [{"draw_index": index, **answer.result()} for index, answer in answers]
            )
            if done % 50 == 0 or done == len(asked):
                construction.tracker.event(stage, judged=done, of=len(asked))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def labeled_responses(construction: Construction, kind: str) -> list[Row]:
    """The response rows of `kind` in drawing order, with `question`, `pair`, `coherence` and
    `kept`. Without a rule, `hit` is the behavior score against `behavior_threshold` (None
    without a score), and the score is under `behavior`."""
    settings = construction.settings
    labeled: list[Row] = []
    for question, pair in construction.parts:
        rows = run_record.read_jsonl(construction.path("responses", kind, question, pair))
        answers = run_record.read_jsonl(construction.path(COHERENCE, kind, question, pair))
        assert [row["draw_index"] for row in rows] == [answer["draw_index"] for answer in answers]
        if construction.rule is None:
            judged = run_record.read_jsonl(construction.path(BEHAVIOR, kind, question, pair))
            assert [row["draw_index"] for row in rows] == [one["draw_index"] for one in judged]
            scores = [one[BEHAVIOR] for one in judged]
            rows = [
                {
                    **row,
                    "hit": None if score is None else score >= settings.behavior_threshold,
                    BEHAVIOR: score,
                }
                for row, score in zip(rows, scores, strict=True)
            ]
        for row, answer in zip(rows, answers, strict=True):
            kept = is_kept(
                row["hit"],
                answer["coherence"],
                positive=kind != NEGATIVE,
                threshold=settings.coherence_threshold,
            )
            labeled.append(
                {
                    **row,
                    "question": question,
                    "pair": pair,
                    "coherence": answer["coherence"],
                    "kept": kept,
                }
            )
    return labeled


def mean_activation_of_kept(
    construction: Construction, kind: str, labeled: Sequence[Row], layers: Sequence[int]
) -> dict[int, torch.Tensor] | None:
    """The mean activation of the kept responses of `kind` at each layer: float32 [hidden size],
    summed in float64. None if none was kept. One forward pass per kept response."""
    kept = [row for row in labeled if row["kept"]]
    if not kept:
        return None
    hidden_size = int(construction.original.model.config.hidden_size)
    totals = {layer: torch.zeros(hidden_size, dtype=torch.float64) for layer in layers}
    with construction.ablated(kind):
        for row in kept:
            activations = mean_activations(
                construction.original,
                construction.prompt_ids(kind, int(row["question"]), int(row["pair"])),
                row["token_ids"],
                layers,
            )
            for layer in layers:
                totals[layer] += activations[layer].double()
    return {layer: (total / len(kept)).float() for layer, total in totals.items()}


def build(
    original: original_model.OriginalModel,
    prompts: construction_prompts.ConstructionPrompts,
    rule: str | None,
    judge: coherence.Judge,
    judge_prompt: str,
    *,
    directory: pathlib.Path,
    settings: Settings,
    ablation: Ablation | None,
    refusal_direction_passes: int,
    judge_model: str,
    judge_concurrency: int = 100,
    tracker: tracking.Tracker | None = None,
) -> dict[str, Any]:
    """Build the vectors of `prompts`' behavior (module docstring) and write them with their cost
    into `directory` (`vectors.save`). Returns the summary also written to `summary.json`.

    `rule` is the task table's rule, or None for a behavior labeled by the judge prompt of
    `prompts`. `judge_prompt` is the coherence prompt. `refusal_direction_passes` is charged to
    every behavior of the directory (0 without a direction). A vector is built if the filter kept
    one of its positive responses and one negative response. Raises ConstructionError if the
    inputs do not fit one another or the directory."""
    if (rule is None) == (prompts.judge_prompt is None):
        raise ConstructionError(
            f"the behavior {prompts.behavior!r} needs one of a rule and a judge prompt in its "
            "construction prompts, which says whether a response shows it"
        )
    if WITH_ABLATION in settings.vectors and ablation is None:
        raise ConstructionError(
            "the vector with ablation needs the model's refusal direction; without it, build "
            "the other vector alone (--set vectors=without_ablation)"
        )
    layer_count = len(hook.decoder_layers(original.model))
    layers = list(range(layer_count)) if settings.layers is None else list(settings.layers)
    if any(layer >= layer_count for layer in layers):
        raise ConstructionError(
            f"the model has {layer_count} layers; the settings ask for {layers}"
        )
    construction = Construction(
        original=original,
        prompts=prompts,
        rule=None if rule is None else rules.parse_rule(rule),
        settings=settings,
        ablation=ablation,
        folder=directory / CONSTRUCTION_FOLDER / prompts.behavior,
        tracker=tracker if tracker is not None else tracking.Tracker(),
    )
    check_source(construction, judge_model, judge_prompt)
    draw_construction_responses(construction)
    judge_construction_responses(construction, judge, judge_prompt, concurrency=judge_concurrency)
    if prompts.judge_prompt is not None:
        judge_construction_responses(
            construction,
            judge,
            prompts.judge_prompt,
            concurrency=judge_concurrency,
            stage=BEHAVIOR,
        )
    labeled = {kind: labeled_responses(construction, kind) for kind in settings.kinds()}
    means = {
        kind: mean_activation_of_kept(construction, kind, rows, layers)
        for kind, rows in labeled.items()
    }
    tensors: dict[str, dict[int, torch.Tensor]] = {}
    negative = means[NEGATIVE]
    for vector in settings.vectors:
        positive = means[POSITIVE_RESPONSES[vector]]
        if positive is not None and negative is not None:
            tensors[vector] = {layer: positive[layer] - negative[layer] for layer in layers}
    built = steering_vectors.ConstructionCost(
        refusal_direction_passes=refusal_direction_passes,
        construction_responses=sum(len(rows) for rows in labeled.values()),
        kept_responses={
            kind: sum(row["kept"] for row in labeled.get(kind, []))
            for kind in steering_vectors.KEPT_RESPONSES
        },
    )
    steering_vectors.save(directory, prompts.behavior, tensors, built)
    spend = ledger.Ledger()
    charges.apply(spend, cost.vector_construction(built))
    # One call for the coherence of each response, and one more for a behavior that the judge
    # labels.
    by_judge = prompts.judge_prompt is not None
    judged = built.construction_responses * (2 if by_judge else 1)
    asked_for = "coherence and behavior judge" if by_judge else "coherence judge"
    spend.charge(
        ledger.Category.UNCOUNTED,
        inferences=0,
        note=f"{asked_for} {judge_model}: {judged} calls, not counted",
    )
    threshold = settings.coherence_threshold
    summary = {
        "behavior": prompts.behavior,
        "model": original.name,
        "model_revision": original.revision,
        "settings": run_record.settings_json(settings),
        "responses": {
            kind: {
                "responses": len(rows),
                "hits": sum(row["hit"] is True for row in rows),
                "coherent": sum(coherence.is_coherent(row["coherence"], threshold) for row in rows),
                "kept": sum(row["kept"] for row in rows),
            }
            for kind, rows in labeled.items()
        },
        "vectors": {
            vector: {
                "built": vector in tensors,
                "norms": {
                    str(layer): float(tensor.norm())
                    for layer, tensor in tensors.get(vector, {}).items()
                },
            }
            for vector in settings.vectors
        },
        "construction_cost": built.to_json(),
        "judge": {"model": judge_model, "calls": judged},
        "ledger": run_record.ledger_record(spend),
    }
    run_record.write_json(construction.folder / SUMMARY_FILE, summary)
    construction.tracker.event(
        "vectors", behavior=prompts.behavior, built=sorted(tensors), kept=built.kept_responses
    )
    return summary
