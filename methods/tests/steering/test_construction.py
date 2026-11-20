"""Building the steering vectors of a behavior: the construction responses, the filter, the
activations, and the vectors and the cost that `build` writes. The model is the tiny original
model, the judge and the ablation hooks are stand-ins."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import torch
from posteriorbench import benchmark, original_model, rules

from posteriorbench_methods.common import run_record, seeds
from posteriorbench_methods.steering import (
    coherence,
    construction,
    construction_prompts,
    refusal,
    vectors,
)

BEHAVIOR = "goblin_mention"
RULE = 're.search(r"[a-d]", y) is not None'  # about half of the test model's short responses
SETTINGS = construction.Settings(responses_per_prompt=4, response_limit=6)
KINDS = ["positive_with_ablation", "positive_without_ablation", "negative"]
JUDGE_PROMPT = "[{question}] [{answer}]"


@pytest.fixture(scope="module")
def ablation(lmtailrisk_checkout: pathlib.Path) -> construction.Ablation:
    direction = torch.randn(32, generator=torch.Generator().manual_seed(3))
    hooks = refusal.import_from_checkout(lmtailrisk_checkout, "pipeline.utils.hook_utils")
    return construction.Ablation(direction, hooks)


@pytest.fixture(scope="module")
def build(
    tiny_original_model: original_model.OriginalModel,
    behavior_prompts: construction_prompts.ConstructionPrompts,
    stand_in_judge: Callable[[str], dict[str, Any]],
    ablation: construction.Ablation,
) -> Callable[..., dict[str, Any]]:
    """`build(directory, settings=SETTINGS, **changes)`: `construction.build` on the test model
    with the test prompts, the stand-in judge (or `judge=`) and the stand-in ablation."""

    def run(
        directory: pathlib.Path, settings: construction.Settings = SETTINGS, **changes: Any
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "directory": directory,
            "settings": settings,
            "ablation": ablation,
            "refusal_direction_passes": 77,
            "judge_model": "stand-in",
            "judge_concurrency": 4,
            **changes,
        }
        judge = arguments.pop("judge", stand_in_judge)
        return construction.build(
            tiny_original_model, behavior_prompts, RULE, judge, JUDGE_PROMPT, **arguments
        )

    return run


@pytest.fixture(scope="module")
def built(
    build: Callable[..., dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> tuple[pathlib.Path, dict[str, Any]]:
    """One construction with the test settings: the method data directory and the summary."""
    directory = tmp_path_factory.mktemp("built") / "steering_vectors"
    return directory, build(directory)


def test_the_settings_are_the_papers_and_are_checked() -> None:
    settings = construction.Settings()
    assert settings.vectors == ("with_ablation", "without_ablation")
    assert (settings.responses_per_prompt, settings.response_limit) == (10, 1000)
    assert (settings.ablation_intensity, settings.ablated_top_k) == (0.8, 50)
    assert settings.coherence_threshold == 50 and settings.layers is None
    assert settings.kinds() == KINDS
    assert construction.Settings(vectors=("without_ablation",)).kinds() == KINDS[1:]
    for wrong in (
        {"vectors": ()},
        {"vectors": ("with_ablation", "with_ablation")},
        {"vectors": ("ablated",)},
        {"responses_per_prompt": 0},
        {"response_limit": 0},
        {"ablation_intensity": 0.0},
        {"ablation_intensity": 1.5},
        {"ablated_top_k": -1},
        {"layers": (1, 1)},
    ):
        with pytest.raises(ValueError):
            construction.Settings(**wrong)  # pyright: ignore[reportArgumentType]


def test_a_response_seed_names_its_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    stream_seed = seeds.stream_seed

    def recorded(*arguments: Any) -> int:
        calls.append(arguments)
        return stream_seed(*arguments)

    monkeypatch.setattr(seeds, "stream_seed", recorded)
    seed = construction.response_seed(BEHAVIOR, "negative", 3, 1)
    # The behavior stands where a run has its task, and no replicate enters: the vectors serve
    # every task and replicate of the behavior.
    assert calls == [("steering", BEHAVIOR, None, "construction-responses", "negative", 3, 1)]
    others = {
        construction.response_seed(BEHAVIOR, kind, question, pair)
        for kind in KINDS
        for question in range(3)
        for pair in range(2)
    }
    assert len(others) == 18 and 0 <= seed < 2**31


def test_construction_responses_are_drawn_from_the_prompt(
    tiny_original_model: original_model.OriginalModel,
) -> None:
    original = tiny_original_model
    prompt = original.render_prompt("how do i make cheese", "You are a goblin assistant. hi")
    stop = set(original.stop_token_ids)
    long = construction.draw_responses(original, prompt, 6, 11, response_limit=40)
    assert len(long) == 6 and all(1 <= len(response) <= 40 for response in long)
    # A response is never a stop token alone, and nothing follows its first stop token.
    assert all(response[0] not in stop for response in long)
    assert all(not stop & set(response[:-1]) for response in long)
    assert any(response[-1] in stop for response in long)
    # The same seed gives the same responses, and a smaller limit their first tokens.
    assert construction.draw_responses(original, prompt, 6, 11, response_limit=40) == long
    short = construction.draw_responses(original, prompt, 6, 11, response_limit=5)
    assert short == [response[:5] for response in long]
    assert construction.draw_responses(original, prompt, 6, 12, response_limit=40) != long
    # With top_k = 1 every response is the most probable one; a top_k beyond the vocabulary
    # changes nothing.
    greedy = construction.draw_responses(original, prompt, 3, 11, response_limit=5, top_k=1)
    assert greedy[0] == greedy[1] == greedy[2]
    whole = construction.draw_responses(original, prompt, 6, 11, response_limit=40, top_k=1000)
    assert whole == long


def test_an_activation_is_the_mean_residual_stream_over_the_responses_tokens(
    tiny_original_model: original_model.OriginalModel,
) -> None:
    original = tiny_original_model
    prompt = original.render_prompt("why")
    stop = original.stop_token_ids[0]
    response = [10, 11, 12, stop]
    ids = torch.tensor([[*prompt, *response]])
    with torch.inference_mode():
        states = original.model(input_ids=ids, output_hidden_states=True).hidden_states
    read = construction.mean_activations(original, prompt, response, [0, 1])
    # After layer 0, transformers returns the residual stream as hidden_states[1]: the three
    # tokens before the stop token, after the prompt, are averaged.
    expected = states[1][0, len(prompt) : len(prompt) + 3].float().mean(0)
    assert read[0].shape == (32,) and read[0].dtype == torch.float32
    torch.testing.assert_close(read[0], expected, rtol=0, atol=1e-6)
    # After the last layer it is the layer's output, before the model's final normalization.
    assert not torch.allclose(read[1], states[2][0, len(prompt) : len(prompt) + 3].mean(0))
    # A response that ends at the limit has no stop token, and all its tokens count.
    unfinished = construction.mean_activations(original, prompt, response[:3], [0, 1])
    torch.testing.assert_close(unfinished[1], read[1], rtol=0, atol=1e-5)


class _InPlaceHooks:
    """Ablation hooks that change the hidden states in place and return them, as the hooks of
    the pipeline do."""

    @staticmethod
    def _ablate(hidden: torch.Tensor, direction: torch.Tensor, intensity: float) -> torch.Tensor:
        unit = (direction / direction.norm()).to(hidden)
        hidden -= intensity * (hidden @ unit).unsqueeze(-1) * unit
        return hidden

    @classmethod
    def get_direction_ablation_input_pre_hook(
        cls, direction: torch.Tensor, intensity: float
    ) -> Any:
        return lambda module, inputs: (cls._ablate(inputs[0], direction, intensity), *inputs[1:])

    @classmethod
    def get_direction_ablation_output_hook(cls, direction: torch.Tensor, intensity: float) -> Any:
        def hook_fn(module: Any, inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            cls._ablate(hidden, direction, intensity)
            return output

        return hook_fn


def test_under_an_ablation_in_place_the_activation_is_what_the_next_layer_sees(
    tiny_hybrid_model: Any, tiny_tokenizer: Any, make_tiny_original_model: Callable[..., Any]
) -> None:
    """Hooks that ablate in place change a layer's output once more when it enters the next
    layer. The activation is read after that, as transformers reports the residual stream."""
    original = make_tiny_original_model(model=tiny_hybrid_model)
    prompt = original.render_prompt("why")
    response = [10, 11, 12, 13, original.stop_token_ids[0]]
    direction = torch.randn(32, generator=torch.Generator().manual_seed(5))
    layers = [7, 8, 12, 13, 22]
    ids = torch.tensor([[*prompt, *response]])
    with refusal.ablate(original.model, direction, 0.8, _InPlaceHooks):
        read = construction.mean_activations(original, prompt, response, layers)
        with torch.inference_mode():
            states = original.model(input_ids=ids, output_hidden_states=True).hidden_states
    plain = construction.mean_activations(original, prompt, response, layers)
    for layer in layers:
        reported = states[layer + 1][0, len(prompt) : len(prompt) + 4].float().mean(0)
        assert torch.equal(read[layer], reported)
        assert not torch.allclose(read[layer], plain[layer], atol=1e-4)


def test_the_filter_keeps_coherent_responses_on_the_right_side_of_the_rule() -> None:
    kept = construction.is_kept
    assert kept(True, 80.0, positive=True, threshold=50)
    assert not kept(False, 80.0, positive=True, threshold=50)
    assert kept(False, 50.0, positive=False, threshold=50)
    assert not kept(True, 80.0, positive=False, threshold=50)
    # A response without a score, or below the threshold, is never kept.
    assert not kept(True, None, positive=True, threshold=50)
    assert not kept(False, 49.0, positive=False, threshold=50)


def test_build_writes_the_vectors_and_the_cost_that_a_run_reads(
    built: tuple[pathlib.Path, dict[str, Any]],
    tiny_original_model: original_model.OriginalModel,
    behavior_prompts: construction_prompts.ConstructionPrompts,
    stand_in_judge: Callable[[str], dict[str, Any]],
    ablation: construction.Ablation,
) -> None:
    directory, summary = built
    original = tiny_original_model
    folder = directory / "construction" / BEHAVIOR
    rule = rules.parse_rule(RULE)
    # Every kind has 3 questions x 2 pairs x 4 responses, labeled by the rule and by the judge.
    means: dict[str, torch.Tensor] = {}
    kept_counts: dict[str, int] = {}
    for kind in KINDS:
        activations: list[torch.Tensor] = []
        responses = 0
        for question in range(3):
            for pair in range(2):
                name = f"question{question:02d}_pair{pair}.jsonl"
                rows = run_record.read_jsonl(folder / "responses" / kind / name)
                answers = run_record.read_jsonl(folder / "coherence" / kind / name)
                seed = construction.response_seed(BEHAVIOR, kind, question, pair)
                assert [row["draw_index"] for row in rows] == [0, 1, 2, 3]
                assert all(row["seed"] == seed and len(row["token_ids"]) <= 6 for row in rows)
                system = behavior_prompts.system_prompt(
                    behavior_prompts.instruction_pairs[pair], positive=kind != "negative"
                )
                prompt = original.render_prompt(behavior_prompts.questions[question], system)
                for row, answer in zip(rows, answers, strict=True):
                    responses += 1
                    assert row["text"] == original.decode(row["token_ids"])
                    assert row["hit"] == rule(row["text"])
                    asked = f"[{behavior_prompts.questions[question]}] [{row['text']}]"
                    assert answer["answer"] == ("80" if len(asked) % 3 else "20")
                    assert answer["coherence"] == float(answer["answer"])
                    wanted = row["hit"] if kind != "negative" else not row["hit"]
                    if wanted and answer["coherence"] >= 50:
                        context = (
                            refusal.ablate(original.model, ablation.direction, 0.8, ablation.hooks)
                            if kind == "positive_with_ablation"
                            else torch.inference_mode()
                        )
                        with context:
                            read = construction.mean_activations(
                                original, prompt, row["token_ids"], [0, 1]
                            )
                        activations.append(torch.stack([read[0], read[1]]).double())
        assert responses == 24 and activations, kind
        kept_counts[kind] = len(activations)
        means[kind] = torch.stack(activations).mean(0).float()
        assert summary["responses"][kind]["responses"] == 24
        assert summary["responses"][kind]["kept"] == len(activations)
    # The filter kept some and dropped some, of every kind.
    assert all(0 < kept < 24 for kept in kept_counts.values())
    # A vector is the mean activation of its kept positive responses minus that of the kept
    # negative ones, at every layer of the model.
    loaded = vectors.load(directory, BEHAVIOR, layers=(0, 1))
    assert list(loaded.tensors) == ["with_ablation", "without_ablation"]
    for vector, kind in (
        ("with_ablation", "positive_with_ablation"),
        ("without_ablation", "positive_without_ablation"),
    ):
        expected = means[kind] - means["negative"]
        assert sorted(loaded.tensors[vector]) == [0, 1]
        for layer in (0, 1):
            torch.testing.assert_close(
                loaded.tensors[vector][layer], expected[layer], rtol=0, atol=1e-6
            )
        assert summary["vectors"][vector]["built"] is True
    assert not torch.equal(
        loaded.tensors["with_ablation"][1], loaded.tensors["without_ablation"][1]
    )
    # The cost is what a run of the behavior is charged; the judge's calls are recorded and
    # count as nothing.
    assert loaded.cost.to_json() == {
        "refusal_direction_passes": 77,
        "construction_responses": 72,
        "kept_responses": kept_counts,
    }
    assert summary["construction_cost"] == loaded.cost.to_json()
    assert summary["judge"] == {"model": "stand-in", "calls": 72}
    assert summary["ledger"]["totals"]["inferences"] == 77 + 72 + sum(kept_counts.values())
    assert summary["ledger"]["charges"][-1] == {
        "category": "uncounted",
        "inferences": 0.0,
        "budget": 0.0,
        "note": "coherence judge stand-in: 72 calls, not counted",
    }
    assert run_record.read_json(folder / "summary.json") == summary


def test_a_construction_that_stopped_goes_on_where_it_was(
    build: Callable[..., dict[str, Any]],
    stand_in_judge: Callable[[str], dict[str, Any]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = build(tmp_path)
    before = vectors.load(tmp_path, BEHAVIOR, layers=(0, 1))
    folder = tmp_path / "construction" / BEHAVIOR
    # The files of one question and pair are gone, and the judge is down: only the responses of
    # that question and pair are drawn again, and the construction stops at the judge.
    (folder / "responses" / "negative" / "question01_pair0.jsonl").unlink()
    (folder / "coherence" / "negative" / "question01_pair0.jsonl").unlink()
    seeds_drawn: list[int] = []
    draw_responses = construction.draw_responses

    def recorded(*arguments: Any, **options: Any) -> list[list[int]]:
        seeds_drawn.append(arguments[3])
        return draw_responses(*arguments, **options)

    def judge_that_is_down(prompt: str) -> dict[str, Any]:
        raise coherence.JudgeError("the judge did not answer")

    monkeypatch.setattr(construction, "draw_responses", recorded)
    with pytest.raises(coherence.JudgeError):
        build(tmp_path, judge=judge_that_is_down)
    assert seeds_drawn == [construction.response_seed(BEHAVIOR, "negative", 1, 0)]
    # With the judge back, only the four missing answers are asked for, and the construction
    # ends with the same vectors.
    asked: list[str] = []

    def judge(prompt: str) -> dict[str, Any]:
        asked.append(prompt)
        return stand_in_judge(prompt)

    assert build(tmp_path, judge=judge) == summary
    assert len(asked) == 4 and len(seeds_drawn) == 1
    after = vectors.load(tmp_path, BEHAVIOR, layers=(0, 1))
    for vector, by_layer in before.tensors.items():
        assert all(torch.equal(after.tensors[vector][layer], by_layer[layer]) for layer in by_layer)
    # Other settings of the responses are refused in this directory.
    with pytest.raises(construction.ConstructionError, match="holds the construction responses"):
        build(tmp_path, dataclasses.replace(SETTINGS, response_limit=7))


def test_a_vector_whose_responses_the_filter_dropped_is_not_built(
    build: Callable[..., dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    """The stand-in judge gives no response a score of 90, so no response is kept. The behavior
    then has no vectors, and its runs are still charged for the construction responses."""
    summary = build(tmp_path, dataclasses.replace(SETTINGS, coherence_threshold=90.0))
    assert [entry["built"] for entry in summary["vectors"].values()] == [False, False]
    loaded = vectors.load(tmp_path, BEHAVIOR, layers=(0, 1))
    assert loaded.tensors == {}
    assert loaded.cost.construction_responses == 72
    assert set(loaded.cost.kept_responses.values()) == {0}


def test_the_vector_without_ablation_is_built_alone_without_a_refusal_direction(
    build: Callable[..., dict[str, Any]],
    built: tuple[pathlib.Path, dict[str, Any]],
    tmp_path: pathlib.Path,
) -> None:
    alone = dataclasses.replace(SETTINGS, vectors=("without_ablation",), layers=(1,))
    summary = build(tmp_path, alone, ablation=None, refusal_direction_passes=0)
    assert list(summary["responses"]) == KINDS[1:] and list(summary["vectors"]) == [KINDS[1][9:]]
    assert not (tmp_path / "construction" / BEHAVIOR / "responses" / KINDS[0]).exists()
    loaded = vectors.load(tmp_path, BEHAVIOR, layers=(1,))
    assert list(loaded.tensors) == ["without_ablation"]
    assert sorted(loaded.tensors["without_ablation"]) == [1]
    assert loaded.cost.refusal_direction_passes == 0
    assert loaded.cost.construction_responses == 48
    assert loaded.cost.kept_responses["positive_with_ablation"] == 0
    # It is the vector that the construction of both vectors builds: the two share no draw.
    both = vectors.load(built[0], BEHAVIOR, layers=(1,))
    assert torch.equal(loaded.tensors["without_ablation"][1], both.tensors["without_ablation"][1])
    # The vector with ablation needs the direction, and a layer must be one of the model's.
    with pytest.raises(construction.ConstructionError, match="needs the model's refusal direction"):
        build(tmp_path / "other", SETTINGS, ablation=None)
    with pytest.raises(construction.ConstructionError, match="the model has 2 layers"):
        build(tmp_path / "other", dataclasses.replace(SETTINGS, layers=(1, 2)))
    assert not (tmp_path / "other").exists()


def test_the_rule_of_a_behavior_is_that_of_its_tasks() -> None:
    bench = benchmark.Benchmark()
    assert construction.behavior_rule(bench, "goblin_mention") == '"goblin" in y.lower()'
    assert construction.behavior_rule(bench, "em_dash") == rules.PAPER_RULES["em_dash"]
    with pytest.raises(construction.ConstructionError, match="have 0 rules"):
        construction.behavior_rule(bench, "goblin")


def test_the_ablation_is_opened_from_the_direction_of_the_directory(
    steering_original: original_model.OriginalModel,
    lmtailrisk_checkout: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    both, alone = construction.Settings(), construction.Settings(vectors=("without_ablation",))
    # A directory without a direction serves the vector without ablation only.
    assert construction.open_ablation(alone, tmp_path, None, "tiny") == (None, 0)
    with pytest.raises(refusal.RefusalError, match="holds no refusal direction"):
        construction.open_ablation(both, tmp_path, lmtailrisk_checkout, "tiny")
    record = refusal.find_and_save(steering_original, lmtailrisk_checkout, tmp_path)
    ablation, passes = construction.open_ablation(both, tmp_path, lmtailrisk_checkout, "tiny")
    assert ablation is not None and passes == record["forward_passes"]
    assert torch.equal(ablation.direction, refusal.load(tmp_path)[0])
    assert hasattr(ablation.hooks, "get_direction_ablation_input_pre_hook")
    # Every behavior of a directory with a direction is charged for it.
    assert construction.open_ablation(alone, tmp_path, None, "tiny") == (None, passes)
    with pytest.raises(construction.ConstructionError, match="name the checkout"):
        construction.open_ablation(both, tmp_path, None, "tiny")
    with pytest.raises(construction.ConstructionError, match="is that of 'tiny', not of 'other'"):
        construction.open_ablation(both, tmp_path, lmtailrisk_checkout, "other")


BEHAVIOR_PROMPT = "<{question}> <{answer}>"


def test_a_behavior_without_a_rule_is_labeled_by_the_judge(
    tiny_original_model: original_model.OriginalModel,
    behavior_prompts: construction_prompts.ConstructionPrompts,
    stand_in_judge: Callable[[str], dict[str, Any]],
    chat_response: Callable[..., dict[str, Any]],
    tmp_path: pathlib.Path,
) -> None:
    """The construction prompts of a behavior that a language model judges hold its judge
    prompt. The judge is then asked twice about every response, and a response shows the
    behavior if its score on the behavior is at least 50."""
    judged = dataclasses.replace(behavior_prompts, judge_prompt=BEHAVIOR_PROMPT)

    def judge(prompt: str) -> dict[str, Any]:
        if not prompt.startswith("<"):
            return stand_in_judge(prompt)  # the coherence prompt
        answer = prompt.split("> <", 1)[1][:-1]
        if "e" in answer:
            return chat_response("REFUSAL")  # no number: the response has no score
        return chat_response("70" if any(letter in answer for letter in "abcd") else "20")

    alone = dataclasses.replace(SETTINGS, vectors=("without_ablation",))
    summary = construction.build(
        tiny_original_model,
        judged,
        None,
        judge,
        JUDGE_PROMPT,
        directory=tmp_path,
        settings=alone,
        ablation=None,
        refusal_direction_passes=0,
        judge_model="stand-in",
        judge_concurrency=4,
    )
    folder = tmp_path / "construction" / BEHAVIOR
    kept: dict[str, int] = {}
    for kind in KINDS[1:]:
        kept[kind] = 0
        for question in range(3):
            for pair in range(2):
                name = f"question{question:02d}_pair{pair}.jsonl"
                rows = run_record.read_jsonl(folder / "responses" / kind / name)
                scores = run_record.read_jsonl(folder / "behavior" / kind / name)
                coherent = run_record.read_jsonl(folder / "coherence" / kind / name)
                # No rule labels a response when it is drawn.
                assert all(row["hit"] is None for row in rows)
                for row, score, answer in zip(rows, scores, coherent, strict=True):
                    text = row["text"]
                    if "e" in text:
                        assert score["behavior"] is None and score["answer"] == "REFUSAL"
                        continue  # without a score a response is on neither side
                    shows = any(letter in text for letter in "abcd")
                    assert score["behavior"] == (70.0 if shows else 20.0)
                    wanted = shows if kind != "negative" else not shows
                    kept[kind] += wanted and answer["coherence"] >= 50
        assert summary["responses"][kind]["kept"] == kept[kind] > 0
    loaded = vectors.load(tmp_path, BEHAVIOR, layers=(0, 1))
    assert list(loaded.tensors) == ["without_ablation"]
    assert loaded.cost.kept_responses == {"positive_with_ablation": 0, **kept}
    # Two calls of the judge for each of the 48 responses, none of them counted.
    assert summary["judge"] == {"model": "stand-in", "calls": 96}
    assert summary["ledger"]["charges"][-1]["note"] == (
        "coherence and behavior judge stand-in: 96 calls, not counted"
    )
    assert run_record.read_json(folder / "source.json")["rule"] is None
    # A behavior has a rule or a judge prompt, and not both.
    for prompts, rule in ((judged, RULE), (behavior_prompts, None)):
        with pytest.raises(construction.ConstructionError, match="one of a rule and a judge"):
            construction.build(
                tiny_original_model,
                prompts,
                rule,
                judge,
                JUDGE_PROMPT,
                directory=tmp_path / "other",
                settings=alone,
                ablation=None,
                refusal_direction_passes=0,
                judge_model="stand-in",
            )


def test_a_response_without_a_score_on_the_behavior_is_on_neither_side() -> None:
    kept = construction.is_kept
    assert not kept(None, 80.0, positive=True, threshold=50)
    assert not kept(None, 80.0, positive=False, threshold=50)
