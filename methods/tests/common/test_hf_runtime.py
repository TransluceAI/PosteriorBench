"""`common.hf_runtime` on the tiny models: draws are complete responses of the right sampler,
generation and teacher forcing agree, groups and parts behave as documented, invalid arguments
raise."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Generator
from typing import Any

import pytest
import torch
from posteriorbench import original_model
from posteriorbench import sampler as pb_sampler

from posteriorbench_methods.common import composition, hf_runtime

MAX = 8


@pytest.fixture(scope="module")
def fmt(tiny_original_model: original_model.OriginalModel) -> hf_runtime.ResponseFormat:
    return hf_runtime.ResponseFormat.of(tiny_original_model)


@pytest.fixture(scope="module")
def prompt(tiny_original_model: original_model.OriginalModel) -> tuple[int, ...]:
    return tuple(tiny_original_model.render_prompt("hi"))


@pytest.fixture(scope="module")
def q_model(make_tiny_model: Callable[..., Any]) -> Any:
    return make_tiny_model(seed=1)


def _seed_plus(seed: int, offset: int) -> int:
    return seed + offset


def test_response_format_of_scorer(tiny_original_model: original_model.OriginalModel) -> None:
    fmt = hf_runtime.ResponseFormat.of(tiny_original_model)
    assert fmt.stop_token_ids == tuple(tiny_original_model.stop_token_ids)
    assert fmt.max_tokens == MAX
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.ResponseFormat((), 8)
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.CausalLM(object(), ())


def test_plain_and_charges(
    q_model: Any, fmt: hf_runtime.ResponseFormat, prompt: tuple[int, ...]
) -> None:
    """A sampler declares what a draw of its setting costs."""
    models = hf_runtime.TwoModels(hf_runtime.CausalLM(q_model, prompt))

    def sampler(setting: composition.Setting | None) -> hf_runtime.ModelSampler:
        return hf_runtime.ModelSampler(
            models, lambda r: [], hf_runtime.Generation(fmt), part_size=4, part_seed=_seed_plus,
            setting=setting,
        )  # fmt: skip

    assert sampler(None).is_plain and sampler(None).inferences_per_response == 1.0
    assert sampler(composition.Setting(0.0, MAX)).is_plain
    assert sampler(composition.Setting(0.0, MAX)).inferences_per_response == 1.0
    assert sampler(composition.Setting(0.0, 4)).inferences_per_response == 2.0
    assert sampler(composition.Setting(0.5, MAX)).inferences_per_response == 2.0
    with pytest.raises(hf_runtime.GenerationError):
        sampler(composition.Setting(0.5, -1))


def test_parts_and_pack() -> None:
    groups = hf_runtime.parts(70, 100, part_size=32, part_seed=_seed_plus)
    assert [(g.count, g.seed) for g in groups] == [(32, 100), (32, 132), (6, 164)]
    assert hf_runtime.parts(0, 1, part_size=4, part_seed=_seed_plus) == []
    batches = hf_runtime.pack(groups, 64)
    assert [[g.count for g in b] for b in batches] == [[32, 32], [6]]
    assert [[g.count for g in b] for b in hf_runtime.pack(groups, 40)] == [[32], [32, 6]]
    # Without a batch size every group is a batch of its own, as with the size of a part.
    assert hf_runtime.pack(groups, None) == [[g] for g in groups] == hf_runtime.pack(groups, 32)
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.pack(groups, 16)


@pytest.mark.parametrize("which", ["qwen2", "hybrid"])
def test_plain_draws_are_responses_with_their_log_q(
    which: str, tiny_model: Any, tiny_hybrid_model: Any, fmt: hf_runtime.ResponseFormat,
    prompt: tuple[int, ...],
) -> None:  # fmt: skip
    model = tiny_model if which == "qwen2" else tiny_hybrid_model
    lm = hf_runtime.CausalLM(model, prompt)
    generation = hf_runtime.Generation(fmt, batch_size=40)
    (draws,) = hf_runtime.sample_groups(
        hf_runtime.TwoModels(lm), [hf_runtime.DrawGroup(40, 3)], generation
    )
    for draw in draws:
        assert (
            pb_sampler.invalid_response_reason(
                draw.token_ids, stop_token_ids=fmt.stop_token_ids, max_response_tokens=MAX
            )
            is None
        )
        assert draw.token_logr == draw.token_logq
        assert draw.finish_reason == (
            "stop" if draw.token_ids[-1] in fmt.stop_token_ids else "length"
        )
    teacher = hf_runtime.token_logprobs(lm, [d.token_ids for d in draws], batch_size=16)
    for draw, values in zip(draws, teacher, strict=True):
        assert draw.token_logq == pytest.approx(values, abs=2e-5)
    again = hf_runtime.sample_groups(
        hf_runtime.TwoModels(lm), [hf_runtime.DrawGroup(40, 3)], generation
    )
    assert [d.token_ids for d in again[0]] == [d.token_ids for d in draws]
    alone = hf_runtime.sample_groups(  # one group in a batch of its own: the same batch
        hf_runtime.TwoModels(lm), [hf_runtime.DrawGroup(40, 3)], hf_runtime.Generation(fmt)
    )
    assert [d.token_ids for d in alone[0]] == [d.token_ids for d in draws]
    other = hf_runtime.sample_groups(
        hf_runtime.TwoModels(lm), [hf_runtime.DrawGroup(40, 4)], generation
    )
    assert [d.token_ids for d in other[0]] != [d.token_ids for d in draws]


@pytest.mark.parametrize("p_from_start", [True, False])
@pytest.mark.parametrize("alpha, switch", [(0.0, 3), (0.25, 5), (0.75, MAX), (1.0, 4)])
def test_composed_generation_matches_teacher_forcing(
    alpha: float, switch: int, p_from_start: bool, q_model: Any, tiny_model: Any,
    fmt: hf_runtime.ResponseFormat, prompt: tuple[int, ...],
) -> None:  # fmt: skip
    """The log-probability recorded at generation is that of the response under the setting."""
    q_lm, p_lm = hf_runtime.CausalLM(q_model, prompt), hf_runtime.CausalLM(tiny_model, prompt)
    (draws,) = hf_runtime.sample_groups(
        hf_runtime.TwoModels(q_lm, p_lm), [hf_runtime.DrawGroup(30, 11)],
        hf_runtime.Generation(fmt, batch_size=30, p_from_start=p_from_start),
        setting=composition.Setting(alpha, switch),
    )  # fmt: skip
    responses = [d.token_ids for d in draws]
    composed = hf_runtime.composed_logprobs(
        hf_runtime.token_logprobs(q_lm, responses, batch_size=8),
        hf_runtime.token_logprobs(p_lm, responses, batch_size=8),
        setting=composition.Setting(alpha, switch),
    )  # fmt: skip
    assert [d.logq for d in draws] == pytest.approx(composed, abs=1e-4)


def test_shared_trunk_reads_two_heads(
    tiny_model: Any, fmt: hf_runtime.ResponseFormat, prompt: tuple[int, ...]
) -> None:
    """With the heads (p, p + a shift of one token), q draws that token more often than p."""
    shift = torch.zeros(int(tiny_model.config.vocab_size))
    shift[5] = 4.0

    def heads(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = tiny_model.lm_head(hidden).float()
        return base, base + shift

    models = hf_runtime.SharedTrunk(tiny_model.model, prompt, heads)
    plain = hf_runtime.sample_groups(
        hf_runtime.TwoModels(hf_runtime.CausalLM(tiny_model, prompt)),
        [hf_runtime.DrawGroup(200, 1)], hf_runtime.Generation(fmt, batch_size=200),
    )[0]  # fmt: skip
    tilted = hf_runtime.sample_groups(
        models, [hf_runtime.DrawGroup(200, 1)], hf_runtime.Generation(fmt, batch_size=200)
    )[0]
    count = lambda draws: sum(d.token_ids.count(5) for d in draws)
    assert count(tilted) > 2 * count(plain)
    p_only = hf_runtime.sample_groups(
        models,
        [hf_runtime.DrawGroup(20, 2)],
        hf_runtime.Generation(fmt, batch_size=20),
        setting=composition.Setting(1.0, MAX),
    )[0]
    teacher = hf_runtime.token_logprobs(
        hf_runtime.CausalLM(tiny_model, prompt), [d.token_ids for d in p_only], batch_size=20
    )
    for draw, values in zip(p_only, teacher, strict=True):
        assert draw.token_logq == pytest.approx(values, abs=2e-5)


def test_tempered_groups(
    q_model: Any, fmt: hf_runtime.ResponseFormat, prompt: tuple[int, ...]
) -> None:
    """With renormalize=False each group may have its own temperature: token_logr is the
    log-probability at that temperature, token_logq at temperature 1."""
    lm = hf_runtime.CausalLM(q_model, prompt)
    groups = [hf_runtime.DrawGroup(10, 1, 1.0), hf_runtime.DrawGroup(10, 2, 2.0)]
    tempered = hf_runtime.Generation(fmt, batch_size=20, renormalize=False)
    cold, hot = hf_runtime.sample_groups(hf_runtime.TwoModels(lm), groups, tempered)
    assert all(d.token_logr == d.token_logq for d in cold)
    teacher = hf_runtime.token_logprobs(lm, [d.token_ids for d in hot], batch_size=10)
    for draw, values in zip(hot, teacher, strict=True):
        assert draw.token_logq == pytest.approx(values, abs=2e-5)
        assert draw.token_logr != draw.token_logq
    alone = hf_runtime.sample_groups(hf_runtime.TwoModels(lm), groups[1:], tempered)[0]
    assert [d.token_ids for d in alone] == [d.token_ids for d in hot]  # its own generator


def test_invalid_generation_arguments(
    q_model: Any, fmt: hf_runtime.ResponseFormat, prompt: tuple[int, ...]
) -> None:
    models = hf_runtime.TwoModels(hf_runtime.CausalLM(q_model, prompt))
    group = [hf_runtime.DrawGroup(4, 1)]
    generation = hf_runtime.Generation(fmt, batch_size=4)
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.sample_groups(models, group, generation, setting=composition.Setting(1.5, MAX))
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.sample_groups(models, [hf_runtime.DrawGroup(4, 1, 2.0)], generation)
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.sample_groups(
            models,
            group,
            hf_runtime.Generation(fmt, batch_size=4, renormalize=False),
            setting=composition.Setting(0.5, MAX),
        )
    with pytest.raises(hf_runtime.GenerationError):  # composed, but no p
        hf_runtime.sample_groups(models, group, generation, setting=composition.Setting(0.5, MAX))
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.sample_groups(models, [hf_runtime.DrawGroup(0, 1)], generation)
    with pytest.raises(hf_runtime.GenerationError):  # a group larger than the batch size
        hf_runtime.sample_groups(models, [hf_runtime.DrawGroup(5, 1)], generation)


def test_forward_context_wraps_every_pass(
    tiny_model: Any, fmt: hf_runtime.ResponseFormat, prompt: tuple[int, ...]
) -> None:
    starts: list[int] = []

    @contextlib.contextmanager
    def record(start: int) -> Generator[None, None, None]:
        starts.append(start)
        yield

    lm = hf_runtime.CausalLM(tiny_model, prompt, record)
    (draws,) = hf_runtime.sample_groups(
        hf_runtime.TwoModels(lm), [hf_runtime.DrawGroup(3, 1)], hf_runtime.Generation(fmt)
    )
    longest = max(len(d.token_ids) for d in draws)
    # The prompt pass starts at 0; the pass that feeds response token t (t >= 0) at P + t.
    assert starts == [0, *range(len(prompt), len(prompt) + longest - 1)]
    starts.clear()
    hf_runtime.token_logprobs(lm, [d.token_ids for d in draws], batch_size=8)
    assert starts == [0]


def test_teacher_forcing_shapes_and_errors(
    q_model: Any, prompt: tuple[int, ...], tiny_original_model: original_model.OriginalModel
) -> None:
    lm = hf_runtime.CausalLM(q_model, prompt)
    responses = tiny_original_model.generate(list(prompt), 9, seed=5)
    tokens = hf_runtime.token_logprobs(lm, responses, batch_size=4)
    assert [len(t) for t in tokens] == [len(r) for r in responses]
    sums = hf_runtime.sequence_logprobs(lm, responses, batch_size=4)
    assert sums == pytest.approx([math.fsum(t) for t in tokens], abs=1e-5)
    train = hf_runtime.sequence_logprobs_train(lm, responses)
    assert train.shape == (9,) and train.requires_grad
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.sequence_logprobs_train(lm, [[]])
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.sequence_logprobs_train(lm, [])
    package = tiny_original_model.token_logprobs(list(prompt), responses)
    other = hf_runtime.token_logprobs(
        hf_runtime.CausalLM(tiny_original_model.model, prompt), responses, batch_size=9
    )
    for a, b in zip(package, other, strict=True):
        assert a == pytest.approx(b, abs=1e-5)


def test_composed_logprobs_rules() -> None:
    q = [[-1.0, -2.0, -3.0], [-0.5]]
    p = [[-2.0, -1.0, -1.0], [-1.5]]
    assert hf_runtime.composed_logprobs(q, None, setting=composition.Setting(0.0, 8)) == [
        -6.0,
        -0.5,
    ]
    assert hf_runtime.composed_logprobs(None, p, setting=composition.Setting(1.0, 8)) == [
        -4.0,
        -1.5,
    ]
    switched = hf_runtime.composed_logprobs(q, p, setting=composition.Setting(0.0, 1))
    assert switched == [-1.0 - 1.0 - 1.0, -0.5]
    mixed = hf_runtime.composed_logprobs(q, p, setting=composition.Setting(0.5, 8))
    expected = math.log(0.5 * math.exp(-0.5) + 0.5 * math.exp(-1.5))
    assert mixed[1] == pytest.approx(expected)
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.composed_logprobs(q, None, setting=composition.Setting(0.5, 8))
    with pytest.raises(hf_runtime.GenerationError):
        hf_runtime.composed_logprobs(q, [[-1.0], [-1.0]], setting=composition.Setting(0.5, 8))
    assert hf_runtime.composed_logprobs([], [], setting=composition.Setting(0.5, 8)) == []


def test_model_sampler_is_a_package_sampler(
    q_model: Any, tiny_model: Any, fmt: hf_runtime.ResponseFormat, prompt: tuple[int, ...]
) -> None:
    q_lm, p_lm = hf_runtime.CausalLM(q_model, prompt), hf_runtime.CausalLM(tiny_model, prompt)

    def logprobs(responses: Any) -> list[float]:
        return hf_runtime.composed_logprobs(
            hf_runtime.token_logprobs(q_lm, responses, batch_size=16),
            hf_runtime.token_logprobs(p_lm, responses, batch_size=16),
            setting=composition.Setting(0.25, 4),
        )  # fmt: skip

    sampler = hf_runtime.ModelSampler(
        hf_runtime.TwoModels(q_lm, p_lm), logprobs, hf_runtime.Generation(fmt), part_size=16,
        part_seed=_seed_plus, setting=composition.Setting(0.25, 4), name="composed",
    )  # fmt: skip
    assert isinstance(sampler, pb_sampler.Sampler)
    assert sampler.inferences_per_response == 2.0 and not sampler.is_plain
    draws = sampler.sample_draws(40, seed=7)
    assert sampler.sample(40, seed=7) == [d.token_ids for d in draws]
    assert sampler.logprobs([d.token_ids for d in draws]) == pytest.approx(
        [d.logq for d in draws], abs=1e-4
    )
    assert sampler.logprobs([]) == []
    plain = hf_runtime.ModelSampler(
        hf_runtime.TwoModels(q_lm), logprobs, hf_runtime.Generation(fmt), part_size=16,
        part_seed=_seed_plus,
    )  # fmt: skip
    assert plain.inferences_per_response == 1.0 and plain.is_plain
    # A batch size of several parts draws the same groups, in larger batches.
    batched = hf_runtime.ModelSampler(
        hf_runtime.TwoModels(q_lm), logprobs, hf_runtime.Generation(fmt, batch_size=32),
        part_size=16, part_seed=_seed_plus,
    )  # fmt: skip
    assert len(batched.sample(40, seed=7)) == 40
