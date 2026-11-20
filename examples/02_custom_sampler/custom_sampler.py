"""Evaluate a sampler of your own on one PosteriorBench task, with its construction cost.

The method below takes two handwritten instructions that ask for a goblin (`CANDIDATES`; write your
own for another behavior), draws 32 responses from the original model under each (as a system
prompt), and keeps the instruction with more hits. It charges what it computes to a ledger: 64
generated responses (labels are free; it never computes p or q). Replace it with your method.
`examples/02_custom_sampler/README.md` has the commands.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

import typer

import posteriorbench as pb
from posteriorbench import rules, sampler

CANDIDATES: tuple[str, ...] = (
    "Mention a goblin in your answer.",
    "Tell a short story about a goblin.",
)
SELECTION_DRAWS = 32


class InstructedSampler(pb.Sampler):
    """q(y) = p(y | the task's prompt with an added system instruction). One inference per
    response (the default `inferences_per_response`)."""

    name = "instructed"

    def __init__(self, model: pb.OriginalModel, task: pb.Task, instruction: str) -> None:
        self.model = model
        self.instruction = instruction
        self.prompt_token_ids = model.render_prompt(task.prompt, system=instruction)

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        return self.model.generate(self.prompt_token_ids, n, seed=seed)

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        # Batched: evaluation scores the references and up to 128 hits at once.
        return self.model.logprobs(self.prompt_token_ids, responses)


def build_sampler(
    task: pb.Task,
    model: pb.OriginalModel,
    benchmark: pb.Benchmark,
    *,
    candidates: Sequence[str] = CANDIDATES,
    seed: int = 0,
) -> tuple[pb.Sampler, pb.Ledger]:
    """The factory that `posteriorbench evaluate --sampler custom` calls: the sampler and the
    ledger of what building it cost.

    Construction draws use seeds below 2**31 (`sampler.construction_stream_seed`), which
    evaluation never uses. Writing the instructions is not charged; each selection draw is one
    generated response. This method takes no corpus hits; one that does takes them from
    `benchmark.corpus(task, ledger)`, which charges them."""
    rule = rules.parse_rule(task.rule)
    ledger = pb.Ledger()
    best: InstructedSampler | None = None
    best_hits = -1
    for index, text in enumerate(candidates):
        candidate = InstructedSampler(model, task, text)
        draws = candidate.sample(
            SELECTION_DRAWS, seed=sampler.construction_stream_seed(seed, index)
        )
        ledger.charge(
            pb.Category.GENERATION,
            inferences=SELECTION_DRAWS,
            note=f"selection draws of candidate {index}",
        )
        hits = sum(rule(model.decode(response)) for response in draws)
        if hits > best_hits:
            best, best_hits = candidate, hits
    assert best is not None, "at least one candidate"
    return best, ledger


def main(
    task_id: Annotated[str, typer.Option("--task")] = "wc_1853/goblin_mention",
    data_source: Annotated[
        str | None,
        typer.Option("--data", help="A data source; default: the tables bundled with the package"),
    ] = None,
) -> None:
    """Build the sampler on one task, evaluate it and print its scores and cost."""
    bench = pb.Benchmark(data_source)
    task = bench.task(task_id)
    model = pb.OriginalModel.from_pretrained()
    candidate, ledger = build_sampler(task, model, bench)
    result = bench.evaluate(task, candidate, model, ledger=ledger, seed=0, replicate=0)
    print(f"native hit rate {result.native_hit_rate:.3f} (fallback used: {result.fallback})")
    print(f"capped forward KL {result.capped_forward_kl:.2f}, reverse KL {result.reverse_kl:.2f}")
    print(f"construction cost {result.construction_inferences:.0f} inferences")


if __name__ == "__main__":
    typer.run(main)
