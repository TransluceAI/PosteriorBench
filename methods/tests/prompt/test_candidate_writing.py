"""Writing candidates in three stages, with a stand-in for the writer LLM in this process."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from posteriorbench import benchmark

from posteriorbench_methods.prompt import candidate_writing, pool, writer, writer_messages

TASK_ID = "wc_1853/goblin_mention"
BEHAVIOR = writer_messages.behavior_of("goblin_mention")
EXPLANATION = writer_messages.Explanation("what it is", "why it matters")
SETTINGS = candidate_writing.Settings(scenarios_per_call=5, concurrency=4)
# One call at a time. The stand-in numbers its scenario calls in the order in which they reach
# it, which is the order of the calls only if no two of them are open at the same time.
ONE_AT_A_TIME = dataclasses.replace(SETTINGS, concurrency=1)


@dataclasses.dataclass
class StandIn:
    """A writer in this process. It answers with the family's `stand_in_answer`, or with what
    `scenarios(call)` and `system_prompt(scenario)` return, where None is a call that gave no
    answer. It keeps the messages and the token limit of every call."""

    answer: Callable[[list[dict[str, str]], int], str]
    scenarios: Callable[[int], str | None] | None = None
    system_prompt: Callable[[str], str | None] | None = None
    explanation: str | None = "<behavior_understanding>what it is</behavior_understanding>"
    model: str = "stand-in"
    calls: list[tuple[str, int]] = dataclasses.field(default_factory=lambda: [])
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _scenario_calls: int = 0

    def complete(self, messages: Sequence[dict[str, str]], *, max_tokens: int) -> str | None:
        system, user = messages[0]["content"], messages[1]["content"]
        stage = (
            "explanation"
            if "BloomUnderstanding" in system
            else "scenarios" if "BloomIdeator" in system else "system prompt"
        )
        with self._lock:
            self.calls.append((stage, max_tokens))
            call = self._scenario_calls
            self._scenario_calls += stage == "scenarios"
        if stage == "explanation":
            return self.explanation
        if stage == "scenarios" and self.scenarios is not None:
            return self.scenarios(call)
        if stage == "system prompt" and self.system_prompt is not None:
            scenario = re.search(r"this steering scenario: (.*?)\n\nThis is merely", user, re.S)
            assert scenario is not None
            return self.system_prompt(scenario.group(1))
        return self.answer(list(messages), call)

    def count(self, stage: str) -> int:
        return sum(name == stage for name, _ in self.calls)


@pytest.fixture
def make_stand_in(
    answer_of_stand_in: Callable[[list[dict[str, str]], int], str]
) -> Callable[..., StandIn]:
    """`make_stand_in(scenarios=..., system_prompt=..., explanation=...)`: a stand-in writer."""

    def make(**answers: Any) -> StandIn:
        return StandIn(answer_of_stand_in, **answers)

    return make


def tagged(*scenarios: str) -> str:
    return "\n".join(f"<scenario>{scenario}</scenario>" for scenario in scenarios)


def write(
    llm: StandIn,
    count: int,
    existing: Sequence[str] = (),
    settings: candidate_writing.Settings = ONE_AT_A_TIME,
) -> candidate_writing.Outcome:
    return candidate_writing.write_candidates(
        llm, BEHAVIOR, EXPLANATION, "hi", count=count, existing=existing, settings=settings
    )


def test_the_defaults_are_the_settings_of_the_papers_candidates() -> None:
    assert dataclasses.asdict(candidate_writing.Settings()) == {
        "scenarios_per_call": 50,
        "temperature": 1.0,
        "explanation_tokens": 4000,
        "scenario_tokens": 32768,
        "system_prompt_tokens": 4000,
        "concurrency": 64,
        "attempts": 40,
        "timeout": 300.0,
    }
    with pytest.raises(ValueError, match="scenarios_per_call must be positive"):
        candidate_writing.Settings(scenarios_per_call=0)


def test_every_scenario_call_asks_for_the_same_number_and_each_scenario_gets_a_prompt(
    make_stand_in: Callable[..., StandIn]
) -> None:
    llm = make_stand_in()
    outcome = write(llm, count=12)
    # 12 candidates at 5 scenarios per call: three calls of five, and the first twelve scenarios.
    assert (outcome.scenario_calls, outcome.scenarios) == (3, 12)
    assert (llm.count("scenarios"), llm.count("system prompt")) == (3, 12)
    assert [new.scenario.splitlines()[-1] for new in outcome.written] == [
        f"scenario {call} {place}" for call in range(3) for place in range(5)
    ][:12]
    assert len({new.system_prompt for new in outcome.written}) == 12
    assert (outcome.without_system_prompt, outcome.repeated) == (0, 0)
    limits = {stage: limit for stage, limit in llm.calls}
    assert limits == {"scenarios": 32768, "system prompt": 4000}


def test_candidates_keep_the_order_of_their_scenarios_whichever_call_ends_first(
    make_stand_in: Callable[..., StandIn]
) -> None:
    """With several calls open at the same time, the call for a later scenario can end before
    the call for an earlier one."""
    release_first = threading.Event()

    def system_prompt(scenario: str) -> str:
        if scenario == "s0":
            assert release_first.wait(timeout=10), "the later calls were not made meanwhile"
        elif scenario == "s3":
            release_first.set()
        return f"<system_prompt>the prompt of {scenario}</system_prompt>"

    llm = make_stand_in(
        scenarios=lambda call: tagged(*(f"s{i}" for i in range(5))), system_prompt=system_prompt
    )
    outcome = write(llm, count=5, settings=SETTINGS)
    assert [(new.scenario, new.system_prompt) for new in outcome.written] == [
        (f"s{i}", f"the prompt of s{i}") for i in range(5)
    ]


def test_an_answer_gives_no_more_scenarios_than_it_was_asked_for(
    make_stand_in: Callable[..., StandIn]
) -> None:
    llm = make_stand_in(
        scenarios=lambda call: tagged(*(f"s{call}{i}" for i in range(7))),
        system_prompt=lambda scenario: f"<system_prompt>the prompt of {scenario}</system_prompt>",
    )
    outcome = write(llm, count=10)
    assert [new.scenario for new in outcome.written] == [
        *(f"s0{i}" for i in range(5)),
        *(f"s1{i}" for i in range(5)),
    ]


def test_nothing_is_asked_again_so_fewer_candidates_may_come_back(
    make_stand_in: Callable[..., StandIn]
) -> None:
    answers = {0: tagged("a", "b", "c"), 1: None, 2: "no scenario here", 3: tagged("d", "e")}
    prompts = {"a": "<system_prompt>p</system_prompt>", "b": None, "c": "no tags"}
    prompts |= {"d": "<system_prompt>p</system_prompt>", "e": "<system_prompt>old</system_prompt>"}
    llm = make_stand_in(scenarios=answers.__getitem__, system_prompt=prompts.__getitem__)
    outcome = write(llm, count=20, existing=["old"])
    assert (outcome.scenario_calls, outcome.scenarios) == (4, 5)
    assert llm.count("system prompt") == 5
    # b has no answer and c no tags; d repeats a, and e repeats a candidate that the task has.
    assert outcome.written == (pool.Written("p", "a"),)
    assert (outcome.without_system_prompt, outcome.repeated) == (2, 2)


def test_no_answer_to_any_scenario_call_is_an_error(make_stand_in: Callable[..., StandIn]) -> None:
    with pytest.raises(candidate_writing.WritingError, match="none of the 2 calls"):
        write(make_stand_in(scenarios=lambda call: None), count=6)
    assert write(make_stand_in(), count=0) == candidate_writing.Outcome((), 0, 0, 0, 0)


def test_a_writer_that_cannot_be_called_stops_the_writing(
    make_stand_in: Callable[..., StandIn]
) -> None:
    def refuse(scenario: str) -> str:
        raise writer.WriterError("the writer answered status 401")

    with pytest.raises(writer.WriterError, match="401"):
        write(make_stand_in(system_prompt=refuse), count=12)


def test_a_task_gets_the_candidates_that_it_lacks(
    make_stand_in: Callable[..., StandIn], bench: benchmark.Benchmark, tmp_path: pathlib.Path
) -> None:
    task = bench.task(TASK_ID)
    path = tmp_path / "candidates.parquet"
    llm = make_stand_in()
    first = candidate_writing.add_to_pool_file(
        path, task, llm, candidates=7, new_batch=False, settings=SETTINGS
    )
    assert first.to_json() == {
        "task_id": TASK_ID,
        "candidates_before": 0,
        "candidates": 7,
        "asked": 7,
        "batch": 0,
        "explanation_calls": 1,
        "scenario_calls": 2,
        "system_prompt_calls": 7,
        "without_system_prompt": 0,
        "repeated": 0,
        "writer_calls": 10,
    }
    assert llm.calls[0] == ("explanation", 4000)
    # The task has what was asked for: no call. More is asked for: only what is lacking.
    same = candidate_writing.add_to_pool_file(
        path, task, llm, candidates=7, new_batch=True, settings=SETTINGS
    )
    assert (same.asked, same.writer_calls, same.batch, same.outcome) == (0, 0, None, None)
    more = candidate_writing.add_to_pool_file(
        path, task, llm, candidates=10, new_batch=True, settings=SETTINGS
    )
    assert (more.candidates_before, more.asked, more.candidates, more.batch) == (7, 3, 10, 1)
    assert (more.explanation_calls, more.writer_calls) == (0, 1 + 3)
    candidates = pool.load_candidates(path, TASK_ID)
    assert [c.batch for c in candidates] == [0] * 7 + [1] * 3
    assert len({c.system_prompt for c in candidates}) == 10


def test_one_explanation_serves_every_task_of_a_behavior_and_every_later_call(
    make_stand_in: Callable[..., StandIn], bench: benchmark.Benchmark, tmp_path: pathlib.Path
) -> None:
    task = bench.task(TASK_ID)
    other = dataclasses.replace(task, task_id="wc_0001/goblin_mention", prompt="hello")
    path = tmp_path / "candidates.parquet"
    answer = (
        "<behavior_understanding>it is this</behavior_understanding>"
        "<scientific_motivation>it matters</scientific_motivation>"
    )
    llm = make_stand_in(explanation=answer)
    for chosen in (task, other, task):
        candidate_writing.add_to_pool_file(
            path, chosen, llm, candidates=3, new_batch=False, settings=SETTINGS
        )
    assert llm.count("explanation") == 1
    kept = candidate_writing.explanations_file(path)
    assert kept == tmp_path / "candidates.explanations.json"
    assert json.loads(kept.read_text()) == {
        "goblin_mention": {
            "understanding": "it is this",
            "motivation": "it matters",
            "writer_model": "stand-in",
            "answer": answer,
        }
    }
    # A later writer reads the explanation that is kept, here from a file named by the caller.
    later = make_stand_in(explanation=None)
    elsewhere = tmp_path / "elsewhere.parquet"
    candidate_writing.add_to_pool_file(
        elsewhere, task, later, candidates=2, new_batch=False, settings=SETTINGS, explanations=kept
    )
    assert later.count("explanation") == 0
    assert not candidate_writing.explanations_file(elsewhere).exists()


def test_no_explanation_no_candidates(
    make_stand_in: Callable[..., StandIn], bench: benchmark.Benchmark, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "candidates.parquet"
    with pytest.raises(candidate_writing.WritingError, match="no explanation"):
        candidate_writing.add_to_pool_file(
            path,
            bench.task(TASK_ID),
            make_stand_in(explanation=None),
            candidates=3,
            new_batch=False,
            settings=SETTINGS,
        )
    assert not path.exists() and not candidate_writing.explanations_file(path).exists()
    broken = tmp_path / "broken.json"
    broken.write_text('{"goblin_mention": {"understanding": "only this"}}')
    with pytest.raises(candidate_writing.WritingError, match="not a file of explanations"):
        candidate_writing.read_explanations(broken)
