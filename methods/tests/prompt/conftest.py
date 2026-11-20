"""Fixtures of the prompt family's tests: the tiny original model with the benchmark's response
limit and the family's batch size, a dataset of one task of that model, a small pool of
candidate prompts for the task, and a stand-in for the server of the writer LLM that listens on
this machine only."""

from __future__ import annotations

import dataclasses
import http.server
import json
import pathlib
import re
import string
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from posteriorbench import benchmark, original_model

from posteriorbench_methods.common import run_context, tracking

TASK_ID = "wc_1853/goblin_mention"
# Eight candidates; the first and the seventh are the same prompt.
POOL = ("be brief", "say a", "talk about cats", "zzz", "say aaa", "reply", "be brief", "abc")


@pytest.fixture(scope="session")
def scorer(make_tiny_original_model: Callable[..., Any]) -> original_model.OriginalModel:
    """The tiny original model with responses of at most 50 tokens, so that the 16 composition
    settings are those of the benchmark, and with the family's batch size."""
    return make_tiny_original_model(max_response_tokens=50, batch_size=64)


@pytest.fixture(scope="session")
def bench(
    scorer: original_model.OriginalModel, make_tiny_dataset: Callable[..., pathlib.Path]
) -> benchmark.Benchmark:
    """A dataset with the task TASK_ID of `scorer`: the response contains "a" (about a third of
    the responses under any prompt)."""
    return benchmark.Benchmark(make_tiny_dataset(scorer, task_id=TASK_ID, draws=1500))


@pytest.fixture(scope="session")
def make_pool_file(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[..., pathlib.Path]:
    """A factory of pool files: `make_pool_file(prompts, task_id=TASK_ID, batches=None)` writes
    `prompts` as the candidates of the task, with the column `batch` if `batches` gives the batch
    of each candidate, and returns the path."""

    def make(
        prompts: tuple[str, ...], task_id: str = TASK_ID, batches: tuple[int, ...] | None = None
    ) -> pathlib.Path:
        rows: list[dict[str, str | int]] = [
            {
                "task_id": task_id,
                "candidate_index": i,
                "candidate_id": f"c{i:03d}",
                "system_prompt": s,
            }
            for i, s in enumerate(prompts)
        ]
        if batches is not None:
            for row, batch in zip(rows, batches, strict=True):
                row["batch"] = batch
        path = tmp_path_factory.mktemp("pool") / "pool.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    return make


@pytest.fixture(scope="session")
def pool_file(make_pool_file: Callable[..., pathlib.Path]) -> pathlib.Path:
    """The pool file of the tiny task: the eight candidates of POOL."""
    return make_pool_file(POOL)


@pytest.fixture(scope="session")
def make_context(
    scorer: original_model.OriginalModel, bench: benchmark.Benchmark, pool_file: pathlib.Path
) -> Callable[..., run_context.RunContext]:
    """A factory of run contexts on the tiny task: `make_context(out, replicate=0, compose=True,
    method_data=<the pool file>)`, with 64 evaluation draws."""

    def make(
        out: pathlib.Path,
        *,
        replicate: int = 0,
        compose: bool = True,
        method_data: pathlib.Path | None = pool_file,
    ) -> run_context.RunContext:
        options = run_context.RunOptions(
            task_id=TASK_ID,
            out=out,
            replicate=replicate,
            compose=compose,
            evaluation_draws=64,
            method_data=method_data,
        )
        return run_context.RunContext(
            family="prompt",
            method="prompt",
            options=options,
            bench=bench,
            task=bench.task(TASK_ID),
            original=scorer,
            load_model_copy=lambda: scorer,
            tracker=tracking.Tracker(),
        )

    return make


def stand_in_answer(messages: list[dict[str, str]], call: int) -> str:
    """The answer of the stand-in writer to the messages of one stage. `call` counts the calls
    for scenarios, so that every list of scenarios is its own.

    Stage 1: an explanation in its two tags. Stage 2: as many scenarios as the messages ask for,
    named by `call` and their place. Stage 3: a system prompt of lowercase letters that names the
    scenario, so it is a text of the tiny tokenizer and differs between scenarios."""
    system, user = messages[0]["content"], messages[1]["content"]
    if "BloomUnderstanding" in system:
        return (
            "<behavior_understanding>what it is</behavior_understanding>\n"
            "<scientific_motivation>why it matters</scientific_motivation>"
        )
    if "BloomIdeator" in system:
        asked = re.search(r"EXACTLY (\d+) scenarios", user)
        assert asked is not None
        return "\n".join(
            f"<scenario>\n**Scenario {place + 1}**\nscenario {call} {place}\n</scenario>"
            for place in range(int(asked.group(1)))
        )
    scenario = re.search(r"scenario (\d+) (\d+)", user)
    assert scenario is not None, "a system prompt is asked for one of the stand-in's scenarios"
    letters = string.ascii_lowercase
    name = letters[int(scenario.group(1)) % 26] + letters[int(scenario.group(2)) % 26]
    return f"<system_prompt>\nsay {name}\n</system_prompt>"


@pytest.fixture(scope="session")
def answer_of_stand_in() -> Callable[[list[dict[str, str]], int], str]:
    """`stand_in_answer`, for a test that builds a stand-in writer of its own."""
    return stand_in_answer


@dataclasses.dataclass
class WriterServer:
    """A stand-in for the server of the writer LLM: it listens on this machine, keeps the
    requests it gets, and answers each with `respond`.

    url: the address to give a writer.
    requests: every request so far, as (headers with lowercase names, body).
    respond: respond(body, n) -> (status, answer body), where n counts the requests from 0. A
        test sets it; the default answers with `stand_in_answer`."""

    url: str = ""
    requests: list[tuple[dict[str, str], dict[str, Any]]] = dataclasses.field(
        default_factory=lambda: []
    )
    respond: Callable[[dict[str, Any], int], tuple[int, dict[str, Any]]] | None = None
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _scenario_calls: int = 0

    def handle(self, headers: dict[str, str], body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self._lock:
            index = len(self.requests)
            self.requests.append((headers, body))
            call = self._scenario_calls
            if "BloomIdeator" in body["messages"][0]["content"]:
                self._scenario_calls += 1
        if self.respond is not None:
            return self.respond(body, index)
        return 200, chat_answer(stand_in_answer(body["messages"], call))


def chat_answer(content: str | None) -> dict[str, Any]:
    """The body of a chat-completions answer with one choice."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture
def writer_server() -> Iterator[WriterServer]:
    """A `WriterServer` at an address of this machine, for one test."""
    stand_in = WriterServer()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            headers = {name.lower(): value for name, value in self.headers.items()}
            status, answer = stand_in.handle(headers, body)
            encoded = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    stand_in.url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield stand_in
    server.shutdown()
    server.server_close()
    thread.join()
