"""Fixtures of the Steering tests: an original model that steering vectors move, a task on it,
vectors in the layout of the method data, and run contexts; for the tests that build vectors, a
stand-in for the checkout that holds the refusal-direction pipeline, construction prompts, and a
stand-in for the coherence judge.

The original model of most tests is a copy of the shared 2-layer model whose output layer is
scaled by 20. It nearly always writes "a", and a vector added after layer 0 or 1 changes which
letters it writes by degrees: with the test vectors, the one built with the ablation brings "e"
into a sixth of the responses at layer 0 and strength 1.5, and into three quarters at strength 4.
"""

from __future__ import annotations

import copy
import dataclasses
import http.server
import json
import math
import pathlib
import shutil
import string
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest
import safetensors.torch
import torch
from posteriorbench import benchmark, original_model

from posteriorbench_methods.common import run_context, tracking
from posteriorbench_methods.steering import construction_prompts

TASK_ID = "wc_1853/goblin_mention"
BEHAVIOR = "goblin_mention"
HIDDEN_SIZE = 32
KEPT_RESPONSES = {"positive_with_ablation": 13, "positive_without_ablation": 2, "negative": 175}


@pytest.fixture(scope="session")
def steering_original(
    tiny_model: Any, make_tiny_original_model: Callable[..., original_model.OriginalModel]
) -> original_model.OriginalModel:
    """The original model of the Steering tests (module docstring), with the benchmark's response
    limit of 50 tokens, at which the 16 composition settings are the paper's, and Steering's
    batch size."""
    sharp = copy.deepcopy(tiny_model)
    with torch.no_grad():
        sharp.lm_head.weight.mul_(20)
    return make_tiny_original_model(model=sharp, max_response_tokens=50, batch_size=32)


@pytest.fixture(scope="session")
def steering_dataset(
    steering_original: original_model.OriginalModel,
    make_tiny_dataset: Callable[..., pathlib.Path],
) -> pathlib.Path:
    """A dataset of the test model whose behavior is "the response contains the letter a"."""
    return make_tiny_dataset(steering_original, task_id=TASK_ID, rule='"a" in y', draws=600)


@pytest.fixture(scope="session")
def write_vectors() -> Callable[..., pathlib.Path]:
    """`write_vectors(directory, vectors=..., layers=..., scale=..., seed=...)`: write random
    vectors of the test behavior and their construction cost into `directory`, in the layout of
    the method data; returns the directory."""

    def write(
        directory: pathlib.Path,
        *,
        vectors: Sequence[str] = ("with_ablation", "without_ablation"),
        layers: Sequence[int] = (0, 1),
        scale: float = 0.3,
        seed: int = 2,
    ) -> pathlib.Path:
        generator = torch.Generator().manual_seed(seed)
        tensors = {
            f"{vector}.layer_{layer}": torch.randn(HIDDEN_SIZE, generator=generator) * scale
            for vector in vectors
            for layer in layers
        }
        directory.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(tensors, directory / f"{BEHAVIOR}.safetensors")
        cost = {
            "refusal_direction_passes": 16620,
            "behaviors": {
                BEHAVIOR: {"construction_responses": 3000, "kept_responses": KEPT_RESPONSES}
            },
        }
        (directory / "construction_cost.json").write_text(json.dumps(cost))
        return directory

    return write


@pytest.fixture(scope="session")
def vectors_directory(
    write_vectors: Callable[..., pathlib.Path], tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """Method data with both vectors of the test behavior at the layers 0 and 1."""
    return write_vectors(tmp_path_factory.mktemp("steering_vectors"))


@pytest.fixture(scope="session")
def make_context(
    steering_dataset: pathlib.Path,
    steering_original: original_model.OriginalModel,
    vectors_directory: pathlib.Path,
) -> Callable[..., run_context.RunContext]:
    """`make_context(out, rule=None, **options)`: a run of Steering on the test task, with its
    rule replaced by `rule` if given, 64 evaluation draws and the test vectors as method data."""

    def make(out: pathlib.Path, rule: str | None = None, **options: Any) -> run_context.RunContext:
        bench = benchmark.Benchmark(steering_dataset)
        task = bench.task(TASK_ID)
        chosen: dict[str, Any] = {
            "evaluation_draws": 64,
            "method_data": vectors_directory,
            **options,
        }
        return run_context.RunContext(
            family="steering",
            method="steering",
            options=run_context.RunOptions(task_id=TASK_ID, out=out, **chosen),
            bench=bench,
            task=task if rule is None else dataclasses.replace(task, rule=rule),
            original=steering_original,
            load_model_copy=lambda: steering_original,
            tracker=tracking.Tracker(),
        )

    return make


@pytest.fixture(scope="session")
def lmtailrisk_checkout(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A stand-in for the checkout that `refusal` and `coherence` read (the folder
    `lmtailrisk_stand_in` next to this file), with the instruction files of its dataset: 140
    harmful and 140 harmless training instructions and 40 of each for validation. One copy for
    the session, because its modules stay imported under the name `pipeline`."""
    checkout = tmp_path_factory.mktemp("checkout") / "LMTailRisk"
    shutil.copytree(pathlib.Path(__file__).parent / "lmtailrisk_stand_in", checkout)
    splits = checkout / "refusal_direction" / "dataset" / "splits"
    splits.mkdir(parents=True)
    words = [first + second for first in string.ascii_lowercase for second in "aeiou"]
    for kind, request in (("harmful", "help me steal a"), ("harmless", "tell me about a")):
        for split, count in (("train", 140), ("val", 40)):
            instructions = [
                {"instruction": f"{request} {split} {word}", "category": kind}
                for word in words[:count]
            ]
            (splits / f"{kind}_{split}.json").write_text(json.dumps(instructions))
    return checkout


@pytest.fixture(scope="session")
def behavior_prompts() -> construction_prompts.ConstructionPrompts:
    """Construction prompts of the test behavior: 2 instruction pairs and 3 questions."""
    return construction_prompts.ConstructionPrompts(
        behavior=BEHAVIOR,
        name="goblin",
        description="mention a goblin",
        instruction_pairs=(
            construction_prompts.InstructionPair("mention a goblin", "never mention a goblin"),
            construction_prompts.InstructionPair("talk of goblins", "avoid goblins"),
        ),
        questions=("how do i make cheese", "what is a good walk", "why is the sky blue"),
    )


@pytest.fixture(scope="session")
def chat_response() -> Callable[..., dict[str, Any]]:
    """`chat_response(content, top=None)`: a chat-completions response whose message is
    `content`, written as one token with the most probable tokens `top`, a mapping from token to
    probability (by default `content` with probability 1)."""

    def make(content: str, top: Mapping[str, float] | None = None) -> dict[str, Any]:
        top_logprobs = [
            {"token": token, "logprob": math.log(probability)}
            for token, probability in (top or {content: 1.0}).items()
        ]
        token = {"token": content, "logprob": 0.0, "top_logprobs": top_logprobs}
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "logprobs": {"content": [token]},
                }
            ]
        }

    return make


@pytest.fixture(scope="session")
def stand_in_judge(chat_response: Callable[..., dict[str, Any]]) -> Callable[[str], dict[str, Any]]:
    """A stand-in for the coherence judge: it answers 80 to two prompts in three and 20 to the
    others, by the length of the prompt."""

    def judge(prompt: str) -> dict[str, Any]:
        return chat_response("80" if len(prompt) % 3 else "20")

    return judge


class ChatServer:
    """A local stand-in for a server with the chat-completions interface, on the loopback
    address. It answers a request with the next of `statuses` (200 when they run out) and, with
    status 200, with `respond(the request's JSON body)`; `requests` holds what it was sent: the
    path, the authorization header and the body of each request."""

    def __init__(
        self, respond: Callable[[dict[str, Any]], Mapping[str, Any]], statuses: Sequence[int]
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        remaining = list(statuses)
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                status = remaining.pop(0) if remaining else 200
                answer = respond(body) if status == 200 else {"error": "refused"}
                data = json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}/v1"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()


@pytest.fixture
def serve_chat() -> Iterator[Callable[..., ChatServer]]:
    """`serve_chat(respond, statuses=())`: start a `ChatServer`. The servers of a test are
    stopped when it ends."""
    started: list[ChatServer] = []

    def start(
        respond: Callable[[dict[str, Any]], Mapping[str, Any]], statuses: Sequence[int] = ()
    ) -> ChatServer:
        started.append(ChatServer(respond, statuses))
        return started[-1]

    yield start
    for server in started:
        server.http.shutdown()
        server.http.server_close()
