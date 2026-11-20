"""The writer behind the chat-completions interface, against a stand-in server on this
machine."""

from __future__ import annotations

from typing import Any

import pytest

from posteriorbench_methods.prompt import writer

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u — 你"}]


def make_writer(server: Any, waits: list[float], *, key: str | None = None) -> Any:
    """A writer of the stand-in server with three attempts per call, whose waits between
    attempts are recorded in `waits` and not waited."""
    return writer.ChatCompletionsWriter(
        server.url,
        "writer-model",
        key=key,
        temperature=1.0,
        attempts=3,
        timeout=10.0,
        sleep=waits.append,
    )


def answer(content: str | None) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_a_call_posts_the_messages_and_returns_the_content(writer_server: Any) -> None:
    writer_server.respond = lambda body, n: (200, answer("the answer"))
    waits: list[float] = []
    llm = make_writer(writer_server, waits)
    assert llm.model == "writer-model"
    assert llm.complete(MESSAGES, max_tokens=4000) == "the answer"
    ((headers, body),) = writer_server.requests
    assert body == {
        "model": "writer-model",
        "messages": MESSAGES,
        "max_tokens": 4000,
        "temperature": 1.0,
    }
    assert headers["content-type"] == "application/json" and "authorization" not in headers
    assert waits == []


def test_the_key_is_read_from_an_environment_variable_and_sent_as_a_bearer_token(
    writer_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STAND_IN_KEY", raising=False)
    assert writer.key_from_environment("STAND_IN_KEY") is None
    monkeypatch.setenv("STAND_IN_KEY", "a-key-of-the-test")
    key = writer.key_from_environment("STAND_IN_KEY")
    writer_server.respond = lambda body, n: (200, answer("ok"))
    make_writer(writer_server, [], key=key).complete(MESSAGES, max_tokens=1)
    ((headers, _),) = writer_server.requests
    assert headers["authorization"] == "Bearer a-key-of-the-test"


def test_a_call_that_fails_is_tried_again(writer_server: Any) -> None:
    replies = [(500, {"error": "busy"}), (200, answer(None)), (200, answer("third time"))]
    writer_server.respond = lambda body, n: replies[n]
    waits: list[float] = []
    assert make_writer(writer_server, waits).complete(MESSAGES, max_tokens=1) == "third time"
    assert len(writer_server.requests) == 3
    assert waits == [5.0, 10.0]


def test_the_wait_between_attempts_grows_to_half_a_minute() -> None:
    assert [writer.wait_before_attempt(attempt) for attempt in (2, 3, 7, 8, 40)] == [
        5.0,
        10.0,
        30.0,
        30.0,
        30.0,
    ]


def test_a_call_gives_no_answer_after_its_last_attempt(writer_server: Any) -> None:
    writer_server.respond = lambda body, n: (429, {"error": "rate limit"})
    waits: list[float] = []
    assert make_writer(writer_server, waits).complete(MESSAGES, max_tokens=1) is None
    assert len(writer_server.requests) == 3 and len(waits) == 2
    # Nothing listens at this address: the connection fails at every attempt.
    nowhere = writer.ChatCompletionsWriter(
        "http://127.0.0.1:9",
        "m",
        key=None,
        temperature=1.0,
        attempts=2,
        timeout=2.0,
        sleep=waits.append,
    )
    assert nowhere.complete(MESSAGES, max_tokens=1) is None


def test_a_request_that_the_server_refuses_is_not_sent_again(writer_server: Any) -> None:
    writer_server.respond = lambda body, n: (400, {"error": {"message": "max_tokens too large"}})
    assert make_writer(writer_server, []).complete(MESSAGES, max_tokens=10**9) is None
    assert len(writer_server.requests) == 1


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_wrong_key_address_or_model_stops_the_command(writer_server: Any, status: int) -> None:
    writer_server.respond = lambda body, n: (status, {"error": {"message": "no such thing"}})
    with pytest.raises(writer.WriterError) as raised:
        make_writer(writer_server, []).complete(MESSAGES, max_tokens=1)
    message = str(raised.value)
    assert f"answered status {status}: " in message and "no such thing" in message
    assert "\n" not in message
    assert len(writer_server.requests) == 1


def test_a_writer_has_an_http_address_and_at_least_one_attempt() -> None:
    with pytest.raises(writer.WriterError, match="http"):
        writer.ChatCompletionsWriter(
            "file:///etc", "m", key=None, temperature=1.0, attempts=1, timeout=1.0
        )
    with pytest.raises(ValueError, match="at least one attempt"):
        writer.ChatCompletionsWriter(
            "http://127.0.0.1:9", "m", key=None, temperature=1.0, attempts=0, timeout=1.0
        )
