"""The coherence judge: its prompt, the score of its answer, and the call to its server. The
server of these tests is a local stand-in on the loopback address (`serve_chat`)."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from posteriorbench_methods.steering import coherence


def test_the_score_is_the_mean_number_under_the_judges_distribution(
    chat_response: Callable[..., dict[str, Any]],
) -> None:
    response = chat_response("80", {"80": 0.5, "90": 0.3, "eighty": 0.15, "101": 0.05})
    top = coherence.answer_top_logprobs(response)
    assert [item["token"] for item in top] == ["80", "90", "eighty", "101"]
    # Words and numbers above 100 are left out, and the rest is weighted by probability.
    assert coherence.score(top) == pytest.approx((80 * 0.5 + 90 * 0.3) / 0.8)
    assert coherence.score(coherence.answer_top_logprobs(chat_response(" 7 "))) == 7
    # Less than a quarter of the probability on numbers gives no score.
    assert coherence.score([{"token": "80", "logprob": -2.0}]) is None
    assert coherence.score([]) is None


def test_an_answer_that_is_not_a_number_has_no_score(
    chat_response: Callable[..., dict[str, Any]],
) -> None:
    refusal = chat_response("I cannot rate this.")
    assert coherence.answer_top_logprobs(refusal) == []
    assert coherence.judge_response(lambda prompt: refusal, "{question}{answer}", "q", "a") == {
        "coherence": None,
        "answer": "I cannot rate this.",
        "top_logprobs": [],
    }
    assert not coherence.is_coherent(None, 50) and not coherence.is_coherent(49.9, 50)
    assert coherence.is_coherent(50.0, 50)


def test_the_answer_follows_the_last_channel_marker(
    chat_response: Callable[..., dict[str, Any]],
) -> None:
    response = chat_response("85")
    response["choices"][0]["message"]["content"] = "analysis: it rambles<|message|>85"
    assert coherence.answer(response) == "85"
    assert coherence.score(coherence.answer_top_logprobs(response)) == 85


def test_a_response_without_the_answers_log_probabilities_is_an_error(
    chat_response: Callable[..., dict[str, Any]],
) -> None:
    split = chat_response("85")
    split["choices"][0]["logprobs"]["content"] = [
        {"token": "8", "top_logprobs": []},
        {"token": "5", "top_logprobs": []},
    ]
    with pytest.raises(coherence.JudgeError, match="is not one token of its response"):
        coherence.answer_top_logprobs(split)
    bare = chat_response("85")
    bare["choices"][0]["logprobs"] = None
    with pytest.raises(coherence.JudgeError, match="no token log probabilities"):
        coherence.answer_top_logprobs(bare)
    with pytest.raises(coherence.JudgeError, match="has no message"):
        coherence.answer({"error": "overloaded"})


def test_the_judges_prompt_is_read_from_the_checkout_or_from_a_file(
    lmtailrisk_checkout: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    from_checkout = coherence.read_prompt(lmtailrisk_checkout, None)
    assert "{question}" in from_checkout and "{answer}" in from_checkout
    own = tmp_path / "prompt.txt"
    own.write_text("Is [{answer}] a coherent answer to [{question}]? Answer 0 to 100.")
    assert coherence.read_prompt(lmtailrisk_checkout, own) == own.read_text()
    with pytest.raises(coherence.JudgeError, match="needs its prompt"):
        coherence.read_prompt(None, None)
    with pytest.raises(coherence.JudgeError, match="does not exist"):
        coherence.read_prompt(None, tmp_path / "absent.txt")
    with pytest.raises(coherence.JudgeError, match="the checkout has no"):
        coherence.read_prompt(tmp_path, None)
    own.write_text("Rate the answer.")
    with pytest.raises(coherence.JudgeError, match="needs the placeholders"):
        coherence.read_prompt(None, own)
    own.write_text("Rate {answer} to {question} as {json}.")
    with pytest.raises(coherence.JudgeError, match="braces other than"):
        coherence.read_prompt(None, own)


def test_the_judge_is_asked_through_the_chat_completions_interface(
    serve_chat: Callable[..., Any], chat_response: Callable[..., dict[str, Any]]
) -> None:
    server = serve_chat(lambda body: chat_response("70"))
    judge = coherence.ChatServer(server.url, "judge-model", api_key="secret")
    judged = coherence.judge_response(judge, "Q: {question} A: {answer}", "why", "because")
    assert judged["coherence"] == 70 and judged["answer"] == "70"
    (request,) = server.requests
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == "Bearer secret"
    assert request["body"] == {
        "model": "judge-model",
        "messages": [{"role": "user", "content": "Q: why A: because"}],
        "max_tokens": 10000,
        "temperature": 0.0,
        "seed": 0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    # Without a key no authorization is sent.
    coherence.ChatServer(server.url + "/", "judge-model")("again")
    assert server.requests[-1]["authorization"] is None


def test_a_request_that_fails_is_sent_again_and_a_refused_one_is_not(
    serve_chat: Callable[..., Any], chat_response: Callable[..., dict[str, Any]]
) -> None:
    def respond(body: dict[str, Any]) -> dict[str, Any]:
        return chat_response("70")

    busy = serve_chat(respond, statuses=(503, 429))
    judge = coherence.ChatServer(busy.url, "judge-model", attempts=3, pause_seconds=0.0)
    assert coherence.answer(judge("prompt")) == "70" and len(busy.requests) == 3
    down = serve_chat(respond, statuses=(500, 500))
    with pytest.raises(coherence.JudgeError, match="did not answer .status 500"):
        coherence.ChatServer(down.url, "judge-model", attempts=2, pause_seconds=0.0)("prompt")
    refusing = serve_chat(respond, statuses=(400, 200))
    with pytest.raises(coherence.JudgeError, match="status 400"):
        coherence.ChatServer(refusing.url, "judge-model", attempts=3, pause_seconds=0.0)("p")
    assert len(refusing.requests) == 1
    with pytest.raises(coherence.JudgeError, match="http or https URL"):
        coherence.ChatServer("file:///etc/passwd", "judge-model")
