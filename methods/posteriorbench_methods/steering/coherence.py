"""The coherence judge of the construction responses; only coherent ones enter a vector.

The judge rates a response from 0 to 100 and answers with the number alone. The score is not the
number it writes but its mean under the judge's distribution: the numbers 0 to 100 among the
TOP_LOGPROBS most probable tokens at the answer's position, weighted by their probabilities. There
is no score if the answer is not a number or the numbers hold less than MIN_NUMERIC_PROBABILITY.
A behavior that no rule defines is scored the same way, with its own judge prompt. The judge is
another model, so its calls are recorded and not counted as inferences.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import re
import runpy
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from posteriorbench_methods.common import errors

PROMPT_FILE = pathlib.Path("persona_vectors") / "eval" / "prompts.py"  # in the checkout
PROMPT_NAME = "coherence_0_100"
# The request for one judgment.
TEMPERATURE = 0.0
SEED = 0
TOP_LOGPROBS = 20
MAX_TOKENS = 10_000  # room for a judge that reasons before it answers
# A score needs this much probability on the numbers 0 to 100.
MIN_NUMERIC_PROBABILITY = 0.25
# Some servers return a reasoning model's channel markers in the content; the answer follows
# the last of them.
FINAL_CHANNEL_MARKER = "<|message|>"

Judge = Callable[[str], Mapping[str, Any]]
"""Maps the judge's prompt for one response to the server's chat-completions response."""


class JudgeError(errors.MethodsError, RuntimeError):
    """A missing or malformed judge prompt, a server that does not answer, or a response without
    the answer's log probabilities."""


def read_prompt(checkout: pathlib.Path | None, prompt_file: pathlib.Path | None) -> str:
    """The text of `prompt_file`, else PROMPT_NAME from PROMPT_FILE of `checkout`; reading the
    checkout's file executes it. Raises JudgeError if neither is given or the prompt lacks the
    placeholders `{question}` and `{answer}`."""
    if prompt_file is not None:
        if not prompt_file.is_file():
            raise JudgeError(f"{prompt_file} does not exist; it holds the judge's prompt")
        prompt = prompt_file.read_text()
    elif checkout is not None:
        path = checkout / PROMPT_FILE
        if not path.is_file():
            raise JudgeError(f"the checkout has no {path}, which holds the judge's prompt")
        prompt = runpy.run_path(str(path))["Prompts"][PROMPT_NAME]
    else:
        raise JudgeError(
            "the coherence judge needs its prompt: name the checkout that holds it "
            "(--lmtailrisk-checkout) or a text file (--coherence-prompt)"
        )
    if not isinstance(prompt, str) or "{question}" not in prompt or "{answer}" not in prompt:
        raise JudgeError("the judge's prompt needs the placeholders {question} and {answer}")
    try:
        prompt.format(question="", answer="")
    except (KeyError, IndexError, ValueError) as error:
        raise JudgeError(
            "the judge's prompt has braces other than {question} and {answer}"
        ) from error
    return prompt


@dataclasses.dataclass(frozen=True)
class ChatServer:
    """The judge behind `<url>/chat/completions` (`url` e.g. "http://localhost:8000/v1").
    `api_key` is sent as a bearer token if given. A request that fails in transport or with
    status 429 or 5xx is sent again after `pause_seconds`, `attempts` times in all."""

    url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 900.0
    attempts: int = 8
    pause_seconds: float = 15.0

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise JudgeError(f"the judge's address is an http or https URL, got {self.url!r}")
        if self.attempts < 1:
            raise JudgeError("the judge is asked at least once")

    def __call__(self, prompt: str) -> Mapping[str, Any]:
        """The server's response to `prompt` as the only user message. Raises JudgeError if the
        server refuses the request or does not answer in `attempts` tries."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        failure = ""
        for attempt in range(self.attempts):
            if attempt:
                time.sleep(self.pause_seconds)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                failure = f"status {error.code}: {error.read().decode(errors='replace')[:500]}"
                if error.code != 429 and error.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                failure = repr(error)
        raise JudgeError(f"the judge at {self.url} did not answer ({failure})")


def answer(response: Mapping[str, Any]) -> str:
    """The message content after the last FINAL_CHANNEL_MARKER, stripped."""
    try:
        content = response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise JudgeError(f"the judge's response has no message ({response!r:.300})") from error
    return str(content).rsplit(FINAL_CHANNEL_MARKER, 1)[-1].strip()


def answer_top_logprobs(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The top tokens (`token`, `logprob`) at the last position whose token is the answer; [] if
    the answer is not a number of one to three digits. Raises JudgeError if the response has no
    log probabilities or no single token is the answer."""
    number = answer(response)
    if not re.fullmatch(r"\d{1,3}", number):
        return []
    try:
        tokens = response["choices"][0]["logprobs"]["content"]
        for token in reversed(tokens):
            if token["token"].strip() == number:
                return [dict(item) for item in token["top_logprobs"]]
    except (KeyError, IndexError, TypeError) as error:
        raise JudgeError(
            "the judge's response has no token log probabilities; the server must return "
            "logprobs and top_logprobs"
        ) from error
    raise JudgeError(
        f"the judge's answer {number!r} is not one token of its response; the coherence score "
        "needs a judge whose tokenizer writes each number up to 100 as one token"
    )


def score(top_logprobs: Sequence[Mapping[str, Any]]) -> float | None:
    """The probability-weighted mean of the tokens that are integers 0 to 100; None if they hold
    less than MIN_NUMERIC_PROBABILITY."""
    total = 0.0
    weighted = 0.0
    for item in top_logprobs:
        try:
            number = int(item["token"])
        except ValueError:
            continue
        if 0 <= number <= 100:
            probability = math.exp(item["logprob"])
            total += probability
            weighted += number * probability
    return weighted / total if total >= MIN_NUMERIC_PROBABILITY else None


def score_response(
    judge: Judge, prompt: str, question: str, text: str, *, name: str
) -> dict[str, Any]:
    """Ask `judge` about response `text` to `question` with a 0-to-100 `prompt`. Returns the
    score (or None) under `name`, the judge's `answer`, and the `top_logprobs` it came from."""
    response = judge(prompt.format(question=question, answer=text))
    top = answer_top_logprobs(response)
    return {name: score(top), "answer": answer(response), "top_logprobs": top}


def judge_response(judge: Judge, prompt: str, question: str, text: str) -> dict[str, Any]:
    """`score_response` with the coherence prompt, under the key `coherence`."""
    return score_response(judge, prompt, question, text, name="coherence")


def is_coherent(coherence: float | None, threshold: float) -> bool:
    """A finite score of at least `threshold`."""
    return coherence is not None and math.isfinite(coherence) and coherence >= threshold
