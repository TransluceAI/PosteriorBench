"""The writer LLM, behind any server with the OpenAI chat-completions interface.

A call that fails (server error, rate limit, timeout, broken connection, empty answer) is tried
again, and after its last attempt gives no answer. A request the server refuses as it is gives no
answer at once; a refused key, address or model name raises WriterError.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from posteriorbench_methods.common import errors, tracking
from posteriorbench_methods.prompt import writer_messages

# Status codes of a request that may succeed when it is sent again.
_TRY_AGAIN = frozenset({408, 409, 425, 429})
# Status codes that say the key, the address or the model name is wrong.
_MISCONFIGURED = frozenset({401, 403, 404})
_LONGEST_WAIT = 30.0  # seconds


class WriterError(errors.MethodsError, RuntimeError):
    """The server refuses the key, or does not know the address or the model."""


class Writer(Protocol):
    """An LLM that answers chat messages."""

    @property
    def model(self) -> str:
        """The name of the model."""
        ...

    def complete(
        self, messages: Sequence[writer_messages.Message], *, max_tokens: int
    ) -> str | None:
        """The answer to `messages`, or None if every attempt failed. Raises WriterError if the
        writer is misconfigured."""
        ...


def key_from_environment(variable: str) -> str | None:
    """The value of `variable`, or None if unset or empty (a server of your own may need none)."""
    return os.environ.get(variable) or None


def wait_before_attempt(attempt: int) -> float:
    """Seconds before attempt 2, 3, ...: 5 more after each failure, at most 30."""
    return min(_LONGEST_WAIT, 5.0 * (attempt - 1))


class ChatCompletionsWriter:
    """A writer that posts to `<url>/chat/completions` (`url` e.g. http://localhost:8000/v1).
    `key` is sent as a bearer token if given; `timeout` is in seconds per attempt.

    An instance holds no connection and no state that a call changes, so calls may run in several
    threads. Raises WriterError for a url that is not http or https."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        key: str | None,
        temperature: float,
        attempts: int,
        timeout: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if attempts < 1:
            raise ValueError(f"a call has at least one attempt, got {attempts}")
        if not url.startswith(("http://", "https://")):
            raise WriterError(f"the writer's address is an http or https address, got {url!r}")
        self.endpoint = url.rstrip("/") + "/chat/completions"
        self._model = model
        self._key = key
        self.temperature = temperature
        self.attempts = attempts
        self.timeout = timeout
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self, messages: Sequence[writer_messages.Message], *, max_tokens: int
    ) -> str | None:
        """The content of the first choice; None if every attempt failed, each failure logged as
        a warning. Raises WriterError on status 401, 403 or 404."""
        body = json.dumps(
            {
                "model": self._model,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self._key is not None:
            headers["Authorization"] = f"Bearer {self._key}"
        log = logging.getLogger(tracking.LOGGER_NAME)
        for attempt in range(1, self.attempts + 1):
            if attempt > 1:
                self._sleep(wait_before_attempt(attempt))
            request = urllib.request.Request(self.endpoint, data=body, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return _content(json.loads(response.read()))
            except urllib.error.HTTPError as error:
                reason = f"status {error.code}: {_on_one_line(error.read())}"
                if error.code in _MISCONFIGURED:
                    raise WriterError(f"the writer at {self.endpoint} answered {reason}") from error
                if error.code < 500 and error.code not in _TRY_AGAIN:
                    # The server refuses this request, and would refuse it again.
                    log.warning("a writer call was refused (%s)", reason)
                    return None
            except (OSError, http.client.HTTPException, ValueError) as error:
                # No connection, a timeout, a broken or unreadable answer, an answer without
                # content.
                reason = f"{type(error).__name__}: {error}"
            log.warning("attempt %d of a writer call failed (%s)", attempt, reason)
        return None


def _content(answer: Any) -> str:
    """The first choice's text; ValueError if there is none, which counts as a failed attempt."""
    try:
        content = answer["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("the answer has no choice with a message") from error
    if not isinstance(content, str):
        raise ValueError("the answer has no content")
    return content


def _on_one_line(body: bytes, limit: int = 300) -> str:
    """The first `limit` characters of an error body, on one line."""
    return " ".join(body.decode(errors="replace").split())[:limit]
