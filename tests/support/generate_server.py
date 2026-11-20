"""A stand-in for a server with SGLang's `/generate` interface, answered by the tiny model on a
local port."""

from __future__ import annotations

import contextlib
import http.server
import json
import threading
from collections.abc import Generator
from typing import Any

from posteriorbench import original_model


@contextlib.contextmanager
def serve_generate(
    model: original_model.OriginalModel, *, drop_last_token: bool = False
) -> Generator[tuple[str, list[dict[str, Any]]]]:
    """Serve `POST /generate` on 127.0.0.1 while the context is open. Yields the server's base
    URL and the list of request bodies received so far.

    A request is answered as SGLang answers it: one output per row of `input_ids`, each with
    `output_ids` (a response of `model`'s network sampled with the request's stop tokens and
    token limit) and `text`. With `drop_last_token`, every `output_ids` lacks its last token, so
    it is not a complete response."""
    requests: list[dict[str, Any]] = []
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            parameters = body["sampling_params"]
            with lock:  # the tiny model samples with torch's global generator
                requests.append(body)
                network = original_model.OriginalModel(
                    model.model,
                    model.tokenizer,
                    max_response_tokens=int(parameters["max_new_tokens"]),
                    stop_token_ids=parameters["stop_token_ids"],
                )
                responses = network.generate(
                    body["input_ids"][0],
                    len(body["input_ids"]),
                    seed=len(requests),
                    temperature=float(parameters["temperature"]),
                )
            outputs = [
                {
                    "output_ids": response[:-1] if drop_last_token else response,
                    "text": model.decode(response),
                    "meta_info": {},
                }
                for response in responses
            ]
            payload = json.dumps(outputs).encode()
            self.send_response(200 if self.path == "/generate" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            """Print nothing for a request."""

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
