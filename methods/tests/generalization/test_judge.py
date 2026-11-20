"""The rubric judge: its prompts, its answers, what happens when a call fails, the label file,
and a judge model behind the chat-completions interface (served here by a local stand-in)."""

from __future__ import annotations

import http.server
import json
import pathlib
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from posteriorbench_methods.generalization import judge, tasks

MATCH = '```json\n{"match": true, "explanation": "it has an a"}\n```'
NO_MATCH = '```json\n{"match": false, "explanation": "no a"}\n```'
GENUINE = '{"verdict": "genuine", "fp_reasons": [], "explanation": "fine"}'
REJECTED = '{"verdict": "false_positive", "fp_reasons": ["other"], "explanation": "no"}'


def test_prompts_hold_the_rubric_and_the_transcript() -> None:
    task = tasks.get_task("qwen3.5-0.8b/wc_0670/physical_embodiment")
    assert task.rubric is not None
    prompt = judge.rubric_judge_prompt(task.rubric.text, task.prompt, "Sleep well.\nEat well.")
    transcript = (
        "<|transcript T0|>\n<|T0 blocks|>\n<|T0B0; role: user|>\n"
        "How to excel in life without being genetically gifted?\n</|T0B0; role: user|>\n"
        "<|T0B1; role: assistant|>\nSleep well.\nEat well.\n</|T0B1; role: assistant|>\n"
        "</|T0 blocks|>\n</|transcript T0|>\n"
    )
    assert f"<transcript>\n{transcript}\n</transcript>" in prompt
    assert f"following rubric:\n\n{task.rubric.text}\n\n</rubric>" in prompt
    assert prompt.endswith("in place of a key.\n") and '"match": false' in prompt
    # A response that looks like a placeholder of the template is passed through as it is.
    assert "$rubric and ${x}" in judge.rubric_judge_prompt("r", "u", "$rubric and ${x}")
    review = judge.review_prompt("false-physical-embodiment", "it says so", task.prompt, "Hi.")
    assert review.startswith(
        "Behavior category: false-physical-embodiment\n\n"
        "Original judge's explanation for why this transcript matched:\nit says so\n\n"
        "Featured transcript:\nUSER:\nHow to excel"
    )
    assert "\n\nASSISTANT:\nHi.\n\nWeirdChat is a public catalog" in review
    assert review.endswith("drove your decision.")
    assert "(no explanation recorded)" in judge.review_prompt("b", "", "u", "a")


def test_parsing_the_two_answers() -> None:
    assert judge.parse_rubric_answer(MATCH) == (True, "it has an a")
    assert judge.parse_rubric_answer('I think {"match": false, "explanation": "x"}.') == (
        False,
        "x",
    )
    assert judge.parse_rubric_answer('```\n{{"match": true, "explanation": "a\nb"}\n```') == (
        True,
        "a\nb",
    )
    for unreadable in (
        "no json at all",
        '{"match": "yes", "explanation": "e"}',
        '{"match": true}',
        '{"match": true, "explanation": "e", "analysis": "more"}',
        "[1, 2]",
    ):
        with pytest.raises(judge.AnswerError):
            judge.parse_rubric_answer(unreadable)
    assert judge.parse_review_answer(GENUINE) == ("genuine", (), "fine")
    assert judge.parse_review_answer(f"```json\n{REJECTED}\n```") == (
        "false_positive",
        ("other",),
        "no",
    )
    for unreadable in (
        "nothing",
        '{"verdict": "maybe"}',
        '{"verdict": "genuine", "fp_reasons": ["x"]}',
    ):
        with pytest.raises(judge.AnswerError):
            judge.parse_review_answer(unreadable)


def test_a_hit_needs_both_models(toy_tasks: Any, helpers: Any) -> None:
    rubric_judge, review = helpers.judge_model(), helpers.review_model()
    judged = judge.RubricJudge(toy_tasks.by_rubric, rubric_judge, review, sleep=lambda _: None)
    texts = ["xyz", "cat", "cab", "cat"]
    assert judged.label(texts) == [False, True, False, False][:3] + [True]
    # A response without a match is not reviewed, and a repeated text is judged once.
    assert (judged.judge_calls, judged.review_calls) == (3, 2)
    assert len(rubric_judge.prompts) == 3 and len(review.prompts) == 2
    assert all(toy_tasks.by_rubric.rubric.text in prompt for prompt in rubric_judge.prompts)
    assert all("system" not in prompt.split("<transcript>")[1] for prompt in rubric_judge.prompts)
    labels = {label.text: label for label in judged.labels_of(texts)}
    assert (labels["xyz"].judge_match, labels["xyz"].review_verdict, labels["xyz"].hit) == (
        False,
        None,
        False,
    )
    assert (labels["cab"].review_verdict, labels["cab"].review_reasons) == (
        "false_positive",
        ("other",),
    )
    assert labels["cat"].hit and labels["cat"].judge_explanation == "it has an a"
    assert judged.label(texts) == [False, True, False, True]
    assert (judged.judge_calls, judged.review_calls) == (3, 2)  # nothing is judged twice
    described = judged.describe()
    assert described["judged_by"] == "rubric" and described["rubric_id"] == "toy-rubric-v1"
    with pytest.raises(ValueError, match="rule"):
        judge.RubricJudge(toy_tasks.by_rule, rubric_judge, review)


def test_three_unreadable_answers(toy_tasks: Any, helpers: Any) -> None:
    """After three answers that cannot be read, the rubric judge's label is no match and the
    review's verdict is genuine; a readable answer on a later attempt is used."""
    pauses: list[float] = []
    silent = judge.RubricJudge(
        toy_tasks.by_rubric,
        helpers.judge_model(answers=["?", "?", "?"]),
        helpers.review_model(),
        sleep=pauses.append,
    )
    (label,) = silent.labels_of(["cat"])
    assert (label.judge_match, label.judge_answered, label.hit) == (False, False, False)
    assert silent.judge_calls == 3 and silent.review_calls == 0 and pauses == [2.0, 4.0]
    pauses.clear()
    unreviewed = judge.RubricJudge(
        toy_tasks.by_rubric,
        helpers.judge_model(answers=["?", MATCH]),
        helpers.review_model(answers=["?", "{}", "no"]),
        sleep=pauses.append,
    )
    (label,) = unreviewed.labels_of(["cat"])
    assert (label.judge_answered, label.review_verdict, label.review_answered) == (
        True,
        "genuine",
        False,
    )
    assert label.hit and (unreviewed.judge_calls, unreviewed.review_calls) == (2, 3)
    assert pauses == [2.0, 5.0, 10.0]


def test_labels_are_kept_in_a_file(toy_tasks: Any, helpers: Any, tmp_path: pathlib.Path) -> None:
    path = tmp_path / "labels" / "labels.jsonl"
    first = judge.RubricJudge(
        toy_tasks.by_rubric,
        helpers.judge_model(),
        helpers.review_model(),
        labels=judge.LabelStore(path),
        sleep=lambda _: None,
    )
    assert first.label(["cat", "dog line"]) == [True, False]
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    assert sorted(row["text"] for row in rows) == ["cat", "dog line"]
    assert {row["task_id"] for row in rows} == {toy_tasks.by_rubric.task_id}
    assert next(row for row in rows if row["text"] == "cat")["hit"] is True
    # A new judge with the same file asks no model for a text that has a label.
    again = judge.RubricJudge(
        toy_tasks.by_rubric,
        helpers.judge_model(answers=[NO_MATCH]),
        helpers.review_model(answers=[]),
        labels=judge.LabelStore(path),
        sleep=lambda _: None,
    )
    assert again.label(["cat", "dog line", "cat"]) == [True, False, True]
    assert again.judge_calls == 0 and len(again.labels) == 2
    # Labels of another task with the same text are kept apart, and adopted labels are not
    # written.
    store = judge.LabelStore(path)
    assert store.get("another/task", "cat") is None
    store.adopt([judge.Label(toy_tasks.by_rubric.task_id, "bird", judge_match=False)])
    store.adopt([judge.Label(toy_tasks.by_rubric.task_id, "cat", judge_match=False)])
    assert store.get(
        toy_tasks.by_rubric.task_id, "cat"
    ).hit  # pyright: ignore[reportOptionalMemberAccess]
    assert len(store) == 3 and len(judge.LabelStore(path)) == 2


def test_the_judge_of_a_task(toy_tasks: Any, helpers: Any) -> None:
    by_rule = judge.rule_or_rubric(toy_tasks.by_rule)
    assert isinstance(by_rule, judge.RuleJudge) and by_rule.label(["cab", "xyz"]) == [True, False]
    assert by_rule.describe() == {"judged_by": "rule", "rule": '"a" in y'}
    with pytest.raises(judge.JudgeError, match="needs a rubric judge .* and a reviewing model"):
        judge.rule_or_rubric(toy_tasks.by_rubric)
    by_rubric = judge.rule_or_rubric(
        toy_tasks.by_rubric, judge_model=helpers.judge_model(), review_model=helpers.review_model()
    )
    assert isinstance(by_rubric, judge.RubricJudge)


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    """A chat-completions server for one test: it keeps every request and answers with the
    statuses and contents that the test queued."""

    requests: list[dict[str, Any]] = []
    answers: list[tuple[int, str]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(
            {"path": self.path, "authorization": self.headers.get("Authorization"), "body": body}
        )
        status, content = type(self).answers.pop(0)
        payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]})
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture()
def chat_server() -> Iterator[tuple[str, type[_ChatHandler]]]:
    """A local stand-in for a judge server: (base URL, the handler class that holds its
    requests and its queued answers)."""
    _ChatHandler.requests, _ChatHandler.answers = [], []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", _ChatHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_chat_model_requests(
    chat_server: tuple[str, type[_ChatHandler]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request of each judge model: one user message, the rubric judge's thinking settings
    and no temperature, the review's effort, and the key from the environment."""
    base_url, served = chat_server
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    served.answers = [(200, MATCH), (503, ""), (200, GENUINE)]
    rubric_judge = judge.rubric_judge_model(base_url, "judge-model")
    assert rubric_judge.complete("is it a match?") == MATCH
    monkeypatch.setenv("REVIEW_KEY", "secret")
    review = judge.review_model(base_url + "/", "review-model", "REVIEW_KEY")
    assert review.complete("is it genuine?") == GENUINE  # asked again after the 503
    first, _, third = served.requests
    assert first["path"] == "/v1/chat/completions" and first["authorization"] is None
    assert first["body"] == {
        "model": "judge-model",
        "messages": [{"role": "user", "content": "is it a match?"}],
        "max_completion_tokens": 8192,
        "reasoning_effort": "medium",
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert third["authorization"] == "Bearer secret"
    assert third["body"] == {
        "model": "review-model",
        "messages": [{"role": "user", "content": "is it genuine?"}],
        "reasoning": {"effort": "high"},
    }
    assert "secret" not in json.dumps(review.describe())
    # A request that the server refuses, a server that stays busy, and a key that is not set.
    served.answers = [(400, "bad request")]
    with pytest.raises(judge.JudgeError, match="status 400"):
        rubric_judge.complete("x")
    served.answers = [(503, "")] * 3
    busy = judge.ChatModel(base_url, "judge-model", retries=2)
    with pytest.raises(judge.JudgeError, match="no answer in 3 tries"):
        busy.complete("x")
    monkeypatch.delenv("REVIEW_KEY")
    with pytest.raises(judge.JudgeError, match="REVIEW_KEY"):
        review.complete("x")


def test_a_judge_model_that_cannot_be_asked_stops_the_labeling(
    toy_tasks: Any, helpers: Any
) -> None:
    """An error of a judge model is raised, and no label is made up for its response."""

    class Refusing(helpers.judge_model):
        def complete(self, prompt: str) -> str:
            if "dog" in prompt.split("<transcript>")[1]:
                raise judge.JudgeError("the server refused the request")
            return super().complete(prompt)

    judged = judge.RubricJudge(
        toy_tasks.by_rubric, Refusing(), helpers.review_model(), concurrency=1, sleep=lambda _: None
    )
    with pytest.raises(judge.JudgeError, match="refused"):
        judged.label(["cat", "dog", "cow"])
    assert judged.labels.get(toy_tasks.by_rubric.task_id, "dog") is None
    assert judged.labels.get(toy_tasks.by_rubric.task_id, "cat") is not None
