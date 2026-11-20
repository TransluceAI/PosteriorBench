"""The examples and the documented code run against the current interface.

The README's quickstart, the quickstart notebook and the documented commands read the tables
bundled with the package, take corpus hits from the Hub and load Qwen3.5-0.8B. Here the bundled
tables are those of the tiny synthetic dataset (whose task has the documented id), the Hub dataset
id is served from the same dataset through a stand-in for `huggingface_hub.hf_hub_download`, and
`OriginalModel.from_pretrained` returns the tiny model, so everything runs end to end without
network access or a GPU. One test runs the README's quickstart on the real bundled tables up to
the point where it loads the model. The Python blocks of `docs/` run in the same way.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import re
import shlex
import sys
import textwrap
from types import ModuleType
from typing import Any

import pytest
import typer
from support import hub_stub
from support.datasets import DOCUMENTED_TASK_ID as DOCUMENTED_TASK
from typer.testing import CliRunner

import posteriorbench
from posteriorbench import benchmark, cli, data, original_model, results

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples" / "02_custom_sampler" / "custom_sampler.py"
NOTEBOOK = REPOSITORY / "examples" / "01_quickstart" / "quickstart.ipynb"
DOCS = sorted((REPOSITORY / "docs").glob("*.md"))
DOCUMENTS = [
    REPOSITORY / "README.md",
    *sorted((REPOSITORY / "examples").glob("*/README.md")),
    REPOSITORY / "tools" / "README.md",
    *DOCS,
]


def load_module(path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while it is executed
    spec.loader.exec_module(module)
    return module


def notebook_cells(kind: str) -> list[str]:
    """The sources of the notebook's cells of one kind ("code" or "markdown"), in order."""
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == kind]


def python_blocks(markdown: str) -> list[str]:
    """The ```python blocks of a Markdown text, in order."""
    return re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL)


def documented_commands(markdown: str) -> list[str]:
    """The commands in the ```bash blocks of a Markdown text, with continuation lines joined."""
    commands: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", markdown, flags=re.DOTALL):
        lines = block.replace("\\\n", " ").splitlines()
        commands.extend(line.strip() for line in lines if line.strip())
    return commands


def normalized(text: str) -> str:
    return " ".join(text.replace("\\\n", " ").split())


@pytest.fixture()
def offline_hub(
    monkeypatch: pytest.MonkeyPatch, tiny_model: Any, tiny_dataset: pathlib.Path
) -> data.Task:
    """Read the bundled tables from the tiny dataset, serve `Transluce/PosteriorBench` (the hit
    files) from it, and load the tiny model instead of Qwen3.5-0.8B."""
    monkeypatch.setattr(data, "BUNDLED_SOURCE", tiny_dataset)
    hub_stub.install_offline_hub(monkeypatch, tiny_dataset)

    def from_pretrained(*args: Any, **kwargs: Any) -> original_model.OriginalModel:
        return tiny_model

    monkeypatch.setattr(original_model.OriginalModel, "from_pretrained", from_pretrained)
    return data.get_task(data.load_tasks(tiny_dataset), DOCUMENTED_TASK)


def test_example_method_is_charged_and_evaluated(
    tiny_model: Any, tiny_dataset: pathlib.Path
) -> None:
    example = load_module(EXAMPLE)
    bench = benchmark.Benchmark(tiny_dataset)
    task = bench.task(DOCUMENTED_TASK)
    candidate, spend = example.build_sampler(task=task, model=tiny_model, benchmark=bench)
    assert isinstance(spend, posteriorbench.Ledger)
    assert spend.inferences == 2 * example.SELECTION_DRAWS == 64
    result = bench.evaluate(task, candidate, tiny_model, ledger=spend, seed=0, replicate=0)
    assert result.task_id == task.task_id
    assert math.isfinite(result.capped_forward_kl) and math.isfinite(result.reverse_kl)
    assert result.construction_inferences == 64.0 and result.inferences_per_response == 1.0


def test_quickstart_notebook_runs(offline_hub: data.Task) -> None:
    namespace: dict[str, Any] = {}
    for index, source in enumerate(notebook_cells("code")):
        exec(compile(source, f"{NOTEBOOK.name}, code cell {index}", "exec"), namespace)
    assert namespace["task"] == offline_hub
    result = namespace["result"]
    assert isinstance(result, results.EvaluationResult) and result.task_id == DOCUMENTED_TASK
    assert result.method is None and result.construction_inferences == 0.0
    assert math.isfinite(result.capped_forward_kl) and math.isfinite(result.reverse_kl)
    fallback = namespace["fallback"]
    assert isinstance(fallback, results.EvaluationResult) and fallback.fallback
    assert namespace["ledger"].inferences == pytest.approx(3 / offline_hub.probability)


def test_readme_quickstart_runs(offline_hub: data.Task) -> None:
    namespace: dict[str, Any] = {}
    quickstart = python_blocks((REPOSITORY / "README.md").read_text())[0]
    exec(compile(quickstart, "README quickstart", "exec"), namespace)
    assert namespace["task"] == offline_hub
    result = namespace["result"]
    assert isinstance(result, results.EvaluationResult) and result.task_id == DOCUMENTED_TASK
    assert result.construction_inferences == 0.0 and result.dataset is not None
    assert result.dataset["source"] == "bundled"


class NeedsModel(Exception):
    """Raised where the quickstart would load Qwen3.5-0.8B."""


def test_readme_quickstart_needs_no_download_before_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the real bundled tables, the README's quickstart reaches the line that loads the model
    without downloading any dataset file."""
    import huggingface_hub

    def no_download(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("the quickstart downloaded a dataset file before loading the model")

    def needs_model(*args: Any, **kwargs: Any) -> original_model.OriginalModel:
        raise NeedsModel

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    monkeypatch.setattr(original_model.OriginalModel, "from_pretrained", needs_model)
    namespace: dict[str, Any] = {}
    quickstart = python_blocks((REPOSITORY / "README.md").read_text())[0]
    with pytest.raises(NeedsModel):
        exec(compile(quickstart, "README quickstart", "exec"), namespace)
    task = namespace["task"]
    assert isinstance(task, data.Task) and task.task_id == DOCUMENTED_TASK and task.benchmark_task


def test_package_docstring_shows_the_readme_quickstart() -> None:
    """The quickstart is stated once: the package docstring repeats the README's block."""
    docstring = posteriorbench.__doc__
    assert docstring is not None
    tour = docstring.split("Quickstart::\n", 1)[1].split("\n\nThe names above", 1)[0]
    quickstart = python_blocks((REPOSITORY / "README.md").read_text())[0]
    assert textwrap.dedent(tour).strip() == quickstart.strip()


@pytest.mark.parametrize("document", DOCS, ids=lambda path: path.name)
def test_docs_python_blocks_run(document: pathlib.Path, offline_hub: data.Task) -> None:
    """Every Python block of a docs page runs, in order, after the README's quickstart (the
    pages continue from its `bench`, `task`, `model` and `candidate`)."""
    namespace: dict[str, Any] = {}
    quickstart = python_blocks((REPOSITORY / "README.md").read_text())[0]
    exec(compile(quickstart, "README quickstart", "exec"), namespace)
    for index, block in enumerate(python_blocks(document.read_text())):
        exec(compile(block, f"{document.name}, block {index}", "exec"), namespace)


def test_readme_sampler_skeleton_is_a_sampler() -> None:
    """The README's second block defines a `Sampler` subclass with the interface's two methods."""
    namespace: dict[str, Any] = {}
    skeleton = python_blocks((REPOSITORY / "README.md").read_text())[1]
    exec(compile(skeleton, "README sampler skeleton", "exec"), namespace)
    sampler_class = namespace["MySampler"]
    assert issubclass(sampler_class, posteriorbench.Sampler)
    assert sampler_class().inferences_per_response == 1.0


def test_readme_loop_over_the_tasks_runs_and_aggregates(
    offline_hub: data.Task, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The README's third block, run after its quickstart, writes one result per task and
    replicate, and the `aggregate` command the README shows next reads the file."""
    monkeypatch.chdir(tmp_path)  # the block writes my_method.jsonl here
    readme = (REPOSITORY / "README.md").read_text()
    namespace: dict[str, Any] = {}
    blocks = python_blocks(readme)
    exec(compile(blocks[0], "README quickstart", "exec"), namespace)
    exec(compile(blocks[2], "README loop over the tasks", "exec"), namespace)
    tasks = namespace["bench"].tasks()
    lines = (tmp_path / "my_method.jsonl").read_text().splitlines()
    assert len(lines) == 3 * len(tasks) > 0
    (command,) = re.findall(r"```sh\n(uv run posteriorbench aggregate .*?)\n```", readme)
    outcome = CliRunner().invoke(cli.app, shlex.split(command)[3:])
    assert outcome.exit_code == 0, f"{outcome.output}\n{outcome.exception!r}"
    (summary,) = json.loads(outcome.output)
    assert (summary["method"], summary["tasks"]) == ("my-method", len(tasks))
    assert summary["replicates_per_task"] == 3


def run_documented_command(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one documented `uv run ...` command in this process (`posteriorbench` through its
    typer app, `python <script>` through the script's typer app or its `main`), with a documented
    PYTHONPATH prepended to sys.path. Fails on any other command shape."""
    words = shlex.split(command)
    while words and re.fullmatch(r"[A-Z_]+=\S*", words[0]):
        name, value = words.pop(0).split("=", 1)
        assert name == "PYTHONPATH", f"unsupported environment variable in {command!r}"
        monkeypatch.syspath_prepend(str(REPOSITORY / value))
    assert words[:2] == ["uv", "run"], f"documented commands start with `uv run`: {command!r}"
    program, *arguments = words[2:]
    if program == "posteriorbench":
        app = cli.app
    else:
        assert program == "python" and arguments, f"unsupported command {command!r}"
        script, *arguments = arguments
        module = load_module(REPOSITORY / script)
        app = getattr(module, "app", None)
        if app is None:
            app = typer.Typer()
            app.command()(module.main)
    outcome = CliRunner().invoke(app, arguments)
    assert outcome.exit_code == 0, f"{command}\n{outcome.output}\n{outcome.exception!r}"


@pytest.mark.parametrize(
    "document", [*DOCUMENTS, NOTEBOOK], ids=lambda path: str(path.relative_to(REPOSITORY))
)
def test_documented_commands_run(
    document: pathlib.Path,
    offline_hub: data.Task,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    if document == NOTEBOOK:
        commands = [c for cell in notebook_cells("markdown") for c in documented_commands(cell)]
    else:
        commands = documented_commands(document.read_text())
    monkeypatch.chdir(tmp_path)  # result files land here
    monkeypatch.delitem(sys.modules, "custom_sampler", raising=False)
    for command in commands:
        run_documented_command(command, monkeypatch)


def test_readme_documents_commands() -> None:
    assert documented_commands((REPOSITORY / "README.md").read_text())
