"""Which task information the code that builds a sampler reads.

A method builds its sampler from the task's prompt and rule, from responses that it draws and pays
for, and from the corpus hits that it buys. It does not see what the evaluation holds out, and it
does not know p̂(S). A static check over every module of the package (parsed with `ast`, never
imported) keeps it so:

1. No module reads held-out evaluation data: the references, the fallback response or the
   evaluation file.
2. Only `common/run_context.py` calls the package's evaluation, which loads that data itself.
3. p̂(S) (`Task.hit_count`, `Task.probability`, `log_event_probability`) and the number of corpus
   hits (`CorpusAccess.available`) are read only at the sites listed in `TASK_INFORMATION`.

A module that needs another read is added to the lists below with its reason.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "posteriorbench_methods"
FAMILIES = {"sft", "prompt", "steering", "gfn", "ctl", "dpg"}

# Held-out data and its loader: never read by any module of the package.
EVALUATION_DATA = {
    "load_evaluation_data",
    "EvaluationData",
    "evaluation_data",
    "EVALUATION_FILE",
    "references",
    "fallback_text",
    "fallback_token_ids",
    "fallback_logp",
}

# The package's evaluation entry points (they load the held-out data themselves).
EVALUATION_ENTRY = {
    "evaluate",
    "evaluate_both",
    "no_sampler_result",
    "fallback_result",
}

# The one module that evaluates a sampler, after a method has built it, and writes its record.
EVALUATION_GLUE = {"common/run_context.py"}

# Task information that only some modules may read: name -> {module: reason}.
TASK_INFORMATION: dict[str, dict[str, str]] = {
    "hit_count": {},
    "log_event_probability": {},
    "probability": {
        "sft/draws_from_p.py": "SFT's draws are taken from the corpus: a draw is a hit with "
        "probability p̂(S); the method sees only where its hits are",
    },
    "available": {
        "sft/draws_from_p.py": "the number of corpus hits that the draws can take",
    },
}


# Folders that are not method code. `generalization` defines tasks outside the benchmark and
# evaluates samplers on them: like the package itself, it holds the held-out data, p̂(S) and the
# evaluation, and the three checks below do not apply to it.
TASKS_AND_EVALUATION = {"generalization"}


def modules() -> list[pathlib.Path]:
    """Every module of the package that is method code: all but TASKS_AND_EVALUATION."""
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if path.relative_to(PACKAGE).parts[0] not in TASKS_AND_EVALUATION
    )


def test_the_checks_cover_every_family() -> None:
    packages = {path.parent.name for path in PACKAGE.glob("*/__init__.py")}
    assert packages == FAMILIES | {"common", "analysis"} | TASKS_AND_EVALUATION
    checked = {relative(path).split("/")[0] for path in modules()}
    assert packages - TASKS_AND_EVALUATION <= checked


def read_names(path: pathlib.Path) -> set[str]:
    """Attribute names read, names used, names imported, and getattr(...) string attributes."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
            if node.asname is not None:
                names.add(node.asname)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def attribute_reads(path: pathlib.Path) -> set[str]:
    """Attribute names read with `x.name` or getattr(x, "name")."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def relative(path: pathlib.Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


def held_out_reads(path: pathlib.Path) -> set[str]:
    return read_names(path) & EVALUATION_DATA


def evaluation_calls(path: pathlib.Path) -> set[str]:
    if relative(path) in EVALUATION_GLUE:
        return set()
    return attribute_reads(path) & EVALUATION_ENTRY


def unlisted_task_information(path: pathlib.Path) -> set[str]:
    reads = attribute_reads(path)
    return {
        name
        for name, allowed in TASK_INFORMATION.items()
        if name in reads and relative(path) not in allowed
    }


def test_no_module_reads_held_out_evaluation_data() -> None:
    found = {relative(p): sorted(held_out_reads(p)) for p in modules() if held_out_reads(p)}
    assert not found


def test_only_evaluation_glue_calls_the_evaluation() -> None:
    found = {relative(p): sorted(evaluation_calls(p)) for p in modules() if evaluation_calls(p)}
    assert not found


def test_task_information_is_read_only_at_allowlisted_sites() -> None:
    found = {
        relative(p): sorted(unlisted_task_information(p))
        for p in modules()
        if unlisted_task_information(p)
    }
    assert not found


def test_the_checks_catch_a_read(tmp_path: pathlib.Path) -> None:
    """The three checks flag a construction module that reads p̂(S), a reference and the
    evaluation (a stand-in module, not part of the package)."""
    module = tmp_path / "construction.py"
    module.write_text(
        "from posteriorbench.data import load_evaluation_data\n"
        "def build(task, bench, data, access):\n"
        "    n = int(1 / task.probability) + getattr(access, 'available')\n"
        "    first = data.references[0]\n"
        "    return bench.evaluate_both(task, n, first)\n"
    )
    assert held_out_reads(module) == {"load_evaluation_data", "references"}
    assert attribute_reads(module) & EVALUATION_ENTRY == {"evaluate_both"}
    assert {"probability", "available"} <= attribute_reads(module)


def test_allowlisted_sites_exist() -> None:
    """Every allowlisted read is still made (a stale entry would hide a moved read)."""
    for name, allowed in TASK_INFORMATION.items():
        for module in allowed:
            assert name in attribute_reads(PACKAGE / module), f"{module} no longer reads {name}"
    for module in EVALUATION_GLUE:
        assert attribute_reads(PACKAGE / module) & EVALUATION_ENTRY, module
