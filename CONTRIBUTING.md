# Contributing

## Setup

```sh
uv sync
```

installs the package, its locked dependencies and the development tools into `.venv`. The paper's methods in `methods/` are a project of their own, with their own environment and checks ([`methods/CONTRIBUTING.md`](methods/CONTRIBUTING.md)); the commands below do not cover them.

## Checks

Run these before you open a pull request; CI runs the same commands.

```sh
uv run black --check .
uv run isort --check-only .
uv run pyright
uv run pytest
```

- The tests run offline in well under a minute on a tiny random model. The tests of `tests/test_hub_dataset.py` read the dataset from the Hugging Face Hub and skip if it cannot be read; the tests marked `real_model` need Qwen3.5-0.8B and are skipped unless `POSTERIORBENCH_REAL_MODEL` is set (see `tests/test_real_model.py`).
- Code blocks in `README.md`, `docs/` and the examples are executed by `tests/test_examples.py`. If you change one, keep it runnable.
- `uvx pre-commit install` sets up the formatting hooks of `.pre-commit-config.yaml`.

## Conventions

- Black with a line length of 100, isort, strict pyright. Import modules, not names, inside the package.
- Errors a caller can cause raise a subclass of `posteriorbench.PosteriorBenchError` or a built-in `ValueError` / `TypeError`, and the docstring says which.
- A change to the fields of `EvaluationResult` or to the dataset's files is a change of the result or data format: describe it in `CHANGELOG.md`.

## Releases

A version is the value in `posteriorbench/_version.py`, an entry in `CHANGELOG.md` and a git tag `v<version>`.
