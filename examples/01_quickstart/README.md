# Quickstart

[`quickstart.ipynb`](quickstart.ipynb) walks through the `posteriorbench` package: browsing the benchmark's tasks; inspecting a task as a method sees it; evaluating a sampler on the task, reading its cost and its KL divergences from the behavior posterior; and taking corpus hits, which are charged to a construction ledger.

To run the notebook yourself, install dependencies with `uv sync` and start Jupyter with `uv run --with jupyter jupyter lab` (Jupyter is not among the package's dependencies). The last part downloads corpus hits from the dataset on the Hugging Face Hub. Evaluating a sampler runs Qwen3.5-0.8B, so we recommend a GPU: one evaluation takes about a minute and a half on an H100 and about an hour on a laptop CPU, and prints nothing while it runs.
