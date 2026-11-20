# Contributing

## Setup and checks

In `methods/` of a clone of the repository:

```sh
uv sync
uv run black --check .
uv run isort --check-only .
uv run pyright
uv run python scripts/make_rank_table.py --check
uv run pytest
```

`uv sync` installs the methods, the benchmark package from the directory above and the locked versions (torch 2.11.0, transformers 5.13.1) with Python 3.12 (set in `.python-version`); Python 3.11 and later are supported. A method's draws depend on these versions. It also installs the development tools and matplotlib, which the scripts in `figures/` and their tests use; `uv sync --extra wandb` adds what `--wandb-project` needs. CI runs the same commands. `make_rank_table.py --check` compares the stored tables of the threshold test with the computed ones. The tests run offline on a CPU, with tiny random models and a synthetic task (`tests/conftest.py`); they take about two and a half minutes on a laptop.

## What a family is

A family is a folder `posteriorbench_methods/<family>/` with one method or several that share their code:

```text
__init__.py
settings.py    Settings: a frozen dataclass whose defaults are the settings you report
run.py         run(context, settings) -> list[run_record.RunSummary]
cost.py        what the method pays for, as functions that return lists of charges.ChargeSpec
cli.py         app = cli_options.family_app("<family>", help=..., methods={...})
reselect.py    in a family that trains: the rules that choose a checkpoint again
README.md      the method, a run, its settings, its cost, what it writes, compute and disk
```

`run` receives a `run_context.RunContext`: the benchmark, the task, the original model, the replicate and the run options. It builds the method's sampler and writes one record for each method and setting that the settings ask for. Everything else is shared, in `posteriorbench_methods/common/` (its README describes the modules):

- `cli_options.family_app` gives the family the commands `run` and `config`, with the options of every family. A setting is changed with `--set NAME=VALUE` or `--config FILE`, so a field of `Settings` is a bool, an int, a float, a str, or a tuple of these.
- A family may add commands of its own to the group that `family_app` returns, as the prompt family does with `write-candidates` and the steering family with `build-vectors`.
- `context.stream_seed(owner, stream, *index)` gives the seed of each random stream from the method, the task, the replicate and the stream's name. Do not seed anything from another source.
- `hf_runtime` draws responses from a transformers model and computes log q of given responses, alone or composed with the original model.
- `charges` records what the method spends in the run's ledger. Every pass of a model over a response is charged.
- `submission` chooses the sampler to submit among the method's candidates: state the method's `submission.Rule` once, rank with `rank_from_selection_draws` (or `rank_candidates`), and test with `submit`.
- `training_run` keeps the checkpoints of a method that trains, with their selection draws, and says which are the candidates at a budget.
- `reselection` chooses the checkpoint of a finished training run again by another rule: the submission rule with a 1% threshold, or a rule of the method's own. A family gives `MethodEntry` its rules by name, each a function with the signature of `run`, and `family_app` adds the command `reselect`.
- `context.label(responses)` says which responses are hits of the task's behavior. With it, and with `context.scoring_batch` and `context.generation_batch` for the size of a forward pass, a method also runs on the tasks of `generalization/`, which have other original models, longer responses and labels from a language model: `MethodEntry(..., generalization_tasks=True)` (`common/README.md`, "Tasks beyond the benchmark").
- `context.evaluate_and_record(sampler, ...)` evaluates the submitted sampler and writes its record. `context.record_no_sampler` writes the record of a run that has no sampler, and `context.record_reused` records an evaluation again at another cost.
- `composition.compose` composes the submitted sampler with the original model and writes the record of `<method>+composition`. The family gives it the selection draws, the threshold-test draws and the sampler of a composition setting.
- `errors.MethodsError` is the base class of the errors that the person at the command line can repair. A command prints such an error as one line, as it does for the package's `PosteriorBenchError`; any other error is a traceback.

## Adding a method

1. Write the family's folder. `posteriorbench_methods/steering/` is a short family without training and `posteriorbench_methods/sft/` a short one that trains. The smallest `run.py` evaluates one sampler and writes its record:

   ```python
   def run(context: run_context.RunContext, settings: Settings) -> list[run_record.RunSummary]:
       spend = ledger.Ledger()  # posteriorbench.ledger; charges.apply(spend, ...) records a charge
       recorded = context.evaluate_and_record(
           my_sampler(context),  # a posteriorbench.sampler.Sampler
           method="mymethod",
           setting=run_record.setting_label(),  # "none"; setting_label(budget=B) gives "B=..."
           settings=settings,
           spend=spend,
       )
       return [recorded.summary]
   ```

   and `cli.py` is

   ```python
   app = cli_options.family_app(
       "mymethod",
       help="My method.",
       methods={"mymethod": run_context.MethodEntry(settings.Settings(), run.run)},
   )
   ```

   The run options are in `context.options`, and three of them are the family's part: reading `method_data`, keeping checkpoint files as `keep_checkpoints` says (`training_run.samplers_to_keep`), and leaving out the composed method without `compose`. A check that needs neither the task nor the model, such as method data that is not given, goes in `MethodEntry(..., check=...)`, which runs before they are loaded.
2. Write `<method>+composition` when `context.options.compose` is set: `composition.compose`, with the three functions that `sft/run.py` and `sft/samplers.py` give it. `tests/test_records.py` expects it of every family.
3. If the method trains, let its checkpoint be chosen again: a `reselect.py` with one function for each rule, given to `MethodEntry(..., reselect={...})`. `sft/reselect.py` has the 1% threshold alone, and `ctl/reselect.py` an own rule as well. A function reads the training run with `reselection.read_training_run`, which trains nothing, and writes its records with `reselection.record_threshold_1pct` or `reselection.record_own_rule`.
4. Write the records as the other families do, so that one reader serves all of them (`common/README.md`, "Records"): the selection draws and the threshold-test draws of a method that ranks candidates go to the draws files `selection` and `threshold_tests`, with the keys of `run_record.DRAW_KEYS`; `details.candidates` lists the candidates; a checkpoint's id is `training_run.checkpoint_id(update)`.
5. Add the family's `app` to `FAMILIES` in `posteriorbench_methods/cli.py`.
6. Add tests under `tests/<family>/`, the family to `FAMILIES` in `tests/test_cli.py` and `tests/test_task_information.py`, and a run that takes seconds to `RUNS` in `tests/test_records.py`. For a family that trains, also add a setting of its training run to `TRAINING_SETTING` there, its rules to `RULES` in `tests/test_cli.py`, and its own rule to `OWN_RULE` in `tests/test_records.py`.
7. Add the family to the table of `README.md`, to `ATTRIBUTION.md` and to `docs/paper_map.md`.
8. To have the method in `compute-per-inference`, say what it trains in `TRAINS` of `posteriorbench_methods/analysis/operations.py`, and, if it trains something that no method there trains, what a pass over it costs (`analysis/flops.py`) and how its operations are timed (`analysis/gpu_times.py`, `analysis/time_operations.py`). `cap-sensitivity` and `estimate-ps` read the records of any method.

Three rules hold for every method, and tests keep them:

- A method reads the task's prompt and rule, the responses it draws, and the corpus hits it pays for. It does not read the references, the fallback response or p̂(S) (`tests/test_task_information.py`).
- Only `common/run_context.py` calls the package's evaluation.
- Every draw that a record stores carries its token ids.

## Conventions

- Black with a line length of 100, isort, strict pyright. Import modules, not names.
- Names follow the paper: candidate, selection draws, hit fraction, selection signal, ranking, threshold test, submit, best-observed, composition, replicate, training run.
- Docstrings and comments say what the code does and why.
