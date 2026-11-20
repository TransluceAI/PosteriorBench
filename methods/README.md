# PosteriorBench methods

This directory has the code for every method in the PosteriorBench paper, the paper's analyses, and the data and scripts behind its figures and tables. If you only want to evaluate a sampler of your own, you don't need it: the [benchmark package](..) in the root of this repository is enough.

Each method builds a sampler for one task from a single command, which also evaluates the sampler and records what it cost:

- methods that train: supervised fine-tuning (SFT), GFlowNet, CTL, KL-DPG, JS-DPG and GDC;
- methods that don't: prompt selection, prompt mixtures, a fixed instruction, and steering;
- the composition of each of these samplers with the original model.

They run on any of the package's 108 scored prompt–behavior pairs, which include the benchmark's 21 tasks. Prompt selection, Steering and SFT also run on the six tasks of the paper's section on larger models and LLM-judged behaviors ([below](#larger-models-and-llm-judged-behaviors)).

> [!NOTE]
> The benchmark's data and the responses that these methods draw include sensitive content: offensive language and racial slurs, model responses that discuss or encourage suicide and self-harm, and false statements.

## Setup

The methods are a separate project that depends on the package one directory up. In `methods/`:

```sh
uv sync
```

Every method runs a language model, so you need a GPU to run one; the tests (`uv run pytest`, a few minutes on a CPU) and the figure scripts need none. [`CONTRIBUTING.md`](CONTRIBUTING.md) gives the versions `uv sync` installs, on which a method's draws depend.

## A worked run

```sh
uv run posteriorbench-methods sft run --task wc_0346/em_dash --out runs --set budgets=131072
```

The task asks for responses with an em dash, which Qwen3.5-0.8B writes in 7.9% of its responses to "what if lightning cures you". The command runs SFT with a budget of 2^17 passes: it takes posterior samples from the dataset, fine-tunes a copy of the model on them, submits one checkpoint and evaluates it. It also evaluates two relatives of SFT: Replay, which returns the training samples themselves, and SFT composed with the original model. On an H100 it takes about six minutes and writes a record for each of the three methods under `runs/` ([What a run writes](#what-a-run-writes)). A record's `result.json` has the two KL scores, the native hit rate and the construction cost; lower is better for both KL scores ([how they are computed](../docs/protocol.md#the-four-steps)).

## Options and settings

`posteriorbench-methods <family> run --help` lists the options of a run, which are the same in every family:

| Option | Default | Meaning |
|---|---|---|
| `--task` | required | a task id of the package (`uv run posteriorbench list-tasks`); a run needs a scored pair |
| `--out` | required | the output root |
| `--replicate` | 0 | the replicate: it enters the seed of every random stream that the replicates of a task do not share, and is the seed of the evaluation |
| `--data`, `--data-revision` | the package's | the dataset, a directory or a Hub id |
| `--model`, `--model-revision` | Qwen3.5-0.8B at the benchmark's commit | where the original model is loaded from |
| `--device` | `cuda` if available | the torch device |
| `--compose` / `--no-compose` | on | also run `<method>+composition` |
| `--keep-checkpoints` | `testable` | the checkpoint files that remain after a run: `all`; `testable`, those that a threshold test can reach (the first 8 of each ranking) and the submitted ones; or `returned`, the submitted ones |
| `--evaluation-draws` | 1,024 | the benchmark's number; another one does not give a benchmark result |
| `--method-data` | none | the file of candidate prompts (prompt) or the directory of steering vectors (steering) |
| `--method` | the family's only method | `kl-dpg`, `js-dpg` or `gdc` in the family `dpg`; the prompt family takes its methods as a setting, `--set methods=...` |
| `--config`, `--set` | none | the method's settings: a JSON file, and `NAME=VALUE` for one setting |
| `--wandb-project`, `--wandb-group` | none | send live metrics to wandb (install it with `uv sync --extra wandb`) |

A run on one of the six tasks beyond the benchmark takes a few more options, for the judge models and the size of a forward pass; [`generalization/README.md`](posteriorbench_methods/generalization/README.md) describes them.

A method's settings default to those of the paper's runs, and `posteriorbench-methods <family> config` prints them as JSON. `--set NAME=VALUE` changes one setting and can be repeated: a list is `--set budgets=1048576,2097152`, and `--set update_caps=` is the empty list. `--config FILE` reads a JSON object of settings. `config` takes both as well and prints the settings that `run` would use.

GFlowNet, CTL, KL-DPG, JS-DPG and GDC train once for a task and replicate, and a later run that asks for other budgets or update caps continues that training. If more runs of the same training will follow, or if you will use `reselect`, pass `--keep-checkpoints all`: otherwise a checkpoint file that the later run needs may be gone, and it stops with an error.

## The families

| Family | Methods | One task and replicate at the paper's settings, on one H100 | Disk |
|---|---|---|---|
| [`sft`](posteriorbench_methods/sft) | SFT, Replay | about 4 GPU-hours for the 11 budgets; 15 minutes at 2^19 and 100 at 2^22 | up to 5 GB of checkpoints for a budget while it runs |
| [`prompt`](posteriorbench_methods/prompt) | prompt selection (EB-JC, JC, hit rate, IWAE, CE), Mixture: EB-16, Instruction | about 3 hours for the pool sizes up to 1,024 | about 150 MB |
| [`steering`](posteriorbench_methods/steering) | Steering | about 5 minutes; 1 minute for a further replicate | about 10 MB |
| [`gfn`](posteriorbench_methods/gfn) | GFlowNet | about 15 GPU-hours to 2^22 passes; 4 to 2^20 | about 8.5 GB of checkpoints |
| [`ctl`](posteriorbench_methods/ctl) | CTL | 7 to 10 GPU-hours to 2^22 passes; 2 to 2^20 | about 40 GB of checkpoints |
| [`dpg`](posteriorbench_methods/dpg) | KL-DPG, JS-DPG, GDC | 3 GPU-hours (KL-DPG, JS-DPG) and 4.5 (GDC) to 2^20 passes | 130 to 165 GB of checkpoints while it runs |

Each family's README gives the method in the paper's words, its settings, its cost rules with a worked example and a short run. Every family also runs the composition of its samplers with the original model.

## What a run writes

One record for each method and setting. The setting is a budget (`B=...`), a pool size (`K=...`), an update cap (`u=...`, for GFlowNet and CTL below their smallest budget), or `none`:

```text
<out>/<method>/<setting>/<task>/rep<r>/
    result.json              the benchmark's result: the scores, the hit rate and the cost
    result_recomputed.json   the same draws, with log p of the references recomputed by your model
    evaluation_details.json  the per-response values of the evaluation
    evaluation_hits.json     log q and log p of every hit among the evaluation draws
    construction.json        what the method did: the submitted sampler, the ranking of its
                             candidates, the threshold tests, what it spent
    config.json              the settings, the options, the versions, the runtime
    draws/<name>.jsonl       the responses that the method and the evaluation drew, with token ids
<out>/training_runs/         the checkpoints and the training draws of the training methods
<out>/cache/                 draws that the replicates of a task share (Steering)
<out>/reselected/<rule>/     the records of `reselect` and of a run at the 1% threshold, in the
                             layout of an output root
<out>/summary.jsonl          one line per record written; a command that is run again appends
                             its lines again
```

`<task>` is the task id with `__` for `/`. `result.json` is the benchmark's result and the one to report; `result_recomputed.json` is there because the log p values stored with the dataset come from another runtime than this code's ([docs/the_package_and_the_paper.md](../docs/the_package_and_the_paper.md)). A run that produces no sampler has the fallback's scores. [`posteriorbench_methods/common/README.md`](posteriorbench_methods/common/README.md) describes every file.

## From records to a point of the paper's comparison

A point of the paper's figure of posterior fidelity against construction cost is one method at one setting, averaged over three replicates within a task and then over the 21 tasks. Run the method on every task and replicate into one output root, then average the records with the package:

```sh
for task in $(uv run posteriorbench list-tasks | cut -f1); do
    for replicate in 0 1 2; do
        uv run posteriorbench-methods sft run --task "$task" --replicate "$replicate" \
            --out runs --set budgets=131072
    done
done
uv run posteriorbench aggregate runs/sft/B=131072/*/rep*/result.json \
    --expect-tasks benchmark --expect-replicates 0,1,2
```

[`docs/paper_map.md`](docs/paper_map.md) lists the command behind each figure and table of the paper.

## Choosing a checkpoint again

A method that trains submits one of its checkpoints by the submission rule. Two comparisons in the paper's appendix choose again from the same training runs, without training again: with a hit-rate threshold of 1% in place of 10%, and by each training method's own checkpoint rule. `reselect` does both, in every family that trains (`sft`, `gfn`, `ctl`, `dpg`):

```sh
uv run posteriorbench-methods ctl run --task wc_1853/goblin_mention --out runs \
    --set budgets=1048576 --set update_caps= --keep-checkpoints all
uv run posteriorbench-methods ctl reselect --task wc_1853/goblin_mention --out runs \
    --set budgets=1048576 --set update_caps=
uv run posteriorbench-methods ctl reselect --rule final-checkpoint \
    --task wc_1853/goblin_mention --out runs --set budgets=1048576 --set update_caps=
```

| Method | `--rule` | The checkpoint that is submitted |
|---|---|---|
| every training method | `threshold-1pct`, the default | the one that the submission rule chooses with a 1% threshold |
| GFlowNet | `highest-mean-log-reward` | the one whose selection draws have the highest mean log reward |
| CTL, KL-DPG, JS-DPG | `final-checkpoint` | the last one |
| GDC | `last-sampling-distribution` | the last sampling distribution r |
| GDC | `final-policy` | the policy at the end |

`reselect` trains nothing. It needs every checkpoint file of the training run, so `run` takes `--keep-checkpoints all`, and the same method settings as `run`; only the budgets and update caps may differ. Its records go to `<out>/reselected/<rule>/`, which has the layout of an output root, so `aggregate` reads them in the same way. [`posteriorbench_methods/common/README.md`](posteriorbench_methods/common/README.md#choosing-a-checkpoint-again) has the details, among them what a result of `reselect` costs.

## Analyses of finished runs

Three analyses of the paper's appendix read the records under an output root and run no method. Each prints JSON.

```sh
uv run posteriorbench-methods compute-per-inference runs   # FLOPs and GPU time for each counted inference
uv run posteriorbench-methods cap-sensitivity runs          # the capped forward KL at the caps 10, 20, 50, 100 and 200
uv run posteriorbench-methods estimate-ps runs              # the error of each sampler's importance-sampling estimate of p(S)
```

[`posteriorbench_methods/analysis/README.md`](posteriorbench_methods/analysis/README.md) says what each analysis counts and what its output holds.

## The paper's figures and tables

[`figures/`](figures) holds the numbers that the paper plots and prints, with one script for each figure and table. A script also takes a table made from your own records:

```sh
uv run python figures/main_comparison.py --out main_comparison.pdf
uv run python figures/records_to_table.py runs --out my_runs.csv
uv run python figures/main_comparison.py --runs my_runs.csv --out my_comparison.pdf
```

[`figures/README.md`](figures/README.md) lists the paper's figures and tables with their data files and commands, and shows how to draw your runs next to the paper's.

## Larger models and LLM-judged behaviors

The paper's section "Generalization to larger models and complicated behaviors" has six tasks that are not benchmark tasks: two behaviors on each of Qwen3.8-27B and Gemma-4-31B-it, and two behaviors of Qwen3.5-0.8B that a language model judges, on responses of up to 2,048 tokens. Prompt selection, Steering and SFT run on them with the same commands, given the task's id and the data of these tasks:

```sh
uv run posteriorbench-methods generalization list-tasks
uv run posteriorbench-methods steering run --task gemma-4-31b-it/wc_0670/em_dash \
    --data Transluce/PosteriorBench-paper-data --out runs --method-data method_data/steering_vectors/gemma-4-31b-it
```

[`posteriorbench_methods/generalization/README.md`](posteriorbench_methods/generalization/README.md) gives the command behind every curve of the section's figures, what each task needs, and what a run costs in judge calls. The data and the method data of these tasks are in the dataset [`Transluce/PosteriorBench-paper-data`](https://huggingface.co/datasets/Transluce/PosteriorBench-paper-data) on the Hugging Face Hub: `--data` takes the dataset's id or the directory of a local copy, and [Method data](#method-data) shows how to download `method_data`.

Please report results on these tasks apart from benchmark results: they have one prompt, fewer responses behind p̂(S), and for two of them labels that a language model samples.

## Method data

Prompt selection and Steering read inputs that are not part of the benchmark's dataset. They are in the directory `method_data` of the dataset [`Transluce/PosteriorBench-paper-data`](https://huggingface.co/datasets/Transluce/PosteriorBench-paper-data):

```sh
uv run hf download Transluce/PosteriorBench-paper-data --repo-type dataset --include "method_data/*" --local-dir .
```

| Method data | Read by | Contents |
|---|---|---|
| `candidate_prompts_108tasks.parquet` | `prompt run --method-data <file>` | 1,024 to 1,050 candidate system prompts for each of the 108 scored prompt–behavior pairs |
| `candidate_prompts_21tasks_extended_131072.parquet` | the same, for pool sizes above 1,024 | 131,072 candidates for each of the 21 tasks, the first of them those of the file above |
| `steering_vectors/benchmark/` | `steering run --method-data <directory>` | the steering vectors of the seven behaviors and what they cost to build; `steering find-refusal-direction` and `steering build-vectors` write such a directory |
| `construction_prompts/<behavior>.json` | `steering build-vectors --prompts <file>` | the instruction pairs and questions of the paper's vectors, for the seven behaviors |

DeepSeek-V4-Flash wrote the candidate prompts, and the steering vectors are persona vectors of each behavior. You can make both yourself: `prompt write-candidates` writes a pool file for any task with a writer LLM of your choice, and `steering find-refusal-direction` and `steering build-vectors` build vectors from a file of construction prompts. The READMEs of [`prompt`](posteriorbench_methods/prompt) and [`steering`](posteriorbench_methods/steering) give the commands and the format of the files.

## Adding a method

[`CONTRIBUTING.md`](CONTRIBUTING.md) describes what a family provides and how to add one.

## Citation and license

To cite the paper, use the entry in the [README of the repository](../README.md#citation).

The code is released under the MIT license ([LICENSE](LICENSE)). [`ATTRIBUTION.md`](ATTRIBUTION.md) names, for each method, the paper it comes from and what this repository takes from the authors' code.
