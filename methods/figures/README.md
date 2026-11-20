# The paper's figures and tables

This folder holds the numbers behind the figures and tables of the PosteriorBench paper, and one script for each that draws the figure or prints the table. The numbers are the paper's results, so you can redraw a figure, average the runs in another way, or put your own method next to the paper's, without running anything on a GPU. Each script also takes a table made from runs of your own.

**Content warning.** `data/sample_responses.jsonl` holds model responses with offensive language. It holds no response of the two behaviors that a language model judges.

## Run a script

The scripts need matplotlib, which `uv sync` installs. From `methods/`:

```sh
uv run python figures/main_comparison.py --out main_comparison.pdf
```

draws the paper's comparison of posterior fidelity against construction cost from `data/main_comparison_runs.csv.gz`. Every figure script takes `--out` (a `.pdf` or a `.png`), and `--help` lists its options.

## The paper's figures and tables

A figure or table is named by its number in the paper. Paths are relative to this folder.

| Paper item | Shows | Data | Command |
|---|---|---|---|
| Figure 3 | Event probabilities p(S) of the 108 prompt–behavior pairs with at least 500 hits | the package's task table | `event_probabilities.py` |
| Figure 5 | Posterior fidelity against construction cost on 21 tasks | `main_comparison_runs.csv.gz` | `main_comparison.py` |
| Figure 6 | Random responses on one profanity task from the true posterior and three samplers | `sample_responses.jsonl`, `main_comparison_runs.csv.gz` | `sample_responses.py --figure 6` |
| Figure 7 | Prompt selection, Instruction and Steering, with and without composition, on three models | `main_comparison_runs.csv.gz`, `larger_models_runs.csv` | `larger_models.py` |
| Figure 8 | Posterior fidelity against construction cost on the two LLM-judged behaviors | `llm_judged_runs.csv` | `llm_judged_behaviors.py` |
| Figure 11 | Training methods at about 5 × 10^5 inferences on the 21 tasks, by event probability | `main_comparison_runs.csv.gz` | `training_by_probability.py` |
| Figure 12 | Prompt selection with four selection signals on the 108 prompt–behavior pairs | `training_free_108pairs_runs.csv.gz` | `selection_signals.py` |
| Figure 13 | Prompt mixtures on the 108 prompt–behavior pairs | the same | `prompt_mixtures.py` |
| Figure 14 | The training-free methods on the 108 pairs, by event probability | the same | `training_free_by_probability.py` |
| Figure 15 | The error of each sampler's estimate of p(S) on the 21 tasks | `estimating_ps_runs.csv.gz`, `estimating_ps_samplers.csv.gz` | `estimating_ps.py` |
| Figure 16 | The two LLM-judged behaviors with a 1% hit-rate threshold | `llm_judged_runs.csv` | `llm_judged_behaviors.py --threshold 0.01` |
| Figures 17 to 21 | Sample responses on five tasks, one per sampler | `sample_responses.jsonl`, `main_comparison_runs.csv.gz` | `sample_responses.py --figure 17` (to `21`) |
| Table 2 | The 21 tasks, with p̂(S) from the corpus | the package's task table | `table_tasks.py` |
| Table 3 | Rules on the decoded response, and the number of pairs of each behavior | the package's task table | `table_rules.py` |
| Table 4 | Compute per counted inference | `compute_per_inference.csv` | `table_compute_per_inference.py` |
| Table 6 | Instruction texts | the package and `posteriorbench_methods/generalization` | `table_instructions.py` |
| Table 7 | The submission rule against each method's own checkpoint rule | `checkpoint_rules_cells.csv`; `main_comparison_runs.csv.gz`, `reselected_runs.csv` | `table_checkpoint_rules.py` |
| Table 8 | Training methods with the 10% threshold and with a 1% threshold | `threshold_1pct_cells.csv`; `main_comparison_runs.csv.gz`, `reselected_runs.csv` | `table_threshold_1pct.py` |
| Table 9 | The tasks on larger models and LLM-judged behaviors | `posteriorbench_methods/generalization` | `table_generalization_tasks.py` |
| Table 10 | System prompts that three selection signals return from the same 1,024 candidates | `selected_prompts.jsonl` | `selected_prompts.py` |

Tables 1 and 5 hold descriptions and formulas, not results. Figures 1, 2, 4, 9 and 10 are drawings in the paper's source and have no data.

`sample_responses.py --figure` takes the number of a figure of sample responses, 6 or 17 to 21, as `sample_responses.jsonl` gives it in its key `figure`. Without the option the script prints all six.

## The data

A table of runs has one row per run: one method at one setting on one task and replicate. The columns have the names of a record's `result.json`:

| Column | Meaning |
|---|---|
| `method`, `setting` | as in a record: `sft` at `B=524288`, `prompt-ebjc` at `K=1024`, `gfn` at `u=8` (a record of GFlowNet says `gflownet`), `steering` at `none`; a composed method ends in `+composition` |
| `task_id`, `replicate` | the task id of the package, and the replicate (0, 1 or 2) |
| `construction_inferences` | what the method spent to build the sampler |
| `capped_forward_kl`, `reverse_kl` | the two scores, in nats; the fallback's scores for a run scored as the fallback |
| `native_hit_rate` | the share of hits among the evaluation draws |
| `fallback` | `true` if the run was scored as the fallback |

| File | Rows | What it holds |
|---|---:|---|
| `main_comparison_runs.csv.gz` | 13,482 | every run of the main comparison: 214 points (a method at a setting) × 21 tasks × 3 replicates |
| `training_free_108pairs_runs.csv.gz` | 18,252 | the 108 pairs: prompt selection with EB-JC, JC, hit rate and IWAE at K = 1 to 1,024, Instruction with and without composition, and Steering, three replicates each; and seven prompt mixtures at K = 16, 64, 256 and 1,024, one replicate |
| `estimating_ps_runs.csv.gz` | 12,789 | every run of the main comparison except Replay's, with the absolute log error of its estimate of p(S) and whether the estimate is the run's own or the fallback's |
| `estimating_ps_samplers.csv.gz` | 4,968 | every sampler that passed the hit-rate threshold: its reverse KL and its signed log error |
| `larger_models_runs.csv` | 840 | the two tasks of Qwen3.8-27B and of Gemma-4-31B-it, three replicates each: prompt selection with EB-JC, hit rate and IWAE at K = 1 to 1,024, Instruction and Steering, and the composition of each |
| `llm_judged_runs.csv` | 276 | the runs on the two LLM-judged tasks, with the 10% threshold (150) and with the 1% threshold (126); the column `threshold` says which |
| `reselected_runs.csv` | 474 | runs whose checkpoint was chosen by another rule, named in the column `rule`: the six training methods with the 1% threshold (`threshold-1pct`, 63 runs each), and on 12 tasks at replicate 0 each method's own rule and, for KL-DPG and JS-DPG, the submission rule on the same trainings (`submission-rule`) |

The other tables hold the plotted points of a figure (a mean and its standard error, as `<column>` and `<column>_se`) or the cells of a table, with all their digits:

| File | Rows | What it holds |
|---|---:|---|
| `main_comparison_points.csv` | 214 | the points of Figure 5 |
| `training_by_probability_points.csv` | 24 | the points of Figure 11: six methods × four bins of p(S) |
| `selection_signals_points.csv` | 44 | the points of Figure 12 |
| `prompt_mixtures_points.csv` | 28 | the points of Figure 13 |
| `training_free_by_probability_points.csv` | 20 | the points of Figure 14 |
| `larger_models_points.csv` | 144 | the points of Figure 7, 48 per model |
| `llm_judged_points.csv` | 92 | the points of Figure 8 (50) and Figure 16 (42) |
| `estimating_ps_points.csv` | 203 | the points of the left panel of Figure 15 |
| `compute_per_inference.csv` | 6 | the cells of Table 4, each with the setting of the point where it is highest |
| `checkpoint_rules_cells.csv` | 6 | the cells of Table 7 |
| `threshold_1pct_cells.csv` | 12 | the cells of Table 8 |

For every figure and table above, a test recomputes these points and cells from the runs and compares them with the stored ones (`tests/figures`). The right panel of Figure 15 is drawn from its stored points.

Three things to know about the tables of runs:

- The 108-pair table has no `construction_inferences`: the paper reports no cost on the 108 pairs.
- `estimating_ps_samplers.csv.gz` names the task and the method of a sampler, not its setting and replicate.
- In Table 7, the own-rule runs of KL-DPG and JS-DPG come from separate trainings with the settings of the main comparison. Their rows of the submission rule in `reselected_runs.csv` are from those trainings, and differ from the main comparison's runs of the same tasks.

In the main comparison, GDC has points at the budgets 2^12 and 2^15 to 2^20. A mixture of one prompt (`mixture-uniform1`) is the prompt that EB-JC selection submits, and its runs repeat those of `prompt-ebjc` at replicate 0.

## Your own runs

`records_to_table.py` writes a table of runs from the records of an output root, and the scripts read it with `--runs`:

```sh
uv run python figures/records_to_table.py runs --out my_runs.csv
uv run python figures/main_comparison.py --runs my_runs.csv --out my_comparison.pdf
```

A point is averaged as `posteriorbench aggregate` averages: the replicates of a task first, then the tasks with equal weights, with the standard error across tasks. On the two LLM-judged tasks a point is one task, and its standard error is across the three replicates. A script draws those of its figure's methods and settings that the table holds, and says so in one line if the table holds none of them. `main_comparison.py` draws every method; a method that the paper does not have is drawn in a colour of its own, with its name as its label.

The axes are the paper's, and the paper's points are averages over many tasks. A run on one task can score outside them: `main_comparison.py` and `estimating_ps.py` widen a logarithmic y axis for a value below or above the paper's range and name on stderr the points that such an axis cannot show (a score that is not positive, a cost outside the cost axis), and the other figure scripts widen the range of a panel that a value lies outside. `llm_judged_behaviors.py` marks a value above a panel with an arrow, as the paper's figure does.

To draw a sampler that you evaluated with the package next to the paper's methods, give the results that your loop wrote (the package's README, "Running the benchmark"):

```sh
uv run python figures/records_to_table.py ../my_method.jsonl --out my_method.csv
uv run python figures/main_comparison.py --add my_method.csv --out with_my_method.pdf
```

`records_to_table.py` reads a file that ends in `.jsonl` as one result per line, as `posteriorbench aggregate` does. `--add` draws the paper's runs and the table's; a method of the table that the paper's runs also have is drawn as `<method> (added)`.

The two tables that compare rules read the column `rule` of the table, which `records_to_table.py` fills for the records under `<out>/reselected/`:

```sh
uv run python figures/table_threshold_1pct.py --runs my_runs.csv
uv run python figures/table_checkpoint_rules.py --runs my_runs.csv
```

Two scripts read the output of an analysis command in place of a table of runs:

```sh
uv run posteriorbench-methods estimate-ps runs --runs --samplers --out ps.json
uv run python figures/estimating_ps.py --analysis ps.json
uv run posteriorbench-methods compute-per-inference runs --out compute.json
uv run python figures/table_compute_per_inference.py --analysis compute.json
```

## Sample responses

`sample_responses.jsonl` holds the 60 responses that the paper shows, each with the number of its figure (the key `figure`), task, method, setting and replicate; a posterior reference has the method `reference`. The scores printed next to a response come from the runs of the main comparison. A response is whole, up to the 50-token limit, as the model wrote it; the paper's figures also remove Markdown and replace characters that its fonts cannot print.

Each response is the first hit of its sampler's 1,024 evaluation draws in a fixed random order, skipping responses with sexual content or racial stereotypes. To read hits of a record of your own in the same way:

```sh
uv run python figures/sample_responses.py --record runs/sft/B=131072/wc_0346__em_dash/rep0 --count 5
```

This reads the record's `draws/evaluation.jsonl` and decodes the token ids of its hits with the tokenizer of the benchmark's model. The prompt that prompt selection submitted is `submitted.system_prompt` in the record's `construction.json`.

## Files

| File | What it holds |
|---|---|
| `paper_runs.py` | reading and writing a table, averaging over replicates and tasks, the bins of p(S) |
| `paper_style.py` | the colour, marker and label of each method, and the parts that several figures share |
| `records_to_table.py` | from records, or from results of the package, to a table of runs |
| one script per figure and table | as in the table above |
| `data/` | the paper's numbers |
