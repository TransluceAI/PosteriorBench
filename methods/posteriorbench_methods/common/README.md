# common

The code that the method families share. A family's own folder holds its method: how it builds its candidate samplers. What every method does the same way is here: the options of a run, the seeds, the records, the charges, the submission rule, composition, generation with a model, and the files of a training run.

## What a family provides

```text
posteriorbench_methods/<family>/
    settings.py    Settings, a frozen dataclass whose defaults are the settings of the paper's runs
    run.py         run(context, settings) -> list[run_record.RunSummary]
    cost.py        the method's charges, as lists of charges.ChargeSpec
    cli.py         app = cli_options.family_app(...)
    reselect.py    in a family that trains: the rules that choose a checkpoint again
```

`run` receives a `run_context.RunContext`: the benchmark, the task, the original model, the replicate and the run options. It builds the method's candidates, chooses one by the submission rule, and writes one record for each method and setting it was asked for (`context.evaluate_and_record`, `context.record_no_sampler`, `context.record_reused`, and `composition.compose` for the composed method). `cli_options.family_app` turns this into the commands `run` and `config`, with the same options in every family. A family that trains also has `reselect.py`, the rules by which the checkpoint of a finished training run is chosen again, and with it the command `reselect`. `methods/CONTRIBUTING.md` says how to add a family.

## Modules

| Module | What it holds |
|---|---|
| `run_context.py` | `RunOptions`, the options of every `run` command; `RunContext`, what `run` receives; `MethodEntry`, one method of a family; opening a run on a task of the package or of the generalization section |
| `cli_options.py` | `family_app`, the commands `run`, `config` and `reselect` of a family; method settings from `--config FILE` and `--set NAME=VALUE` |
| `errors.py` | `MethodsError`, the base class of the errors that a command reports as one line that begins with `Error:`: a mistake that the user can repair, such as an unknown task, method data that is not given, or a training run on disk that was started with other settings |
| `data_source.py` | `reading`, which reports a dataset on the Hub that is not in the local cache while the Hub cannot be reached as such an error |
| `seeds.py` | `stream_seed(owner, task_id, replicate, stream, *index)`, the seed of one random stream of a run; `substream_seed` for the parts of a stream |
| `run_record.py` | where a run writes and what the files hold: `<out>/<method>/<setting>/<task>/rep<r>/`, strict JSON, `summary.jsonl` |
| `charges.py` | `ChargeSpec`, `apply`, and the charges every family makes: selection draws, threshold tests, the selection draws of composition |
| `signals.py` | the selection signals JC and EB-JC, from selection draws (`SelectionDraws`) |
| `submission.py` | the submission rule: `Rule`, `rank_candidates`, `rank_from_selection_draws`, `submit`; `at_one_percent`, a rule at the 1% threshold |
| `rank_table.py` | `threshold_test(rank)`, the threshold test at any rank; the tables in `data/` and how they are computed |
| `composition.py` | composition with the original model: the 16 settings, `choose_setting`, its charges, `should_compose`; `compose`, the composition step of a run |
| `hf_runtime.py` | generation, log q of given responses, and composed sampling with a transformers model; `ModelSampler` |
| `training_run.py` | a training run on disk: checkpoints, their selection draws, the candidates at each budget, kept files, continuing |
| `reselection.py` | choosing the checkpoint of a finished training run again: the submission rule at a 1% threshold, reading a training run without training, the records of a rule |
| `tracking.py` | progress as JSON log lines, with the same names in every family (`update`, `checkpoint`, `record`, `budget_spent`), and as a wandb run with `--wandb-project` |
| `data/rank_table.json` | the threshold tests at ranks 1 to 1,024 for the 10% threshold |
| `data/rank_table_1pct.json` | the threshold tests at ranks 1 to 16 for a 1% threshold |

## The submission rule

A method that has several candidate samplers submits one of them by the paper's submission rule, in `submission.py`:

1. Each candidate has selection draws, m responses with h hits among them.
2. `rank_from_selection_draws` ranks the candidates. A candidate whose hit fraction h / m is above 15% is ranked by the selection signal (EB-JC); the others follow in decreasing order of hits. `rank_candidates` does the same for a method that computes its scores itself.
3. `submit` tests the candidates against the hit-rate threshold in rank order and submits the first that passes. The threshold test at rank j draws n_j new responses and passes with at least k_j hits: a one-sided binomial test of q(S) <= 10% at level 0.05 / (j (j + 1)), with power 0.95 at q(S) = 20% (rank 1: 25 hits among 164 draws).
4. If no candidate passes, the tested candidate with the highest hit fraction in its test is submitted, the best-observed candidate.

A method states its settings of the rule once, as a `submission.Rule`. The families differ in three of them: how many candidates are tested at most, how ties are ordered, and what happens to candidates that are one sampler (checkpoints with the same weights, prompts that render to the same tokens).

| | Tests at most | A tie goes to | Candidates that are one sampler |
|---|---|---|---|
| SFT, GFlowNet, KL-DPG, JS-DPG, GDC | 8 | equal scores: more selection hits; equal hits: the higher JC; then the earlier candidate | are ranked once; the one that JC ranks first stands for them in the EB-JC fit |
| CTL | 8 | the same | are ranked once; the earliest stands for them in the EB-JC fit |
| prompt selection and mixtures | all; a mixture stops when s candidates have passed | the earlier candidate of the pool | are ranked once, at the best of their ranks; each is in the EB-JC fit with its own selection draws |
| Steering | all; a candidate without a selection hit is not ranked | the earlier candidate | do not occur |
| composition settings | all | the earlier setting of the grid | do not occur |

`posteriorbench-methods threshold-test --rank j` prints the test at rank j.

**A 1% threshold.** The paper's appendix repeats its comparison on the two LLM-judged tasks with every hit-rate threshold divided by ten. `submission.at_one_percent` gives a family's rule at that threshold: a candidate is ranked by the signal if its hit fraction is above 1.5%, the tests are those of `data/rank_table_1pct.json` (rank 1: 28 hits among 1,857 draws), and the family's other settings stay as they are. A rule that otherwise tests until a candidate passes, as those of prompt selection, Steering and composition do, tests at most 16 candidates at 1%, which is the length of that table. A method that does not train takes the threshold as its setting `threshold` and asks the run for its rule (`context.at_threshold`, `context.rule_at_threshold`); a method that trains chooses its checkpoint again at 1% with `reselect`. Both write their records under `<out>/reselected/threshold-1pct/`.

## Tasks beyond the benchmark

A run is opened on a task of the package, or on one of the six tasks of [`generalization`](../generalization), which have another original model, a response limit of up to 2,048 tokens, or labels from a language model. `MethodEntry(..., generalization_tasks=True)` says that a method runs on them; `prompt`, `steering` and `sft` do, as in the paper. Such a method asks the run for what depends on the task:

- `context.label(responses)` says which responses are hits: by the task's rule, or by the two judge models of a task that a language model judges. A method does not call the rule itself.
- `context.scoring_batch(n)` and `context.generation_batch(n)` give the number of responses in one forward pass: the family's own number n on a benchmark task, and at most the limit of the task's original model on these tasks.
- `context.original.max_response_tokens` is the task's response limit. Composition takes its last L from it.

On a benchmark task these label by the task's rule and return the family's own numbers. The options that only these tasks read, the judge models and the two batch sizes, are refused on a task of the package.

## Composition

Under a composition setting (alpha, L), each of the first L tokens of a response is drawn from (1 - alpha) q + alpha p, where q is the sampler and p the original model, and p writes the rest. `composition.choose_setting` chooses among the 16 settings: it gives each of them 16 selection draws, ranks the settings by EB-JC and tests them in rank order, all by the submission rule. A draw of a composed setting runs the sampler and the original model and costs two inferences.

`composition.compose` is this step for a run. A family gives it the method's record and three functions, the selection draws, the threshold-test draws and the sampler of a setting; it charges composition on top of the method's cost and writes the record of `<method>+composition`. A sampler that did not pass its own threshold test, or that is the original model, is not composed, and its record then repeats the method's evaluation with the status `not_composed`. So does a record whose chosen setting is the plain one, alpha = 0 with L at the response limit, at the cost with composition.

## Training runs

A method that trains saves checkpoints while it trains, each with its selection draws, in the directory of its training run (`training_run.py`). The run at a budget submits one of the checkpoints that the training had saved within that budget, so one training run gives the method's results at several budgets. A later run with a larger budget continues the same training run.

## Choosing a checkpoint again

`reselect` takes a training run that `run` has finished and chooses among the same candidates by another rule, without training (`reselection.py`). A family states its rules by name in `MethodEntry.reselect`, each with a function that has the signature of `run`; the family's `reselect.py` holds them.

| Rule | What is submitted | Threshold tests | Threshold of the evaluation |
|---|---|---|---|
| `threshold-1pct`, in every family that trains | the checkpoint that the submission rule chooses when a candidate whose hit fraction is above 1.5% is ranked by the signal | at most 8, at the 1% threshold, from random streams of their own | 1% |
| a method's own rule: `highest-mean-log-reward` (GFlowNet), `final-checkpoint` (CTL, KL-DPG, JS-DPG), `last-sampling-distribution` and `final-policy` (GDC) | the checkpoint that the rule names | none | the benchmark's 10% |

A family's settings of the submission rule stay as they are at 1% (`submission.at_one_percent`): how many candidates are tested, how ties are ordered, and how candidates that are one sampler are treated. The candidates at a budget or an update cap are those of `run`, and a result costs what the training had spent when it saved the last of them, plus its threshold tests. The tests at 1% draw more than the 1,912 responses that a budget sets aside (the eight of them up to 21,946), so this cost is not held to the budget.

`reselect` reads the training run that `run` left under `--out`, and it needs three things from that run. The checkpoint files must all be there, so `run` takes `--keep-checkpoints all`. The method's settings (`--set`, `--config`) must be those of `run`, since they name the training run; only the budgets and update caps may differ, which say where to choose. And the training must have gone as far as these budgets and update caps need. `reselect` then chooses among the candidates that `run` had at each budget or update cap, from the selection draws that the training run stores. It trains nothing, composes nothing and leaves the records of `run` as they are.

The records go to `<out>/reselected/<rule>/`, which has the layout of an output root, so they are averaged in the same way (`uv run posteriorbench aggregate runs/reselected/threshold-1pct/ctl/B=1048576/*/rep*/result.json`). A record at the 1% threshold has `threshold` in its `details`, before the entries of the family's record of `run`. A record of an own rule has `own_rule` there, the status `own_rule`, and no ranking.

## Records

A record is one method at one setting on one task and replicate, `<out>/<method>/<setting>/<task>/rep<r>/` (`run_record.py`). Every family writes the same files:

| File | What it holds |
|---|---|
| `result.json` | the package's result, with the log p of the references that the dataset ships: the two KL scores, the hit rate, `construction_inferences` |
| `result_recomputed.json` | the same draws, with log p of the references recomputed by the original model; absent for a run without a sampler |
| `evaluation_details.json` | the per-response values of both results; absent for a run without a sampler |
| `evaluation_hits.json` | `draw_index`, `logq` and `logp` of every hit among the evaluation draws. The evaluation computes log q and log p for at most 128 hits of a sampler that passed; the others are scored when the record is written (`run_record.score_every_hit`), which draws nothing and is not charged. Absent for a run without a sampler |
| `construction.json` | `method`, `setting`, `task_id`, `replicate`; `status`: `passed`, `best_observed` or `no_candidate` from the submission rule, or `no_selection`, `no_sampler`, `not_composed`, `own_rule`; `submitted`, the submitted candidate with its threshold test; `ranking`; `submission`, every threshold test; `ledger`, what the method spent; `details`, the family's own entries |
| `config.json` | `family`, `method`, `setting`, `task_id`, `replicate`, `settings`, `options`, `versions`, `scorer` (the runtime of the original model) |
| `draws/<name>.jsonl` | one row per response, always with `token_ids`: `selection` and `threshold_tests` in the record of a method that ranks candidates, `composition_selection` and `composition_threshold_tests` in a composed record, the family's own files, and `evaluation` |

The rows of the four draws files of the submission rule have the same keys in every family (`run_record.DRAW_KEYS`). A row of `selection` has `candidate_id`, `draw_index`, `token_ids`, `hit`, `logq`, `logp` and `seed`; `logq` and `logp` are null where the method did not compute them, such as log p of a miss. A row of `threshold_tests` has `candidate_id`, `rank`, `draw_index`, `token_ids`, `hit` and `seed`. The two files of a composed record have `setting` in place of `candidate_id`. A family may add keys of its own, which its README names; GFlowNet and CTL add the response `text`.

A checkpoint's id is `update_000128` for the checkpoint saved after 128 updates, in every family. GDC has two kinds of checkpoint, its sampling distribution r and its policy, and writes `r_update_000010` and `pi_update_000010`. In a ranking and in the draws files, a checkpoint's `candidate_id` is this id. `details.candidates` holds one object for each candidate that a record chose from: the rows of the training run's index for a method that trains, and the vector, layer, strength and selection hits of each candidate for Steering. A pool of prompt selection is not listed there; its candidates are those of `draws/selection.jsonl`.

A composed record's `submitted` holds the chosen `setting`, and its `details` hold `plain` (the status of the method's submission and what the method submitted) and `composition` (the settings, their ranking and tests and the chosen setting, or the `reason` why the sampler was not composed). `result.json` is written last, so a record that has one is complete, and `<out>/summary.jsonl` gets one line for each record written.

## Randomness

Every random stream of a run has a name, and its seed is computed from the method, the task, the replicate and that name (`seeds.stream_seed`), so a method runs on any task and two runs of a command have the same seeds.

Whether two such runs give the same records depends on the method. The training of GFlowNet and of CTL does not repeat exactly on a GPU: from the first update that changes the weights, the weights differ in their last bits from run to run. The draws can then differ from run to run (their READMEs). GFlowNet + composition and CTL + composition take their seeds from the weights of the submitted checkpoint, so their selection draws, and the setting they choose, differ from run to run.
