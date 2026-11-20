# SFT, Replay and SFT + composition

SFT collects posterior samples by rejection sampling from the original model and fine-tunes a copy of the model on them by maximum likelihood. It makes all of its draws before it trains, so a run first divides its budget between draws and training. It then trains the last layers of the model on the hits among its draws, saves checkpoints while it trains, and submits one of them by the submission rule. Replay is an ablation that shows what training adds beyond the samples themselves: it returns one of SFT's training responses uniformly at random. SFT + composition composes the checkpoint that SFT submits with the original model.

SFT trains separately at each budget. At a budget of B passes, a run does the following.

1. It divides the budget (`budget_rule.py`). It sets 1,912 passes aside for its threshold tests, and the training share of the rest pays for training at two passes per response, forward and backward; training always covers at least one epoch over the hits. It also sets passes aside for the selection draws of its checkpoints. What is left pays for draws from the original model, at one pass each.
2. It takes its draws from the corpus, which is a sample of the original model (`draws_from_p.py`): each draw is a hit with probability p̂(S), and the hits are corpus hits of the task. A run with n draws therefore trains on k ~ Binomial(n, p̂(S)) hits. The rule of step 1 sees where the run's hits are among its draws, never p̂(S).
3. It trains on the k hits in batches of min(k, 32) with AdamW (`train.py`). It saves a checkpoint after 32, 64, 128, ... training responses and after the last one (a run with fewer than 32 hits also saves one after its first epoch), and gives each checkpoint 128 selection draws.
4. It ranks the checkpoints by EB-JC, tests at most the first 8 in rank order and submits one (`selection.py`). This is the record of `sft`.
5. Replay is evaluated on the same k hits, at the cost of the n draws.
6. If the submitted checkpoint passed its threshold test, the submission rule chooses one of the 16 composition settings for it (`common/composition.py`). This is the record of `sft+composition`.

A budget that does not pay for a single hit gives no sampler: the three methods are then recorded as the fallback, at a cost of B.

## Run

```sh
uv run posteriorbench-methods sft run --task wc_1853/goblin_mention --out runs --replicate 0
```

This runs the 11 budgets one after the other; `--set budgets=1048576` runs one. `--no-compose` leaves out SFT + composition. The run reads the task's hit file from the dataset (`--data`); it needs no method data. The methods README lists the options that all families share and says how settings are changed.

A run that is interrupted while it trains continues after its last checkpoint when the same command is run again. A budget whose training is complete is not trained again, and its checkpoint is submitted and evaluated again from the files on disk.

`reselect` needs every checkpoint file: make the run above with `--keep-checkpoints all`.

```sh
uv run posteriorbench-methods sft reselect --task wc_1853/goblin_mention --out runs \
    --set budgets=524288
```

## Training on a whole corpus

On the paper's two LLM-judged tasks, whose responses have up to 2,048 tokens, SFT does not decide how many draws to make: it takes the whole corpus of the task, 3.4 million responses, and trains on every hit in it that is not a reference.

```sh
uv run posteriorbench-methods sft run --task qwen3.5-0.8b/wc_0670/physical_embodiment \
    --data Transluce/PosteriorBench-paper-data --out runs --set training_set=corpus --no-compose \
    --judge-url http://localhost:8000/v1 --judge-model google/gemma-4-31B-it \
    --review-url https://openrouter.ai/api/v1 --review-model openai/gpt-6-sol \
    --review-key-env REVIEW_KEY --labels labels.jsonl
```

With `training_set=corpus` a run has one budget and does not read `budgets` (`budget_rule.divide_corpus`):

1. Its draws are the D responses of the corpus, and its training set the k corpus hits that a method may take. They are the same in every replicate; the replicates differ in the order of training and in their draws from the checkpoints.
2. The draws are the share 1 - s of a budget B = D / (1 - s), and the training share s pays for training at two passes per response: s B / 2 responses, read as E whole epochs over the k hits, with E the nearest whole number. With D = 3.4 million and s = 1/4, B is 4,533,334 and the training share pays for 566,667 responses: 814 epochs over the 696 hits of direct encouragement (566,544 responses), and 198 over the 2,860 of physical embodiment (566,280).
3. It saves a checkpoint after the first epoch, after the whole number of epochs nearest to 32, 64, 128, ... training responses, and after the last epoch: after epochs 1, 3, 6, 12, 24, 47, 94, 188, 377, 753 and 814 on direct encouragement.
4. It submits one checkpoint by the submission rule, as at any budget. The records have the setting `B=4533334`. The run also writes Replay's record on the same hits. The paper reports SFT alone on these tasks, and the command above leaves composition out.

The budget names the run and is not a limit: nothing is set aside for selection draws and tests, and the run is charged what it uses. The corpus costs what the benchmark charges for corpus hits, 1 / p̂(S) draws for each hit taken; the task's references are not among them, so this is less than D (3,113,684 for the 696 hits of direct encouragement).

A batch of 32 responses of 2,048 tokens does not fit in one forward pass. An update passes its batch through the model in parts of at most `update_tokens` tokens, counted as the responses in the part times the longest of them with its prompt, and adds the gradients of the parts, so the update is that of the whole batch. The default, 11,104 tokens, holds 5 responses of 2,048 tokens, and the benchmark's batches are far below it. The task's other forward passes are bounded by the run ([`generalization`](../generalization), "The size of a forward pass").

`sft reselect --set training_set=corpus`, with the options of the run, chooses the checkpoint of this training run again at the 1% threshold.

## Choosing a checkpoint again

`reselect` chooses the checkpoint of finished runs again with a hit-rate threshold of 1% in place of 10%: in the hit fraction above which a checkpoint is ranked by EB-JC (1.5%), in the threshold tests and in the evaluation (the methods README, "Choosing a checkpoint again"). SFT trains separately at each budget, so `reselect` reads the training run of each of its `budgets` under `--out`, which `run` must have written with `--keep-checkpoints all` and the same settings; `--set` and `--config` work for `reselect` as they do for `run`. A budget whose draws held no hit has no checkpoint to choose, and `reselect` stops there with an error. Its cost is the cost of the run until its training ended, plus the tests at 1%. SFT has no checkpoint rule of its own, so the 1% threshold is the only rule of its `reselect`.

## Settings

| Setting | Default (the paper's runs) | Meaning |
|---|---|---|
| `budgets` | 2^12, 2^13, ..., 2^22 | the training budgets B, in passes of a model over one response |
| `training_set` | `draws` | what a run trains on: `draws`, the hits among the draws that its budget pays for; or `corpus`, every hit of the task's corpus that is not a reference ([Training on a whole corpus](#training-on-a-whole-corpus)) |
| `training_share` | 0.25 | share of a budget that pays for training |
| `learning_rate` | 1e-6 | learning rate of AdamW, constant |
| `weight_decay` | 0 | weight decay of AdamW |
| `gradient_clip` | 1 | largest gradient norm of an update |
| `trainable_last_layers` | 4 | number of decoder layers that are trained, counted from the last of the model's 24 |
| `batch_size` | 32 | most training responses in one update |
| `update_tokens` | 11,104 | most tokens in one forward pass of an update; a batch that holds more is computed in parts |
| `selection_draws` | 128 | selection draws of each checkpoint |

## Cost

| Event | Inference count | Budget count |
|---|---|---|
| a draw from the original model | 1 | 1 |
| an update on b training responses | b | 2b |
| the 128 selection draws of a checkpoint, h of them hits | 128 + h | 256 + h |
| a threshold test of n draws | n | n |
| the selection draws of composition, h of them hits | 496 + h | 496 + h |
| a threshold test of a composed setting, n draws | 2n | 2n |

An update is counted for its forward pass, and its backward pass is charged to the budget alone. A checkpoint needs log q of its selection draws from a second pass over them, which is charged to the budget alone, and log p of the hits. A test of the plain composition setting, which is the checkpoint itself, costs n. Replay costs the draws. A run on a whole corpus pays 1 / p̂(S) for each hit that it takes in place of the first row.

Example, at B = 2^20 = 1,048,576. The share for training pays for 130,833 responses (261,666 passes). The rule sets aside 14 x 384 passes for checkpoints and 1,912 for tests, which leaves 779,622 draws. On a task with p̂(S) = 0.001 these hold about 780 hits, so training reads them about 168 times and saves 13 checkpoints. If the checkpoints have 400 selection hits in all and the first test passes, SFT's inference count is 779,622 + 130,833 + 13 x 128 + 400 + 164 = 912,683, and its budget count is 779,622 + 261,666 + 13 x 256 + 400 + 164 = 1,045,180. Replay's inference count is 779,622.

## What a run writes

One record for each method and budget, `<out>/<method>/B=<budget>/<task>/rep<r>/`, with the methods `sft`, `replay` and `sft+composition`. The methods README describes the files of a record. In this family:

- `draws/training_set.jsonl` (`sft`, `replay`): the hits among the run's draws, each with its place among the draws (`draw_index`) and its row in the corpus (`corpus_row`).
- `draws/selection.jsonl` and `draws/threshold_tests.jsonl` (`sft`): the selection draws of every checkpoint, with log q and, on hits, log p, and the draws of every threshold test. `candidate_id` is the checkpoint, `update_000128` for the one saved after 128 updates.
- `draws/composition_selection.jsonl` and `draws/composition_threshold_tests.jsonl` (`sft+composition`): the same for the composition settings, with the setting in `setting`.
- `details` of `construction.json`: `division` (draws, hits, training responses, epochs, batch, the updates with a checkpoint), `trained_parameters` and `candidates` (the checkpoints, as the rows of the training run's index) for `sft`; `composition` (the 16 settings, their ranking and tests, the chosen setting) and `plain` (the status of SFT's submission and its checkpoint) for `sft+composition`. A composed record whose status is `not_composed`, or whose chosen setting is the plain one, holds SFT's evaluation.

The training run of each budget is in `<out>/training_runs/sft/<task>/rep<r>/B=<budget>/`: the checkpoints (the trained layers only), their selection draws and the index of the checkpoints (`common/training_run.py`).

`reselect` writes the record of `sft` at each budget under `<out>/reselected/threshold-1pct/`, which has the layout of an output root. The `details` of its `construction.json` hold `threshold` (0.01), `division` and `candidates`; `draws/selection.jsonl` holds the selection draws of the checkpoints, those of `run`, and `draws/threshold_tests.jsonl` the draws of the tests at 1%.

## Compute and disk

On one H100, a run of the three methods at one budget takes about 15 minutes at B = 2^19 (up to 40 minutes on some tasks), about an hour at 2^21 and about 100 minutes at 2^22. The runs of the paper (21 tasks, 11 budgets, 3 replicates) took about 260 GPU-hours.

A checkpoint holds the last 4 layers, 332 MB. A budget saves up to 16 checkpoints, about 5 GB at 2^22, of which `--keep-checkpoints testable` keeps at most 8 when the budget is done. While it trains, a run also keeps the state from which it can continue, about 1 GB.

## Paper results this produces

With the default settings, `run` on the 21 tasks and the replicates 0, 1 and 2 produces the points of SFT, Replay and SFT + composition in the paper's main comparison. The paper's comparison of the training methods at about 5 x 10^5 inferences reads SFT's records at B = 2^19, and `reselect` at B = 2^19 produces SFT's row of the comparison at a 1% threshold. The runs of the tuning grid are `run` on the two tuning pairs, `wc_0346/goblin_mention` and `wc_1852/chinese_character`, at B = 2^20, with other values of `learning_rate`, `trainable_last_layers` and `training_share`.

`posteriorbench-methods compute-per-inference` on these records gives SFT's row of the paper's table of compute per counted inference (the methods README, "Analyses of finished runs"). For the FLOPs of a backward pass, SFT trains the last 4 layers, 83,018,720 of the model's 752,393,024 parameters, and nothing below them needs a gradient. The GPU time of its training is priced by whole updates at the run's batch size, min(k, 32) for k hits (`division.batch`).

`run --set training_set=corpus` on the two LLM-judged tasks of the generalization section, with the replicates 0, 1 and 2, produces SFT's points on those behaviors, and `reselect --set training_set=corpus` its points at the 1% threshold ([`generalization`](../generalization), "Running the paper's methods"). The paper trains no method on the two larger models.
