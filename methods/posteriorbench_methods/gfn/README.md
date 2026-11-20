# GFlowNet and GFlowNet + composition

GFlowNet fine-tuning (Hu et al., 2024) trains a policy, a copy of the original model with low-rank adapters, to generate each response with probability proportional to a reward. The reward of a response y is p(y) if y is a hit and 1e-8 p(y) if it is a miss, where p is the original model, so the policy's hits follow the posterior. The loss is subtrajectory balance. It computes a flow for every prefix of a response, from the reward of the response that ends there and the policy's probability of ending there, and it asks that along every segment of the response the flow of the first prefix times the policy's probability of the segment equal the flow of the last. Training mixes fresh responses of the policy, drawn at sampling temperatures that vary from group to group, with the highest-reward responses found so far, replayed from a buffer. The policy is saved as a checkpoint at intervals, and a run submits one of the checkpoints by the submission rule, testing at most the first 8 of its ranking. GFlowNet + composition composes the submitted checkpoint with the original model.

The code follows the algorithm for GFlowNet fine-tuning in the appendix of the PosteriorBench paper: `training.py` is the loop, `subtb.py` the loss, `reward.py` the reward, `replay.py` the buffer, and `policy.py` the adapters.

## Run

```sh
uv run posteriorbench-methods gfn run --task wc_1853/goblin_mention --out runs \
    --set budgets= --set update_caps=8,16,32
```

This trains 32 updates and writes the records of GFlowNet and GFlowNet + composition at the three update caps. Without `--set`, the run trains to 2^22 passes and writes the records at the paper's nine points.

Two runs with the same seeds draw the same responses up to the first update that changes the weights, which is the second, since the first has learning rate 0. From then on the weights differ in their last bits from run to run, and the draws can differ with them. GFlowNet + composition takes its seeds from the weights of the submitted checkpoint, so its draws differ from run to run.

GFlowNet needs no method data. The methods README lists the options that all families share and says how settings are changed.

`reselect` needs every checkpoint file: make the run above with `--keep-checkpoints all`.

```sh
uv run posteriorbench-methods gfn reselect --task wc_1853/goblin_mention --out runs \
    --set budgets= --set update_caps=8,16,32
uv run posteriorbench-methods gfn reselect --rule highest-mean-log-reward \
    --task wc_1853/goblin_mention --out runs --set budgets= --set update_caps=8,16,32
```

`reselect` chooses the checkpoint again at each update cap and budget of the settings, from the training run that `run` left under `--out` with `--keep-checkpoints all` and the same settings (the methods README, "Choosing a checkpoint again"). Without `--rule` it repeats the submission rule with a hit-rate threshold of 1% in place of 10%. With `--rule highest-mean-log-reward` it applies GFlowNet's own rule: among the candidates, it submits the checkpoint whose 128 selection draws have the highest mean log reward, log p(y) for a hit and log p(y) + log epsilon for a miss, and the earliest of them if several have the same mean. That checkpoint is submitted without a threshold test.

## Settings

`settings.py` holds the settings, with the values of the paper's runs as defaults.

| Setting | Default (the paper's runs) | Meaning |
|---|---|---|
| `budgets` | 2^20, 2^21, 2^22 | training budgets B, in passes, at which the run submits a sampler |
| `update_caps` | 8, 16, 32, 64, 128, 256 | update caps u at which the run submits a sampler |
| `training_budgets` | 2^20, 2^21, 2^22 | the budgets that the training run is trained under; it ends at the largest |
| `epsilon` | 1e-8 | the reward of a miss is epsilon p(y) |
| `lora_rank`, `lora_alpha` | 64, 16 | rank and scale of the adapters |
| `learning_rate` | 1e-4 | AdamW, weight decay 0.01, no gradient clipping |
| `warmup_updates` | 20 | update u has the learning rate 1e-4 (u - 1) / 20 until u = 21 |
| `groups_per_update` | 25 | groups of responses in one update |
| `responses_per_group` | 20 | responses in one group |
| `replay_probability` | 0.25 | a group is replayed from the buffer with this probability |
| `tempered_probability` | 0.666 | a fresh group is drawn at a temperature T ~ U[0.5, 2] with this probability, else at T = 1 |
| `temperature_low`, `temperature_high` | 0.5, 2 | the range of T |
| `buffer_size` | 50 | responses in the replay buffer |
| `buffer_similarity` | 0.25 | responses of n and m tokens with edit distance below 0.25 (n + m) are near-duplicates |
| `selection_draws` | 128 | selection draws of each checkpoint |
| `checkpoint_updates` | 1, 2, 4, 8, 16, 20, 32, 64, 128 | a checkpoint is saved after these updates |
| `checkpoint_interval` | 32 | and after every 32nd update from then on |
| `checkpoint_budget_counts` | 2^10, 2^12, ..., 2^20 | and after the first update at which the budget count has reached each of these |
| `generation_batch_size` | 500 | responses generated in one forward pass |
| `forward_batch_size` | 20 | responses in one teacher-forced pass of the policy |

What the settings do not say:

- The adapters sit on the four attention projections of a full-attention layer, on `in_proj_qkv` and `out_proj` of a linear-attention layer, and on `up_proj` and `down_proj` of every feed-forward network: 108 modules in Qwen3.5-0.8B. Their second factor starts at zero, and dropout is off everywhere.
- The first update has learning rate 0. The checkpoint after it is the original model, and is not composed.
- The buffer drops a response whose text, without periods and surrounding whitespace, it already holds. Of two near-duplicates it keeps the one with the higher reward. When it is full, the response with the lowest reward leaves. Only fresh responses are offered to it.
- The loss handles the two stop tokens and the response limit (`subtb.py`): the flow of a prefix sums the reward and the policy's probability over both stop tokens, one more term per prefix sets the flows computed from each stop token equal, and a response that reaches the response limit ends there with probability one. The loss of a group is divided by its number of terms and by the groups of an update.

### Budgets and update caps

The training run closes each of its `training_budgets`: just before the budget count could pass B - 1,912, it saves the policy as a checkpoint if the schedule has not just done so. The 1,912 passes are what the threshold tests of a run can draw. The run at budget B then submits one of the checkpoints saved within B - 1,912 passes. The run at update cap u submits one of the checkpoints saved in the first u updates, and costs what the training had spent by then plus its threshold tests.

A run trains only as far as its `budgets` and `update_caps` need, and every run of a task and replicate shares one training run. `--set budgets=1048576 --set update_caps=` trains as far as the 2^20 budget needs, and a later run with larger budgets continues from there, with the seeds of a training that never stopped. A budget must be one of the `training_budgets`. These belong to the training run, like every setting but `budgets` and `update_caps`, so a run that changes them needs another output directory.

## Cost

`cost.py` states what GFlowNet pays for. The inference count is the cost that the benchmark reports. The budget count also holds the passes that the inference count leaves out, and it decides how many updates fit in a budget.

| Event | Inferences | Passes |
|---|---|---|
| a fresh training response: generation and p(y); the loss's forward pass is counted with the generation | 2 | 5, with q(y) by teacher forcing, the loss's forward pass and its backward pass |
| a replayed training response: the loss's forward pass | 1 | 2, with the backward pass |
| the selection draws of a checkpoint, with h hits: 128 generations and p on the hits | 128 + h | 384, with p on the misses and q of the draws |
| a threshold-test draw | 1 | 1 |
| composition: the selection draws of the 16 settings, with h hits | 496 + h | 752, with p on the misses |
| a threshold-test draw of composition | 2, or 1 under the plain setting | the same |

For example, an update has 500 responses, of which 375 are fresh on average: 2 x 375 + 125 = 875 inferences and 5 x 375 + 2 x 125 = 2,125 passes. About 490 updates fit in 2^20 passes. A run at update cap 32 whose first candidate passes its threshold test costs what the training had spent when it saved the checkpoint of update 32, plus 164 inferences.

`reselect` costs what the training had spent when it saved the last candidate of the point, plus its threshold tests. A test at 1% draws 1,857 responses at rank 1 and 3,269 at rank 8, so a result at a budget can cost more than the budget. GFlowNet's own rule has no test.

## What a run writes

A record of `gflownet`, and with composition one of `gflownet+composition`, at each point: `<out>/<method>/u=<update cap>/<task>/rep<r>/` and `<out>/<method>/B=<budget>/<task>/rep<r>/`. Besides the files of every record:

- `draws/selection.jsonl` (`gflownet`) holds the selection draws of the point's candidates, with the `candidate_id` of their checkpoint, log q and log p.
- `draws/threshold_tests.jsonl` (`gflownet`) holds the draws of the threshold tests, with the `rank` and `candidate_id` of the tested checkpoint.
- `draws/composition_selection.jsonl` and `draws/composition_threshold_tests.jsonl` (`gflownet+composition`) hold the draws of composition, with their `setting`.
- Every row of these four files also has the response `text`, and a threshold-test row `logq`, the log q with which the response was generated.
- `details` in `construction.json` holds `update_cap` or `budget`. A `gflownet` record adds `candidates`, the checkpoints it chose from as rows of the training run's index. A composed record adds `plain`, the status of GFlowNet's submission, and `composition`, the outcome of choosing the setting, or the reason why the checkpoint was not composed.

`reselect` writes a record of `gflownet` at each point under `<out>/reselected/threshold-1pct/` or `<out>/reselected/highest-mean-log-reward/`, each with the layout of an output root. A record at the 1% threshold has the files and the `details` of the `gflownet` record of `run`, with `threshold` (0.01) before them. A record of the own rule has `own_rule` and `mean_log_reward`, the mean of every candidate, in its `details`; its status is `own_rule`, it has no ranking and no `draws/threshold_tests.jsonl`, and its `draws/selection.jsonl` holds the selection draws that the rule read.

The training run is in `<out>/training_runs/gflownet/<task>/rep<r>/`:

- `index.jsonl`: the checkpoints, each with its update, the budget count and ledger when it was saved, and the budgets it closes.
- `checkpoints/update_<n>.pt`: the adapters of a checkpoint.
- `selection/update_<n>.jsonl`: its selection draws, with log q and log p.
- `training/update_<n>.jsonl`: the responses that update n trained on, with their group, whether they were replayed, the group's seed and temperature, log p, and log q under the policy before the update.
- `settings.json`, `resume.pt`, `done.json`: what the training run was started with, what it needs to continue, and where it stopped.

## Compute and disk

On one H100, an update of 500 responses takes about 25 seconds and the 128 selection draws of a checkpoint about 10 more. Training to 2^20 passes takes about 4 GPU-hours and to 2^22 about 15. A run to update cap 32 with its three points, composition and evaluation takes about 20 minutes. Peak GPU memory is about 25 GiB, which includes the copy of the original model that computes p.

A checkpoint file is about 121 MB. A training run to 2^22 saves about 70 of them, and `resume.pt` is about 360 MB while training is unfinished. The training draws take less than 1 MB per update.

## Paper results this produces

The defaults produce GFlowNet's and GFlowNet + composition's points in the paper's figure of posterior fidelity against construction cost: for each task and replicate, the six update caps and the three budgets. The paper's comparisons of the training methods at about 5 x 10^5 inferences read GFlowNet's records at B = 2^20. There, `reselect --set budgets=1048576 --set update_caps=` produces GFlowNet's row of the comparison at a 1% threshold, on the 21 tasks and the three replicates, and the same with `--rule highest-mean-log-reward` its row of the comparison of the submission rule with each method's own rule, which the paper makes on replicate 0 of 12 tasks. The runs of the paper's tuning grid are `run` on the two tuning pairs, `wc_0346/goblin_mention` and `wc_1852/chinese_character`, with `--set budgets=1048576 --set update_caps=` and other values of `learning_rate`, `replay_probability` and `responses_per_group`, the setting that carries the responses per update: 5, 20 or 80 responses in each of the 25 groups.

`posteriorbench-methods compute-per-inference` on these records gives GFlowNet's row of the paper's table of compute per counted inference (the methods README, "Analyses of finished runs"). For the FLOPs, the adapters are what trains, 30,277,632 parameters at rank 64 on 108 projections; the frozen model below them still passes the gradient on. The analysis reads a fresh training response as the count does, as one generation and one backward pass: the teacher-forced pass for its log q and the separate forward pass of its loss, which this loop makes and charges to the budget alone, are not in the compute.

## Attribution

GFlowNet fine-tuning is the method of Hu et al., "Amortizing intractable inference in large language models" (ICLR 2024). `replay.py` adapts the replay buffer's `add` and the text normalization of `add_batch` from `next_sentence/utils.py`, and `policy.disable_dropout` adapts the dropout removal of `next_sentence/train.py`, both from [GFNOrg/gfn-lm-tuning](https://github.com/GFNOrg/gfn-lm-tuning) at commit 8c64035 (MIT License, Copyright (c) 2023 GFNOrg; `THIRD_PARTY_LICENSES/GFNOrg-gfn-lm-tuning-MIT.txt`). The rest is our own code.
