# CTL and CTL + composition

Contrastive twist learning (CTL; Zhao et al., 2024) keeps the original model frozen and learns a twist, a function ψ that scores every prefix of a response. The twisted model draws each token a with probability proportional to p(a | y_<t) ψ_t(y_<t a), where p is the original model. If the twist were the probability that a response with this prefix ends up a hit, the twisted model would sample the posterior exactly. Here the twist is a feed-forward network on the frozen model's last hidden state, and its output is added to the model's next-token logits. An update draws responses from the twisted model itself, weights them once toward the posterior and once toward the prefixes that the current twist defines, and moves the twist along the difference. The twist is saved as a checkpoint at intervals, and a run submits one of the checkpoints by the submission rule, testing at most the first 8 of its ranking. CTL + composition composes the submitted checkpoint with the original model.

The code follows the algorithm for CTL in the appendix of the PosteriorBench paper: `twist.py` is the twist network, the weights ω⁺ and ω⁻ and the loss, `twisted_model.py` the twisted model and its draws, and `training.py` the loop.

## Run

```sh
uv run posteriorbench-methods ctl run --task wc_1853/goblin_mention --out runs \
    --set budgets= --set update_caps=4,8,16,32
```

This trains 32 updates and writes the records of CTL and CTL + composition at the four update caps. Without `--set`, the run trains to 2^22 passes and writes the records at the paper's nine points. `--no-compose` leaves out CTL + composition.

A run that is interrupted while it trains continues after its last checkpoint when the same command is run again. Two runs with the same seeds draw the same responses in their first update. After it their weights can differ in the last bits, because torch does not fix the order in which it adds up the gradient of the network's last layer. CTL + composition takes its seeds from the weights of the submitted checkpoint, so its draws differ from run to run.

CTL needs no method data. The methods README lists the options that all families share and says how settings are changed.

`reselect` needs every checkpoint file: make the run above with `--keep-checkpoints all`.

```sh
uv run posteriorbench-methods ctl reselect --task wc_1853/goblin_mention --out runs \
    --set budgets= --set update_caps=4,8,16,32
uv run posteriorbench-methods ctl reselect --rule final-checkpoint \
    --task wc_1853/goblin_mention --out runs --set budgets= --set update_caps=4,8,16,32
```

`reselect` chooses the checkpoint again at each update cap and budget of the settings, from the training run that `run` left under `--out` with `--keep-checkpoints all` and the same settings (the methods README, "Choosing a checkpoint again"). Without `--rule` it repeats the submission rule with a hit-rate threshold of 1% in place of 10%. With `--rule final-checkpoint` it applies CTL's own rule, which returns the twist as it is when training ends: it submits the last candidate, the checkpoint after update u at an update cap u and the checkpoint that closed the budget at a budget, without a threshold test.

## Settings

`settings.py` holds the settings, with the values of the paper's runs as defaults.

| Setting | Default (the paper's runs) | Meaning |
|---|---|---|
| `budgets` | 2^20, 2^21, 2^22 | training budgets B, in passes, at which the run submits a sampler |
| `update_caps` | 4, 8, 16, 32, 64, 128 | update caps u at which the run submits a sampler |
| `training_budgets` | 2^20, 2^21, 2^22 | the budgets that the training run is trained under; it ends at the largest |
| `responses_per_update` | 1,000 | responses that the twisted model draws for one update |
| `learning_rate` | 1e-4 | Adam, constant, with no weight decay and no gradient clipping |
| `epsilon` | 1e-16 | target weight of a miss, so that an update without a hit has weights |
| `selection_draws` | 128 | selection draws of each checkpoint |
| `checkpoint_updates` | 1, 2, 4, ..., 128 | a checkpoint is saved after these updates |
| `checkpoint_interval` | 32 | and after every 32nd update from then on |

The learning rate and the responses per update are those of the command that Zhao et al. released for a threshold on a toxicity score, their target closest to a behavior rule.

What the settings do not say:

- The twist network has three linear layers with ReLUs between them: from the hidden size of the model (1,024) to itself twice, and then to the 248,320 rows of the model's output layer. It runs in float32 and starts with Xavier-normal weights and zero biases.
- An update is one step of Adam on the mean of the loss over the 50 token positions. At the positions after its end, a response counts as a whole. The weights use log q as the twisted model generated the responses, and the gradient uses the hidden states kept from their generation, so an update runs the model once per response.
- The twisted model draws over every row of the output layer, which has 243 more rows than the tokenizer defines tokens. The original model's text interface never produces such a token, so a response has p = 0 from its first undefined token on, in the weights of an update and on the selection draws (`p_scores.py`). The benchmark's evaluation scores such a token like any other.
- Responses are generated 64 at a time, each batch from a random generator of its own, and log q of given responses is computed 8 at a time. Both numbers are constants in `twisted_model.py`; the draws depend on the first.

### Budgets and update caps

The training run closes each of its `training_budgets`. Before an update, if the update and a checkpoint after it would take the budget count beyond B - 1,912, the twist as it is becomes a checkpoint, unless the schedule has just saved it (`budget_rule.py`). The 1,912 passes are what the threshold tests of a run can draw. The run at budget B then submits one of the checkpoints saved within B - 1,912 passes, and the run at update cap u one of the checkpoints saved in the first u updates.

Every update takes 4,000 passes and every checkpoint 384, whatever the responses are. Training under 2^20, 2^21 and 2^22 passes therefore ends after 260, 521 and 1,044 updates on every task, and the run at an update cap u with c checkpoints has the candidates and the cost of a run at the budget 4,000 u + 384 c + 1,912: update cap 4 stands for the budget 19,064. The paper's points below 2^20 are the six update caps.

A run trains only as far as its `budgets` and `update_caps` need, and every run of a task and replicate shares one training run. `--set budgets=1048576 --set update_caps=` trains the 260 updates of the 2^20 budget, and a later run with larger budgets continues from there, with the seeds of a training that never stopped. A budget must be one of the `training_budgets`. These belong to the training run, like every setting but `budgets` and `update_caps`, so a run that changes them needs another output directory.

## Cost

`cost.py` states what CTL pays for. The inference count is the cost that the benchmark reports. The budget count also holds the passes that the inference count leaves out, and it decides how many updates fit in a budget.

| Event | Inferences | Passes |
|---|---|---|
| an update on 1,000 responses: their generation, and p of each | 2,000 | 4,000, with the twist network's forward and backward pass |
| the 128 selection draws of a checkpoint, with h hits: their generation, and p on the hits | 128 + h | 384, with p on the misses and log q of the draws |
| a threshold-test draw | 1 | 1 |
| composition: the selection draws of the 16 settings, with h hits | 496 + h | 496 + h |
| a threshold-test draw of composition | 2, or 1 under the plain setting | the same |

An update needs p of every response, hit or miss, for its weights. The twist network runs on the model's hidden states and is small next to the model, so the inference count leaves its passes out. A composed draw needs one pass of the frozen model for q and p together, and is charged two inferences like every composed draw of the benchmark.

For example, training under 2^20 passes runs 260 updates and saves 13 checkpoints, twelve on the schedule and one that closes the budget: 260 x 4,000 + 13 x 384 = 1,044,992 passes. If the checkpoints have 686 selection hits in all and the first candidate passes its threshold test of 164 draws, the run at 2^20 costs 260 x 2,000 + 13 x 128 + 686 + 164 = 522,514 inferences, about half its budget.

`reselect` costs what the training had spent when it saved the last candidate of the point, plus its threshold tests. A test at 1% draws 1,857 responses at rank 1 and 3,269 at rank 8, so a result at a budget can cost more than the budget. With the first test passed, the example above costs 522,350 + 1,857 = 524,207 inferences at the 1% threshold, and 522,350 under CTL's own rule, which has no test.

## What a run writes

A record of `ctl`, and with composition one of `ctl+composition`, at each point: `<out>/<method>/u=<update cap>/<task>/rep<r>/` and `<out>/<method>/B=<budget>/<task>/rep<r>/`. Besides the files of every record:

- `draws/selection.jsonl` (`ctl`) holds the selection draws of the point's candidates, with the `candidate_id` of their checkpoint, log p, and log q by teacher forcing.
- `draws/threshold_tests.jsonl` (`ctl`) holds the draws of the threshold tests, with the `rank` and `candidate_id` of the tested checkpoint. A test draws in parts of 32 responses, and `seed` is the seed of a draw's part.
- `draws/composition_selection.jsonl` and `draws/composition_threshold_tests.jsonl` (`ctl+composition`) hold the draws of composition, with their `setting`.
- Every row of these four files also has the response `text`, and a row of `draws/threshold_tests.jsonl` has `logq`, the log q with which the response was generated.
- `details` in `construction.json` holds `budget`, or `update_cap` and the budget it stands for, `equivalent_budget`. A `ctl` record adds `candidates`, the checkpoints it chose from as rows of the training run's index. A composed record adds `plain`, the status of CTL's submission and its checkpoint, and `composition`, the outcome of choosing the setting, or the reason why the checkpoint was not composed.

A composed record whose status is `not_composed`, or whose chosen setting is the plain one, holds CTL's evaluation: the first at CTL's cost, the second with the cost of composition added.

`reselect` writes a record of `ctl` at each point under `<out>/reselected/threshold-1pct/` or `<out>/reselected/final-checkpoint/`, each with the layout of an output root. A record at the 1% threshold has the files and the `details` of the `ctl` record of `run`, with `threshold` (0.01) before them. A record of the own rule has `own_rule` there; its status is `own_rule`, and it has no ranking and no draws but those of the evaluation.

The training run is in `<out>/training_runs/ctl/<task>/rep<r>/`:

- `index.jsonl`: the checkpoints, each with its update, the budget count and ledger when it was saved, and the budgets it closes.
- `checkpoints/update_<n>.pt`: the twist network of a checkpoint.
- `selection/update_<n>.jsonl`: its selection draws.
- `training/update_<n>.jsonl`: the responses of update n, with log q and log p of every token.
- `settings.json`, `resume.pt`, `done.json`: what the training run was started with, what it needs to continue, and where it stopped.

## Compute and disk

On one H100, an update takes about 24 seconds and the selection draws of a checkpoint about 7 more. Training to 2^20 passes takes about 2 GPU-hours and to 2^22 between 7 and 10. Each point then takes about a minute for its threshold tests and evaluation, and about another for its composition. A run of the update caps up to 32 takes about 20 minutes. The runs of the paper (21 tasks, 3 replicates) took about 600 GPU-hours. Peak GPU memory is about 16 GB: two copies of the model, the twist network with the state of Adam, and the hidden states of an update's 1,000 responses.

A checkpoint file holds the twist network, 1.03 GB. A training run to 2^22 saves 39 of them, about 40 GB, of which `--keep-checkpoints testable` keeps at most 8 for each point when the run ends. `resume.pt` is about 3.1 GB while training is unfinished, and the training draws take about 4 MB per update.

## Paper results this produces

With the default settings, `run` on the 21 tasks and the replicates 0, 1 and 2 produces the points of CTL and CTL + composition in the paper's figure of posterior fidelity against construction cost: for each task and replicate, the six update caps and the three budgets. The paper's comparisons of the training methods at about 5 x 10^5 inferences read CTL's records at B = 2^20. There, `reselect --set budgets=1048576 --set update_caps=` produces CTL's row of the comparison at a 1% threshold, on the 21 tasks and the three replicates, and the same with `--rule final-checkpoint` its row of the comparison of the submission rule with each method's own rule, which the paper makes on replicate 0 of 12 tasks. The runs of the paper's tuning grid are `run` on the two tuning pairs, `wc_0346/goblin_mention` and `wc_1852/chinese_character`, with `--set budgets=1048576 --set update_caps=` and other values of `learning_rate` and `responses_per_update`.

`posteriorbench-methods compute-per-inference` on these records gives CTL's row of the paper's table of compute per counted inference (the methods README, "Analyses of finished runs"). The twist network, 256,627,200 parameters, is what trains. The inference count leaves its passes out; the analysis adds them: its output layer over the vocabulary at every generated token, and its update on the kept hidden states of the 1,000 responses.
