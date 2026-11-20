# KL-DPG, JS-DPG and GDC

KL-DPG and JS-DPG are f-DPG (Go et al., 2023) with two divergences. f-DPG fine-tunes a policy, which starts as a copy of the original model, toward the posterior by minimizing an f-divergence between the two on the policy's own samples: the forward KL from the posterior to the policy (KL-DPG) or the Jensen-Shannon divergence (JS-DPG). GDC (Khalifa et al., 2021) trains the policy by importance-weighted maximum likelihood on samples of a sampling distribution r, which starts as the original model and becomes a copy of the policy whenever a fit check finds the policy closer to the posterior. The three methods train every weight of the model, save checkpoints while they train, and submit one of the checkpoints by the submission rule. Each method + composition composes the submitted checkpoint with the original model.

A method trains once for a task and replicate, toward a training budget of 2^20 passes, and the run at a budget B submits one of the checkpoints that this training saved within B. A run does the following.

1. It trains the policy as far as its largest budget needs (`fdpg.py`, `gdc.py`, which state the updates). KL-DPG and JS-DPG save a checkpoint before the first update, after updates 1, 2, 4, 8 and 16, and then after every 32nd update. GDC's candidates for submission are the original model, r whenever a fit check replaces it, the policy at every fit check that keeps r, and the policy after the last update. Training ends when another update might not fit in the training budget, and earlier if the budgets of the run need no more of it (`budget_rule.py`).
2. At each budget B, the candidates are the checkpoints that the training saved while its budget count was at most B - 1,912; the 1,912 passes are set aside for the threshold tests. The run ranks them by EB-JC, tests at most the first 8 in rank order and submits one (`selection.py`). This is the record of the method. Budgets with the same candidates give the same record.
3. If the submitted checkpoint passed its threshold test and is not the original model, the submission rule chooses one of the 16 composition settings for it (`common/composition.py`). This is the record of `<method>+composition`.

A budget that is too small for any candidate, below 2,296 passes with the default settings, gives no sampler, and both methods are recorded as the fallback at a cost of B.

## Run

```sh
uv run posteriorbench-methods dpg run --method kl-dpg --task wc_1853/goblin_mention --out runs \
    --set budgets=32768
```

`--method` is `kl-dpg`, `js-dpg` or `gdc`. Without `--set`, the run trains to 2^20 passes and writes the records at the nine budgets; `--no-compose` leaves out the composed method. The methods need no method data. The methods README lists the options that all families share and says how settings are changed.

Every run of a method on a task and replicate uses one training run. A later run with a larger budget continues the training where it stopped, with the seeds of a training that never stopped, and so does a run that was interrupted.

`reselect` needs every checkpoint file: make the run above with `--keep-checkpoints all`.

```sh
uv run posteriorbench-methods dpg reselect --method kl-dpg --task wc_1853/goblin_mention \
    --out runs --set budgets=32768
uv run posteriorbench-methods dpg reselect --method gdc --rule final-policy \
    --task wc_1853/goblin_mention --out runs --set budgets=32768
```

`reselect` chooses the checkpoint again at each budget of the settings, from the training run that `run` left under `--out` with `--keep-checkpoints all` and the same settings (the methods README, "Choosing a checkpoint again"). Without `--rule` it repeats the submission rule with a hit-rate threshold of 1% in place of 10%. With `--rule` it applies the method's own rule, which submits one of the candidates without a threshold test:

| Method | `--rule` | The candidate that is submitted |
|---|---|---|
| KL-DPG, JS-DPG | `final-checkpoint` | the last candidate: at the training budget, the policy after the last update |
| GDC | `last-sampling-distribution` | the last candidate that is a sampling distribution r, the one that GDC draws from at the end |
| GDC | `final-policy` | the last candidate, which is the policy at the end: a sampling distribution that a fit check has just replaced is a copy of the policy |

Budgets with the same candidates get the same choice. A budget that is too small for any candidate has no checkpoint to choose, and `reselect` stops with an error.

## Settings

KL-DPG and JS-DPG (`settings.FdpgSettings`):

| Setting | Default (the paper's runs) | Meaning |
|---|---|---|
| `objective` | `kl` or `js` | the divergence: KL-DPG or JS-DPG. `--method` sets it, and a run refuses another value |
| `budgets` | 2^12, 2^13, ..., 2^20 | the budgets B at which the run submits a checkpoint, in passes of a model over one response |
| `training_budget` | 2^20 | the budget of the training run; no budget may exceed it |
| `responses_per_update` | 256 | responses drawn from the policy for one update |
| `learning_rate` | 1.41e-5 | learning rate of Adam after the warmup |
| `warmup_updates` | 100 | the learning rate rises linearly from 0 over this many updates |
| `adam_beta1`, `adam_beta2`, `adam_epsilon` | 0.9, 0.999, 1e-8 | Adam, without weight decay or gradient clipping |
| `baseline_decay` | 0.99 | JS-DPG: share of its old value that the baseline keeps at each of its updates |
| `baseline_group` | 64 | JS-DPG: the baseline is updated once for every 64 responses of an update |
| `selection_draws` | 128 | selection draws of each checkpoint |

GDC (`settings.GdcSettings`) has `budgets`, `training_budget`, `learning_rate`, `warmup_updates`, the three settings of Adam and `selection_draws` with the same values, and:

| Setting | Default (the paper's runs) | Meaning |
|---|---|---|
| `responses_per_update` | 2,048 | responses drawn from r for one update |
| `loss_divisor` | 256 | an update's loss is minus the sum of (p / r) log pi over its hits, divided by this number: the original code averages over 256 responses at a time and adds up the 8 averages |
| `fit_check_interval` | 10 | a fit check follows every 10th update |
| `fit_check_draws` | 5,120 | new responses that a fit check draws from r |

`config` takes `--method` too: `posteriorbench-methods dpg config --method gdc` prints GDC's settings.

Every setting but `budgets` belongs to the training run, so a run that changes one of them needs another output directory. The first update has learning rate 0, so the checkpoint after it is the original model again. An update of KL-DPG or JS-DPG steps on the mean over its responses. The numbers of responses in one forward pass are fixed in `samplers.py`: 128 when responses are generated and 32 when given responses are scored or trained on.

## Cost

| Event | Inference count | Budget count |
|---|---|---|
| an update of KL-DPG or JS-DPG on 256 draws, h of them hits | 512 + h | 1,024 (KL-DPG), 1,280 (JS-DPG) |
| the same before the first hit of the training | 256 | 768 |
| the 128 selection draws of a checkpoint, h of them hits | 128 + h | 384 |
| an update of GDC on 2,048 draws, h of them hits | 2,048 + 2h | 2,048 + 4h |
| a fit check of GDC on 5,120 draws, h of them hits | 5,120 + 2h | 5,120 + 3h |
| a threshold test of n draws | n | n |
| the selection draws of composition, h of them hits | 496 + h | 1,264 |
| a threshold test of a composed setting, n draws | 2n | 2n |

The inference count has the generation of each draw, p on the hits, and the forward pass of a gradient step. The budget count has every pass that a run makes (`cost.py`). An update of KL-DPG computes p of all its draws and runs a forward and a backward pass over them, and JS-DPG one more pass for its coefficients; before the first hit there is no gradient step. A checkpoint computes p of all its selection draws and gets their log q from a pass of its own, and so does composition, with both models. GDC computes p, r and the policy on hits only. A test of the plain composition setting, which is the checkpoint itself, costs n.

A run at budget B costs what the training had spent when it saved the last candidate of B, whichever candidate is submitted, plus its threshold tests. So does `reselect`. A test at 1% draws 1,857 responses at rank 1 and 3,269 at rank 8, more than the 1,912 passes that B sets aside for all its tests, so a result at the 1% threshold can cost more than B. An own rule has no test.

Example, KL-DPG at B = 2^15 = 32,768. After update 16 the training has saved 6 checkpoints and its budget count is 16 x 1,024 + 6 x 384 = 18,688. The next checkpoint follows update 32 and would come at 18,688 + 16 x 768 + 384 = 31,360 passes at the least, beyond 32,768 - 1,912 = 30,856, so training stops. If the 16 updates drew 1,300 hits, the selection draws 250, and the first threshold test passes, the inference count is 16 x 512 + 1,300 + 6 x 128 + 250 + 164 = 10,674 and the budget count 18,688 + 164 = 18,852.

## What a run writes

One record for each method and budget, `<out>/<method>/B=<budget>/<task>/rep<r>/`, with the methods `kl-dpg`, `js-dpg` or `gdc` and the same with `+composition`. The methods README describes the files of a record. In this family:

- `draws/selection.jsonl` (the method): the selection draws of the checkpoints that the budget chose from, with log q and log p, by `candidate_id`. A checkpoint's id is `update_000016` for the policy of KL-DPG or JS-DPG after 16 updates, and `r_update_000010` or `pi_update_000010` for GDC's r or policy after 10 updates.
- `draws/threshold_tests.jsonl` (the method): the draws of every threshold test, with the `candidate_id` and `rank` of the tested checkpoint.
- `draws/composition_selection.jsonl` and `draws/composition_threshold_tests.jsonl` (the composed method): the draws of composition, each with its `setting`.
- `details` of `construction.json`: `candidates`, the checkpoints that the budget chose from as rows of the training run's index, for the method; `plain` (the status of the method's submission and its checkpoint) and `composition` (the 16 settings, their ranking and tests, the chosen setting) for the composed method. A composed record whose status is `not_composed` gives the reason in `composition`, `plain_not_passed` or `original_model`, and holds the method's evaluation; so does one whose chosen setting is the plain one.

`reselect` writes a record of the method at each budget under `<out>/reselected/<rule>/`, with the layout of an output root; the rule is `threshold-1pct` unless `--rule` names another. A record at the 1% threshold has the files and the `details` of the method's record of `run`, with `threshold` (0.01) before `candidates`. A record of an own rule has `own_rule` there; its status is `own_rule`, and it has no ranking and no draws but those of the evaluation.

The training run is in `<out>/training_runs/<method>/<task>/rep<r>/`:

- `index.jsonl`: the checkpoints in the order saved, each with its update, the budget count and ledger when it was saved, and its selection hits. `extra` holds log Z for KL-DPG and JS-DPG; for GDC, whether the candidate is r or the policy, and the fit check that it follows.
- `checkpoints/<id>.pt`: every weight of the model. Checkpoints with the same weights are one file.
- `selection/<id>.jsonl`: the checkpoint's selection draws, with log q and log p.
- `training/update_<n>.jsonl`: the draws of update n. For KL-DPG and JS-DPG a row has log p and the policy's log-probability of the response, as generation recorded it (`logq_generation`) and from the update's forward pass (`logq`). For GDC a hit has `logp`, `logr` and `logpi`. `training/fit_check_<n>.jsonl` holds the draws of GDC's fit check after update n.
- `settings.json`, `resume.pt`, `done.json`: what the training run was started with, what it needs to continue, and where it stopped.

## Compute and disk

On one H100, training to 2^20 passes takes about 3 GPU-hours for KL-DPG (about 1,000 updates) and JS-DPG (about 800), and about 4.5 for GDC (at most 400 updates). A run at B = 2^15 alone takes about 7 minutes for KL-DPG or JS-DPG and about 10 for GDC, with the threshold tests, composition and the evaluation. A run holds the original model, the policy with Adam's two moments and, for GDC, r; the paper's runs used one 80 GB GPU.

A checkpoint file is about 4 GB. Training to 2^20 saves up to 38 checkpoints for KL-DPG, 32 for JS-DPG and 41 for GDC, so up to about 150 GB, 130 GB and 165 GB, before `--keep-checkpoints testable` or `returned` deletes most of them when the run ends. While training is unfinished, `resume.pt` takes about 10 GB (KL-DPG, JS-DPG) or 14 GB (GDC).

## Paper results this produces

With the default settings, `run` with each of the three methods on the 21 tasks and the replicates 0, 1 and 2 produces the points of KL-DPG, JS-DPG, GDC and their compositions in the paper's main comparison. GDC's first fit check follows update 10, so at 2^12, 2^13 and 2^14 its only candidate is the original model and the three records are the same; the paper shows this point once. The paper's comparisons of the training methods at about 5 x 10^5 inferences read these records at B = 2^20 for KL-DPG and JS-DPG and at B = 2^19 for GDC. At these budgets, `reselect` produces the three methods' rows of the comparison at a 1% threshold, on the 21 tasks and the three replicates, and `reselect --rule` their rows of the comparison of the submission rule with each method's own rule, which the paper makes on replicate 0 of 12 tasks. The runs of the tuning grid are `run` on the two tuning pairs, `wc_0346/goblin_mention` and `wc_1852/chinese_character`, with `--set budgets=1048576` and other values of `learning_rate`, `responses_per_update`, and for GDC `warmup_updates` and `fit_check_draws`.

`posteriorbench-methods compute-per-inference` on these records gives the three methods' rows of the paper's table of compute per counted inference (the methods README, "Analyses of finished runs"). The three methods train all of the model's 752,393,024 weights, so the backward pass of a gradient step costs two forward passes; KL-DPG and JS-DPG take one on every response of an update that has a gradient step, GDC on the hits of an update.

The paper's section on larger models and LLM-judged behaviors has no result of these methods, and this family does not run on the tasks of that section.

## Attribution

f-DPG is the method of Go et al., "Aligning Language Models with Preferences through f-divergence Minimization" (ICML 2023), and GDC that of Khalifa et al., "A Distributional Approach to Controlled Text Generation" (ICLR 2021). The code is our own. Where the two method papers leave a detail open, the code follows their authors' implementations, [naver/disco](https://github.com/naver/disco) and [naver/gdc](https://github.com/naver/gdc): JS-DPG updates its baseline before it subtracts it, once for every 64 responses, and GDC divides its loss by 256.
