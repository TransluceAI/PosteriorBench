# Analyses of finished runs

Three analyses of the paper's appendix read the records that runs have written, and run no method: how much a method computes for each inference that it is charged, how the scores change with the cap C of the forward KL, and how well a submitted sampler estimates p(S). All take records as their arguments: an output root, whose records are all read, a record directory, or a record's `result.json`. A point holds at most one record for a task and replicate, so the records of `run` and of each rule of `reselect` (`<out>/reselected/<rule>`) are analysed in separate calls.

## Compute per counted inference

The benchmark's cost is the inference count: one inference for generating a response, for p(y) or q(y) of a response, and for the forward pass of a gradient step on a response. The count leaves out the backward pass of a gradient step, and it counts a response of a method's sampler like a response of the original model. The paper's table of compute per counted inference says how far the count is from what a run computes, in two measures:

- **FLOPs / count.** One inference is one forward pass of the original model over one response.
- **GPU time / count.** One inference is the time to generate one response with the original model.

```sh
uv run posteriorbench-methods compute-per-inference runs
```

prints, as JSON, one entry for each method and one for each point (a method at a setting) under `runs`:

```text
methods   method, points, runs,
          flops_per_inference, flops_per_inference_at:        the highest value among the
                                                              method's points, and its setting
          gpu_time_per_inference, gpu_time_per_inference_at:  the same for GPU time
          largest_run_flops_per_inference, largest_run_gpu_time_per_inference
points    method, setting, runs, construction_inferences (the mean over the runs),
          flops_per_inference, flops_per_inference_by_convention, gpu_time_per_inference
not_covered   the records that were left out, by method
```

`--runs` adds the values of every single run with its operations, `--flops-only` leaves out the GPU time, and `--out FILE` writes the JSON to a file. The paper's table is the `methods` list of the runs of its main comparison, on the 21 tasks and three replicates. The command reads the sizes of the model from its configuration and the length of each task's prompt from the dataset; `--model` and `--data` name them as they do for `run`.

For a point, the compute of its runs is added up and divided by their inference counts, added up. This is the ratio of the two averages over tasks and replicates, so a point of the comparison whose cost is the average count has this compute per inference. A method's value is the highest value among its points.

### What the compute of a run is

`operations.py` reads from a record what the run did, in responses:

| What a run did | Where the record has it |
|---|---|
| draws from the original model | SFT: `division.draws`; Replay and the methods of the prompt family: every response they generate |
| draws from the method's sampler | the other generated responses, threshold tests included |
| counted log-probabilities | the ledger's `p_scoring` and `q_scoring` |
| counted forward passes of gradient steps | the ledger's `training_forward` |
| responses of gradient steps, each with a backward pass | SFT, KL-DPG, JS-DPG, GDC: those of the counted forward passes. GFlowNet: these and its fresh training responses, whose forward pass is counted with their generation. CTL: its training responses, on which it updates the twist network |

The first four add up to the run's inference count. The ledger of a record is in its `result.json`; the division of SFT's budget and the candidates of GFlowNet and CTL, whose selection draws tell the training responses from the other generated responses, are in the `details` of its `construction.json`.

The compute of a run is the cost of these operations, each with the model that runs it. A run also makes passes that are not among them: a teacher-forced pass for log q of a sampler's own draws, p on responses where the method does not use it, JS-DPG's pass for its coefficients, and for a fresh training response of GFlowNet a pass for log q and a forward pass of the loss, which the authors' code avoids by computing the loss while it generates. A method need not make these passes, the inference count leaves them out for that reason, and the compute leaves them out too. They are in the budget count of a record.

A run whose budget was too small for a sampler is charged its budget. SFT and Replay had made their draws by then, and that is their compute.

### FLOPs

`flops.py` counts a pass in multiply-accumulates, as in Kaplan et al. (2020, "Scaling laws for neural language models"): a linear map with W weights costs W for each token in the forward pass, and in the backward pass W for the gradient of its input and W for the gradient of its weights. A frozen map needs no gradient of its weights, and nothing below the lowest trained weight needs a gradient at all. The sizes of the model come from its configuration; Qwen3.5-0.8B has 18 linear-attention and 6 full-attention layers and 752,393,024 parameters. Every response is counted with the response limit of 50 tokens, after the prompt of its task (16 to 30 tokens on the 21 tasks).

| Method | What trains | A draw of its sampler | The backward pass of a gradient step |
|---|---|---:|---:|
| SFT | the last 4 layers, 83,018,720 parameters | 1 | 0.54 |
| GFlowNet | low-rank adapters, 30,277,632 parameters; the model is frozen | 1.04 | 1.09 |
| CTL | the twist network, 256,627,200 parameters, on hidden states kept from generation | 1.22 | 0.005 for a whole update |
| KL-DPG, JS-DPG, GDC | all of the model's weights | 1 | 2 |
| Replay, prompt selection, prompt mixtures | nothing | 1 | |

The numbers are in forward passes of the original model over the same response, for a prompt of 24 tokens. A backward pass over all weights costs two forward passes. A gradient step of KL-DPG, JS-DPG or GDC on a response comes with a counted generation and a counted forward pass, so it counts 2 and computes 4, and their compute stays below twice their count whatever the run. CTL's twist network runs its output layer over the vocabulary at every generated token, which the count leaves out.

A pass can be counted in more than one way. The command computes four (`flops.CONVENTIONS`): with the output layer at every position of the prompt and the response, as a teacher-forced pass runs it (`every_position`); with the output layer only at the response's positions (`output_layer_where_read`); without the operations that have no weights, which are the products of full attention, the state of linear attention and its convolution (`weights_only`); and with the prompt run once and kept, so that it costs nothing in a pass (`prompt_cached`). `flops_per_inference` is the largest of the four, as in the paper's table, and `flops_per_inference_by_convention` has each.

### GPU time

The GPU time of a run is the time of its operations, each priced from a measured time (`gpu_times.py`):

| Operations of a run | Priced at the time of |
|---|---|
| draws from the original model | generation by the original model, 500 responses in a forward pass |
| draws from the sampler of SFT, KL-DPG, JS-DPG, GDC | the same, 128 in a forward pass |
| draws from GFlowNet's policy, from CTL's twisted model | generation by that model, 500 and 64 in a forward pass |
| counted log-probabilities | a teacher-forced forward pass of the original model over 32 responses |
| SFT's training | its updates, each measured whole at its batch size of 1 to 32 responses |
| CTL's training | the update of the twist network on the 1,000 responses of one update, measured whole |
| training of GFlowNet, KL-DPG, JS-DPG, GDC | the forward pass of a gradient step for each counted one; the forward and backward pass minus the forward pass for each response of a gradient step, 20 (GFlowNet) or 32 responses at a time; one step of the optimizer for each update |

The batch sizes are those at which the methods run these operations here. One inference is priced at the time to generate one response with the original model, 500 at a time.

`data/h100_times.json` holds the measurements behind the paper's table, made on one NVIDIA H100 with float32 weights, torch 2.11.0 and transformers 5.13.1, on the prompt of `wc_0532/profanity` (24 tokens, the median of the 21 tasks) and responses of 50 tokens. A method is priced from one session of measurements, the last that has every measurement it needs.

To measure the times on your GPU:

```sh
uv run posteriorbench-methods time-operations --out my_times.json
uv run posteriorbench-methods compute-per-inference runs --times my_times.json
```

`time-operations` times every operation of the table above with this repository's own functions (`time_operations.py`), six times after two warm-up calls by default, and writes the medians as one session. It takes about four minutes on an H100. Its largest operation, a gradient step on all weights over 32 responses, needs about 37 GiB of GPU memory.

Two of the operations need a word on what is timed. SFT's updates are timed first, on a copy of the model that nothing else has used, with SFT's own step, optimizer and learning rate, as a run of SFT makes them; at a small batch an update is bound by the time to issue its operations, not by the GPU. An update issues about 90,000 of them, so its time depends on the machine's CPU and its load. The update of the twist is timed from tensors that are on the device already. An update of CTL first moves log p of its 1,000 responses there, one response at a time; this is not part of the time.

Three things to know when reading the GPU time:

- Every time is measured with transformers in float32, the engine of this repository.
- A small backward pass takes longer than its FLOPs suggest, and generation takes about twice as long as a teacher-forced pass over the same response at these batch sizes. The method with the most FLOPs per inference is therefore not the one with the most GPU time.
- A run of SFT with few hits trains in small batches, at which an update on one response takes up to twelve times as long as at 32. A single such run can take more than twice its count in GPU time; the points do not.

### What is covered

The methods above. A record of a composed method (`<method>+composition`), of Steering or of Instruction is left out and counted under `not_covered`: a composed response runs two models for part of its tokens, Steering's vectors are built from longer responses, and Instruction costs nothing. The paper's table does not have them either.

## Sensitivity to the cap C

The capped forward KL of a sampler that passed the threshold is the mean over the 250 references of min(t, C), where t = log(p_S(y) / q_S(y)) is the reference's term and C = 50 is the benchmark's cap. The paper checks that its comparisons do not depend on this choice.

```sh
uv run posteriorbench-methods cap-sensitivity runs
```

prints, for each of the caps 10, 20, 50, 100 and 200 (`--caps`):

```text
share_at_cap  cap, samplers, references, references_at_cap, share: the references whose term is
              at or above the cap, pooled over the samplers that passed the threshold; and the
              same for each method (by_method)
points        cap, method, setting, and the summary that `posteriorbench aggregate` gives:
              tasks, replicates_per_task, results, passed_threshold, and the mean and standard
              error over tasks of the capped forward KL at this cap and of the other scores
```

`--runs` adds the capped forward KL of every run at each cap, `--allow-unequal-replicates` averages tasks with different numbers of replicates, and `--out FILE` writes the JSON to a file.

Nothing is run again. A record keeps the terms of its references (`forward_terms` in `evaluation_details.json`), and its score at another cap is their mean after capping. A run that was scored as the fallback has rho log rho + (1 - rho) C, where rho is the `fallback_rho` of its result. The other scores do not depend on the cap. Before it uses a record's terms, the command checks that they give the record's own score at the record's own cap.

The share leaves out the samplers that were scored exactly over a finite support, which is Replay: none of its training responses is a reference, so all of its terms are infinite and its capped forward KL is C.

## Estimating p(S) with a sampler

A posterior sampler also gives an estimate of the event probability. From n draws y_1, ..., y_n of a sampler q, importance sampling estimates

    p_IS(S) = (1 / n) sum_i 1[y_i in S] p(y_i) / q(y_i),

which is unbiased when q can generate every hit that p can. The paper's appendix measures the log error of this estimate, log p_IS(S) - log p̂(S), where p̂(S) is the task's estimate from the corpus, for the sampler that each run submitted.

```sh
uv run posteriorbench-methods estimate-ps runs
```

prints, for the first 960 of each sampler's 1,024 evaluation draws, the number of draws that the paper's analysis uses (`--draws`):

```text
points            method, setting, tasks, replicates_per_task, runs,
                  own_estimate_passed, own_estimate_failed, fallback_estimate, unavailable:
                      how many runs are scored by which estimate,
                  construction_inferences: mean and standard_error,
                  abs_log_error: mean and standard_error of the absolute log error
samplers_summary  samplers: the samplers that passed the threshold;
                  spearman_abs_log_error_capped_forward_kl, spearman_abs_log_error_reverse_kl;
                  need_fewer_draws, need_fewer_draws_median_log_error
unavailable       the runs whose own estimate the records do not hold the scores for
not_covered       the records that were left out, by method
```

`--samplers` adds one row for each sampler that passed the threshold: `log_error`, the `reverse_kl` and `capped_forward_kl` of its evaluation, `reverse_kl_all_hits`, `native_hit_rate`, `draws_needed` and `records`. `--runs` adds the estimate of every run, `--allow-unequal-replicates` averages tasks with different numbers of replicates, and `--out FILE` writes the JSON to a file.

In the paper's figure, the left panel plots `abs_log_error` of each point against its `construction_inferences`, with a band of one standard error, and the right panel plots `log_error` against `reverse_kl` for every row of `--samplers`, with the line log error = -reverse KL.

**Which estimate a run is scored by.** A run with a hit among the draws is scored by its own estimate, whether or not its sampler passed the hit-rate threshold. A run without such a hit, and a run that produced no sampler, takes the estimate of the fallback, which always returns its one response a: that estimate is p(a), and its log error, log(p(a) / p̂(S)), is minus the fallback's reverse KL. A point averages the absolute log error as `posteriorbench aggregate` averages a score, over the replicates of a task and then over the tasks, and its cost in the same way.

**The bound.** With H hits among the n draws, log p_IS(S) = log(H / n) + log mean(p / q over the hits). By Jensen's inequality this is at least log(H / n) + mean(log p - log q), so log error >= -reverse KL, where the reverse KL is estimated from all the hits among the same draws: `reverse_kl_all_hits`. The benchmark's `reverse_kl` averages over at most 128 hits of all 1,024 draws, so a sampler can lie slightly below the line when it is drawn against that one.

**How many draws a sampler needs.** `draws_needed` is exp(capped forward KL) / hit rate, with the two scores of the evaluation. `samplers_summary` counts the samplers that need fewer draws than the estimate reads and gives the median of their log errors, and the rank correlations of the absolute log error with the two KL scores.

**What a record holds.** The estimate needs log q and log p of every hit among the draws. The benchmark's evaluation computes them only for the hits that its reverse KL is averaged over, at most 128, and for none of the hits of a sampler below the threshold. `run` therefore scores the other hits after the evaluation, with the submitted sampler and the original model, and writes all of them to `evaluation_hits.json`. This draws nothing and is not construction cost. A record without `evaluation_hits.json` has its own estimate only if the evaluation scored every hit among the draws; otherwise the run counts as `unavailable`, and its point has no average.

Replay is left out (`not_covered`): it returns its training samples, and its evaluation draws nothing. Composed methods are covered like the others; a composed record that repeats its method's evaluation counts once among the samplers (`records` says how many records hold a sampler's evaluation).
