# The evaluation protocol

`Benchmark.evaluate(task, sampler, model, ledger=..., seed=..., replicate=...)` scores one sampler q on one task. The task fixes a prompt, a behavior S and the estimate p̂(S) of the probability that a response of the original model p to the prompt shows the behavior. The posterior p_S is p restricted to the responses with the behavior.

## The four steps

1. Draw 1,024 new responses from q and label them with the task's rule. Their hit fraction q̂(S) is the sampler's native hit rate.
2. If q̂(S) is below 10% (fewer than 103 hits), the sampler fails and receives the scores of the fallback (below).
3. Otherwise compute p(y) and q(y) for the task's 250 references, which are held-out samples of the posterior, and for up to 128 of the sampler's hits from step 1, chosen at random.
4. Estimate two divergences between the posterior and q restricted to its hits:
   - the capped forward KL, the mean over the references of min{log p(y) − log q(y) − log p̂(S) + log q̂(S), C} with C = 50; a reference with q(y) = 0 contributes C;
   - the reverse KL, the mean over the scored hits of log q(y) − log p(y), minus log q̂(S), plus log p̂(S).

Lower is better for both. Both are 0 when q restricted to its hits equals the posterior, and near 0 an estimate can come out slightly negative.

## The fallback

The fallback is a sampler that always returns one fixed hit a, stored with the task. With ρ = p(a) / p̂(S), its capped forward KL is ρ log ρ + (1 − ρ) C and its reverse KL is −log ρ. A sampler that fails step 2 gets these scores, and its result has `fallback` set. `Benchmark.fallback_result(task)` returns them without a model:

```python
fallback = bench.fallback_result(task)
print(fallback.capped_forward_kl, fallback.reverse_kl)
```

The result of `fallback_result` scores the fallback as a sampler of its own, so besides `fallback` it has `passed_threshold` set and a `native_hit_rate` of 1: every draw of the fallback is a hit.

A run that ends without a sampler is scored as the fallback as well: `Benchmark.no_sampler_result(task, budget=B, seed=..., replicate=...)` at a cost equal to the run's budget B, or with `budget=None` and a ledger at what the run spent. From the command line, `posteriorbench evaluate --task <id> --sampler fallback` prints the fallback's result.

## The result

`bench.evaluate` returns an `EvaluationResult`. For the README's quickstart, `result.to_dict()` looks like this (some fields left out):

```json
{
 "task_id": "wc_1853/goblin_mention",
 "kind": "sampler",
 "replicate": 0,
 "draws": 1024,
 "hits": 414,
 "native_hit_rate": 0.404296875,
 "passed_threshold": true,
 "fallback": false,
 "capped_forward_kl": 14.589206465512763,
 "reverse_kl": 18.276529268147385,
 "construction_inferences": 0.0,
 "inferences_per_response": 1.0,
 "reference_logp": "shipped"
}
```

- `native_hit_rate` is the share of hits among the draws of step 1. Below 10% the sampler fails: `fallback` is true and the two KL scores are the fallback's (on this task a capped forward KL of 50.0 and a reverse KL of 150.9).
- `capped_forward_kl` and `reverse_kl` are the two estimates of step 4, in nats. The forward KL penalizes a sampler that makes the references unlikely; with the cap, a reference the sampler cannot produce costs 50 instead of infinity. The reverse KL penalizes hits of the sampler that are unlikely under the posterior.
- `construction_inferences` is the method's cost, the total of its ledger (here 0: a handwritten instruction is not charged).
- `reference_logp` says which log p of the references the forward KL used ([The log p of the references](#the-log-p-of-the-references)).

## Responses

A response is a list of generated token ids as generation returns it: 1 to 50 tokens, cut after the first `<|im_end|>` or `<|endoftext|>` or at 50 tokens, so a response shorter than 50 tokens ends with a stop token. Its likelihood includes that stop token. A shorter sequence without a stop token is a prefix, not a response: evaluation raises `SamplerError` for it, and nothing appends a stop token. Labels are computed on the response's text, `tokenizer.decode(token_ids, skip_special_tokens=True)`. The prompt that a response continues is `task.prompt_token_ids`.

## Samplers with a finite support

A sampler whose distribution is a known finite set of responses may return it from `finite_support()`; `pb.FiniteSampler` does. If every response in the set is a hit, the sampler is scored exactly over that set: its hit rate is 1, its forward estimate is the one of step 4, and its reverse KL is the exact expectation over the set. A set that contains a miss is not used, and the sampler is evaluated from draws like any other.

## Seeds

An evaluation with seed s calls `Sampler.sample(1024, seed=2**31 + s)`. A method that calls `Sampler.sample` while it builds its sampler uses seeds below 2**31 (`posteriorbench.sampler.construction_stream_seed`), so the evaluation draws are always new. Evaluate replicate r with seed r; the paper's main comparison uses three replicates (0, 1 and 2).

## Settings

`pb.EvaluationSettings` holds the number of draws (1,024), the threshold (10%), the number of hits scored for the reverse KL (128) and the cap C (50). Pass other values as `settings=` (a threshold as a `fractions.Fraction`) to study how a result depends on them, for example the cap:

```python
capped_at_20 = bench.evaluate(
    task, candidate, model, ledger=pb.Ledger(), seed=0, replicate=0,
    settings=pb.EvaluationSettings(cap=20.0),
)
```

Results that are compared must use the same settings.

## The log p of the references

The forward KL needs p(y) of the references. By default evaluation uses the values stored with the dataset (`reference_logp="shipped"`), so log p of the references does not depend on the machine or the library versions that evaluate; log q of the references and the hit rate come from the sampler and its draws. With `reference_logp="recomputed"` the model passed to `evaluate` computes them. `Benchmark.evaluate_both` returns both results from one set of draws:

```python
both = bench.evaluate_both(task, candidate, model, ledger=pb.Ledger(), seed=0, replicate=0)
print(both["shipped"].capped_forward_kl, both["recomputed"].capped_forward_kl)
```

The log p of the sampler's own hits, which the reverse KL needs, is always computed by the model. The log p of the fallback response is always the stored value.

## Averaging

To compare methods, average each method's results over replicates within a task, then over tasks with equal weights; the standard error is that of the task means. The command `posteriorbench aggregate` does this separately for each method and setting, and so does `posteriorbench.aggregate.aggregate` on each list of results in the dictionary that `aggregate.group_results(results)` returns: pass `method="..."` to `evaluate`, and `setting="..."` for a variant or a budget, so that your results are kept apart from other methods'. Both refuse a group of results that mixes caps, shipped and recomputed reference log p, or numbers of draws, and with `expected_tasks` / `--expect-tasks benchmark` a group that misses one of the 21 tasks.
