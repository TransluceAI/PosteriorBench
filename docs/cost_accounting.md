# Construction cost

A method's construction cost is counted in inferences of the original model and of samplers built from it. The package cannot meter compute done outside it, so a method charges what it computes to a `pb.Ledger` and passes the ledger to `Benchmark.evaluate`. The result reports the ledger's total as `construction_inferences` and its breakdown as `ledger`. Evaluation itself is never charged.

## What counts

- Generating one response costs one inference (`Category.GENERATION`). So does computing p(y) or q(y) of one response (`P_SCORING`, `Q_SCORING`) and the forward pass of a gradient step on one response (`TRAINING_FORWARD`).
- A sampler gets q(y) of a response it generates itself for free. p(y) is paid only on the responses where the method uses it.
- A hit taken from the corpus costs 1/p̂(S) inferences, the expected number of draws of the original model that rejection sampling spends per hit. `Benchmark.corpus(task, ledger)` charges it (`CORPUS_HITS`).
- A draw made to test a candidate against the hit-rate threshold costs what the candidate spends to generate one response (`THRESHOLD_TEST`): one for a plain sampler, two for a sampler that runs both a trained model and the original model.
- Not counted: backward passes and other passes that the inference count leaves out (`BUDGET_ONLY`, inferences 0), calls to other models such as an LLM that writes candidate prompts (`UNCOUNTED`, recorded only), and evaluation.
- A cost shared by several runs is charged in full to every task and replicate (`SHARED`).

A submitted sampler may spend at most two inferences to generate one response (`Sampler.inferences_per_response`: 1 by default; override the property for a sampler that runs two models). A mixture of samplers costs one.

## Charging

Every charge has an inference amount and a budget amount. The inference count is the cost the benchmark reports. The budget count is a second total for methods that train under a fixed budget B of passes: it also counts the passes that the inference count leaves out, such as backward passes, and `pb.Ledger(budget_limit=B)` stops a run that would exceed B. A method without such a budget can ignore it: `budget` defaults to `inferences`. The category of a charge only labels the breakdown.

```python
ledger = pb.Ledger()
ledger.charge(pb.Category.GENERATION, inferences=16 * 64, note="16 selection draws for each of 64 prompts")
ledger.charge(pb.Category.P_SCORING, inferences=37, note="p on the selection hits")
ledger.charge(pb.Category.THRESHOLD_TEST, inferences=164, note="threshold test of the first candidate")
ledger.charge(pb.Category.BUDGET_ONLY, inferences=0, budget=512, note="backward passes")
print(ledger.inferences, ledger.budget)
```

`pb.Ledger(budget_limit=B)` refuses a charge that would take the budget count beyond B (`posteriorbench.ledger.BudgetExceeded`). A training run that is evaluated at several budgets can store `ledger.totals()` with each checkpoint and continue from it with `pb.Ledger.resume(totals)`.

## Corpus hits

```python
ledger = pb.Ledger()
access = bench.corpus(task, ledger)   # reads the pair's hit file
hits = access.take_first(64)          # 64 hits, charged 64 / p̂(S)
print(len(hits[0].token_ids), ledger.inferences)
```

`bench.corpus` reads the hit file from the dataset on the Hugging Face Hub (one file, downloaded on first use) or, for `pb.Benchmark(hits_source="path/to/dataset")`, from a local copy. Each distinct hit is charged once per access object, and each call of `bench.corpus` returns a new access, so keep one per task and ledger. The hits are in random order, so the first n are a uniform sample. The pair's references are never handed out, and by default neither are hits whose token ids equal a reference's; a result built with `exclude_reference_duplicates=False` should say so.

## Reporting

Report `construction_inferences` and the ledger's breakdown with the scores, and say which costs your method had outside the ledger, if any.
