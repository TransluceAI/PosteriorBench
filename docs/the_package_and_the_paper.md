# The package and the paper

This page relates the package to [the paper](https://arxiv.org/abs/2611.25735): what it covers, the runtime it uses, and where the paper's definitions are in the code.

## Scope

The package implements the paper's benchmark: the tasks on Qwen3.5-0.8B with seven rule-defined behaviors, the evaluation protocol and the cost accounting. It includes the original model and the Instruction sampler as reference samplers. The paper's other methods are not part of it, and neither are the paper's experiments with larger models and with behaviors judged by a language model. They are in [`methods/`](../methods) of this repository, a project of its own that depends on the package; [`methods/docs/paper_map.md`](../methods/docs/paper_map.md) gives the command for each figure and table of the paper.

## Runtime

`uv sync` installs the versions the package is tested with (`uv.lock`): torch 2.11.0, transformers 5.13.1 and numpy 2.3.5, with Python 3.12 (set in `.python-version`); Python 3.11 and later are supported. On Linux the lock takes torch from the CUDA 12.8 wheel index. `pip install .` also works and resolves the version ranges in `pyproject.toml`; on Linux it takes the PyPI build of torch, which needs a newer NVIDIA driver than the CUDA 12.8 build in the lock. `pb.OriginalModel` computes p with transformers in float32. Every result records its runtime in `scorer`: model commit, dtype, device, library versions.

The log p values stored with the dataset (of the references and of the fallback responses) were computed with another runtime, SGLang 0.5.10 in bfloat16, which also generated the corpus. The two runtimes give different values:

- With the default `reference_logp="shipped"`, the forward KL uses the stored log p of the references, which is the same for everyone. Log q of the references and the hit rate still come from your sampler and your draws.
- With `reference_logp="recomputed"`, your model scores the references. On the 250 references of `wc_1853/goblin_mention`, the recomputed values differ from the stored ones by 0.18 nats per reference on average (at most 0.70).
- The reverse KL always uses the log p of the sampler's own hits, computed by your model.
- The fallback's scores always use the stored log p of the fallback response.

Evaluation draws depend on the device, the library versions and the batch size (`pb.OriginalModel.from_pretrained(batch_size=...)`, 16 by default), so two machines give different draws for the same seed. Compare means over tasks and replicates, not single runs.

If the `flash-linear-attention` or `causal-conv1d` packages are installed, Qwen3.5's linear-attention layers run other kernels than without them; `scorer` records whether they were importable. Without them, transformers may print when the model loads that its fast path is not available; the package is tested without them.

## Where the paper's definitions are in the code

| In the paper | In the package |
|---|---|
| evaluation protocol (draws, threshold, likelihoods, the two estimates) | `posteriorbench.protocol.evaluate`, `posteriorbench.scoring`, `posteriorbench.estimators` |
| capped forward KL, reverse KL, cap C = 50, threshold 10% | `estimators.capped_forward_kl`, `estimators.reverse_kl`, `estimators.CAP`, `estimators.HIT_RATE_THRESHOLD` |
| fallback and its scores | `estimators.fallback_scores`, `Benchmark.fallback_result`, `Benchmark.no_sampler_result` |
| exact scoring of a sampler that replays stored hits | `scoring.score_finite_support`, `pb.FiniteSampler` |
| construction cost and the limit of two inferences per response | `posteriorbench.ledger`, `Benchmark.corpus`, `sampler.MAX_INFERENCES_PER_RESPONSE` |
| averaging over replicates and tasks | `posteriorbench.aggregate` |
| the seven behaviors and their rules | `posteriorbench.rules.PAPER_RULES`, the `rule` column of the task table |
| the 21 tasks | `Benchmark.tasks()` |
| the Instruction texts of the seven behaviors | `posteriorbench.instructions.INSTRUCTIONS` |
