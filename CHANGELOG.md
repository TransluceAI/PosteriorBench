# Changelog

## 1.0.0

The package at this version:

- `posteriorbench.Benchmark`: the benchmark's 21 tasks (and the 108 scored pairs, and all 140 prompt–behavior pairs), corpus hits charged to a ledger, and evaluation. The task table and the evaluation file are bundled, so listing tasks and evaluating a sampler need no download.
- `Sampler`: the interface a method submits (`sample`, `logprobs`, `inferences_per_response`); `FiniteSampler` for samplers with a known finite support.
- `EvaluationResult`: native hit rate, capped forward KL, reverse KL, construction cost, and the record of how they were computed. `reference_logp` says whether the forward KL used the log p of the references shipped with the dataset or recomputed it; `kind` says whether a sampler, a run without a sampler or the fallback was scored.
- `Ledger`: the counting rules for construction cost.
- `OriginalModel` (Qwen3.5-0.8B with Hugging Face transformers) and the reference samplers `original_model_sampler`, `system_prompt_sampler` and `instruction_sampler`.
- `posteriorbench.scoring`: the benchmark's arithmetic as functions of numbers.
- The command-line tool `posteriorbench` with `list-tasks`, `evaluate` and `aggregate`.
- `tools/` (in the repository, not in the installed package): building hit files, checking labels and adding a behavior.
- `tools/` also states how the dataset was made, step by step in `docs/how_the_dataset_was_made.md`: choosing the prompts, sampling the corpus, choosing the scored pairs, the tasks and the references, and checking the fallback responses.

It reads dataset version v1.0.0.
