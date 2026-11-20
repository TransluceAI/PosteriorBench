# Custom sampler

[`custom_sampler.py`](custom_sampler.py) builds and evaluates a sampler of your own on one task. Its method takes two handwritten instructions that ask for a goblin (`CANDIDATES`; write your own for another behavior), draws 32 responses from the original model under each (as a system prompt), and keeps the instruction with more hits. It charges those 64 generated responses to a `posteriorbench.Ledger`, which evaluation reports as the construction cost. Replace it with your method, and charge everything it computes (see [docs/cost_accounting.md](../../docs/cost_accounting.md)).

```bash
uv run python examples/02_custom_sampler/custom_sampler.py --task wc_1853/goblin_mention
```

The command-line tool evaluates the same method through `build_sampler(task, model, benchmark)`, which returns the sampler and its ledger, imported from this directory:

```bash
PYTHONPATH=examples/02_custom_sampler uv run posteriorbench evaluate \
    --task wc_1853/goblin_mention --sampler custom \
    --factory custom_sampler:build_sampler --out result.json
```

## Samplers that need no class

`pb.system_prompt_sampler(model, task, text)` is the model under a system prompt of yours. For a model of your own with the benchmark's tokenizer, a fine-tuned copy for example, `pb.OriginalModel(your_model, tokenizer)` wraps a transformers causal LM (it gives `generate`, `logprobs`, `decode` and `render_prompt`), and `pb.original_model_sampler(wrapped, task)` is that model on the task's prompt.

To label a draw while you build a sampler, apply the task's rule to its text: `pb.rules.parse_rule(task.rule)(model.decode(token_ids))`. Labels cost nothing.
