# PosteriorBench

[📄 Paper](https://arxiv.org/abs/2611.25735) · [🤗 Dataset](https://huggingface.co/datasets/Transluce/PosteriorBench)

PosteriorBench is a benchmark for sampling rare behaviors of a language model. A task is a prompt and a behavior the model shows only rarely in response to it: Qwen3.5-0.8B, for example, mentions a goblin in 0.08% of its responses to one of our prompts. The goal is a sampler that produces such responses (*hits*) often, and with the same distribution as the model's own hits, which we call the *behavior posterior*. A sampler is scored by its hit rate, its forward and reverse KL divergence from the posterior, and how many model inferences it took to build.

The benchmark has 21 tasks on Qwen3.5-0.8B, whose behavior probabilities $\hat{p}(S)$ range from $10^{-5}$ to $0.1$. We sampled 50 million responses for each prompt, so every task comes with real posterior samples: some are held out for evaluation, and the rest are available to methods as training data, at a cost.

The package at the root of this repository, `posteriorbench`, is all you need to evaluate a sampler of your own. The code for the methods we compare in the paper, and the scripts behind its figures and tables, are in [`methods/`](methods).

> [!NOTE]
> PosteriorBench includes sensitive content: offensive language and racial slurs, model responses that discuss or encourage suicide and self-harm, and false statements.

## Setup

```sh
uv sync
```

Evaluation runs Qwen3.5-0.8B, so we recommend a GPU. [docs/the_package_and_the_paper.md](docs/the_package_and_the_paper.md#runtime) gives the versions `uv sync` installs and how to install with pip.

## Quickstart

```python
import posteriorbench as pb

bench = pb.Benchmark()                           # the bundled task table and evaluation file
task = bench.task("wc_1853/goblin_mention")      # prompt, behavior, rule, p̂(S)
model = pb.OriginalModel.from_pretrained()       # downloads Qwen3.5-0.8B on first use

candidate = pb.instruction_sampler(model, task)  # a simple sampler; put yours here
result = bench.evaluate(task, candidate, model, ledger=pb.Ledger(), seed=0)
print(result.native_hit_rate, result.capped_forward_kl, result.reverse_kl)
```

This evaluates a simple baseline, the model with an instruction that asks for the behavior, and prints its hit rate, capped forward KL and reverse KL (lower is better for both). On an H100 it takes about a minute and a half and prints roughly `0.40 14.6 18.3`.

[`examples/01_quickstart`](examples/01_quickstart) goes through the same steps in a notebook.

## Evaluating your own sampler

A sampler draws responses to the task's prompt and can say how likely any response is under it:

```python
import posteriorbench as pb


class MySampler(pb.Sampler):
    name = "my-sampler"             # recorded in the result

    def sample(self, n, *, seed):   # n responses, each a list of generated token ids
        ...

    def logprobs(self, responses):  # log q(y) of each response; -inf if q cannot produce it
        ...
```

The `ledger` you pass to `bench.evaluate` records what your method spent to build the sampler, in model inferences. Posterior samples to train on come from `bench.corpus(task, ledger)`, which charges each hit at $1/\hat{p}(S)$ inferences, what rejection sampling would have cost. [`examples/02_custom_sampler`](examples/02_custom_sampler) is a small complete method. For the details, see [docs/protocol.md](docs/protocol.md) (what a response is, how the scores are computed) and [docs/cost_accounting.md](docs/cost_accounting.md) (what is charged).

## Running the benchmark

The benchmark is the 21 tasks with three replicates each: 63 evaluations, about an hour and a half on one H100. Continuing from the quickstart:

```python
import json

with open("my_method.jsonl", "w") as out:
    for task in bench.tasks():  # the 21 tasks
        for r in (0, 1, 2):  # three replicates
            ledger = pb.Ledger()
            sampler = pb.instruction_sampler(model, task)  # build yours here and charge `ledger`
            result = bench.evaluate(task, sampler, model, ledger=ledger, seed=r, method="my-method")
            out.write(json.dumps(result.to_dict()) + "\n")
```

```sh
uv run posteriorbench aggregate my_method.jsonl --expect-tasks benchmark --expect-replicates 0,1,2
```

`aggregate` averages over replicates within a task and then over tasks, and prints the mean and standard error of each score. A sampler fails a task if fewer than 10% of its draws are hits. It then gets the fallback's scores on that task, and they count in the average.

When you report results, please give the construction cost next to the two KLs, mention any cost your method had outside the ledger, and use the same evaluation settings on every task.

The same steps work from the command line (`posteriorbench <command> --help` lists the options):

```bash
uv run posteriorbench list-tasks
uv run posteriorbench evaluate --task wc_1853/goblin_mention --sampler instruction --out instruction_r0.json
uv run posteriorbench aggregate instruction_r0.json --expect-replicates 0
```

## Data

The [dataset](https://huggingface.co/datasets/Transluce/PosteriorBench) holds 50 million labeled responses of Qwen3.5-0.8B for each of 20 prompts (146 GB), and the files the benchmark reads. Listing tasks and evaluating need no data download: the task table and the held-out references come with the package. `bench.corpus` downloads a task's hits the first time you use it.

[docs/data.md](docs/data.md) describes the files, [docs/how_the_dataset_was_made.md](docs/how_the_dataset_was_made.md) how we built them, and [docs/the_package_and_the_paper.md](docs/the_package_and_the_paper.md) where each definition of the paper is in the code.

## Citation

```bibtex
@article{xun2026posteriorbench,
  title={PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models},
  author={Zhiyang Xun and Daniel D. Johnson and Sarah Schwettmann and Neil Chowdhury},
  year={2026},
  eprint={2611.25735},
  archivePrefix={arXiv}
}
```

## License

The code is released under the MIT license ([LICENSE](LICENSE)). The two data tables in [`posteriorbench/dataset`](posteriorbench/dataset) are released under the Open Data Commons Attribution License 1.0 ([posteriorbench/dataset/LICENSE](posteriorbench/dataset/LICENSE)), the license of WildChat-1M, whose prompts the dataset uses.
