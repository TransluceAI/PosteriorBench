# The data

The dataset holds, for each of 20 prompts, 50 million responses of Qwen3.5-0.8B with seven behavior labels, and the files the benchmark reads. Every prompt paired with every behavior is a prompt–behavior pair: a `Task` in the package, with id `<prompt_id>/<behavior>`. The 108 pairs with at least 500 hits are scored (`task.scored`), and 21 of them are the benchmark's tasks (`task.benchmark_task`): for each behavior, the scored pairs with the lowest, median and highest p̂(S).

## Files

| File | What it holds | Read by |
|---|---|---|
| `tasks/tasks.parquet` | the 140 pairs: prompt and its token ids, behavior and rule, `num_responses`, `hit_count`, p̂(S) = `hit_count / num_responses`, `scored`, `benchmark_task` | `Benchmark.tasks`, `Benchmark.task` |
| `evaluation/evaluation.parquet` | for each scored pair: the 250 references and the fallback response, with their stored log p | the evaluation methods only |
| `hits/<prompt_id>/<behavior>.parquet` | every hit of a scored pair except its references, in corpus order (3.4 GB in all) | `Benchmark.corpus` |
| `corpus/<prompt_id>/part-*.parquet` | the 50 million labeled responses of each prompt (146 GB in all) | `posteriorbench.data.iter_corpus`, for defining new behaviors |

The first two files are bundled with the package, so `pb.Benchmark()` lists tasks and evaluates samplers without a download. `pb.Benchmark(source)` reads all files from one place instead: a local directory with this layout or a Hub dataset id. A Hub source is read at the dataset version this package reads (`posteriorbench.data.DATASET_REVISION`) unless `revision=` names another. Every result records where its tables came from (`dataset`).

The column names and types of the first three kinds of files are `TASKS_SCHEMA`, `EVALUATION_SCHEMA` and `HITS_SCHEMA` in `posteriorbench.data`. The dataset card describes every column. [how_the_dataset_was_made.md](how_the_dataset_was_made.md) says how the prompts were chosen, how the corpus was sampled and labeled, how the scored pairs, the tasks and the references were chosen, and what makes a fallback response valid.

## What is held out

The task table holds only what a method may see. The references, the fallback response and their stored log p are in the evaluation file, which only the evaluation methods read. A pair's hit file does not hold its references. By default `bench.corpus` also leaves out the hits whose token ids equal a reference's; these are frequent short responses (134 hits on 11 pairs).

The separation is per pair. A response can be a reference of one pair and a hit of another pair on the same prompt, so the other pair's hit file can contain it; this happens on 99 of the 108 scored pairs. The corpus holds every response, references included. Build a pair's sampler only from `bench.corpus(task, ...)` of that pair.

## Reading the corpus

```python
from posteriorbench import data

rows = data.iter_corpus(bench.hits_source, task.prompt_id, columns=["row", "text", task.behavior])
first = next(iter(rows))
print(first["row"], first[task.behavior])
```

`iter_corpus` streams one prompt's rows in file order. Within a prompt the rows are in random order, so any prefix is a uniform sample of the prompt's responses. From the Hub it downloads the prompt's 50 corpus files first.

## Adding a behavior

Any rule on the response text defines a new pair on one of the 20 prompts without sampling the model again: the matching rows of the prompt's corpus are samples of its posterior, and their share is p̂(S). `tools/add_behavior.py` in this repository builds a dataset directory for such a pair, which `pb.Benchmark(directory)` then evaluates samplers on. It holds out the first hits in corpus order as references, computes their log p and the log p of the fallback response you give with the original model, and writes the pair's task table, evaluation file and hit file. The tool reads the prompt's whole corpus and keeps the pair's hits in memory.

```bash
uv run python tools/add_behavior.py --help
```
