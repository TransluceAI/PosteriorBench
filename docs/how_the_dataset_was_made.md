# How the dataset was made

The dataset was made in six steps. This page gives the rule of each step and the script that states it. The scripts are in `tools/` of this repository, not in the installed package; run them from a clone, in its environment. [data.md](data.md) describes the files that the steps produced.

| Step | Rule | Checked against |
|---|---|---|
| 1. The prompts | `tools/choose_prompts.py` | the prompts of the task table (`--check`, with version 5.0.0 of the `datasets` library) |
| 2. The corpus | `tools/sample_corpus.py` | p̂(S) of the task table |
| 3. Labels and p̂(S) | `posteriorbench/rules.py` | the labels of the corpus (`tools/check_labels.py`) |
| 4. The scored pairs and the 21 tasks | `tools/choose_tasks.py` | the `scored` and `benchmark_task` columns of the task table (`--check`) |
| 5. The references | `tools/choose_references.py` | nothing: the references are fixed data |
| 6. The fallback responses | `tools/check_fallbacks.py` | the fallback responses of the evaluation file |

## 1. The prompts

The 20 prompts are first user turns of conversations in [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), at revision `7d6490e462285cf85d91eabea0f9a954fbddcd1f`.

1. The conversations are read as a shuffled stream: version 5.0.0 of the `datasets` library in streaming mode, with `shuffle(seed=0, buffer_size=50_000)`.
2. A conversation's first user turn is kept if it has 10 to 100 characters, WildChat labels it as English, all of its letters are ASCII, WildChat does not mark it as redacted, and no turn kept before it has the same text (compared stripped and casefolded). The first 2,000 kept turns are numbered `wc_0000` to `wc_1999` in the order in which they are kept.
3. The 20 prompts are `random.Random(20).sample` of the 2,000. The task table's `wildchat_conversation_hash` names the conversation of each.

```sh
uv run --with 'datasets==5.0.0' python tools/choose_prompts.py --check
```

prints the 20 prompts and compares their ids, texts and conversations with the task table. `datasets` is not a dependency of the package. The command reads part of WildChat-1M from the Hugging Face Hub, needs no login and takes about two minutes. The order of a streaming shuffle is decided by the `datasets` library; version 5.0.0 gives the dataset's 2,000 turns and its 20 prompts.

## 2. The corpus

For each prompt, 50 million responses of Qwen3.5-0.8B were sampled, every one an independent draw.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, bfloat16 weights, served by SGLang 0.5.10 |
| Prompt | the user turn as the only message, rendered with the model's chat template with thinking off and no system prompt: `prompt_token_ids` of the task table (`pb.OriginalModel.render_prompt` renders it) |
| Sampling | temperature 1, top-p 1, at most 50 new tokens, stopping at `<\|im_end\|>` or `<\|endoftext\|>` |
| Stored | the generated token ids and their text as the server decodes them |

```sh
uv run python tools/sample_corpus.py --server http://localhost:30000 --prompt-id wc_1853 \
    --responses 1000000 --out my_dataset
```

samples in the same way from a server of yours with SGLang's `/generate` interface, labels the responses (step 3) and writes them in the layout of the corpus, `my_dataset/corpus/wc_1853/part-*.parquet` with 1,000,000 rows per file. It prints each behavior's hit fraction next to the task table's p̂(S). The responses cannot be regenerated from a seed, so yours are new draws.

## 3. Labels and p̂(S)

Each behavior is a rule on the response text (`posteriorbench.rules.PAPER_RULES`, the `rule` column of the task table). Every response of the corpus has one label per behavior, and p̂(S) of a prompt–behavior pair is its number of hits divided by its prompt's 50,000,000 responses (`hit_count / num_responses` in the task table). `tools/check_labels.py` applies the rules to corpus rows again and compares the result with the stored labels.

## 4. The scored pairs and the 21 tasks

A pair is scored if it has at least 500 hits, which keeps the estimated relative standard error of p̂(S) below 5%; 108 of the 140 pairs are. The 21 tasks are three scored pairs of each behavior: of the behavior's scored pairs sorted by p̂(S) (pairs with equal p̂(S) by id), the first, the one at index `len // 2` and the last. With an even number of pairs, the median is the upper of the two middle ones.

```sh
uv run python tools/choose_tasks.py --check
```

computes both from the bundled task table and compares them with its `scored` and `benchmark_task` columns.

## 5. The references

Each scored pair has 250 references, chosen by position among its hits and never by content. Of the pair's first 5,000 hits in the order in which they were sampled (all of them if it has fewer), a random half was set aside with `numpy.random.default_rng(0)`, and the references are the first 250 hits of that half. A pair's hit file leaves out its references. The other hits of that half are in the hit file, like any hit that is not a reference.

The dataset's references are fixed data. The corpus is in random order within each prompt, not in the order sampled, so this choice cannot be repeated from the released files. `tools/choose_references.py` states the rule as a function of a pair's number of hits and prints the positions it gives:

```sh
uv run python tools/choose_references.py --hits 1837
```

For a new pair on one of the prompts, `tools/add_behavior.py` holds out the first hits in corpus order, which are a uniform sample as well ([data.md](data.md#adding-a-behavior)).

## 6. The fallback responses

Each scored pair has one fallback response: a hit written in advance by a language model from the prompt and the behavior. A fallback response is valid if its text is a hit of the pair's rule, its token ids are the tokens of the text followed by the end-of-turn token `<|im_end|>`, they are a complete response of at most 50 tokens, and decoded they are still a hit. The 108 fallback responses have 20 to 49 tokens.

```sh
uv run python tools/check_fallbacks.py
```

checks the fallback responses of the bundled evaluation file with the model's tokenizer, which it downloads on first use.
