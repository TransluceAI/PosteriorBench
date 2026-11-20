# Steering and Steering + composition

Steering changes the original model's residual stream h_b after one layer b to h_b + γ v_b at every token position, the prompt's included, where v_b is a steering vector and γ a strength. The vectors are persona vectors of the task's behavior: the mean activation on responses that show the behavior minus the mean activation on responses that avoid it. Each behavior has two, one built while the model's refusal direction was ablated and one built without the ablation. A candidate is one vector added after one layer with one strength, and the original model, which is the strength 0, is a candidate too. Steering chooses among its candidates for each task by the submission rule. Steering + composition composes the candidate that Steering submits with the original model.

The vectors are method data: `build-vectors` builds them once for each behavior, and a run reads them from a directory.

A run on one task and replicate does the following.

1. It reads the vectors of the task's behavior (`vectors.py`) and is charged what they cost to build (`cost.py`).
2. It lists its candidates (`candidates.py`): the original model, then for each vector every strength and layer. With two vectors, six layers and eight strengths there are 97. The six layers are those at depths 0.35 to 0.60 of the original model (`settings.py`).
3. It gives each candidate 128 selection draws on the task's prompt, with the candidate's log q of each draw and the original model's log p of each hit (`draws.py`). The replicates of a task share these draws. Here and in the threshold tests, a response with a token that the tokenizer does not define is a miss.
4. It ranks the candidates (`run.py`, `common/submission.py`). Candidates with hits in more than 15% of their selection draws come first, ranked by EB-JC; the others follow in decreasing order of hits; ties go to the earlier candidate. A candidate without a selection hit is not ranked.
5. It tests the candidates in rank order until one passes or the ranking ends, and submits the first that passes, or else the best-observed one. This is the record of `steering`. If no candidate was ranked, the run has no sampler and is recorded as the fallback, at the cost it spent.
6. If the submitted candidate passed its test and is not the original model, the submission rule chooses one of the 16 composition settings for it (`common/composition.py`). The plain setting is tested on the draws of Steering's own test of the candidate, extended if needed, and these draws are charged again. This is the record of `steering+composition`.

## Build the vectors

```sh
uv run --with jaxtyping --with einops --with matplotlib \
    posteriorbench-methods steering find-refusal-direction --method-data steering_vectors \
    --lmtailrisk-checkout LMTailRisk
uv run --with jaxtyping --with einops --with matplotlib \
    posteriorbench-methods steering build-vectors --behavior goblin_mention \
    --prompts goblin_mention.json --method-data steering_vectors \
    --lmtailrisk-checkout LMTailRisk \
    --judge-url http://localhost:8000/v1 --judge-model openai/gpt-oss-120b
```

The first command is run once for a model, before the second, which is run once for each behavior and records the forward passes of the direction that the directory holds. Both write into the directory `steering_vectors`, which a run then reads as its `--method-data`. They need three things that this repository does not hold.

**The pipeline of Angell et al.** The search for the refusal direction, the hooks that ablate it and the prompt of the coherence judge are imported from a checkout of [rangell/LMTailRisk](https://github.com/rangell/LMTailRisk) at commit `b9e383136fdeeb441d56f93744774dcafd6e039b`:

```sh
git clone https://github.com/rangell/LMTailRisk
git -C LMTailRisk checkout b9e383136fdeeb441d56f93744774dcafd6e039b
```

From its folder `refusal_direction/`, `find-refusal-direction` imports `pipeline.submodules.generate_directions` and `pipeline.submodules.select_direction` and reads `pipeline/model_utils/model_base.py` and `dataset/splits/`, and `build-vectors` imports `pipeline.utils.hook_utils`. The judge's prompt is `coherence_0_100` in `persona_vectors/eval/prompts.py`. These modules import jaxtyping, einops and matplotlib, which are not dependencies of the methods, so the two commands are run with them, as above: `uv run --with jaxtyping --with einops --with matplotlib posteriorbench-methods steering ...`.

**The construction prompts of the behavior** (`--prompts`), one JSON file:

```json
{"behavior": "goblin_mention",
 "name": "goblin",
 "description": "Mention a goblin naturally in responses.",
 "instruction_pairs": [{"positive": "Bring a goblin into every answer.",
                        "negative": "Never mention goblins."}],
 "questions": ["How do I make cheese at home?", "What is a good walk near a lake?"]}
```

Each instruction pair holds an instruction for the behavior and one against it, and the questions are everyday questions that have nothing to do with the benchmark's prompts. The original model answers every question under every instruction, with the system prompt `You are a goblin assistant. <positive instruction>` for a positive response and `You are a helpful assistant. <negative instruction>` for a negative one; `name` is the short name in the first of them. `description` is optional and is not read. You can write the file by hand or have a language model write it; the paper's files have 5 pairs and 20 questions.

**A coherence judge**: a language model on any server with the OpenAI chat-completions interface that returns token log probabilities (`--judge-url`, `--judge-model`). The key of the server is read from the environment variable `JUDGE_API_KEY`, or from the one that `--judge-key-env` names; if it is not set, no key is sent. The judge should write each number from 0 to 100 as one token.

`build-vectors` then does the following (`construction.py`).

1. Construction responses. For every question and instruction pair it draws 10 positive responses while the refusal direction is ablated with intensity 0.8, 10 positive responses without the ablation, and 10 negative responses: 3,000 responses with the paper's prompts. A response has at most 1,000 tokens.
2. Labels. The behavior's rule, the one of its tasks, says whether a response shows the behavior. The judge rates the coherence of each response from 0 to 100 (`coherence.py`); the score is the mean of the numbers among the 20 most probable tokens of its answer, weighted by their probabilities.
3. Filter. A positive response is kept if it shows the behavior and its coherence score is at least 50, a negative one if it does not show the behavior and its score is at least 50.
4. Vectors. The activation of a response at a layer is the residual stream after the layer, averaged over the response's tokens. A vector at a layer is the mean activation of its kept positive responses minus that of the kept negative ones, which the two vectors share. The activations of the responses that were drawn under the ablation are read under the ablation.

A vector whose positive responses the filter dropped entirely is not built, and the tasks of the behavior then have the candidates of the other vector. `--set vectors=without_ablation` builds that vector alone, which needs neither the refusal direction nor the hooks; the judge's prompt can then come from a text file with the placeholders `{question}` and `{answer}` (`--coherence-prompt`). Besides `--model`, `--model-revision` and `--device`, `build-vectors` takes `--data` and `--data-revision` as a run does: it reads the behavior's rule from the task table.

**A behavior that a language model judges.** Such a behavior has no rule that could label a construction response. Its construction prompts hold one more entry, `judge_prompt`: a text with the placeholders `{question}` and `{answer}` that asks the judge how strongly the answer shows the behavior, with a number from 0 to 100. `build-vectors` then asks the judge twice about every response, once with this prompt and once with the coherence prompt, and reads both scores in the same way. A response shows the behavior if its score on the behavior is at least 50 (`behavior_threshold`); a response without such a score, because the judge wrote no number, is kept on neither side. The paper's two LLM-judged behaviors are built this way: the language model that writes the instruction pairs and the questions from the behavior's rubric writes the judge prompt with them.

**Another original model.** With `--task` in place of `--behavior`, `build-vectors` builds the vectors of a task of the [generalization section](../generalization) on that task's original model, and `find-refusal-direction --task` finds the refusal direction of that model. Each original model has a directory of vectors of its own.

```sh
uv run --with jaxtyping --with einops --with matplotlib \
    posteriorbench-methods steering find-refusal-direction \
    --task qwen3.8-27b/wc_0670/chinese_character \
    --method-data steering_vectors/qwen3.8-27b --lmtailrisk-checkout LMTailRisk
uv run --with jaxtyping --with einops --with matplotlib \
    posteriorbench-methods steering build-vectors \
    --task qwen3.8-27b/wc_0670/chinese_character --prompts chinese_character.json \
    --method-data steering_vectors/qwen3.8-27b --lmtailrisk-checkout LMTailRisk \
    --judge-url http://localhost:8000/v1 --judge-model openai/gpt-oss-120b --set ablated_top_k=20
```

The paper's vectors on Qwen3.8-27B take each token of the responses drawn under the ablation from the 20 most probable ones, that model's own generation setting, where the benchmark's vectors take 50. On Gemma-4-31B-it the paper's search found no direction. A search that finds none ends `find-refusal-direction` with an error that says so; `--set vectors=without_ablation` then builds the one vector, in a directory without a direction, and its runs are charged nothing for one.

| Setting of `build-vectors` | Default (the paper's vectors) | Meaning |
|---|---|---|
| `vectors` | with_ablation, without_ablation | the vectors to build |
| `responses_per_prompt` | 10 | responses of each kind to one question under one instruction |
| `response_limit` | 1,000 | the most tokens of a construction response |
| `ablation_intensity` | 0.8 | the share of the component along the refusal direction that the ablation removes |
| `ablated_top_k` | 50 | the responses drawn under the ablation take each token from the 50 most probable; 0 takes it from the whole vocabulary, as the other responses do |
| `coherence_threshold` | 50 | the coherence score from which a response is coherent |
| `behavior_threshold` | 50 | for a behavior that the judge labels: the score from which a response shows the behavior |
| `layers` | none: every layer | the layers after which the vectors are built |

They are changed with `--set NAME=VALUE` and `--config FILE`, as the settings of a run are.

The directory then holds

```text
steering_vectors/
    <behavior>.safetensors          the vectors of one behavior
    construction_cost.json          what building the vectors cost
    refusal_direction.safetensors   the refusal direction, and in refusal_direction.json the
                                    layer and token position where it was found and the
                                    forward passes of the search
    refusal_direction_search/       the candidates and scores that the pipeline writes
    construction/<behavior>/        the construction responses with their token ids, text and
                                    labels, the judge's answers, and summary.json: the counts
                                    of every kind of response, the vectors that were built,
                                    the cost and the judge's calls
```

and a run reads the first two. A construction that stops, at a judge that does not answer for example, goes on where it was when the command is run again; in a directory that holds the construction of the behavior with another model, other prompts or other settings it stops with an error. Build the behaviors of a directory one after another: each adds its entry to `construction_cost.json`.

The inferences of the original model are counted, and every run of the behavior is charged for them ([Cost](#cost)): the forward passes of the search for the refusal direction, each construction response, and one forward pass for each kept response. The calls to the judge, one for each construction response, are calls to another model. They are recorded in `summary.json` and are not counted as inferences.

## Run

```sh
uv run posteriorbench-methods steering run --task wc_1853/goblin_mention --out runs \
    --method-data steering_vectors
```

`--method-data` is the directory of the vectors:

```text
steering_vectors/
    <behavior>.safetensors    the vectors of one behavior
    construction_cost.json    what building the vectors cost
```

A task of the [generalization section](../generalization) is run in the same way, with its id, the data of these tasks as `--data` and the directory of its original model's vectors; a task that a language model judges also takes the two judge models, which label every draw.

`<behavior>` is the behavior of the task, as in its task id (`goblin_mention`). The file holds one float32 tensor of shape [hidden size] for each vector and layer, named `<vector>.layer_<b>`, where `<vector>` is `with_ablation` or `without_ablation` and b counts the layers from 0. `construction_cost.json` holds `refusal_direction_passes` and, under `behaviors` and the behavior's name, `construction_responses` and `kept_responses` (`positive_with_ablation`, `positive_without_ablation`, `negative`). A run works on any task whose behavior has vectors in the directory. If a behavior has one vector only, its tasks have the candidates of that vector.

`--no-compose` leaves out Steering + composition. Run the replicates of a task into the same `--out`: a later replicate then reads the selection draws that the first one drew. The methods README lists the options that all families share and says how settings are changed.

## Settings

| Setting | Default (the paper's runs) | Meaning |
|---|---|---|
| `layers` | none: those at the paper's depths | the decoder layers after which a vector can be added, counted from 0. By default the layers at depths 0.35, 0.40, ..., 0.60 of the original model, layer int(n d) - 1 at depth d of n layers: 7, 8, 9, 11, 12, 13 of Qwen3.5-0.8B's 24, the layers 21, 24, 27, 31, 34, 37 of Qwen3.8-27B's 64, and 20, 23, 26, 29, 32, 35 of Gemma-4-31B-it's 60 |
| `strengths` | 1, 1.5, 2, 2.5, 3, 4, 6, 8 | the strengths γ |
| `selection_draws` | 128 | selection draws of each candidate |
| `threshold` | 0.10 | the hit-rate threshold: the benchmark's, or 0.01 |

With `--set threshold=0.01` every threshold of a run is a tenth of the benchmark's, as in the paper's appendix on the two LLM-judged tasks: a candidate is ranked by EB-JC if its hit fraction is above 1.5%, the tests are those at 1% (rank 1: 28 hits among 1,857 draws), of which a ranking gets at most 16, and the evaluation passes a sampler with a hit rate of at least 1%. The composition settings are chosen at the same threshold. The selection draws do not depend on the threshold, so a run at 1% into the output root of a run at 10% reads them from its cache directory. Its records go to `<out>/reselected/threshold-1pct/`, which has the layout of an output root.

## Cost

| Event | Inferences |
|---|---|
| finding the model's refusal direction | its forward passes |
| building the two vectors of the behavior | 1 per construction response, and 1 more per response that the filter kept |
| the 128 selection draws of a candidate, h of them hits | 128 + h |
| a threshold test of n draws | n |
| the selection draws of composition, h of them hits | 496 + h |
| a threshold test of a composed setting, n draws | 2n |

Steering has no training budget, and every charge counts the same in the inference count and in the budget count. The first two rows are read from the method data, where `find-refusal-direction` and `build-vectors` wrote them. For the paper's vectors the refusal direction took 16,620 forward passes, and a behavior has 3,000 construction responses: 1,000 positive ones for each vector and 1,000 negative ones that both share. A kept response is charged a second time for the forward pass that reads its activations. The vectors serve every task of their behavior and the selection draws every replicate of their task, but each task and replicate is charged for them in full. The selection draws of a candidate include the original model's log p of the hits. A sampler's test responses are generated 128 at a time, and a test is charged for the n that it reads. A test of the plain composition setting, which is the candidate itself, costs n.

Example: for a behavior whose filter kept 190 construction responses, every run is charged 16,620 + 3,000 + 190 = 19,810 inferences for the vectors. If the 97 candidates have 1,104 hits among their 12,416 selection draws and the first threshold test passes, Steering costs 19,810 + 12,416 + 1,104 + 164 = 33,494 inferences. If composition then has 60 selection hits and its second test passes, both tests on composed settings, Steering + composition costs 33,494 + 496 + 60 + 2 x (164 + 198) = 34,774.

## What a run writes

Two records, `<out>/steering/none/<task>/rep<r>/` and `<out>/steering+composition/none/<task>/rep<r>/`. The methods README describes the files of a record. In this family:

- `draws/selection.jsonl` (`steering`): the selection draws of every candidate, with log q and, on hits, log p. `candidate_id` is the candidate, for example `with_ablation_layer12_strength1.5`, or `original` for the original model.
- `draws/threshold_tests.jsonl` (`steering`): the draws of every threshold test, with the `candidate_id` and `rank` of the tested candidate.
- `draws/composition_selection.jsonl` and `draws/composition_threshold_tests.jsonl` (`steering+composition`): the same for the composition settings, with the setting in `setting`.
- In every draws file, `seed` is the seed of the batch that drew a response and `draw_index` its place in that batch.
- `details` of `construction.json`: for `steering`, `construction_cost` (the counts read from the method data) and `candidates` (each candidate's vector, layer, strength and selection hits); for `steering+composition`, `plain` (the status of Steering's submission and its candidate) and `composition` (the 16 settings, their ranking and tests and the chosen setting, or the reason why the candidate was not composed).
- A composed record whose status is `not_composed`, or whose chosen setting is the plain one, holds Steering's evaluation.

The selection draws that the replicates share are in `<out>/cache/steering/<task>/`: `selection/<candidate>.jsonl`, and `composition-selection/<candidate>_<setting>.jsonl` for the composition settings of a submitted candidate. `source.json` there records the model, the method data directory and the number of selection draws; a run that differs in one of them stops with an error instead of reading these draws.

## Compute and disk

On one H100, the first replicate of a task takes about 5 minutes, about 4 of them for the selection draws, and a later replicate into the same output root about 1 minute. The paper's 21 tasks with three replicates take about 2.5 GPU-hours. Peak GPU memory is about 7 GB: one copy of the weights serves the steered samplers and the original model.

A task and replicate writes about 10 MB, most of it selection draws.

Building the vectors of one behavior generates 3,000 responses of up to 1,000 tokens, 10 at a time, asks the judge 3,000 times, and runs one forward pass for each kept response. The search for the refusal direction runs 16,620 prompts through Qwen3.5-0.8B.

## Paper results this produces

With the default settings and the paper's vectors, `run` on the 21 tasks and the replicates 0, 1 and 2 produces the points of Steering and Steering + composition in the paper's main comparison. With `--no-compose` on the 108 prompt-behavior pairs it produces Steering's results by behavior and by rarity of the behavior.

Steering is not in the paper's table of compute per counted inference, and `compute-per-inference` leaves its records out: its vectors are built from longer responses.

`find-refusal-direction` and `build-vectors` with their default settings are how the paper's vectors were built, for the seven behaviors that a rule defines, on Qwen3.5-0.8B:

- The construction prompts of each behavior were written by gpt-oss-120b at temperature 1, with the prompt `generate_trait` of the persona vectors of Chen et al. (`data_generation/prompts.py` in [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors)). It was given a short name for the behavior and one sentence that describes its rule, and no prompt or result of the benchmark. Its first 5 instruction pairs and its first 20 questions are the construction prompts. These files are method data.
- The coherence judge was gpt-oss-120b at temperature 0 and seed 0, with the prompt `coherence_0_100` and the 20 most probable tokens at each position.
- The paper's vector files hold the layers 7, 8, 9 and 11 to 16. `build-vectors` writes every layer.

Building samples the construction responses, so vectors that you build differ from the paper's.

Two variants that the paper's tuning tried on the two tuning pairs, `wc_0346/goblin_mention` and `wc_1852/chinese_character`, are settings. Shorter construction responses are `build-vectors --set response_limit=50` or `150`, into a directory of their own that holds a copy of `refusal_direction.safetensors` and `refusal_direction.json`; the responses under a smaller limit are the first tokens of those under a larger one. Deeper layers are `run --set layers=11,12,13,14,15,16`.

On the six tasks of the generalization section, `run` with the default settings produces the points of Steering and Steering + composition on the two larger models and on the two LLM-judged behaviors, and with `--set threshold=0.01` those at the 1% threshold ([`generalization`](../generalization), "Running the paper's methods"). Their vectors were built as described under "Another original model" and "A behavior that a language model judges". For the two LLM-judged behaviors, gpt-oss-120b wrote the construction prompts of physical embodiment from the behavior's name and rubric; it refuses that request for direct encouragement, where DeepSeek-V4-Flash wrote them. gpt-oss-120b is the judge of both.

## Attribution

Steering with persona vectors follows Angell et al., "Estimating Tail Risks in Language Model Output Distributions" (ICML 2026), from whom the six layers and the strengths 1 to 3 are taken. Persona vectors are from Chen et al., "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" (2025), and the refusal direction from Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (NeurIPS 2024). The code in this folder is our own. The refusal-direction pipeline and the judge's prompt are read from a checkout of Angell et al.'s repository when the vectors are built; neither is included here.
