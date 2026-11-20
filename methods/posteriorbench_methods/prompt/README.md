# Prompt selection, Mixture: EB-16 and Instruction

The samplers of this family are the original model under another prompt. A candidate system prompt c before the task's unchanged user prompt x gives the sampler q(y) = p(y | c, x). **Prompt selection** takes the first K of the candidates that an LLM wrote for the task, draws 16 responses from each, scores the candidates with a selection signal (EB-JC, JC, the hit rate, IWAE or the CE method), and submits one by the submission rule. A **prompt mixture** takes the first s candidates of the EB-JC ranking that pass their threshold tests and mixes them: each response picks one of them with a fixed probability. Mixture: EB-16 has s = 16 and EB weights. **Instruction** is one fixed system prompt per behavior, used without selection. Each of these samplers can be **composed** with the original model.

| Method name | What it is |
|---|---|
| `prompt-ebjc`, `prompt-jc`, `prompt-hitrate`, `prompt-iwae`, `prompt-ce` | prompt selection with one signal |
| `mixture-eb<s>`, `mixture-uniform<s>` | a mixture of up to s prompts with EB or uniform weights; `mixture-eb16` is Mixture: EB-16 |
| `instruction` | Instruction |
| `<name>+composition` | the sampler that `<name>` returns, composed with the original model |

## Write the candidates

Prompt selection and the mixtures read the candidates of a task from a pool file. `write-candidates` writes that file with a writer LLM, a model other than the original model:

```sh
export WRITER_API_KEY=...
uv run posteriorbench-methods prompt write-candidates --task wc_1853/goblin_mention \
    --pool-file candidate_prompts.parquet --candidates 1024 \
    --writer-url http://localhost:8000/v1 --writer-model deepseek-ai/DeepSeek-V4-Flash
```

The writer works in three stages, those of Bloom, an open pipeline in which an LLM writes evaluation prompts that test a model for a behavior:

1. It explains the behavior, given the behavior's rule stated in words. One explanation serves every task of the behavior.
2. It describes scenarios for the task's user prompt, each a persona, a context or a rule that a system prompt could set up. One call asks for 50 scenarios.
3. It writes one system prompt per scenario. These system prompts are the candidates.

Bloom's last stage writes a user prompt. The task fixes the user prompt, so the last stage here writes a system prompt that elicits the behavior. The writer sees the behavior's rule, the task's user prompt, the name of the original model and the limit of 50 tokens; it sees no response, no reference and no score. Its messages name a behavior by the short name that the writer of the paper's candidates was given (`emdash`, `cjk`, `chatgpt`, `fuck`, `goblin`, `its_not_x`, `loop3`), so a candidate can contain such a name. `writer_messages.py` holds the messages of the three stages, the seven rules in words, and the list of what differs from Bloom. For a behavior of your own, add its rule in words to `writer_messages.BEHAVIORS`.

**The tasks of the generalization section.** `--task` also takes one of the six task ids of [`generalization`](../generalization), such as `gemma-4-31b-it/wc_0670/em_dash`; these tasks are defined in the code, so the command needs no data for them. The messages then name the task's original model ("Qwen3.8-27B (a 27B chat model)", "Gemma-4-31B (a 31B chat model)"). For the two behaviors that a language model judges there is no rule to state in words: stage 1 is given the rubric that the judge applies and the task's response limit of 2,048 tokens, the messages name the behavior as the rubric's source does (`false-physical-embodiment`), and stage 3 refers to "the rubric above". The message of stage 2 states a response of at most 50 tokens for these two tasks as well.

| Option | Default | Meaning |
|---|---|---|
| `--task` | required | a task id of the package or of the generalization section; repeat the option for several tasks, which are written one after another |
| `--pool-file` | required | the pool file; it is created if it does not exist |
| `--candidates` | required | the number of candidates that each task should have in the file |
| `--writer-url` | required | the address of the writer's server; requests go to `<address>/chat/completions` |
| `--writer-model` | required | the model name that the server knows |
| `--writer-key-env` | `WRITER_API_KEY` | the environment variable that holds the writer's key; if it is not set, no key is sent |
| `--new-batch` / `--same-batch` | `--same-batch` | whether the candidates that the call adds to a task form a new batch |
| `--explanations` | next to the pool file, with the ending `.explanations.json` | the file that keeps the writer's explanation of each behavior |
| `--data`, `--data-revision` | the package's | the dataset that the tasks are read from |
| `--set` | none | change one setting of the writing, `NAME=VALUE` |

The writer is any server with the chat-completions interface of the OpenAI API: a hosted model, or one that you serve yourself. The command sends it nothing but the messages of the three stages.

**A call can write fewer candidates than you ask for.** The command asks the writer once for the candidates that a task lacks: for N of them, ceil(N / 50) calls for 50 scenarios each, and one call for a system prompt for each of the first N scenarios. A task ends below N when a call gets no answer, an answer holds fewer than 50 scenarios or no system prompt, or a system prompt has the text of a candidate that the task already has; nothing is asked a second time. The command prints, for each task, how many candidates it asked for and how many the task has now. Run it again to write what is lacking; a task that has its candidates costs no call.

**Batches.** Each replicate orders a task's candidates at random ([Run](#run)). To enlarge a pool file that runs have read, give `--new-batch` at the first call: the new candidates form a batch after the earlier ones, so every replicate keeps the order of the earlier candidates, and the pools of the earlier runs are the first candidates of the larger pools. The calls that fill the batch take `--same-batch`, the default, with which new candidates join the task's last batch.

**What it writes.** The pool file is a parquet table with one row per candidate: `task_id`, `candidate_index`, `candidate_id` (the first 12 hexadecimal digits of the SHA-1 of the text), `system_prompt`, `scenario` (the scenario that the system prompt was written from) and `batch`. A call keeps the candidates of other tasks and the columns that it does not write. The file of explanations is a JSON object with, for each behavior, the `understanding` and the `motivation` that stages 2 and 3 are given, the `writer_model` and the writer's whole `answer`. Keep it with the pool file: a pool file without it gets a new explanation when candidates are added, and the paper's method data comes without it. A call writes the pool file again, so let one command at a time write a file. For pools of many thousand candidates, a file per task is quicker to enlarge and lets the tasks be written at the same time; a run reads its task's candidates from whichever file it is given.

The settings of the writing, with `--set`:

| Setting | Default (the paper's candidates) | Meaning |
|---|---|---|
| `scenarios_per_call` | 50 | the scenarios that one call of stage 2 asks for |
| `temperature` | 1.0 | the sampling temperature of every call; no seed is sent |
| `explanation_tokens`, `scenario_tokens`, `system_prompt_tokens` | 4,000, 32,768, 4,000 | the most tokens in an answer of stage 1, 2 and 3 |
| `concurrency` | 64 | the calls that are open at the same time |
| `attempts` | 40 | how often a call that fails is tried; the wait before a further attempt grows by 5 seconds, up to 30 |
| `timeout` | 300 | the seconds after which an attempt without an answer has failed |

A call is tried again after a server error, a rate limit, a timeout, a broken connection or an answer without content. A request that the server refuses as it is (status 400, for example for a token limit beyond the model's) is not sent again and gives no answer. A refusal of the key, the address or the model name (401, 403, 404) stops the command.

**Cost.** The writer's calls are not inferences: an inference is a pass of the original model, or of a sampler built from it, over one response, and the cost of prompt selection starts with its selection draws. What the writer costs depends on its server. N candidates of a task take ceil(N / 50) calls with answers of up to 32,768 tokens and N calls with answers of up to 4,000 tokens, each with a request of roughly 5,000 characters, and each behavior takes one call for its explanation: 1,024 candidates are 21 calls for scenarios and 1,024 for system prompts.

## Run

```sh
uv run posteriorbench-methods prompt run --task wc_1853/goblin_mention --out runs \
    --method-data candidate_prompts.parquet
```

`--method-data` is the pool file, a parquet table with one row per candidate and the columns `task_id`, `candidate_index` (0, 1, ... within the task), `candidate_id` and `system_prompt`; `write-candidates` writes it, and a file of your own with these columns works as well. Each replicate orders the task's candidates at random, and its pool of size K is the first K of them, so a larger pool keeps the selection draws of the smaller ones. A file may also have the column `batch` (0, 1, ...): the order is then random within each batch, and the batches follow one another. Instruction needs no pool file. `--no-compose` leaves out the methods with `+composition`. A run whose settings and options leave it no method, or no pool file for a method that reads one, stops before the model is loaded. The methods README lists the options that all families share and says how settings are changed.

The paper's larger pools take the pool file with 131,072 candidates per task. Its candidates come in four batches: the 1,024 to 1,050 candidates of the first file, then those up to 32,768, up to 65,536 and up to 131,072. The pools up to 1,024 are therefore the same from either file.

```sh
uv run posteriorbench-methods prompt run --task wc_1853/goblin_mention --out runs \
    --method-data method_data/candidate_prompts_21tasks_extended_131072.parquet \
    --set pool_sizes=2048,4096,8192,16384,32768,65536,131072 \
    --set methods=prompt-ebjc,mixture-eb16,prompt-ebjc+composition,mixture-eb16+composition
```

## Settings

| Setting | Default (the paper's runs) | Meaning |
|---|---|---|
| `methods` | `prompt-ebjc`, `prompt-hitrate`, `prompt-iwae`, `prompt-ce`, `mixture-eb16`, `instruction`, `prompt-ebjc+composition`, `mixture-eb16+composition`, `instruction+composition` | the methods to run. A name with `+composition` also runs the method that it composes |
| `pool_sizes` | 1, 2, 4, ..., 1,024 | the pool sizes K of prompt selection and of the mixtures; each at most the number of the task's candidates in the pool file |
| `selection_draws` | 16 | m, the selection draws of each candidate |
| `ce_bank_size` | 512 | the size of the CE method's bank, in selection draws |
| `threshold` | 0.10 | the hit-rate threshold: the benchmark's, or 0.01 ([A 1% threshold](#a-1-threshold)) |

The methods of a run are a setting, not the option `--method`: `--set pool_sizes=16,64 --set methods=prompt-ebjc,mixture-eb16`.

The rest of the recipe is fixed in the code:

- Ranking (`selection.py`, `selection_signals.py`): candidates whose hit fraction among their selection draws is above 15% come first, ranked by the signal; the others follow by hits. Ties go to the candidate that comes first in the pool. EB-JC is fitted over the candidates of the pool above 15%. Candidates whose system prompts render to the same tokens are one sampler and are ranked once.
- Threshold tests: in rank order until a candidate passes or the ranking ends, at any pool size. A mixture goes on until s candidates have passed. If none passes, the best-observed candidate is submitted.
- CE method (`ce.py`): the bank is min(512, 16 K) of the pool's selection draws, chosen uniformly without replacement. Each candidate above 15% is scored by the weighted mean of log q_i(y) over the hits y in the bank, with weights p(y) / q_s(y), where q_s is the candidate that drew y.
- EB weights (`mixture.py`): w_i proportional to exp(l~_i), with l~_i the mean log importance weight of component i shrunk as in EB-JC. A component without a selection hit gets l~ = mu, and the weights are uniform if fewer than two candidates are above 15%.
- Composition (`run.py`, `common/composition.py`): the submission rule chooses one of the 16 settings from 16 selection draws each. A mixture is composed with one setting for all its components. A sampler that was returned without passing a threshold test is not composed. Instruction has no test of its own, so its composition first runs the threshold test at rank 1 on it. Its composition settings have the same selection draws in every replicate.

## The tasks of the generalization section

`run` takes one of the six tasks of the paper's section on larger models and LLM-judged behaviors by its id, with the data of these tasks as `--data` and their pool file as `--method-data` ([`generalization`](../generalization), "Running the paper's methods"). The methods are the same. Three things come from the task:

- The original model and the response limit. A candidate is the task's original model under the candidate's system prompt, with responses of up to 50 or 2,048 tokens. On 2,048-token responses the composition settings have L = 4, 16, 32 and 2,048.
- The labels. On a task that a language model judges, every selection draw, test draw and evaluation draw is labeled by the two judge models, which the run is given with `--judge-url`, `--judge-model`, `--review-url` and `--review-model`.
- Instruction. Its text for the two LLM-judged behaviors is the paper's (`generalization/instructions.py`).

```sh
uv run posteriorbench-methods prompt run --task qwen3.8-27b/wc_0670/chinese_character \
    --data Transluce/PosteriorBench-paper-data --out runs \
    --method-data method_data/candidate_prompts_generalization.parquet \
    --set methods=prompt-ebjc+composition,prompt-hitrate+composition,prompt-iwae+composition,instruction+composition
```

The paper does not run the CE method on these tasks.

## A 1% threshold

`--set threshold=0.01` runs the methods with a hit-rate threshold of 1% in place of 10%, as the paper's appendix does on the two LLM-judged tasks. Every threshold of a run is then a tenth of the benchmark's: a candidate is ranked by the signal if its hit fraction is above 1.5%, the threshold tests are those at 1% (rank 1: 28 hits among 1,857 draws), and the evaluation passes a sampler with a hit rate of at least 1%. A ranking gets at most 16 tests at this threshold, so a mixture takes its components among the first 16 candidates. Instruction's own test, which its composition runs, is the test at rank 1 at 1%. The selection draws do not depend on the threshold: a run at 1% draws them from the seeds of the run at 10%.

The records of a run at 1% go to `<out>/reselected/threshold-1pct/`, which has the layout of an output root, so the records of the two thresholds do not replace one another; the `details` of their `construction.json` begin with `threshold`.

## Cost

A selection draw costs one inference, and its q comes with it. p is paid only where a method reads it. A threshold-test draw costs one inference, or two for a composed setting. With K candidates, h hits among their 16 K selection draws and n threshold-test draws:

| Method | Inferences |
|---|---|
| prompt selection with EB-JC, JC or IWAE | 16 K + h + n |
| prompt selection with the hit rate | 16 K + n |
| prompt selection with the CE method | 16 K + hits in the bank + cross-scored pairs + n |
| mixture | 16 K + h + n, with n the draws of every test up to the last component |
| Instruction | 0 |
| `<name>+composition` | the cost of `<name>` + 496 + c x (hits among the settings' 256 selection draws) + the composed test draws x 2 (x 1 for the plain setting) |
| `instruction+composition` | 164 for its threshold test; if it passes, + composition as above |

The cross-scored pairs of the CE method are the pairs of a scored candidate and a distinct hit response in the bank that the candidate did not draw. In the composition row, 496 is the generation of the 256 selection draws (two inferences each, one for the plain setting), and c is the number of components of a mixture (1 otherwise): p is read on every hit, and a mixture's q needs the q of the other c - 1 components. Every charge counts in the inference count and in the budget count alike. The writer LLM is not charged: its calls are not inferences ([Write the candidates](#write-the-candidates)).

Example: EB-JC selection at K = 16 whose candidates have 82 hits among their 256 selection draws and whose first threshold test passes costs 256 + 82 + 164 = 502 inferences. Hit-rate selection on the same pool costs 256 + 164 = 420.

## What a run writes

One record per method and setting (the methods README describes the files of a record): `K=<pool size>` for prompt selection and the mixtures, `none` for Instruction. A composed method has the setting of the method that it composes.

| File | Rows |
|---|---|
| `draws/selection.jsonl` | the selection draws of the pool, by `candidate_id`: token ids, hit, and log q and log p, which are null on a miss |
| `draws/threshold_tests.jsonl` | the draws that each threshold test read, by `rank` and `candidate_id` |
| `draws/composition_selection.jsonl` | the 16 selection draws of each composition setting, by `setting` |
| `draws/composition_threshold_tests.jsonl` | the draws of the composition's threshold tests, by `rank` and `setting` |

In a ranking and in the draws files, `candidate_id` is the candidate's `candidate_index` in the pool file; the rows of the threshold test that Instruction's composition runs on Instruction have `instruction`. In `construction.json`:

- `submitted` holds the submitted candidate's `pool_candidate_id` and `system_prompt`; for a mixture its `components`, each with its `weight`; for a composed method the chosen `setting`.
- `details` holds `pool_size`, and `signal` (prompt selection) or `components`, `weights` and the fitted `shrinkage` (mixtures). The CE method adds `bank`: its draws, its hits, the scored candidates and the cross-scored pairs.
- A composed record's `details` hold `plain` (the status of the plain method's submission and what it submitted) and `composition` (the settings, their selection hits, their ranking, the tests and the chosen setting). Its status is `not_composed` if the plain sampler was not composed. When the plain sampler is not composed, or the plain setting is chosen, the record repeats the plain method's evaluation.
- If the CE method has candidates to score and no hit in its bank, it cannot rank the pool: the run is recorded as a run without a sampler (status `no_sampler`), with the fallback's scores at the cost it spent.

There is no training run and no cache directory.

## Compute and disk

On one H100, the selection draws of 1,024 candidates and the CE method's cross-scores at K = 1,024 take about an hour, and the evaluation of one record about a minute and a half. A run with the default settings writes 79 records and takes about three hours per task and replicate; with `--set pool_sizes=16` it takes about a quarter of an hour. The selection draws are generated one candidate at a time, 16 responses per call, so their time grows in proportion to K.

A record of prompt selection or of a mixture holds the 16 K selection draws of its pool: about 7 MB at K = 1,024 and about 1 GB at K = 131,072. A run with the default settings writes about 150 MB per task and replicate. A run keeps the selection draws of its largest pool in memory, about 5 GB at K = 131,072, and needs about as much again while it writes a record.

`write-candidates` needs no GPU: it waits for the writer's server, and its time is that of the server. A pool file takes about 0.4 MB per 1,000 candidates.

## Paper results this produces

- The main comparison on the 21 tasks, with replicates 0, 1 and 2: the default settings give EB-JC, hit-rate, IWAE and CE selection and Mixture: EB-16 at K = 1 to 1,024, Instruction, and the three compositions. The second command above gives EB-JC selection and Mixture: EB-16 at K = 2,048 to 131,072. The paper shows the mixture from K = 16 on.
- The runs on the 108 prompt-behavior pairs: `--set methods=prompt-ebjc,prompt-jc,prompt-hitrate,prompt-iwae` for the selection signals, and `--set methods=mixture-uniform1,mixture-uniform4,mixture-uniform16,mixture-uniform64,mixture-eb4,mixture-eb16,mixture-eb64 --set pool_sizes=16,64,256,1024` for the mixtures of other sizes and weights (`mixture-uniform1` is EB-JC selection).
- The candidates themselves, the figure of the pipeline: `write-candidates` with its default settings and DeepSeek-V4-Flash (`deepseek-ai/DeepSeek-V4-Flash`) as the writer. The paper's candidates are the two pool files of the method data (the methods README, "Method data"). The writer samples, so a pool file that you write holds other candidates than the paper's.
- `posteriorbench-methods compute-per-inference` on these records gives the row of prompt selection and Mixture: EB-16 in the paper's table of compute per counted inference (the methods README, "Analyses of finished runs"); composed methods are left out.
- The results on the two larger models and on the two LLM-judged behaviors, at the 10% and at the 1% threshold: the commands of [`generalization`](../generalization), "Running the paper's methods".
