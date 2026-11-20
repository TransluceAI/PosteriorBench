# The paper and the commands

Which command produces which figure or table of the paper, and which script draws or prints it. A command below is run for every task and replicate that the item covers (`--task`, `--replicate`), and its records are averaged with `posteriorbench aggregate` (the methods README, "From records to a point of the paper's comparison"). "The 21 tasks" are those of `posteriorbench list-tasks`, and "the 108 pairs" those of `posteriorbench list-tasks --subset scored`. The paper uses the replicates 0, 1 and 2 unless an item says otherwise.

The last column names the script that draws the figure or prints the table, from the paper's numbers or from a table of your own runs; a script is run as `uv run python figures/<script>`, and [`figures/`](../figures/README.md) describes the numbers and the scripts.

The method data files are named in the methods README. `<108>` stands for `method_data/candidate_prompts_108tasks.parquet`, `<131072>` for `method_data/candidate_prompts_21tasks_extended_131072.parquet`, and `<vectors>` for the directory `method_data/steering_vectors/benchmark`.

The six tasks of the generalization section have ids such as `gemma-4-31b-it/wc_0670/em_dash` (`posteriorbench-methods generalization list-tasks`), and a command on one of them also takes `--data <the data of these tasks>`. `<gen-pool>` stands for their pool file `method_data/candidate_prompts_generalization.parquet`, `<gen-vectors>` for the directory `method_data/steering_vectors/<model>` of the task's original model, and `<judges>` for the options of the two judge models of an LLM-judged task (`--judge-url`, `--judge-model`, `--judge-key-env`, `--review-url`, `--review-model`, `--review-key-env`, `--labels`). [`posteriorbench_methods/generalization/README.md`](../posteriorbench_methods/generalization/README.md) shows the commands in full, with the same placeholders.

## Main text

| Paper item | Command | Script |
|---|---|---|
| Figure 3, p̂(S) of the 108 pairs | the package: `posteriorbench list-tasks --subset scored` | `figures/event_probabilities.py` |
| Table 1, the methods | the six families (the methods README, "The families") | |
| Figure 5, posterior fidelity against construction cost on the 21 tasks: SFT, Replay | `sft run` | `figures/main_comparison.py` |
| Figure 5: GFlowNet | `gfn run` | `figures/main_comparison.py` |
| Figure 5: CTL | `ctl run` | `figures/main_comparison.py` |
| Figure 5: KL-DPG, JS-DPG, GDC | `dpg run --method kl-dpg`, `--method js-dpg`, `--method gdc` | `figures/main_comparison.py` |
| Figure 5: prompt selection (EB-JC, hit rate, IWAE, CE), Mixture: EB-16 and Instruction, up to K = 1,024 | `prompt run --method-data <108>` | `figures/main_comparison.py` |
| Figure 5: EB-JC selection and Mixture: EB-16 at K = 2,048 to 131,072 | `prompt run --method-data <131072> --set pool_sizes=2048,4096,8192,16384,32768,65536,131072 --set methods=prompt-ebjc,mixture-eb16,prompt-ebjc+composition,mixture-eb16+composition` | `figures/main_comparison.py` |
| Figure 5: Steering | `steering run --method-data <vectors>` | `figures/main_comparison.py` |
| Figure 5: every method with composition | the same commands; a run writes `<method>+composition` next to each method unless `--no-compose` is given | `figures/main_comparison.py` |
| Figure 7, the results on Qwen3.5-0.8B | the records of prompt selection, Instruction and Steering of Figure 5 | `figures/larger_models.py` |
| Figure 7, the two larger models: prompt selection with EB-JC, the hit rate and IWAE at K = 1 to 1,024, each with composition; Instruction and Instruction + composition | on the two tasks of Qwen3.8-27B and the two of Gemma-4-31B-it: `prompt run --method-data <gen-pool> --set methods=prompt-ebjc+composition,prompt-hitrate+composition,prompt-iwae+composition,instruction+composition`. The figure draws the composition of EB-JC selection; those of the hit rate and IWAE are in the appendix's text | `figures/larger_models.py` |
| Figure 7, the two larger models: Steering and Steering + composition | on the same four tasks: `steering run --method-data <gen-vectors>` | `figures/larger_models.py` |
| Figure 8, the behaviors judged by a language model: prompt selection with EB-JC, IWAE and the hit rate at K = 16, 64, 256 and 1,024, EB-JC selection + composition, Mixture: EB-16, Instruction and Instruction + composition | on the two tasks of Qwen3.5-0.8B with responses of up to 2,048 tokens: `prompt run --method-data <gen-pool> <judges> --set pool_sizes=16,64,256,1024 --set methods=prompt-ebjc+composition,prompt-iwae,prompt-hitrate,mixture-eb16,instruction+composition` | `figures/llm_judged_behaviors.py` |
| Figure 8: Steering and Steering + composition | `steering run --method-data <gen-vectors> <judges>` | `figures/llm_judged_behaviors.py` |
| Figure 8: SFT | `sft run <judges> --set training_set=corpus --no-compose --keep-checkpoints all` | `figures/llm_judged_behaviors.py` |

With their default settings the commands run every budget, update cap and pool size of Figure 5. The mixture is shown from K = 16 on.

## Appendix

| Paper item | Command or code | Script |
|---|---|---|
| Table 2, the 21 tasks | the package: `posteriorbench list-tasks` | `figures/table_tasks.py` |
| Table 3, the rules | the package: `posteriorbench.rules.PAPER_RULES`, and the `rule` of each task | `figures/table_rules.py` |
| Table 4, compute per counted inference | `posteriorbench-methods compute-per-inference <out>` on the output root with the records of Figure 5: the `methods` of its output are the rows; Replay, prompt selection and Mixture: EB-16, which train nothing, share the table's last row. `time-operations --out <file>` measures the times of the second column on another GPU, for `compute-per-inference --times <file>` (`analysis/README.md`) | `figures/table_compute_per_inference.py` |
| Sensitivity of the comparison to the cap C of the forward KL | `posteriorbench-methods cap-sensitivity <out>` on the output root with the records of Figure 5: `share_at_cap` is the share of the references at or above each cap, and `points` the scores of every point at C = 10, 20, 50, 100 and 200 (`analysis/README.md`) | |
| Figure 15, estimating p(S) with a sampler | `posteriorbench-methods estimate-ps <out> --samplers` on the output root with the records of Figure 5: `points` are the left panel (average absolute log error against construction cost; Replay is left out) and `samplers` the right panel (log error against the reverse KL of each passing sampler), `samplers_summary` the numbers of the text (`analysis/README.md`) | `figures/estimating_ps.py` |
| Figure 9, the submission rule | `common/submission.py`; `posteriorbench-methods threshold-test --rank j` prints the test at rank j | |
| Table 5, the selection signals | `common/signals.py` (JC, EB-JC), `prompt/selection_signals.py` (hit rate, IWAE), `prompt/ce.py` (the CE method) | |
| Algorithm 1, prompt selection; Algorithm 2, prompt mixture | `prompt/selection.py`, `prompt/mixture.py` | |
| Figure 10, writing the candidate prompts | `prompt write-candidates --task <task> --pool-file <file> --candidates <n> --writer-url <address> --writer-model <name>`, with DeepSeek-V4-Flash as the writer; the messages of the three stages are in `prompt/writer_messages.py`. The paper's candidates are the method data `<108>` and `<131072>` | |
| Table 6, the Instruction texts | the package: `posteriorbench.instructions` holds the seven texts of the behaviors that a rule defines; the two of the behaviors judged by a language model are in `generalization/instructions.py` | `figures/table_instructions.py` |
| Algorithm 3, building and adding a steering vector | building: `steering find-refusal-direction --method-data <vectors> --lmtailrisk-checkout <checkout>` once for the model, then `steering build-vectors --behavior <behavior> --prompts <file> --method-data <vectors> --lmtailrisk-checkout <checkout> --judge-url <address> --judge-model <name>` for each behavior (`steering/refusal.py`, `steering/construction.py`); adding: `steering/hook.py` | |
| Algorithms 4 to 8: SFT, GFlowNet, CTL, f-DPG, GDC | `sft/train.py`, `gfn/training.py`, `ctl/training.py`, `dpg/fdpg.py`, `dpg/gdc.py` | |
| Composition | `common/composition.py` | |
| Figure 11, the training methods at about 5 x 10^5 inferences by event probability | the records of Figure 5 at B = 2^19 for SFT and GDC and at B = 2^20 for GFlowNet, CTL, KL-DPG and JS-DPG | `figures/training_by_probability.py` |
| Table 7, the submission rule against each method's own checkpoint rule, on replicate 0 of the 12 tasks below | "Ours": the records of Figure 11, except for KL-DPG and JS-DPG, whose "ours" is the `run` record of the training that their `reselect` below reads. "Own", after `run --keep-checkpoints all`: `gfn reselect --rule highest-mean-log-reward --set budgets=1048576 --set update_caps=`; `ctl reselect --rule final-checkpoint --set budgets=1048576 --set update_caps=`; `dpg reselect --method gdc --rule last-sampling-distribution --set budgets=524288` and the same with `--rule final-policy`; `dpg reselect --method kl-dpg --rule final-checkpoint --set budgets=1048576` and the same with `--method js-dpg` | `figures/table_checkpoint_rules.py` |
| Comparison of SFT with Replay | the `replay` records of `sft run` at B = 2^20, 2^21 and 2^22 | |
| Table 8, a 1% threshold | the 10% columns: the records of Figure 11. The 1% columns, after `run --keep-checkpoints all`: `sft reselect --set budgets=524288`; `gfn reselect --set budgets=1048576 --set update_caps=`; `ctl reselect --set budgets=1048576 --set update_caps=`; `dpg reselect --method gdc --set budgets=524288`; `dpg reselect --method kl-dpg --set budgets=1048576` and the same with `--method js-dpg` | `figures/table_threshold_1pct.py` |
| Figure 12, the selection signals on the 108 pairs | `prompt run --method-data <108> --set methods=prompt-ebjc,prompt-jc,prompt-hitrate,prompt-iwae` | `figures/selection_signals.py` |
| Figure 13, prompt mixtures on the 108 pairs, one replicate | `prompt run --method-data <108> --set methods=mixture-uniform1,mixture-uniform4,mixture-uniform16,mixture-uniform64,mixture-eb4,mixture-eb16,mixture-eb64 --set pool_sizes=16,64,256,1024` | `figures/prompt_mixtures.py` |
| Figure 14, the training-free methods on the 108 pairs by event probability; Steering by behavior | the records of Figures 12 and 13 at K = 1,024; `prompt run --set pool_sizes= --set methods=instruction+composition` for Instruction and its composition; `steering run --method-data <vectors> --no-compose` | `figures/training_free_by_probability.py` |
| Table 9, the tasks of the generalization section | `posteriorbench-methods generalization list-tasks` | `figures/table_generalization_tasks.py` |
| Figure 16, a 1% threshold on the behaviors judged by a language model | the commands of Figure 8 with `--set threshold=0.01` for the two families that do not train: `prompt run --method-data <gen-pool> <judges> --set threshold=0.01 --set pool_sizes=16,64,256,1024 --set methods=prompt-ebjc,prompt-iwae,prompt-hitrate,mixture-eb16,instruction+composition` and `steering run --method-data <gen-vectors> <judges> --set threshold=0.01`; for SFT, after the `sft run` of Figure 8, `sft reselect <judges> --set training_set=corpus`. The records are under `<out>/reselected/threshold-1pct/` | `figures/llm_judged_behaviors.py --threshold 0.01` |
| The candidate prompts and the steering vectors of the generalization tasks | `prompt write-candidates --task <task id>`; `steering find-refusal-direction --task <task id>` and `steering build-vectors --task <task id> --prompts <file>` on the task's original model, with `--set ablated_top_k=20` on Qwen3.8-27B and `--set vectors=without_ablation` on Gemma-4-31B-it, where the search for a refusal direction fails. The paper's are the method data of `generalization/README.md` | |
| Figure 6 and Figures 17 to 21, sample responses | the first hits among the evaluation draws of the records of Figure 5, in a fixed random order: `figures/sample_responses.py --record <record>` prints hits of a record's `draws/evaluation.jsonl` in this way | `figures/sample_responses.py` |
| Table 10, the prompts that three selection signals submit | a record's `construction.json` holds the prompt that prompt selection submitted, `submitted.system_prompt` | `figures/selected_prompts.py` |

`reselect` chooses the checkpoint of a finished training run again (the methods README, "Choosing a checkpoint again"). It takes the task, the replicate, the output root and the settings of the `run` whose training run it reads, and its records are under `<out>/reselected/<rule>/`; without `--rule` the rule is `threshold-1pct`. A `run` with the same `--set` as the `reselect` above trains as far as that budget alone. The 12 tasks of Table 7 are `wc_0346/em_dash`, `wc_0532/profanity`, `wc_0670/chinese_character`, `wc_0670/contrastive_phrasing`, `wc_0842/chatgpt_mention`, `wc_1380/word_repetition`, `wc_1405/chinese_character`, `wc_1569/em_dash`, `wc_1569/goblin_mention`, `wc_1786/goblin_mention`, `wc_1852/chatgpt_mention` and `wc_1853/profanity`.

The tuning grids of the training methods are the families' `run` commands on the two tuning pairs, `wc_0346/goblin_mention` and `wc_1852/chinese_character`, with other values of their settings (`--set`); each family's README names the settings that were tuned.

Steering's variants are `steering build-vectors --set response_limit=50` (or 150) and `steering run --set layers=11,12,13,14,15,16`.

The Instruction results of the generalization section also come from `posteriorbench-methods generalization evaluate --sampler instruction`, which evaluates the Instruction sampler alone.
