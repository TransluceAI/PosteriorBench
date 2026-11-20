# Tools

Scripts for building and checking the dataset. They are part of this repository, not of the
installed package; run them from a clone, in its environment.

| Script | What it does |
|---|---|
| `add_behavior.py` | builds a dataset directory for a new behavior on one of the prompts (references, fallback response, hit file) |
| `build_hits_index.py` | builds the hit files of one prompt from its corpus, leaving out the references |
| `check_fallbacks.py` | checks that the fallback response of every scored pair is a hit and a complete response |
| `check_labels.py` | recomputes the labels of corpus rows from their text and compares them with the stored labels |
| `choose_prompts.py` | chooses the 20 prompts from WildChat-1M and compares them with the task table (needs the `datasets` library; version 5.0.0 gives the dataset's prompts) |
| `choose_references.py` | says which hits of a scored pair are its references, by their position in the order sampled |
| `choose_tasks.py` | computes the scored pairs and the 21 tasks from the task table and compares them with its columns |
| `sample_corpus.py` | samples responses of a prompt from a server with SGLang's `/generate` interface, labels them and writes corpus files |

`check_fallbacks.py`, `choose_prompts.py`, `choose_references.py`, `choose_tasks.py` and
`sample_corpus.py` state the rules by which the dataset was made;
[docs/how_the_dataset_was_made.md](../docs/how_the_dataset_was_made.md) goes through them step by
step.

`uv run python tools/<script>.py --help` lists a script's options.
