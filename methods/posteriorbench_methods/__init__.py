"""The methods of the PosteriorBench paper, built on the `posteriorbench` package.

One subpackage per method family: `sft` (SFT, Replay), `prompt` (prompt selection, prompt
mixtures, Instruction), `steering`, `gfn` (GFlowNet), `ctl` and `dpg` (KL-DPG, JS-DPG, GDC), each
with the composition of its samplers with the original model. `common` holds what the families
share, and `analysis` the analyses of finished runs.
"""

__version__ = "1.0.0"
