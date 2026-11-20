"""PosteriorBench: evaluating samplers of rare behaviors of a language model.

Quickstart::

    import posteriorbench as pb

    bench = pb.Benchmark()                           # the bundled task table and evaluation file
    task = bench.task("wc_1853/goblin_mention")      # prompt, behavior, rule, p̂(S)
    model = pb.OriginalModel.from_pretrained()       # downloads Qwen3.5-0.8B on first use

    candidate = pb.instruction_sampler(model, task)  # a simple sampler; put yours here
    result = bench.evaluate(task, candidate, model, ledger=pb.Ledger(), seed=0)
    print(result.native_hit_rate, result.capped_forward_kl, result.reverse_kl)

The names above and the other names in `__all__` are the main interface. The modules behind them:
`benchmark` (the entry point), `data` (the task table; the evaluation file, for evaluation only),
`corpus` (corpus hits, charged to a ledger), `ledger` (the counting rules), `rules` (behavior
rules), `sampler` (the interface a method submits), `protocol` (the evaluation steps), `scoring`
and `estimators` (the arithmetic), `results` (settings and the result record), `aggregate` (task
averages), `original_model`, `model_samplers` and `instructions` (the original model and the
reference samplers), `errors` (the base exception).
"""

from posteriorbench._version import __version__
from posteriorbench.benchmark import Benchmark
from posteriorbench.corpus import CorpusAccess, CorpusHit
from posteriorbench.data import Task
from posteriorbench.errors import PosteriorBenchError
from posteriorbench.ledger import Category, Ledger, LedgerTotals
from posteriorbench.model_samplers import (
    instruction_sampler,
    original_model_sampler,
    system_prompt_sampler,
)
from posteriorbench.original_model import OriginalModel
from posteriorbench.protocol import Scorer
from posteriorbench.results import EvaluationResult, EvaluationSettings
from posteriorbench.sampler import FiniteSampler, Sampler

__all__ = [
    "Benchmark",
    "Category",
    "CorpusAccess",
    "CorpusHit",
    "EvaluationResult",
    "EvaluationSettings",
    "FiniteSampler",
    "Ledger",
    "LedgerTotals",
    "OriginalModel",
    "PosteriorBenchError",
    "Sampler",
    "Scorer",
    "Task",
    "__version__",
    "instruction_sampler",
    "original_model_sampler",
    "system_prompt_sampler",
]
