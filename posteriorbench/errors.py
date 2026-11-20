"""The base class of the package's exceptions.

Every error the package raises for a misuse of the benchmark or for malformed data derives from
`PosteriorBenchError`, so `except posteriorbench.PosteriorBenchError` catches all of them. The
specific classes live next to the code that raises them: `data.DataError`, `corpus.CorpusError`,
`sampler.SamplerError`, `ledger.BudgetExceeded`, `rules.RuleError`,
`instructions.InstructionError` and `original_model.ResponseError`. Wrong argument values of
ordinary functions (a negative count, an unknown option) raise the built-in `ValueError` or
`TypeError`.
"""

from __future__ import annotations


class PosteriorBenchError(Exception):
    """Base class of the package's exceptions."""
