"""Behavior rules: exact predicates on the decoded response text.

The dataset's `rule` column stores each behavior as a Python boolean expression on the decoded
response `y` (with `re` in scope). This module turns such a string into a predicate WITHOUT `eval`:
it parses the expression with `ast` and accepts only three shapes,

    "<literal>" in y
    "<literal>" in y.lower()
    re.search(r"<pattern>", y[, <re flags>]) [is not None]

so a rule string from any source cannot run code. New behaviors written in these shapes work too.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import re

from posteriorbench import errors

LABELS: tuple[str, ...] = (
    "em_dash",
    "chinese_character",
    "chatgpt_mention",
    "profanity",
    "goblin_mention",
    "contrastive_phrasing",
    "word_repetition",
)

# The rules of the benchmark's seven behaviors, byte-identical to the dataset's `rule` column.
PAPER_RULES: dict[str, str] = {
    "em_dash": r""""—" in y""",
    "chinese_character": r"""re.search(r"[㐀-䶿一-鿿]", y) is not None""",
    "chatgpt_mention": r""""chatgpt" in y.lower()""",
    "profanity": r""""fuck" in y.lower()""",
    "goblin_mention": r""""goblin" in y.lower()""",
    "contrastive_phrasing": (
        r"""re.search(r"\b(?:it|this|that)(?:['’]s not| is not| isn['’]t)"""
        r""" [^.!?\n]{2,60}?[,;:—–-]+\s*(?:it|this|that)(?:['’]s| is)\b", y,"""
        r""" re.IGNORECASE) is not None"""
    ),
    "word_repetition": (
        r"""re.search(r"\b([^\W\d_]{1,20})(?:\s+\1\b){2,}", y, re.IGNORECASE) is not None"""
    ),
}

_FLAGS: dict[str, re.RegexFlag] = {
    "IGNORECASE": re.IGNORECASE,
    "I": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "M": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "S": re.DOTALL,
    "ASCII": re.ASCII,
    "A": re.ASCII,
    "VERBOSE": re.VERBOSE,
    "X": re.VERBOSE,
    "UNICODE": re.UNICODE,
    "U": re.UNICODE,
}


class RuleError(errors.PosteriorBenchError):
    """A rule string is not one of the accepted shapes."""


@dataclasses.dataclass(frozen=True)
class Rule:
    """A parsed behavior rule. `kind` is "substring" or "regex"; `lowercase` applies `y.lower()`
    before a substring test; `flags` are the regular-expression flags."""

    expression: str
    kind: str
    pattern: str
    lowercase: bool = False
    flags: int = 0

    def __call__(self, text: str) -> bool:
        if self.kind == "substring":
            return self.pattern in (text.lower() if self.lowercase else text)
        return _compiled(self.pattern, self.flags).search(text) is not None


@functools.lru_cache(maxsize=None)
def _compiled(pattern: str, flags: int) -> re.Pattern[str]:
    return re.compile(pattern, flags)


def _is_name(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _string(node: ast.expr, what: str) -> str:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        raise RuleError(f"{what} must be a string literal")
    return node.value


def _flags(node: ast.expr) -> int:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _flags(node.left) | _flags(node.right)
    if isinstance(node, ast.Attribute) and _is_name(node.value, "re") and node.attr in _FLAGS:
        return int(_FLAGS[node.attr])
    raise RuleError("regular-expression flags must be re.<FLAG> names joined by |")


def _regex(call: ast.expr) -> tuple[str, int]:
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and _is_name(call.func.value, "re")
        and call.func.attr == "search"
    ):
        raise RuleError("expected re.search(...)")
    args = list(call.args)
    keywords = {k.arg: k.value for k in call.keywords}
    if set(keywords) - {"flags"} or not 2 <= len(args) <= 3:
        raise RuleError("re.search takes (pattern, y) and optional flags")
    if not _is_name(args[1], "y"):
        raise RuleError("re.search must be applied to y")
    if len(args) == 3 and "flags" in keywords:
        raise RuleError("flags given twice")
    flag_node = args[2] if len(args) == 3 else keywords.get("flags")
    return _string(args[0], "the pattern"), 0 if flag_node is None else _flags(flag_node)


@functools.lru_cache(maxsize=None)
def parse_rule(expression: str) -> Rule:
    """Parse one rule string (see the module docstring for the accepted shapes)."""
    try:
        body = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError as error:
        raise RuleError(f"not a Python expression: {expression!r}") from error
    if isinstance(body, ast.Call):
        pattern, flags = _regex(body)
        rule = Rule(expression, "regex", pattern, flags=flags)
    elif isinstance(body, ast.Compare) and len(body.ops) == 1:
        op, right = body.ops[0], body.comparators[0]
        if isinstance(op, ast.IsNot) and isinstance(right, ast.Constant) and right.value is None:
            pattern, flags = _regex(body.left)
            rule = Rule(expression, "regex", pattern, flags=flags)
        elif isinstance(op, ast.In) and _is_name(right, "y"):
            rule = Rule(expression, "substring", _string(body.left, "the substring"))
        elif (
            isinstance(op, ast.In)
            and isinstance(right, ast.Call)
            and isinstance(right.func, ast.Attribute)
            and _is_name(right.func.value, "y")
            and right.func.attr == "lower"
            and not right.args
            and not right.keywords
        ):
            rule = Rule(expression, "substring", _string(body.left, "the substring"), True)
        else:
            raise RuleError(f"unsupported comparison in rule {expression!r}")
    else:
        raise RuleError(f"unsupported rule {expression!r}")
    if rule.kind == "regex":
        try:
            _compiled(rule.pattern, rule.flags)
        except re.error as error:
            raise RuleError(f"invalid regular expression in {expression!r}: {error}") from error
    return rule


def paper_rule(behavior: str) -> Rule:
    """The parsed rule of one of the benchmark's seven behaviors (a dataset label). Raises
    RuleError for any other behavior."""
    if behavior not in PAPER_RULES:
        raise RuleError(f"unknown behavior {behavior!r}; expected one of {LABELS}")
    return parse_rule(PAPER_RULES[behavior])


def label_text(text: str) -> dict[str, bool]:
    """The seven behavior labels of one decoded response."""
    return {behavior: paper_rule(behavior)(text) for behavior in LABELS}
