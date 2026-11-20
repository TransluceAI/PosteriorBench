"""The behavior rules: parsing the released rule strings and the seven paper behaviors."""

from __future__ import annotations

import pytest

from posteriorbench import rules

CASES: dict[str, list[tuple[str, bool]]] = {
    "em_dash": [("a — b", True), ("a – b", False), ("a - b", False), ("x—", True)],
    "chinese_character": [
        ("hello 中文", True),
        ("㐀", True),
        ("䶿", True),
        ("鿿", True),
        ("㏿", False),
        ("䷀", False),
        ("豈", False),
        ("こんにちは", False),
        ("한국어", False),
        ("plain", False),
    ],
    "chatgpt_mention": [("I am ChatGPT.", True), ("CHATGPT", True), ("chat gpt", False)],
    "profanity": [("What the FUCK", True), ("fucking", True), ("f*ck", False)],
    "goblin_mention": [("Goblins!", True), ("hobgoblin", True), ("gob lin", False)],
    "contrastive_phrasing": [
        ("It's not a bug, it's a feature.", True),
        ("This isn't about money — it's about trust.", True),
        ("that is not ok; this is fine", True),
        ("It’s not magic, it’s science", True),
        ("It is not a bug. It is a feature.", False),
        ("It's not x, it's y", False),
        ("it's not good", False),
        ("Its not a bug, its a feature", False),
    ],
    "word_repetition": [
        ("the the the", True),
        ("Go go GO now", True),
        ("no no", False),
        ("1 1 1", False),
        ("ha ha ha!", True),
        ("thethe the the", False),
        ("a_b a_b a_b", False),
    ],
}


@pytest.mark.parametrize("behavior", rules.LABELS)
def test_paper_rules(behavior: str) -> None:
    rule = rules.paper_rule(behavior)
    for text, expected in CASES[behavior]:
        assert rule(text) is expected, (behavior, text)


def test_rule_shapes() -> None:
    assert rules.parse_rule('"x" in y').kind == "substring"
    lower = rules.parse_rule('"x" in y.lower()')
    assert lower.lowercase and lower("X")
    bare = rules.parse_rule('re.search(r"ab+", y)')
    assert bare.kind == "regex" and bare("abbb") and not bare("AB")
    keyword = rules.parse_rule('re.search(r"ab", y, flags=re.I | re.M) is not None')
    assert keyword("AB")


def test_paper_table_escapes_are_equivalent() -> None:
    # The paper writes the Chinese-character class with \u escapes; the dataset with literals.
    escaped = rules.parse_rule(r're.search(r"[㐀-䶿一-鿿]", y)')
    released = rules.paper_rule("chinese_character")
    for text, _ in CASES["chinese_character"]:
        assert escaped(text) == released(text)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "y.startswith('a')",
        "len(y) > 3",
        '"a" in z',
        're.match(r"a", y)',
        're.search(r"a", y) is None',
        're.search(r"(", y)',
        "re.search(pattern, y)",
        '"a" in y and "b" in y',
    ],
)
def test_rejects_other_expressions(expression: str) -> None:
    with pytest.raises(rules.RuleError):
        rules.parse_rule(expression)


def test_label_text() -> None:
    labels = rules.label_text("It's not a goblin, it's ChatGPT — 中 go go go")
    assert labels == {
        "em_dash": True,
        "chinese_character": True,
        "chatgpt_mention": True,
        "profanity": False,
        "goblin_mention": True,
        "contrastive_phrasing": True,
        "word_repetition": True,
    }
