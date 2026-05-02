"""First-int 0-5 parser. Most failure cases come from real Qwen outputs
seen during tuning: stray markdown, model preamble, refusal text.
"""

import pytest

from parser import parse_score


@pytest.mark.parametrize("response,expected", [
    ("3", 3),
    ("0", 0),
    ("5", 5),
    ("4\n", 4),
    (" 2 ", 2),
    ("Score: 4", 4),
    ("4. The explanation correctly...", 4),
    ("**3**", 3),
    ("I would rate this a 5.", 5),
])
def test_valid_scores(response, expected):
    assert parse_score(response) == expected


@pytest.mark.parametrize("response", [
    "",
    None,
    "no number here",
    "6",                       # out of range high
    "-1",                      # out of range negative (regex matches "-1" -> -1, rejected)
    "10",                      # 10 -> first int is 10, rejected
    "fifty",                   # words don't count
    "I cannot rate this.",
])
def test_invalid_returns_none(response):
    assert parse_score(response) is None


def test_first_int_wins():
    # If model says "5 (out of 10)" we want 5, not 10. If it says "I'd say 4
    # but maybe 7", we want 4.
    assert parse_score("5 (out of 10)") == 5
    assert parse_score("I'd say 4 but maybe 7") == 4
