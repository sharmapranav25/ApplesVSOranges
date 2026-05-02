"""Parse the judge model's text response into a 0-5 integer score.

The judge prompt asks for a single digit and nothing else, but instruction-
following is imperfect — Qwen sometimes prepends a thinking-style preamble
or wraps the digit in markdown. Rule: take the FIRST integer in the response,
return it iff it's in [0, 5], else None.

The "first integer" rule (vs. e.g. "last integer") matches the simplest
reading of an instruction-following model that mostly complies but occasionally
starts with "5" then comments. We reject anything not in 0..5 rather than
clamping, so the error log surfaces real prompt-following failures.
"""

from __future__ import annotations

import re

_FIRST_INT_RE = re.compile(r"-?\d+")


def parse_score(response: str | None) -> int | None:
    """Return first integer 0-5 found in `response`, else None."""
    if not response:
        return None
    m = _FIRST_INT_RE.search(response)
    if m is None:
        return None
    try:
        n = int(m.group())
    except ValueError:
        return None
    if 0 <= n <= 5:
        return n
    return None
