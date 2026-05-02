"""Load the rubric prompt files and validate they are in ascending 0->5 order.

The validation matters: paper §4.3 presents the rubric DESCENDING; Appendix A.6
says the LLM-judge gets it ASCENDING (to align with Prometheus's training).
A silent re-order would corrupt every judge call.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"

_LEVEL_RE = re.compile(r"^(\d+)\s+points?:", re.MULTILINE)


def load_rubric(criterion: str) -> str:
    """Return the verbatim rubric text for `criterion`, validated as ascending 0->5."""
    if criterion not in ("accuracy", "completeness"):
        raise ValueError(f"unknown criterion: {criterion!r}")
    path = PROMPTS_DIR / f"rubric_{criterion}.txt"
    text = path.read_text()
    levels = [int(m.group(1)) for m in _LEVEL_RE.finditer(text)]
    if levels != [0, 1, 2, 3, 4, 5]:
        raise ValueError(
            f"rubric {path} not in ascending 0->5 order; got levels={levels}"
        )
    return text.rstrip("\n") + "\n"


def load_judge_template() -> str:
    """Return the verbatim judge prompt template (Appendix A.6)."""
    return (PROMPTS_DIR / "judge_template.txt").read_text()
