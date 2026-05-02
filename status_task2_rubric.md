# Task 2 — Evaluation rubric (accuracy + completeness, 0–5) — ✅ DONE

**Owner:** Pranav (Member 1)
**Last verified:** 2026-04-30

## Goal
Implement the §4.3 rubric — two criteria (accuracy, completeness), 0–5 Likert scale — in a form the judge prompt can consume verbatim.

## What exists
| Path | Notes |
|---|---|
| `prompts/rubric_accuracy.txt` | Verbatim §4.3, **ascending 0→5** |
| `prompts/rubric_completeness.txt` | Verbatim §4.3, **ascending 0→5** |
| `src/rubric.py` | `load_rubric(criterion)` + `load_judge_template()` |
| `tests/test_rubric.py` | 3 tests — all passing |

## Verification
- `pytest tests/test_rubric.py` → **3 passed**
  - Both rubric files parse to ascending levels `[0,1,2,3,4,5]` and contain the verbatim spot-check phrases ("entirely incorrect" / "fully accurate" / "superficial or irrelevant" / "comprehensive").
  - `load_rubric()` raises if anyone re-orders the file (paranoid guard).

## ⚠ Critical: ordering trap (do not "fix")
- Paper §4.3 prints the rubric **descending (5→0)** for human readability.
- Appendix A.6 explicitly says the LLM judge gets it **ascending (0→5)** to match Prometheus's training prompt.
- Our files are ascending; the test enforces this.

## Sample (rubric_accuracy.txt, line 1 + line 6)
```
0 points: The explanation is entirely incorrect, failing to identify the humour or context correctly.
…
5 points: The explanation is fully accurate, correctly identifying the joke's main humour and context without errors.
```

## `is_good` definition (§5.1)
`is_good = (accuracy >= 4) AND (completeness >= 4)` — implemented downstream in metrics/judge analysis, not in the rubric files themselves.

## Nothing left to do here
Files are static; no run step.
