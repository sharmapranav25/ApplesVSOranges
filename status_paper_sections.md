# Member 1 — Paper-section writeups — ❌ NOT STARTED

**Owner:** Pranav (Member 1)
**Last verified:** 2026-04-30

## Goal
Member 1 is also responsible for the writeup of three paper sections, complementing the four code/build tasks:

1. **Related Work** — situate Loakman et al. (2025) against prior pun-explanation / humour-understanding work (Hessel et al. 2023 NYT cartoons, Mittal et al. 2022 ExPUNations, Hossain SemEval-2017 Task 7 pun-detection corpus, Sun et al. ColBERT pun-classification, etc.).
2. **Dataset / Background** — describe the four joke types (homographic, heterographic, non-topical, topical), the 600-row composition, the gold-explanation provenance, and how `new_jokes.csv` slots into the paper's stated 600.
3. **Analysis section** — error analysis (qualitative categorization of where models fail, e.g. surface-form vs cultural reference) **and** metric correlation (Spearman/Pearson of automatic metrics vs Qwen judge ratings, per the paper's claim that automatic metrics correlate poorly with judge scores on humour explanation).

## What exists
**Nothing.** No `*.md`, `*.ipynb`, `*.tex`, or `*.docx` for any of the three sections. Search confirmed:
```bash
$ find . -iname "*analysis*" -o -iname "*related*" -o -iname "*background*" -o -iname "*report*" -o -iname "*writeup*"
# (no results)
```

## What's needed (per section)

### Related Work
- 1–2 paragraphs.
- Inputs available: `Comparing Apples to Oranges.pdf` §2 already cites the relevant prior work — start by re-reading it.
- Recommended: write to `writeup/related_work.md`.

### Dataset / Background
- 1–2 paragraphs + a small composition table.
- Inputs available: `data/jokes.jsonl` (verified 150/type/total 600), CLAUDE.md §"Critical conventions".
- Recommended: write to `writeup/dataset.md`.

### Analysis (error analysis + metric correlation)
This is the most substantive of the three and **blocks on Tasks 3 and 4 outputs**:
- **Error analysis** needs `outputs/ratings_judge.jsonl` (currently missing — and we now judge with 7B, see status_task3_judge.md) to identify low-scoring (joke, model) pairs to inspect. As an interim, can use `outputs/ratings_judge_paper.jsonl` (the paper's own Qwen-**72B** ratings, 9600 rows extracted from CSV) for triage — but call out in the writeup that the final analysis uses 7B, not 72B.
- **Metric correlation** needs *both*:
  - Per-instance metric scores (currently `metrics.py` aggregates per bucket — would need a `--per-instance` flag or a small notebook variant).
  - Per-instance judge scores from `ratings_judge.jsonl` (our 7B). The `ratings_judge_paper.jsonl` (paper's 72B) can be used as an additional reference column to discuss judge-scale effects, which becomes a small bonus contribution.
- Recommended: `writeup/analysis.md` + a supporting notebook like `notebooks/metric_correlation.ipynb`.

## Suggested ordering
1. Run `metrics.py` (free, local) → Task 4 fully closed.
2. Run `judge.py` against OpenRouter (~$5–15) → Task 3 fully closed.
3. Then write the three sections; Analysis becomes feasible only after step 2.

Alternatively, if waiting on the judge run:
1. Write Related Work + Dataset/Background now (they don't depend on outputs).
2. Use `ratings_judge_paper.jsonl` as the source for analysis writing as a **placeholder**, then re-run with our own ratings once they're produced.

## Status
| Section | State |
|---|---|
| Related Work | ❌ not started |
| Dataset / Background | ❌ not started |
| Analysis (error + correlation) | ❌ not started — blocks on judge & metrics outputs |
