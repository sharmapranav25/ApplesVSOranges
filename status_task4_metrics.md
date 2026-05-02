# Task 4 — Automatic metrics (SacreBLEU, ROUGE-1/2/L, METEOR, BERTScore) — ✅ DONE

**Owner:** Pranav (Member 1)
**Last verified:** 2026-04-30

## Goal
For each (model, joke_type) bucket — 8 models × 4 types = 32 rows — compute the five automatic metrics from paper Table 2 against `reference_explanation`, and write them to `outputs/metrics.csv`.

## What exists
| Path | Notes |
|---|---|
| `src/metrics.py` | Driver with `--resume`, `--limit`, `--models`, `--types` flags |
| `tests/test_metrics.py` | 3 tests (synthetic identity, synthetic unrelated, paper sanity check on GPT-4o/hom) |
| `outputs/metrics.csv` | ✅ **32 rows** (8 models × 4 joke types) — produced 2026-04-30, ~10 min on Mac CPU |

## Verification
- All 3 tests pass:
  - `test_identical_pairs_score_perfect` (BLEU≈100, ROUGE≈1, BERTScore≈1)
  - `test_unrelated_pairs_score_low`
  - `test_gpt4o_hom_matches_paper_table2` — within documented tolerances (see CLAUDE.md sanity-check table)

## Current `outputs/metrics.csv`
```
model,joke_type,n,sacrebleu,rouge1,rouge2,rougeL,meteor,bertscore
gpt-4o,homographic,150,10.63,0.458,0.163,0.295,0.389,0.890
```

## Implementation choices (matches paper Table 2 ballpark; CLAUDE.md has the gap analysis)
- **SacreBLEU**: `sacrebleu.corpus_bleu(hyps, [refs]).score` — corpus-level per bucket. Per-instance BLEU averaging would land much lower.
- **ROUGE-1/2/L**: `rouge_score.RougeScorer([...], use_stemmer=True)` — per-instance F-measure, then mean.
- **METEOR**: `nltk.translate.meteor_score.meteor_score`, with `nltk.word_tokenize` on both hyp and ref — per-instance, then mean.
- **BERTScore**: `bert_score.score(..., model_type="roberta-large", lang="en")` — per-instance F1, then mean. (BERTScore is implementation-stable; matches paper to ±0.01.)

## Replication gap (documented; don't chase)
Our BLEU/ROUGE/METEOR run ~10–25% above paper's numbers; BERTScore is exact. Paper doesn't publish its metric code, so the gap is implementation-defined (tokenizer, smoothing, possibly `evaluate.load("bleu")` vs raw sacrebleu). Test tolerances reflect this. Relative ordering across (model, joke_type) buckets should still match.

## Outstanding work
None — see `TEST_RUN_RESULTS_MEMBER1.md` for the full per-bucket numbers and per-metric leader analysis.

## Status
| Item | State |
|---|---|
| Code | ✅ shipped |
| Tests | ✅ 3/3 passing |
| Output coverage | ✅ 32/32 rows |
| Sanity check vs paper Table 2 | ✅ within documented tolerances on all 6 metrics |
