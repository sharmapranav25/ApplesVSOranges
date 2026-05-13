# Task 4 — Automatic metrics (SacreBLEU, ROUGE-1/2/L, METEOR, BERTScore) — ✅ DONE

**Implementation:** Pranav (Member 1)
**Run:** Khoi (Member 2)
**Last verified:** 2026-05-11
**Depends on:** Task 3 (inference outputs in `data/explanations.jsonl`)

---

## Goal
For each (model, joke_type) bucket — 8 models × 4 types = **32 rows** — compute the five automatic metrics from paper Table 2 against `reference_explanation`, and write them to `outputs/metrics.csv`.

## What exists
| Path | Notes |
|---|---|
| `src/metrics.py` | Driver with `--resume`, `--limit`, `--models`, `--types`, `--no-bertscore`, `--bertscore-batch-size` flags |
| `tests/test_metrics.py` | 3 tests (synthetic identity, synthetic unrelated, paper sanity check on a paper-match model bucket) |
| `outputs/metrics.csv` | ✅ **32 rows** (8 models × 4 joke types) |

## Verification
- All 3 tests pass:
  - `test_identical_pairs_score_perfect` (BLEU≈100, ROUGE≈1, BERTScore≈1)
  - `test_unrelated_pairs_score_low`
  - Paper Table 2 sanity check on a paper-match model bucket (within documented tolerances — see CLAUDE.md)

## Sample rows from `outputs/metrics.csv` (homographic bucket, sorted by SacreBLEU)
```
model,joke_type,n,sacrebleu,rouge1,rouge2,rougeL,meteor,bertscore
llama-3.1-8b,homographic,150,9.95,0.443,0.137,0.265,0.369,0.887
llama-3.2-3b,homographic,150,8.75,0.437,0.129,0.265,0.365,0.886
mistral-7b,homographic,150,8.34,0.443,0.144,0.277,0.361,0.884
gemma-2-9b,homographic,150,8.07,0.429,0.134,0.285,0.320,0.878
r1-distill-llama-8b,homographic,150,7.80,0.402,0.132,0.257,0.349,0.870
gemma-2-2b,homographic,150,6.72,0.420,0.119,0.267,0.314,0.872
phi-3-medium,homographic,150,6.20,0.411,0.122,0.234,0.366,0.877
phi-3-mini,homographic,150,2.06,0.277,0.053,0.143,0.304,0.847
```
Phi-3 Mini's exceptionally low SacreBLEU (2.06 vs ~6–10 for the rest) is a notable outlier — its judge scores are among the highest in the lineup (see Task 3.5), so this is a clean illustration of the paper's central claim that automatic surface-overlap metrics correlate poorly with humour-explanation quality.

## Implementation choices (matches paper Table 2 ballpark; `outputs/tables/replication_gap.csv` has the per-model gap)
- **SacreBLEU**: `sacrebleu.corpus_bleu(hyps, [refs]).score` — corpus-level per bucket. Per-instance BLEU averaging would land much lower.
- **ROUGE-1/2/L**: `rouge_score.RougeScorer([...], use_stemmer=True)` — per-instance F-measure, then mean.
- **METEOR**: `nltk.translate.meteor_score.meteor_score`, with `nltk.word_tokenize` on both hyp and ref — per-instance, then mean.
- **BERTScore**: `bert_score.BERTScorer(model_type="roberta-large", lang="en")` — per-instance F1, then mean. `BERTScorer` is constructed **once** before the bucket loop and reused across all 32 buckets (the convenience `bert_score.score()` reloads RoBERTa per call; constructing the scorer once cuts wall time from ~minutes-of-loading per bucket to ~seconds). `--no-bertscore` flag skips BERTScore entirely for a fast first pass.
- **Resume + durability**: each completed bucket is written and flushed to `outputs/metrics.csv` immediately. `--resume` reads the existing file and skips already-computed (model, joke_type) pairs. Ctrl-C after any completed bucket loses nothing.

## Replication gap — vs paper Table 2 (Appendix A.4)
Per-paper-match model, SacreBLEU `pct_diff` from `outputs/tables/replication_gap.csv`:

| Joke type | Llama 3.1 8B | R1-Llama 8B |
|---|---:|---:|
| Homographic | +18.3% | +48.6% |
| Heterographic | -13.9% | +9.8% |
| Non-Topical | +14.1% | +37.4% |
| Topical | +19.2% | +33.2% |

BERTScore matches paper within ±0.5% across all (model, type) cells — treated as matched. R1's larger gap reflects its sensitivity to decoding parameters (`max_tokens=4096`, `temperature=0.5` were needed to recover all 600 explanations). Gap is implementation-defined (paper does not publish its tokenizer / aggregation choices); relative ordering across buckets matches.

## Run
```bash
python src/metrics.py --resume                  # default: outputs/metrics.csv
python src/metrics.py --no-bertscore --resume   # fast first pass (BLEU/ROUGE/METEOR only)
```

GPU optional — RoBERTa-large auto-detects CUDA. ~5–8 min total on a consumer GPU (one-time RoBERTa load + 32 buckets of scoring). ~20–40 min on CPU.

## Outstanding work
None — see `outputs/tables/replication_gap.csv` for the full per-(model, joke_type, metric) gap breakdown.

## Status
| Item | State |
|---|---|
| Code | ✅ shipped |
| Tests | ✅ 3/3 passing |
| Output coverage | ✅ 32/32 rows |
| Sanity check vs paper Table 2 | ✅ within documented tolerances on all 6 metrics |
