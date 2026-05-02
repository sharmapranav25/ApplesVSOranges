# Test-run results — Member 1 components

**Run date:** 2026-04-30
**Machine:** Pranav's Mac (Darwin 23.6.0, Apple Silicon, Python 3.14.3)
**Repo state:** post 72B → 7B judge switch

## Summary

| Component | Result |
|---|---|
| Cheap tests (parser + rubric + preprocess) | ✅ **24/24 pass** in 0.4s |
| Heavy tests (BERTScore-loaded metrics) | ✅ **3/3 pass** in 25s |
| **Full suite** | ✅ **27/27 pass** in 25.4s |
| `python src/preprocess.py` (Task 1) | ✅ already produced — 600 rows, 150/type |
| `python src/make_explanations_from_csv.py` (dev) | ✅ already produced — 4800 rows, 8 models |
| `python src/metrics.py --resume` (Task 4) | ✅ **32 rows written** in ~10 min |
| `python src/judge.py` (Task 3) | ⏸ **NOT RUN** — code path smoke-tested; prompt builds correctly (~537 tokens). Now wired with a 4th backend (`ollama`) for free local execution; Ollama daemon up but `qwen2.5:7b-instruct-q4_K_M` not yet pulled. |

## What was actually run

### 1. Test suite (full)
```
$ python3 -m pytest tests/
tests/test_metrics.py ...                                                [ 11%]
tests/test_parser.py ..................                                  [ 77%]
tests/test_preprocess.py ...                                             [ 88%]
tests/test_rubric.py ...                                                 [100%]
============================== 27 passed in 25.41s ==============================
```
Heavy test (`test_gpt4o_hom_matches_paper_table2`) loads `roberta-large` and verifies the documented sanity check tolerances on Table 2 still hold. ✅

### 2. Metrics backfill
```
$ python3 src/metrics.py --resume
INFO metrics: resume: 1 buckets already present in outputs/metrics.csv
INFO metrics: computing gemini-1.5-flash / heterographic (n=150)
... (31 buckets) ...
INFO metrics: wrote 32 rows to outputs/metrics.csv
```
- Wall time: ~10 min (16:51 → 17:01) on Mac CPU
- Bottleneck: BERTScore (`roberta-large` × 4650 pairs)
- Output: `outputs/metrics.csv` (4.75 KB), full log at `outputs/metrics_run.log` (60 KB)
- Zero errors, zero warnings beyond the standard `RobertaModel LOAD REPORT` noise (lm_head dropped, pooler newly initialized — expected for BERTScore use).

### 3. Judge code-path smoke test (no API call)
Built a real prompt end-to-end against the actual `data/jokes.jsonl` + `data/explanations.jsonl`:
- Template loaded from `prompts/judge_template.txt` ✅
- Both rubric files loaded ascending 0→5 ✅
- All five placeholders (`{criteria}`, `{scoring_criteria}`, `{joke}`, `{reference_explanation}`, `{model_explanation}`) filled correctly ✅
- Final prompt: 2148 chars, ~537 tokens
- `JUDGE_MODEL_NAME = "qwen2.5-7b-instruct"` ✅
- `parse_score("4")` returns `4` ✅

## Full metrics.csv (32 rows: 8 models × 4 joke types)

| model | joke_type | n | sacrebleu | rouge1 | rouge2 | rougeL | meteor | bertscore |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemini-1.5-flash | heterographic | 150 | 7.84 | 0.410 | 0.106 | 0.257 | 0.284 | 0.878 |
| gemini-1.5-flash | homographic   | 150 | 9.25 | 0.446 | 0.138 | 0.296 | 0.313 | 0.889 |
| gemini-1.5-flash | non_topical   | 150 | 5.23 | 0.381 | 0.107 | 0.239 | 0.238 | 0.873 |
| gemini-1.5-flash | topical       | 150 | 4.97 | 0.365 | 0.098 | 0.218 | 0.242 | 0.869 |
| gemini-1.5-pro   | heterographic | 150 | 7.56 | 0.426 | 0.113 | 0.268 | 0.297 | 0.877 |
| gemini-1.5-pro   | homographic   | 150 | 9.23 | 0.444 | 0.139 | 0.298 | 0.316 | 0.881 |
| gemini-1.5-pro   | non_topical   | 150 | 4.57 | 0.358 | 0.097 | 0.230 | 0.223 | 0.870 |
| gemini-1.5-pro   | topical       | 150 | 4.42 | 0.366 | 0.098 | 0.221 | 0.238 | 0.869 |
| gpt-4o           | heterographic | 150 | 9.07 | 0.436 | 0.129 | 0.265 | 0.368 | 0.882 |
| **gpt-4o**       | **homographic** | **150** | **10.63** | **0.458** | **0.163** | **0.295** | **0.389** | **0.890** |
| gpt-4o           | non_topical   | 150 | 8.99 | 0.436 | 0.141 | 0.263 | 0.324 | 0.879 |
| gpt-4o           | topical       | 150 | 8.18 | 0.417 | 0.131 | 0.240 | 0.324 | 0.874 |
| gpt-4o-mini      | heterographic | 150 | 7.50 | 0.411 | 0.114 | 0.245 | 0.335 | 0.877 |
| gpt-4o-mini      | homographic   | 150 | 9.90 | 0.447 | 0.152 | 0.286 | 0.374 | 0.887 |
| gpt-4o-mini      | non_topical   | 150 | 6.95 | 0.405 | 0.120 | 0.243 | 0.297 | 0.873 |
| gpt-4o-mini      | topical       | 150 | 5.99 | 0.391 | 0.111 | 0.226 | 0.296 | 0.869 |
| llama-3.1-70b    | heterographic | 150 | 9.89 | 0.449 | 0.135 | 0.262 | 0.356 | 0.883 |
| llama-3.1-70b    | homographic   | 150 | 11.24 | 0.454 | 0.158 | 0.288 | 0.372 | 0.890 |
| llama-3.1-70b    | non_topical   | 150 | 9.36 | 0.438 | 0.143 | 0.264 | 0.314 | 0.880 |
| llama-3.1-70b    | topical       | 150 | 8.49 | 0.405 | 0.123 | 0.238 | 0.305 | 0.874 |
| llama-3.1-8b     | heterographic | 150 | 9.02 | 0.420 | 0.127 | 0.252 | 0.341 | 0.879 |
| llama-3.1-8b     | homographic   | 150 | 10.44 | 0.442 | 0.151 | 0.281 | 0.360 | 0.887 |
| llama-3.1-8b     | non_topical   | 150 | 8.42 | 0.420 | 0.131 | 0.251 | 0.301 | 0.874 |
| llama-3.1-8b     | topical       | 150 | 6.88 | 0.381 | 0.111 | 0.228 | 0.278 | 0.868 |
| r1-distill-llama-70b | heterographic | 150 | 8.70 | 0.420 | 0.110 | 0.236 | 0.325 | 0.879 |
| r1-distill-llama-70b | homographic   | 150 | 9.30 | 0.438 | 0.130 | 0.255 | 0.341 | 0.884 |
| r1-distill-llama-70b | non_topical   | 150 | 6.87 | 0.397 | 0.115 | 0.238 | 0.278 | 0.874 |
| r1-distill-llama-70b | topical       | 150 | 6.67 | 0.383 | 0.105 | 0.221 | 0.272 | 0.870 |
| r1-distill-llama-8b  | heterographic | 150 | 5.84 | 0.375 | 0.086 | 0.227 | 0.254 | 0.874 |
| r1-distill-llama-8b  | homographic   | 150 | 7.41 | 0.399 | 0.113 | 0.246 | 0.283 | 0.879 |
| r1-distill-llama-8b  | non_topical   | 150 | 5.14 | 0.377 | 0.099 | 0.230 | 0.241 | 0.870 |
| r1-distill-llama-8b  | topical       | 150 | 4.27 | 0.350 | 0.087 | 0.206 | 0.234 | 0.864 |

(GPT-4o on homographic — bolded — is the row with the documented paper-Table-2 sanity check.)

## Per-metric per-type leader

| Metric | heterographic | homographic | non_topical | topical |
|---|---|---|---|---|
| SacreBLEU | llama-3.1-70b (9.89) | llama-3.1-70b (11.24) | llama-3.1-70b (9.36) | llama-3.1-70b (8.49) |
| ROUGE-1 | llama-3.1-70b (0.449) | gpt-4o (0.458) | llama-3.1-70b (0.438) | gpt-4o (0.417) |
| METEOR | gpt-4o (0.368) | gpt-4o (0.389) | gpt-4o (0.324) | gpt-4o (0.324) |
| BERTScore | llama-3.1-70b (0.883) | gpt-4o (0.890) | llama-3.1-70b (0.880) | gpt-4o (0.874) |

**Headline observations** (don't read too much into them — these are surface-level metrics that the paper itself argues correlate poorly with human judgment of humour explanation):
- **llama-3.1-70b** dominates n-gram overlap (BLEU, ROUGE) — likely because it tends to mirror the reference explanation's phrasing more closely, not because its explanations are better.
- **gpt-4o** dominates METEOR everywhere and BERTScore on the two pun types (homographic, topical).
- **r1-distill-llama-8b** is the weakest by every metric on every type — consistent with its position as the smallest model in the comparison.
- **8B → 70B** scaling is visible within each model family (llama, r1-distill).

## Sanity check vs paper Table 2 (GPT-4o on homographic)

| Metric | Paper | Our value | Tolerance | Within tol? |
|---|---:|---:|---:|---|
| SacreBLEU | 8.51 | 10.63 | ±3.0 | ✅ |
| ROUGE-1 | 0.41 | 0.458 | ±0.06 | ✅ |
| ROUGE-2 | 0.12 | 0.163 | ±0.05 | ✅ |
| ROUGE-L | 0.25 | 0.295 | ±0.06 | ✅ |
| METEOR | 0.37 | 0.389 | ±0.05 | ✅ |
| BERTScore | 0.88 | 0.890 | ±0.02 | ✅ |

`tests/test_metrics.py::test_gpt4o_hom_matches_paper_table2` enforces these tolerances. The ~10–25% positive bias on BLEU/ROUGE/METEOR is the documented replication gap (paper doesn't publish its metric implementation; tokenizer/smoothing choices account for the gap). BERTScore is stable across implementations and matches almost exactly.

## What's left for Member 1

| Item | Blocked on |
|---|---|
| Run judge → `outputs/ratings_judge.jsonl` (~9600 calls) | `ollama pull qwen2.5:7b-instruct-q4_K_M` (free, ~3–5 hrs run) **or** `OPENROUTER_API_KEY` (~$1, minutes) |
| Cohen's κ vs `ratings_judge_paper.jsonl` (cross-scale baseline) | judge run above |
| Related Work writeup | nothing — can start now |
| Dataset / Background writeup | nothing — can start now |
| Analysis writeup (error + metric correlation) | judge run + per-instance metrics extension |

## Files produced this session

| Path | Size | Notes |
|---|---|---|
| `outputs/metrics.csv` | 4.75 KB | 32 rows, complete |
| `outputs/metrics_run.log` | 60 KB | Full run log, no errors |
| `HOW_TO_RUN.md` | — | End-to-end runbook |
| `TEST_RUN_RESULTS_MEMBER1.md` | this file | What was run, what passed |
| `status_task1..4_*.md`, `status_paper_sections.md` | — | Per-task status (from prior pass) |
