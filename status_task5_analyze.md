# Task 5 — Analysis pipeline (figures, tables, hypothesis checks) — ✅ DONE

**Owner:** Khoi (Member 2)
**Last verified:** 2026-05-05
**Depends on:** outputs/ratings_judge.jsonl (Task 3 run), outputs/metrics.csv (Task 4)

---

## Goal

Produce all figures and tables needed for the paper's Replication, Results,
and Conclusion sections from the judge ratings and automatic metrics.

---

## What exists

| Path | Notes |
|---|---|
| `src/analyze.py` | Reads `outputs/ratings_judge.jsonl` and `outputs/metrics.csv` and produces all figures and tables for the paper. Does **not** call any model or recompute scores — purely a visualisation and reporting script. |

Follows the same project conventions as `judge.py` and `metrics.py`:
- `argparse` + `main()` entry point
- Imports `JOKE_TYPES`, `CRITERIA` from `schema.py`
- Re-running is always safe — outputs are overwritten
- All input/output paths overridable via flags

---

## How to run

```bash
# From project root with applesVsOranges env active
python src/analyze.py

# Override paths if needed
python src/analyze.py \
    --ratings       outputs/ratings_judge.jsonl \
    --paper-ratings outputs/ratings_judge_paper.jsonl \
    --metrics       outputs/metrics.csv \
    --outdir        outputs
```

No GPU needed. Runs in ~30 seconds.

---

## Outputs

### Summary → `outputs/results_summary.txt`

Full console output saved to file on every run. Contains avg scores,
hypothesis checks, logistic regression, judge agreement, and replication gap.
Overwritten each run.

### Figures → `outputs/figures/`

| File | Description | Paper equivalent |
|---|---|---|
| `fig3b_accuracy.png` | Avg accuracy by model × joke type | Figure 3b |
| `fig3c_completeness.png` | Avg completeness by model × joke type | Figure 3c |
| `fig4_success.png` | Good/poor success rate grid | Figure 4 |
| `fig_judge_comparison.png` | Score distribution: our 7B vs paper 72B | n/a (novel) |

### Tables → `outputs/tables/`

| File | Columns | Notes |
|---|---|---|
| `avg_scores.csv` | model, joke_type, criterion, mean_score | Source for Figure 3b/3c |
| `success_rates.csv` | model, joke_type, success_rate | Source for Figure 4 |
| `replication_gap.csv` | model, joke_type, metric, paper, ours, diff, pct_diff | per paper-match model vs Table 2 |
| `logistic_regression.csv` | feature, beta, se, z, p_value | Replicates paper §5 regression |
| `judge_agreement.csv` | criterion, n_paired, n_models_overlap, pearson_r, mae, exact_match, within_1, binary_agree, cohen_kappa | 7B vs 72B agreement; weak when overlap < 4 models |
| `hypothesis_checks.csv` | hypothesis, description, value_a, value_b, delta, verdict, note | H1–H4 verdicts — dynamic (recomputed from data each run) |

---

## Key findings from outputs

### Hypothesis verdicts (from `hypothesis_checks.csv`)

| H | Description | Δ (accuracy) | Verdict |
|---|---|---:|---|
| H1 | Traditional puns easier than Reddit jokes | 0.024 | NEAR-TIE |
| H2 | Homographic puns easier than heterographic | 0.263 | CONFIRMED |
| H3 | Topical jokes harder than non-topical | 0.151 | CONFIRMED |
| H4 (Gemma 2 9B vs 2B, clean pair) | Larger > smaller success rate | 0.075 | CONFIRMED |
| H4 (Llama 3.1 8B vs 3.2 3B, cross-gen) | Larger > smaller success rate | 0.028 | NEAR-TIE |

H1's near-tie reflects the 7B judge's score concentration around 4, which compresses small joke-type differences. The Llama H4 pair is cross-generation — Llama 3.2 3B is a newer generation than 3.1 8B, so the size effect is confounded by training-data improvements. The Gemma 2 9B vs 2B pair is the clean within-family H4 evidence.

### Figure 4 — Success rates (from `success_rates.csv`)

Per-model overall success rate (avg over 4 joke types):

| Model | Success rate |
|---|---:|
| Llama 3.1 8B | 39.5% |
| Llama 3.2 3B | 36.7% |
| R1-Llama 8B | 36.5% |
| Gemma 2 9B | 18.3% |
| Gemma 2 2B | 10.8% |

Homographic jokes are the easiest type for every model, matching the paper's observation that polysemy is more tractable than phonetic similarity for orthographic-token-trained LLMs. Topical jokes are hardest across the board.

### Figure 3b / 3c — Average scores

| Joke type | Accuracy | Completeness |
|---|---:|---:|
| Homographic | 3.687 | 3.293 |
| Heterographic | 3.424 | 3.060 |
| Non-Topical | 3.607 | 3.076 |
| Topical | 3.456 | 2.964 |

Completeness is consistently lower than accuracy across every (model, joke type) cell — matches the paper's §5 finding that models more often *omit* key details than *hallucinate* incorrect ones.

### Replication gap — `replication_gap.csv`

SacreBLEU vs paper Table 2 (Appendix A.4), per paper-match model:

| Joke type | Llama 3.1 8B | R1-Llama 8B |
|---|---:|---:|
| Homographic | +18.3% | +48.6% |
| Heterographic | -13.9% | +9.8% |
| Non-Topical | +14.1% | +37.4% |
| Topical | +19.2% | +33.2% |

BERTScore matches within ±0.5% across all (model, type) cells — treated as matched. R1's larger gap reflects its sensitivity to decoding parameters (we used `max_tokens=4096, temperature=0.5` to recover all 600 explanations after initial truncation issues). Gap is implementation-defined (paper does not publish its tokenizer / aggregation choices) and does not affect qualitative conclusions.

### Logistic regression — `logistic_regression.csv`

P(good | model size, joke type), with paper coefficients for comparison:

| Feature | β (ours) | β (paper) | p (ours) |
|---|---:|---:|---:|
| is_large (Gemma 9B / Llama 3.1 8B) | +0.046 | +1.707 | n.s. |
| is_heterographic | -0.601 | n/a | <0.001 |
| is_non_topical | -0.648 | -0.511 | <0.001 |
| is_topical | -0.954 | -0.574 | <0.001 |

Joke-type coefficients have the same signs as the paper and are statistically significant at p<0.001, with magnitudes slightly larger than the paper's because the 7B judge is more discriminating across joke types. The `is_large` coefficient is small and non-significant: our within-family size pairs (Gemma 9B vs 2B, Llama 8B vs 3B) are much smaller absolute size differences than the paper's 70B vs 8B pairs, so detecting a size effect requires either larger pairs or a more sensitive judge.

### Judge agreement — `judge_agreement.csv`

| Criterion | n_paired | n_models_overlap | r | MAE | κ | within±1 |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1,200 | 2 | 0.244 | 0.718 | 0.221 | 86.7% |
| Completeness | 1,200 | 2 | 0.291 | 0.654 | 0.273 | 90.2% |

The two-model overlap (Llama 3.1 8B and R1-Llama 8B) limits the agreement to a narrow comparison; this is a weak baseline rather than a true 7B-vs-72B calibration. For a proper calibration, run `src/judge.py` over `data/explanations_paper.jsonl` to produce a "your-judge × paper's-explanations" file and re-run analyze with that as `--ratings`.

## Status

| Item | State |
|---|---|
| Code (`src/analyze.py`) | ✅ shipped and ran |
| Figures (4 files) | ✅ `outputs/figures/` |
| Tables (6 files) | ✅ `outputs/tables/` (includes `hypothesis_checks.csv`) |
| Results summary | ✅ `outputs/results_summary.txt` — saved on every run |
