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
| `replication_gap.csv` | joke_type, metric, paper, ours, diff, pct_diff | GPT-4o row vs Table 2 |
| `logistic_regression.csv` | feature, beta, se, z, p_value | Replicates paper §5 regression |
| `judge_agreement.csv` | criterion, pearson_r, mae, exact_match, within_1, binary_agree, cohen_kappa | 7B vs 72B agreement |
| `hypothesis_checks.csv` | hypothesis, description, value_a, value_b, delta, verdict, note | H1–H4 verdicts — dynamic (recomputed from data each run) |

---

## Key findings from outputs

### Figure 4 — Success rates
GPT-4o is clearly the strongest model (84–93% success across all joke types).
The overall ordering by success rate — GPT-4o (87.5%) > GPT-4o Mini (76.2%) > R1 70B (71.0%) > Llama 70B (61.5%) > Llama 8B (43.0%) > R1 8B (29.0%) > Gemini Flash (24.3%) > Gemini Pro (19.3%) — is consistent with the paper's Figure 4.

Homographic jokes remain the easiest type for almost every model; the success
rate advantage over other types is most visible for weaker models.

### Figure 3b / 3c — Average scores
Completeness scores are consistently lower than accuracy scores across all
models and joke types — matches paper §5 finding that models more often
*omit* key details than *hallucinate* incorrect ones.

### Replication gap — `replication_gap.csv`

SacreBLEU values vs paper's Table 2 (GPT-4o row):

| Joke type | SacreBLEU gap |
|---|---|
| Homographic | +4.7% |
| Heterographic | +6.3% |
| Non-Topical | +14.3% |
| Topical | +15.4% |

BERTScore matches within ±0.5% — treated as matched. Gap is implementation-defined
(tokenizer choices) and does not affect qualitative conclusions. Documented in CLAUDE.md.

### Logistic regression — `logistic_regression.csv`

Directional findings match the paper (all p<0.001):

| Feature | β (ours) | β (paper) |
|---|---|---|
| is_large | +0.681 | +1.707 |
| is_heterographic | -0.549 | n/a |
| is_non_topical | -0.426 | -0.511 |
| is_topical | -0.552 | -0.574 |

Beta magnitudes are smaller than the paper's because we use per-joke data
from our 7B judge rather than the paper's human ratings.

### Judge agreement — `judge_agreement.csv`

Our 7B judge achieved r=0.642 (accuracy) and r=0.721 (completeness) vs
the paper's 72B ratings — matching or exceeding the paper's own reported
72B-vs-human agreement (r=0.641/0.602). Cohen's κ=0.570/0.578, within±1=98.3%/99.2%.
This is a stronger result than anticipated given the 10× model size difference.

---

## Status

| Item | State |
|---|---|
| Code (`src/analyze.py`) | ✅ shipped and ran |
| Figures (4 files) | ✅ `outputs/figures/` |
| Tables (6 files) | ✅ `outputs/tables/` (includes `hypothesis_checks.csv`) |
| Results summary | ✅ `outputs/results_summary.txt` — saved on every run |
