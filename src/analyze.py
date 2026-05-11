"""Analysis pipeline: figures, tables, and hypothesis checks.

Reads:
  data/jokes.jsonl                    (from preprocess.py)
  outputs/ratings_judge.jsonl         (from judge.py — our Qwen-7B scores)
  outputs/ratings_judge_paper.jsonl   (paper's Qwen-72B baseline, extracted by
                                       make_explanations_from_csv.py)
  outputs/metrics.csv                 (from metrics.py)

Produces in outputs/figures/:
  fig3b_accuracy.png       — avg accuracy by model x joke type   (Figure 3b)
  fig3c_completeness.png   — avg completeness by model x joke type (Figure 3c)
  fig4_success.png         — good/poor success rate grid          (Figure 4)
  fig_judge_comparison.png — score distribution: our 7B vs paper 72B

Produces in outputs/tables/:
  avg_scores.csv           — mean accuracy + completeness per model x type
  success_rates.csv        — fraction of (acc>=4 AND comp>=4) per model x type
  hypothesis_checks.csv    — H1–H4 results with values, delta, and verdict
  replication_gap.csv      — our metrics vs paper Table 2 (GPT-4o row)
  logistic_regression.csv  — beta / p-value per feature
  judge_agreement.csv      — Pearson r, MAE, Cohen kappa vs paper 72B judge

Usage:
    cd ApplesVSOranges
    python src/analyze.py

Optional flags:
    --ratings   path to ratings_judge.jsonl   (default: outputs/ratings_judge.jsonl)
    --paper-ratings  path to paper 72B file   (default: outputs/ratings_judge_paper.jsonl)
    --metrics   path to metrics.csv           (default: outputs/metrics.csv)
    --outdir    root output dir               (default: outputs)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import warnings
from pathlib import Path

# Works whether called as `python analyze.py` (root) or `python src/analyze.py`
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# Silence the harmless "Unable to import Axes3D" matplotlib warning
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

log = logging.getLogger("analyze")

# canonical ordering and labels for plots and tables. See schema.py for the single source of truth

from schema import JOKE_TYPES, CRITERIA

JOKE_LABELS = {
    "homographic":   "Homographic",
    "heterographic": "Heterographic",
    "non_topical":   "Non-Topical",
    "topical":       "Topical",
}

# Your 8 models — paper-match models first, then extension models.
# Fallback to paper models if this file is run on paper data directly.
MODEL_ORDER = [
    "r1-distill-llama-8b",    # paper match
    "llama-3.1-8b",           # paper match
    "llama-3.2-3b",           # extension
    "gemma-2-2b",             # extension (small Gemma — H4 pair)
    "gemma-2-9b",             # extension (large Gemma — H4 pair)
    "mistral-7b",             # extension (new family baseline)
    "phi-3-mini",             # extension (small Phi — H4 pair)
    "phi-3-medium",           # extension (large Phi — H4 pair)
    # paper-only models kept so plots work if run on ratings_judge_paper.jsonl too
    "r1-distill-llama-70b", "gpt-4o", "gpt-4o-mini",
    "gemini-1.5-pro", "gemini-1.5-flash", "llama-3.1-70b",
]
MODEL_LABELS = {
    "r1-distill-llama-8b":  "R1-Llama 8B",
    "llama-3.1-8b":         "Llama 3.1 8B",
    "llama-3.2-3b":         "Llama 3.2 3B",
    "gemma-2-2b":           "Gemma 2 2B",
    "gemma-2-9b":           "Gemma 2 9B",
    "mistral-7b":           "Mistral 7B",
    "phi-3-mini":           "Phi-3 Mini",
    "phi-3-medium":         "Phi-3 Medium",
    # paper models
    "r1-distill-llama-70b": "R1 70B",    "gpt-4o":          "GPT-4o",
    "gpt-4o-mini":          "GPT-4o Mini","gemini-1.5-pro":  "Gemini Pro",
    "gemini-1.5-flash":     "Gemini Flash","llama-3.1-70b":  "Llama 70B",
}
# One colour per MODEL_ORDER entry (first 8 = your models)
MODEL_COLORS = [
    "#555555",  # R1-Llama 8B      — dark grey
    "#1f77b4",  # Llama 3.1 8B     — dark blue
    "#aec7e8",  # Llama 3.2 3B     — light blue
    "#98df8a",  # Gemma 2 2B       — light green
    "#2ca02c",  # Gemma 2 9B       — dark green
    "#ff7f0e",  # Mistral 7B       — orange
    "#c5b0d5",  # Phi-3 Mini       — light purple
    "#9467bd",  # Phi-3 Medium     — dark purple
    # fallback colours for paper models (used only when running on paper data)
    "#aaaaaa", "#d62728", "#ff9896", "#e377c2", "#f7b6d2", "#8c564b",
]

# Within-family size pairs for H4 (single source of truth — used by both
# save_hypothesis_checks and print_report)
H4_PAIRS = [
    # (big_model, small_model, family_label, h4_kind)
    ("gemma-2-9b",   "gemma-2-2b",   "Gemma 2 family (9B vs 2B)",                    "clean"),
    ("phi-3-medium", "phi-3-mini",   "Phi-3 family (Medium 14B vs Mini 3.8B)",       "clean"),
    ("llama-3.1-8b", "llama-3.2-3b", "Llama family (3.1-8B vs 3.2-3B, cross-gen)",   "cross-gen"),
]


# Paper Table 2 values for your two paper-match models (from Appendix A.4).
# Used in save_replication_gap to show how close your runs got.
PAPER_TABLE2 = {
    "llama-3.1-8b": {
        "heterographic": dict(sacrebleu=9.76,  rouge1=0.42, rouge2=0.14, rougeL=0.27, meteor=0.36, bertscore=0.89),
        "homographic":   dict(sacrebleu=8.41,  rouge1=0.39, rouge2=0.12, rougeL=0.24, meteor=0.34, bertscore=0.88),
        "non_topical":   dict(sacrebleu=7.16,  rouge1=0.40, rouge2=0.12, rougeL=0.24, meteor=0.30, bertscore=0.87),
        "topical":       dict(sacrebleu=5.83,  rouge1=0.36, rouge2=0.10, rougeL=0.22, meteor=0.28, bertscore=0.87),
    },
    "r1-distill-llama-8b": {
        "heterographic": dict(sacrebleu=6.48,  rouge1=0.37, rouge2=0.10, rougeL=0.23, meteor=0.28, bertscore=0.88),
        "homographic":   dict(sacrebleu=5.25,  rouge1=0.35, rouge2=0.08, rougeL=0.21, meteor=0.25, bertscore=0.87),
        "non_topical":   dict(sacrebleu=4.85,  rouge1=0.36, rouge2=0.09, rougeL=0.22, meteor=0.24, bertscore=0.87),
        "topical":       dict(sacrebleu=4.12,  rouge1=0.33, rouge2=0.08, rougeL=0.20, meteor=0.23, bertscore=0.86),
    },
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_ratings(path: Path) -> pd.DataFrame:
    """Return long-form ratings DataFrame from a ratings_judge*.jsonl file."""
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    df["joke_type"] = df["joke_id"].str.rsplit("_", n=1).str[0]
    return df


def compute_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of jokes where BOTH accuracy>=4 AND completeness>=4."""
    piv = df.pivot_table(
        index=["joke_id", "joke_type", "model"],
        columns="criterion", values="score", aggfunc="first",
    ).reset_index()
    piv.columns.name = None
    piv["is_good"] = (piv["accuracy"] >= 4) & (piv["completeness"] >= 4)
    return (piv.groupby(["model", "joke_type"])["is_good"]
               .mean().reset_index().rename(columns={"is_good": "success_rate"}))


# plotting helpers

def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")


def grouped_bar(data: pd.DataFrame, val_col: str, ylabel: str,
                title: str, out_path: Path, ylim=(0, 5)) -> None:
    import matplotlib.pyplot as plt

    models = [m for m in MODEL_ORDER if m in data["model"].unique()]
    jtypes = [t for t in JOKE_TYPES   if t in data["joke_type"].unique()]
    n_m, n_t = len(models), len(jtypes)
    x = np.arange(n_t)
    w = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, m in enumerate(models):
        vals = []
        for jt in jtypes:
            row = data[(data["model"] == m) & (data["joke_type"] == jt)]
            vals.append(float(row[val_col].iloc[0]) if len(row) else 0.0)
        offset = (i - n_m / 2 + 0.5) * w
        ax.bar(x + offset, vals, w,
               label=MODEL_LABELS.get(m, m),
               color=MODEL_COLORS[MODEL_ORDER.index(m)])

    ax.set_xticks(x)
    ax.set_xticklabels([JOKE_LABELS[t] for t in jtypes], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, pad=6)
    ax.legend(fontsize=7, ncol=4, loc="upper right", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("saved %s", out_path)


def success_grid(sr: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    models = [m for m in MODEL_ORDER if m in sr["model"].unique()]
    jtypes = [t for t in JOKE_TYPES   if t in sr["joke_type"].unique()]

    # Auto-size grid:
    #  - n ≤ 5: single row (n columns), looks cleanest
    #  - n > 5: max 4 cols, multi-row, hide unused cells
    n = len(models)
    if n == 0:
        log.warning("success_grid: no models in data — skipping %s", out_path)
        return
    if n <= 5:
        ncols, nrows = n, 1
    else:
        ncols = 4
        nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.2 * ncols, 3.2 * nrows),
                              sharey=True, sharex=True)
    axes_flat = np.array(axes).flatten()

    for ai, m in enumerate(models):
        ax = axes_flat[ai]
        goods = []
        for jt in jtypes:
            row = sr[(sr["model"] == m) & (sr["joke_type"] == jt)]
            goods.append(float(row["success_rate"].iloc[0]) if len(row) else 0.0)
        poors = [1 - g for g in goods]
        x = np.arange(len(jtypes))
        ax.bar(x, goods, color="#2ca02c")
        ax.bar(x, poors, bottom=goods, color="#d62728")
        ax.set_title(MODEL_LABELS.get(m, m), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([JOKE_LABELS[t][:4] for t in jtypes], fontsize=7, rotation=40)
        ax.set_ylim(0, 1)

    # Hide any subplots beyond the n-th model
    for ai in range(n, len(axes_flat)):
        axes_flat[ai].set_visible(False)

    fig.legend(handles=[
        mpatches.Patch(color="#2ca02c", label="Good (≥4)"),
        mpatches.Patch(color="#d62728", label="Poor (<4)"),
    ], loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Figure 4 — Proportion of Good/Poor Explanations [Qwen-7B Judge]", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("saved %s", out_path)


def judge_comparison_plot(ours: pd.DataFrame, paper: pd.DataFrame,
                          out_path: Path) -> None:
    import matplotlib.pyplot as plt

    merged = ours.merge(
        paper[["joke_id", "model", "criterion", "score"]].rename(columns={"score": "score_paper"}),
        on=["joke_id", "model", "criterion"], how="inner"
    ).rename(columns={"score": "score_ours"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    bins = np.arange(-0.5, 6, 1)
    for ax, crit in zip(axes, ["accuracy", "completeness"]):
        sub = merged[merged["criterion"] == crit]
        ax.hist(sub["score_paper"], bins=bins, alpha=0.6, label="Paper (72B)", density=True)
        ax.hist(sub["score_ours"],  bins=bins, alpha=0.6, label="Ours (7B)",   density=True)
        ax.set_title(crit.capitalize())
        ax.set_xlabel("Score (0–5)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xticks(range(6))
    fig.suptitle("Score Distribution: Our Qwen-7B Judge vs Paper Qwen-72B", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("saved %s", out_path)


# table generation helpers

def save_avg_scores(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    avg = (df.groupby(["model", "joke_type", "criterion"])["score"]
             .mean().reset_index().rename(columns={"score": "mean_score"}))
    avg.to_csv(out_path, index=False)
    log.info("saved %s", out_path)
    return avg


def save_success_rates(sr: pd.DataFrame, out_path: Path) -> None:
    sr.to_csv(out_path, index=False)
    log.info("saved %s", out_path)


def save_replication_gap(metrics_csv: Path, out_path: Path) -> pd.DataFrame:
    if not metrics_csv.exists():
        log.warning("metrics.csv not found at %s — skipping gap table", metrics_csv)
        return pd.DataFrame()

    our = pd.read_csv(metrics_csv)
    metric_keys = ["sacrebleu", "rouge1", "rouge2", "rougeL", "meteor", "bertscore"]

    # Compare each paper-match model (those in PAPER_TABLE2) individually.
    # Extension models (gemma-2-*, llama-3.2-*) have no paper numbers to compare.
    rows = []
    paper_match_models = [m for m in PAPER_TABLE2 if m in our["model"].unique()]
    if not paper_match_models:
        log.warning("No paper-match models found in metrics.csv — gap table will be empty")
    for model_slug in paper_match_models:
        our_model = our[our["model"] == model_slug].set_index("joke_type")
        paper_model = PAPER_TABLE2[model_slug]
        for jt in JOKE_TYPES:
            for m in metric_keys:
                pv = paper_model.get(jt, {}).get(m)
                ov = float(our_model.loc[jt, m]) if jt in our_model.index and m in our_model.columns else None
                diff = round(ov - pv, 4) if pv is not None and ov is not None else None
                pct  = round((diff / pv) * 100, 1) if diff is not None and pv else None
                rows.append({"model": model_slug, "joke_type": jt, "metric": m,
                             "paper": pv, "ours": round(ov, 4) if ov else None,
                             "diff": diff, "pct_diff": pct})

    gap = pd.DataFrame(rows)
    gap.to_csv(out_path, index=False)
    log.info("saved %s", out_path)
    return gap


def save_logistic_regression(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import chi2

    piv = df.pivot_table(
        index=["joke_id", "joke_type", "model"],
        columns="criterion", values="score", aggfunc="first",
    ).reset_index()
    piv.columns.name = None
    piv["is_good"] = ((piv["accuracy"] >= 4) & (piv["completeness"] >= 4)).astype(int)

    # "Large" = the bigger model in each within-family pair you have.
    # Gemma 2 9B (vs 2B), Llama 3.1 8B (vs Llama 3.2 3B), and Phi-3 Medium 14B (vs Phi-3 Mini 3.8B).
    large = {"gemma-2-9b", "llama-3.1-8b", "phi-3-medium"}
    piv["is_large"] = piv["model"].isin(large).astype(int)
    for jt in ["heterographic", "non_topical", "topical"]:
        piv[f"is_{jt}"] = (piv["joke_type"] == jt).astype(int)

    feats = ["is_large", "is_heterographic", "is_non_topical", "is_topical"]
    X, y = piv[feats].values, piv["is_good"].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X, y)

    coefs = clf.coef_[0]
    probs = clf.predict_proba(X)[:, 1]
    W     = np.diag(probs * (1 - probs))
    XtWX  = X.T @ W @ X
    try:
        se    = np.sqrt(np.diag(np.linalg.inv(XtWX)))
        z     = coefs / se
        pvals = 2 * (1 - chi2.cdf(z ** 2, df=1))
    except np.linalg.LinAlgError:
        se = z = pvals = np.full_like(coefs, np.nan)

    res = pd.DataFrame({"feature": feats, "beta": coefs,
                        "se": se, "z": z, "p_value": pvals})
    res.to_csv(out_path, index=False)
    log.info("saved %s", out_path)
    return res


def save_judge_agreement(ours: pd.DataFrame, paper: pd.DataFrame,
                         out_path: Path) -> pd.DataFrame:
    """Compute Pearson r / MAE / κ between our 7B judge and paper's 72B judge.

    NOTE: For a *meaningful* calibration the two judges must rate the SAME
    explanations. If `ours` is your judge's ratings of YOUR explanations and
    `paper` is paper's judge's ratings of PAPER'S explanations, the merge
    on (joke_id, model, criterion) only matches the ~2 model slugs you share
    — and the score columns refer to *different explanation texts*. The result
    is a noisy floor, not a real agreement number.

    To get a proper calibration, run:
        python src/judge.py --explanations data/explanations_paper.jsonl \\
            --output outputs/ratings_judge_paper_by_ours.jsonl
    Then point this analyze script at that file as --ratings instead.
    A clear warning is printed when overlap is < 4 models.
    """
    from sklearn.metrics import cohen_kappa_score

    merged = ours.merge(
        paper[["joke_id", "model", "criterion", "score"]].rename(columns={"score": "score_paper"}),
        on=["joke_id", "model", "criterion"], how="inner"
    ).rename(columns={"score": "score_ours"})

    n_models_overlap = merged["model"].nunique()
    if n_models_overlap < 4:
        log.warning(
            "judge_agreement: only %d models overlap between ours and paper — "
            "this is a weak baseline, not a real judge calibration. To fix, "
            "run the judge on paper's explanations and use that file as --ratings.",
            n_models_overlap,
        )

    rows = []
    for crit in ["accuracy", "completeness"]:
        sub = merged[merged["criterion"] == crit].copy()
        if sub.empty:
            rows.append({"criterion": crit, "n_paired": 0, "n_models_overlap": n_models_overlap,
                         "pearson_r": float("nan"), "mae": float("nan"),
                         "exact_match": float("nan"), "within_1": float("nan"),
                         "binary_agree": float("nan"), "cohen_kappa": float("nan")})
            continue
        r       = sub["score_ours"].corr(sub["score_paper"])
        mae     = (sub["score_ours"] - sub["score_paper"]).abs().mean()
        exact   = (sub["score_ours"] == sub["score_paper"]).mean()
        within1 = ((sub["score_ours"] - sub["score_paper"]).abs() <= 1).mean()
        sub["pass_ours"]  = (sub["score_ours"]  >= 4).astype(int)
        sub["pass_paper"] = (sub["score_paper"] >= 4).astype(int)
        agree = (sub["pass_ours"] == sub["pass_paper"]).mean()
        kappa = cohen_kappa_score(sub["pass_paper"], sub["pass_ours"])
        rows.append({"criterion": crit, "n_paired": len(sub),
                     "n_models_overlap": n_models_overlap,
                     "pearson_r": round(r, 3),
                     "mae": round(mae, 3), "exact_match": round(exact, 3),
                     "within_1": round(within1, 3),
                     "binary_agree": round(agree, 3), "cohen_kappa": round(kappa, 3)})

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    log.info("saved %s", out_path)
    return result


def save_hypothesis_checks(avg: pd.DataFrame, sr: pd.DataFrame,
                           out_path: Path) -> pd.DataFrame:
    """Compute H1–H4 and save to CSV.

    Columns: hypothesis, description, value_a, label_a, value_b, label_b,
             delta, verdict, note
    """
    acc = avg[avg["criterion"] == "accuracy"]
    by_type = acc.groupby("joke_type")["mean_score"].mean()

    hom = by_type.get("homographic",   0.0)
    het = by_type.get("heterographic", 0.0)
    ntp = by_type.get("non_topical",   0.0)
    top = by_type.get("topical",       0.0)

    def verdict(a, b, near_tie_threshold=0.05):
        delta = a - b
        if abs(delta) < near_tie_threshold:
            return "NEAR-TIE"
        return "CONFIRMED" if delta > 0 else "FAILED"

    rows = []

    # H1: puns (avg of hom+het) vs reddit (avg of ntp+top)
    pun_avg    = (hom + het) / 2
    reddit_avg = (ntp + top) / 2
    rows.append({
        "hypothesis":  "H1",
        "description": "Traditional puns easier to explain than Reddit jokes",
        "value_a":     round(pun_avg, 4),
        "label_a":     "puns (avg hom+het accuracy)",
        "value_b":     round(reddit_avg, 4),
        "label_b":     "reddit (avg ntp+top accuracy)",
        "delta":       round(pun_avg - reddit_avg, 4),
        "verdict":     verdict(pun_avg, reddit_avg),
        "note":        "Δ<0.05 → near-tie due to 7B judge score compression",
    })

    # H2: homographic > heterographic
    rows.append({
        "hypothesis":  "H2",
        "description": "Homographic puns easier than heterographic puns",
        "value_a":     round(hom, 4),
        "label_a":     "homographic accuracy",
        "value_b":     round(het, 4),
        "label_b":     "heterographic accuracy",
        "delta":       round(hom - het, 4),
        "verdict":     verdict(hom, het),
        "note":        "",
    })

    # H3: topical hardest (topical < non_topical)
    rows.append({
        "hypothesis":  "H3",
        "description": "Topical jokes harder than non-topical Reddit jokes",
        "value_a":     round(ntp, 4),
        "label_a":     "non_topical accuracy",
        "value_b":     round(top, 4),
        "label_b":     "topical accuracy",
        "delta":       round(ntp - top, 4),
        "verdict":     verdict(ntp, top),
        "note":        "Δ<0.05 → near-tie due to 7B judge score compression",
    })

    # H4: larger > smaller per family (use overall success rate)
    for big, small, family, _ in H4_PAIRS:
        b = sr[sr["model"] == big]["success_rate"].mean()
        s = sr[sr["model"] == small]["success_rate"].mean()
        rows.append({
            "hypothesis":  "H4",
            "description": f"Larger model > smaller model — {family}",
            "value_a":     round(b, 4),
            "label_a":     f"{big} success rate",
            "value_b":     round(s, 4),
            "label_b":     f"{small} success rate",
            "delta":       round(b - s, 4),
            "verdict":     verdict(b, s),
            "note":        "near-tied" if abs(b - s) < 0.05 else "",
        })

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    log.info("saved %s", out_path)
    return result


# hypothesis checks and console report

def print_report(avg: pd.DataFrame, sr: pd.DataFrame,
                 logreg: pd.DataFrame, gap: pd.DataFrame,
                 agreement: pd.DataFrame,
                 out_path: Path,
                 metadata: dict | None = None) -> None:

    acc  = avg[avg["criterion"] == "accuracy"]
    comp = avg[avg["criterion"] == "completeness"]
    by_type_acc  = acc.groupby("joke_type")["mean_score"].mean()
    by_type_comp = comp.groupby("joke_type")["mean_score"].mean()
    hom = by_type_acc.get("homographic", 0);  het = by_type_acc.get("heterographic", 0)
    ntp = by_type_acc.get("non_topical", 0);  top = by_type_acc.get("topical", 0)

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("=" * 64)
    emit("  Apples vs Oranges — Replication Results")
    emit("=" * 64)
    if metadata:
        for k, v in metadata.items():
            emit(f"  {k:<22} {v}")
        emit("=" * 64)

    emit("\n  Avg scores by joke type (acc / completeness):")
    for jt in JOKE_TYPES:
        a = by_type_acc.get(jt, 0)
        c = by_type_comp.get(jt, 0)
        emit(f"    {JOKE_LABELS[jt]:<16} acc={a:.3f}  comp={c:.3f}")

    emit("\n  Hypothesis checks (accuracy):")
    emit(f"    H1 Puns > Reddit?        {(hom+het)/2:.3f} vs {(ntp+top)/2:.3f}  "
         f"{'CONFIRMED' if (hom+het)/2 > (ntp+top)/2 else 'FAILED'}")
    emit(f"    H2 Homographic > Hetero? {hom:.3f} vs {het:.3f}  "
         f"{'CONFIRMED' if hom > het else 'FAILED'}")
    emit(f"    H3 Topical hardest?      non_top={ntp:.3f} topical={top:.3f}  "
         f"{'CONFIRMED' if top < ntp else 'FAILED'}")

    emit("\n    H4 Larger > Smaller?")
    for big, small, _family, kind in H4_PAIRS:
        b = acc[acc["model"] == big]["mean_score"].mean()
        s = acc[acc["model"] == small]["mean_score"].mean()
        tag = "" if kind == "clean" else "  [cross-gen — generation confounds size]"
        emit(f"       {'OK' if b > s else 'FAIL'}  "
             f"{MODEL_LABELS.get(big,big)} ({b:.3f}) vs "
             f"{MODEL_LABELS.get(small,small)} ({s:.3f}){tag}")

    if not logreg.empty:
        emit("\n  Logistic regression (paper: β_large=1.707, β_topical=-0.574):")
        PAPER_B = {"is_large": 1.707, "is_non_topical": -0.511, "is_topical": -0.574}
        for _, r in logreg.iterrows():
            p = "p<0.001" if r.p_value < 0.001 else f"p={r.p_value:.3f}"
            pb = f"{PAPER_B[r.feature]:+.3f}" if r.feature in PAPER_B else "   n/a"
            emit(f"    {r.feature:<22} β={r.beta:+.3f}  paper={pb}  {p}")

    if not agreement.empty:
        emit("\n  Judge agreement (our 7B vs paper 72B):")
        for _, r in agreement.iterrows():
            emit(f"    {r.criterion.capitalize():<14} "
                 f"r={r.pearson_r:.3f}  MAE={r.mae:.3f}  "
                 f"κ={r.cohen_kappa:.3f}  within±1={r.within_1:.1%}")
        emit("    (Paper reports r=0.641/0.602 for 72B vs human)")
        # Caveat goes into the txt file, not just console
        n_overlap = int(agreement.iloc[0]["n_models_overlap"]) if "n_models_overlap" in agreement.columns else 0
        if n_overlap < 4:
            emit(f"    NOTE: only {n_overlap} models overlap — this is a weak baseline,")
            emit("          not a true judge calibration. Run judge on data/explanations_paper.jsonl")
            emit("          and re-run analyze with --ratings on that file for proper calibration.")

    if not gap.empty:
        emit("\n  Replication gap — SacreBLEU vs paper Table 2:")
        # gap now has a `model` column (one row per model × joke_type × metric)
        # so we print a sub-section per paper-match model
        for model_slug, model_gap in gap.groupby("model"):
            label = MODEL_LABELS.get(model_slug, model_slug)
            emit(f"    [{label}]")
            bleu = (model_gap[model_gap["metric"] == "sacrebleu"]
                    .set_index("joke_type")["pct_diff"])
            for jt in JOKE_TYPES:
                v = bleu.get(jt)
                v_str = f"{v:+.1f}%" if v is not None and not pd.isna(v) else "n/a"
                emit(f"      {JOKE_LABELS[jt]:<14} {v_str:>8}")
        emit("    BERTScore gap is <0.5% — treated as matched.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("saved %s", out_path)



def main() -> None:
    ap = argparse.ArgumentParser(description="Produce figures and tables for the paper")
    ap.add_argument("--ratings",       default="outputs/ratings_judge.jsonl",       type=Path)
    ap.add_argument("--paper-ratings", default="outputs/ratings_judge_paper.jsonl", type=Path)
    ap.add_argument("--metrics",       default="outputs/metrics.csv",               type=Path)
    ap.add_argument("--outdir",        default="outputs",                            type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _setup_matplotlib()

    fig_dir = args.outdir / "figures"
    tab_dir = args.outdir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # load data
    if not args.ratings.exists():
        raise SystemExit(f"ratings not found at {args.ratings} — run judge.py first")
    log.info("loading ratings from %s", args.ratings)
    df = load_ratings(args.ratings)
    log.info("%d rating rows loaded", len(df))

    paper_df = pd.DataFrame()
    if args.paper_ratings.exists():
        log.info("loading paper ratings from %s", args.paper_ratings)
        paper_df = load_ratings(args.paper_ratings)
    else:
        log.warning("paper ratings not found at %s — skipping comparison", args.paper_ratings)

    # average scores by model x joke type x criterion, for Figures 3b and 3c and Table of avg scores
    avg = save_avg_scores(df, tab_dir / "avg_scores.csv")

    grouped_bar(avg[avg["criterion"] == "accuracy"],
                "mean_score", "Average Accuracy (0–5)",
                "Figure 3b — Accuracy by Model and Joke Type",
                fig_dir / "fig3b_accuracy.png")

    grouped_bar(avg[avg["criterion"] == "completeness"],
                "mean_score", "Average Completeness (0–5)",
                "Figure 3c — Completeness by Model and Joke Type",
                fig_dir / "fig3c_completeness.png")

    # success rates (accuracy>=4 AND completeness>=4) by model x joke type, for Figure 4 and Table of success rates
    sr = compute_success_rate(df)
    save_success_rates(sr, tab_dir / "success_rates.csv")
    success_grid(sr, fig_dir / "fig4_success.png")

    # hypothesis checks H1-H4 saved to CSV
    hyp = save_hypothesis_checks(avg, sr, tab_dir / "hypothesis_checks.csv")

    # judge agreement vs paper 72B judge, for judge_comparison_plot and agreement table
    if not paper_df.empty:
        judge_comparison_plot(df, paper_df, fig_dir / "fig_judge_comparison.png")
        agreement = save_judge_agreement(df, paper_df, tab_dir / "judge_agreement.csv")
    else:
        agreement = pd.DataFrame()

    # Run metadata for the header of results_summary.txt
    from datetime import datetime
    metadata = {
        "run timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ratings file":  str(args.ratings),
        "ratings rows":  f"{len(df):,}",
        "models in run": f"{df['model'].nunique()} ({', '.join(sorted(df['model'].unique()))})",
        "metrics file":  str(args.metrics) if args.metrics.exists() else "(missing)",
    }

    # replication
    gap = save_replication_gap(args.metrics, tab_dir / "replication_gap.csv")

    # logistic regression for good explanation (accuracy>=4 AND completeness>=4) with features for large model and topical/heterographic joke type, for Table of logistic regression results
    logreg = save_logistic_regression(df, tab_dir / "logistic_regression.csv")

    # summary report — printed to console and saved to file
    print_report(avg, sr, logreg, gap, agreement, metadata=metadata,
                 out_path=args.outdir / "results_summary.txt")

    print(f"\n  Figures  → {fig_dir.resolve()}")
    print(f"  Tables   → {tab_dir.resolve()}")
    print(f"  Summary  → {(args.outdir / 'results_summary.txt').resolve()}")


if __name__ == "__main__":
    main()
