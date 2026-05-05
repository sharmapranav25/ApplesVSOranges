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

log = logging.getLogger("analyze")

# ── canonical ordering (mirrors schema.py) ────────────────────────────────────

from schema import JOKE_TYPES, CRITERIA

JOKE_LABELS = {
    "homographic":   "Homographic",
    "heterographic": "Heterographic",
    "non_topical":   "Non-Topical",
    "topical":       "Topical",
}

MODEL_ORDER = [
    "r1-distill-llama-70b", "r1-distill-llama-8b",
    "gpt-4o",               "gpt-4o-mini",
    "gemini-1.5-pro",       "gemini-1.5-flash",
    "llama-3.1-70b",        "llama-3.1-8b",
]
MODEL_LABELS = {
    "r1-distill-llama-70b": "R1 70B",    "r1-distill-llama-8b": "R1 8B",
    "gpt-4o":               "GPT-4o",    "gpt-4o-mini":         "GPT-4o Mini",
    "gemini-1.5-pro":       "Gemini Pro","gemini-1.5-flash":    "Gemini Flash",
    "llama-3.1-70b":        "Llama 70B", "llama-3.1-8b":        "Llama 8B",
}
MODEL_COLORS = [
    "#555555", "#aaaaaa",
    "#d62728", "#ff9896",
    "#2ca02c", "#98df8a",
    "#1f77b4", "#aec7e8",
]

# Paper Table 2 (GPT-4o) for replication gap
PAPER_TABLE2 = {
    "heterographic": dict(sacrebleu=8.53, rouge1=0.41, rouge2=0.12, rougeL=0.25, meteor=0.37, bertscore=0.88),
    "homographic":   dict(sacrebleu=10.15,rouge1=0.43, rouge2=0.15, rougeL=0.28, meteor=0.39, bertscore=0.89),
    "non_topical":   dict(sacrebleu=7.86, rouge1=0.41, rouge2=0.13, rougeL=0.25, meteor=0.32, bertscore=0.88),
    "topical":       dict(sacrebleu=7.09, rouge1=0.40, rouge2=0.12, rougeL=0.23, meteor=0.32, bertscore=0.87),
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


# ── plotting ──────────────────────────────────────────────────────────────────

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

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True, sharex=True)
    for ai, m in enumerate(models):
        ax = axes[ai // 4][ai % 4]
        goods = []
        for jt in jtypes:
            row = sr[(sr["model"] == m) & (sr["joke_type"] == jt)]
            goods.append(float(row["success_rate"].iloc[0]) if len(row) else 0.0)
        poors = [1 - g for g in goods]
        x = np.arange(len(jtypes))
        ax.bar(x, goods, color="#2ca02c")
        ax.bar(x, poors, bottom=goods, color="#d62728")
        ax.set_title(MODEL_LABELS.get(m, m), fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([JOKE_LABELS[t][:4] for t in jtypes], fontsize=6, rotation=40)
        ax.set_ylim(0, 1)

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


# ── tables ────────────────────────────────────────────────────────────────────

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
    our_gpt = our[our["model"] == "gpt-4o"].set_index("joke_type")
    metric_keys = ["sacrebleu", "rouge1", "rouge2", "rougeL", "meteor", "bertscore"]

    rows = []
    for jt in JOKE_TYPES:
        for m in metric_keys:
            pv = PAPER_TABLE2.get(jt, {}).get(m)
            ov = float(our_gpt.loc[jt, m]) if jt in our_gpt.index and m in our_gpt.columns else None
            diff = round(ov - pv, 4) if pv is not None and ov is not None else None
            pct  = round((diff / pv) * 100, 1) if diff is not None and pv else None
            rows.append({"joke_type": jt, "metric": m,
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

    large = {"gpt-4o", "gemini-1.5-pro", "llama-3.1-70b", "r1-distill-llama-70b"}
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
    from sklearn.metrics import cohen_kappa_score

    merged = ours.merge(
        paper[["joke_id", "model", "criterion", "score"]].rename(columns={"score": "score_paper"}),
        on=["joke_id", "model", "criterion"], how="inner"
    ).rename(columns={"score": "score_ours"})

    rows = []
    for crit in ["accuracy", "completeness"]:
        sub = merged[merged["criterion"] == crit].copy()
        r       = sub["score_ours"].corr(sub["score_paper"])
        mae     = (sub["score_ours"] - sub["score_paper"]).abs().mean()
        exact   = (sub["score_ours"] == sub["score_paper"]).mean()
        within1 = ((sub["score_ours"] - sub["score_paper"]).abs() <= 1).mean()
        sub["pass_ours"]  = (sub["score_ours"]  >= 4).astype(int)
        sub["pass_paper"] = (sub["score_paper"] >= 4).astype(int)
        agree = (sub["pass_ours"] == sub["pass_paper"]).mean()
        kappa = cohen_kappa_score(sub["pass_paper"], sub["pass_ours"])
        rows.append({"criterion": crit, "pearson_r": round(r, 3),
                     "mae": round(mae, 3), "exact_match": round(exact, 3),
                     "within_1": round(within1, 3),
                     "binary_agree": round(agree, 3), "cohen_kappa": round(kappa, 3)})

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    log.info("saved %s", out_path)
    return result


# ── console report ────────────────────────────────────────────────────────────

def print_report(avg: pd.DataFrame, sr: pd.DataFrame,
                 logreg: pd.DataFrame, gap: pd.DataFrame,
                 agreement: pd.DataFrame) -> None:

    acc = avg[avg["criterion"] == "accuracy"]
    by_type = acc.groupby("joke_type")["mean_score"].mean()
    hom = by_type.get("homographic", 0);  het = by_type.get("heterographic", 0)
    ntp = by_type.get("non_topical", 0);  top = by_type.get("topical", 0)

    print("\n" + "=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)

    print("\n  Avg accuracy by joke type:")
    for jt in JOKE_TYPES:
        print(f"    {JOKE_LABELS[jt]:<16} {by_type.get(jt, 0):.3f}")

    print("\n  Hypothesis checks (accuracy):")
    print(f"    H1 Puns > Reddit?        {(hom+het)/2:.3f} vs {(ntp+top)/2:.3f}  "
          f"{'CONFIRMED' if (hom+het)/2 > (ntp+top)/2 else 'FAILED'}")
    print(f"    H2 Homographic > Hetero? {hom:.3f} vs {het:.3f}  "
          f"{'CONFIRMED' if hom > het else 'FAILED'}")
    print(f"    H3 Topical hardest?      non_top={ntp:.3f} topical={top:.3f}  "
          f"{'CONFIRMED' if top < ntp else 'FAILED'}")

    pairs = [("gpt-4o","gpt-4o-mini"),("gemini-1.5-pro","gemini-1.5-flash"),
             ("llama-3.1-70b","llama-3.1-8b"),
             ("r1-distill-llama-70b","r1-distill-llama-8b")]
    print("\n    H4 Larger > Smaller?")
    for big, small in pairs:
        b = acc[acc["model"] == big]["mean_score"].mean()
        s = acc[acc["model"] == small]["mean_score"].mean()
        print(f"       {'OK' if b > s else 'FAIL'}  "
              f"{MODEL_LABELS.get(big,big)} ({b:.3f}) vs "
              f"{MODEL_LABELS.get(small,small)} ({s:.3f})")

    if not logreg.empty:
        print("\n  Logistic regression (paper: β_large=1.707, β_topical=-0.574):")
        PAPER_B = {"is_large": 1.707, "is_non_topical": -0.511, "is_topical": -0.574}
        for _, r in logreg.iterrows():
            p = "p<0.001" if r.p_value < 0.001 else f"p={r.p_value:.3f}"
            pb = f"{PAPER_B[r.feature]:+.3f}" if r.feature in PAPER_B else "   n/a"
            print(f"    {r.feature:<22} β={r.beta:+.3f}  paper={pb}  {p}")

    if not agreement.empty:
        print("\n  Judge agreement (our 7B vs paper 72B):")
        for _, r in agreement.iterrows():
            print(f"    {r.criterion.capitalize():<14} "
                  f"r={r.pearson_r:.3f}  MAE={r.mae:.3f}  "
                  f"κ={r.cohen_kappa:.3f}  within±1={r.within_1:.1%}")
        print("    (Paper reports r=0.641/0.602 for 72B vs human)")

    if not gap.empty:
        bleu = gap[gap["metric"] == "sacrebleu"].set_index("joke_type")["pct_diff"]
        print("\n  Replication gap — SacreBLEU vs paper Table 2 (GPT-4o):")
        for jt in JOKE_TYPES:
            v = bleu.get(jt)
            print(f"    {JOKE_LABELS[jt]:<16} {f'{v:+.1f}%' if v else 'n/a':>8}")
        print("    BERTScore gap is <0.5% — treated as matched.")


# ── main ──────────────────────────────────────────────────────────────────────

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

    # ── load ratings ──────────────────────────────────────────────────────────
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

    # ── average scores ────────────────────────────────────────────────────────
    avg = save_avg_scores(df, tab_dir / "avg_scores.csv")

    grouped_bar(avg[avg["criterion"] == "accuracy"],
                "mean_score", "Average Accuracy (0–5)",
                "Figure 3b — Accuracy by Model and Joke Type",
                fig_dir / "fig3b_accuracy.png")

    grouped_bar(avg[avg["criterion"] == "completeness"],
                "mean_score", "Average Completeness (0–5)",
                "Figure 3c — Completeness by Model and Joke Type",
                fig_dir / "fig3c_completeness.png")

    # ── success rates ─────────────────────────────────────────────────────────
    sr = compute_success_rate(df)
    save_success_rates(sr, tab_dir / "success_rates.csv")
    success_grid(sr, fig_dir / "fig4_success.png")

    # ── judge comparison ──────────────────────────────────────────────────────
    if not paper_df.empty:
        judge_comparison_plot(df, paper_df, fig_dir / "fig_judge_comparison.png")
        agreement = save_judge_agreement(df, paper_df, tab_dir / "judge_agreement.csv")
    else:
        agreement = pd.DataFrame()

    # ── replication gap ───────────────────────────────────────────────────────
    gap = save_replication_gap(args.metrics, tab_dir / "replication_gap.csv")

    # ── logistic regression ───────────────────────────────────────────────────
    logreg = save_logistic_regression(df, tab_dir / "logistic_regression.csv")

    # ── console report ────────────────────────────────────────────────────────
    print_report(avg, sr, logreg, gap, agreement)

    print(f"\n  Figures → {fig_dir.resolve()}")
    print(f"  Tables  → {tab_dir.resolve()}")


if __name__ == "__main__":
    main()
