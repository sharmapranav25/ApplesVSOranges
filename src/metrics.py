"""Task 4: automatic metrics vs. reference_explanation.

For each (model, joke_type) bucket, compute:
- SacreBLEU: corpus-level (sacrebleu.corpus_bleu(hyps, [refs]).score)
- ROUGE-1/2/L: per-instance F-measure (use_stemmer=True), averaged
- METEOR: per-instance (nltk word_tokenize on both), averaged
- BERTScore: per-instance F1 with roberta-large, averaged

Output: outputs/metrics.csv.

Why corpus-level for BLEU but per-instance for the others: that's what matches
the paper's Table 2 numbers (e.g., GPT-4o on hom = 8.51, which is in the typical
range for sacrebleu corpus_bleu on ~150 short references; per-instance averaged
sentence BLEU would land much lower).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import logging
from collections import defaultdict
from pathlib import Path

from io_jsonl import read_jsonl
from schema import MetricRow

log = logging.getLogger("metrics")


def _ensure_nltk() -> None:
    """Download the nltk resources METEOR needs, idempotently."""
    import nltk
    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def compute_bucket_metrics(
    hyps: list[str],
    refs: list[str],
    bertscore_batch_size: int = 32,
) -> dict[str, float]:
    """Compute one row of metrics for a single (model, joke_type) bucket."""
    import sacrebleu
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.meteor_score import meteor_score
    import bert_score

    assert len(hyps) == len(refs) and hyps, "non-empty parallel hyps/refs required"

    # SacreBLEU corpus-level.
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

    # ROUGE per-instance, averaged.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = 0.0, 0.0, 0.0
    for h, r in zip(hyps, refs):
        s = scorer.score(r, h)  # rouge_scorer signature: (target, prediction)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(hyps)
    r1, r2, rl = r1 / n, r2 / n, rl / n

    # METEOR per-instance, averaged.
    m = 0.0
    for h, r in zip(hyps, refs):
        m += meteor_score([nltk.word_tokenize(r)], nltk.word_tokenize(h))
    m /= n

    # BERTScore F1 per-instance, averaged.
    P, R, F1 = bert_score.score(
        hyps, refs,
        model_type="roberta-large",
        lang="en",
        verbose=False,
        batch_size=bertscore_batch_size,
        rescale_with_baseline=False,
    )
    bs = float(F1.mean())

    return {
        "sacrebleu": bleu,
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rl,
        "meteor": m,
        "bertscore": bs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Automatic metrics vs reference_explanation")
    ap.add_argument("--jokes", default="data/jokes.jsonl", type=Path)
    ap.add_argument("--explanations", default="data/explanations.jsonl", type=Path)
    ap.add_argument("--output", default="outputs/metrics.csv", type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap explanations processed (debug)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these model slugs")
    ap.add_argument("--types", nargs="+", default=None,
                    help="restrict to these joke types")
    ap.add_argument("--resume", action="store_true",
                    help="if --output exists, only compute (model, joke_type) buckets not present")
    ap.add_argument("--bertscore-batch-size", type=int, default=32)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _ensure_nltk()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    jokes = {j["id"]: j for j in read_jsonl(args.jokes)}
    if not jokes:
        raise SystemExit(f"no jokes in {args.jokes}; run preprocess.py first")

    explanations = list(read_jsonl(args.explanations))
    if args.limit is not None:
        explanations = explanations[: args.limit]
    if not explanations:
        raise SystemExit(f"no explanations in {args.explanations}")

    # Bucket: (model, joke_type) -> list[(hyp, ref)]
    buckets: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for e in explanations:
        joke = jokes.get(e["joke_id"])
        if joke is None:
            log.warning("unknown joke_id=%r in explanations; skipping", e["joke_id"])
            continue
        if args.models and e["model"] not in args.models:
            continue
        if args.types and joke["type"] not in args.types:
            continue
        buckets[(e["model"], joke["type"])].append((e["explanation"], joke["reference_explanation"]))

    # --resume: skip already-computed buckets.
    existing_keys: set[tuple[str, str]] = set()
    if args.resume and args.output.exists():
        with args.output.open() as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["model"], row["joke_type"]))
        log.info("resume: %d buckets already present in %s", len(existing_keys), args.output)

    rows: list[MetricRow] = []
    # Read prior rows so we preserve them in --resume mode.
    if args.resume and args.output.exists():
        with args.output.open() as f:
            for row in csv.DictReader(f):
                rows.append(MetricRow(
                    model=row["model"], joke_type=row["joke_type"], n=int(row["n"]),
                    sacrebleu=float(row["sacrebleu"]),
                    rouge1=float(row["rouge1"]), rouge2=float(row["rouge2"]),
                    rougeL=float(row["rougeL"]),
                    meteor=float(row["meteor"]), bertscore=float(row["bertscore"]),
                ))

    for (model, jt), pairs in sorted(buckets.items()):
        if (model, jt) in existing_keys:
            continue
        hyps = [h for h, _ in pairs]
        refs = [r for _, r in pairs]
        log.info("computing %s / %s (n=%d)", model, jt, len(pairs))
        m = compute_bucket_metrics(hyps, refs, bertscore_batch_size=args.bertscore_batch_size)
        rows.append(MetricRow(model=model, joke_type=jt, n=len(pairs), **m))

    # Write out.
    fieldnames = ["model", "joke_type", "n", "sacrebleu", "rouge1", "rouge2", "rougeL", "meteor", "bertscore"]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.model, x.joke_type)):
            w.writerow(dc.asdict(r))
    log.info("wrote %d rows to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
