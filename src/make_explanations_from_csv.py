"""Dev helper: derive `data/explanations.jsonl` and (optionally) the paper's
Qwen judge ratings from `new_jokes.csv`. Used until the teammate ships their
own explanations file.

Why this exists: the production pipeline assumes a teammate produces
explanations. While we wait, we want to run end-to-end tests against the
exact strings the paper used so the metric sanity check (GPT-4o on hom)
is reproducible.
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import logging
from pathlib import Path

import pandas as pd

from io_jsonl import write_jsonl
from preprocess import assign_joke_ids, load_typed_csv
from schema import MODEL_SLUG_MAP, PAPER_QWEN_JUDGE_COLS, Explanation, JUDGE_MODEL_NAME, Rating

log = logging.getLogger("make_explanations")


def build_explanations(csv_path: Path) -> list[Explanation]:
    df = load_typed_csv(csv_path)
    ids = assign_joke_ids(df)
    explanations: list[Explanation] = []
    for row_idx, row in df.iterrows():
        joke_id = ids.loc[row_idx]
        for col, slug in MODEL_SLUG_MAP.items():
            text = row.get(col)
            if pd.isna(text) or not str(text).strip():
                continue
            explanations.append(
                Explanation(joke_id=joke_id, model=slug, explanation=str(text).strip())
            )
    return explanations


def build_paper_qwen_ratings(csv_path: Path) -> list[Rating]:
    df = load_typed_csv(csv_path)
    ids = assign_joke_ids(df)
    ratings: list[Rating] = []
    for row_idx, row in df.iterrows():
        joke_id = ids.loc[row_idx]
        for col, (slug, criterion) in PAPER_QWEN_JUDGE_COLS.items():
            v = row.get(col)
            if pd.isna(v):
                continue
            try:
                score = int(v)
            except (TypeError, ValueError):
                continue
            if not (0 <= score <= 5):
                continue
            ratings.append(
                Rating(
                    joke_id=joke_id,
                    model=slug,
                    criterion=criterion,
                    score=score,
                    annotator=JUDGE_MODEL_NAME,
                )
            )
    return ratings


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive explanations.jsonl + paper Qwen ratings from CSV")
    ap.add_argument("--input", default="new_jokes.csv", type=Path)
    ap.add_argument("--output", default="data/explanations.jsonl", type=Path)
    ap.add_argument(
        "--extract-paper-ratings",
        type=Path,
        default=None,
        help="if set, also extract the paper's Qwen2.5-72B judge ratings to this path",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="no-op for this script (full rewrite)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.resume:
        log.info("--resume is a no-op for this dev script; full rewrite always")

    explanations = build_explanations(args.input)
    if args.limit is not None:
        explanations = explanations[: args.limit]
    n = write_jsonl(args.output, (dc.asdict(e) for e in explanations))
    log.info("wrote %d explanations to %s", n, args.output)

    if args.extract_paper_ratings is not None:
        ratings = build_paper_qwen_ratings(args.input)
        if args.limit is not None:
            ratings = ratings[: args.limit]
        n2 = write_jsonl(args.extract_paper_ratings, (dc.asdict(r) for r in ratings))
        log.info("wrote %d paper-Qwen ratings to %s", n2, args.extract_paper_ratings)


if __name__ == "__main__":
    main()
