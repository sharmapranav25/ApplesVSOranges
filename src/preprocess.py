"""Task 1: CSV (`new_jokes.csv`) -> `data/jokes.jsonl`.

One row per joke: {id, type, joke, reference_explanation, source_index}.
600 jokes, 150 each in {homographic, heterographic, non_topical, topical}.
IDs are `{type}_{NNN}` assigned in CSV row order within each type bucket.
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import logging
from pathlib import Path

import pandas as pd

from io_jsonl import read_jsonl_keys, write_jsonl
from schema import JOKE_TYPE_MAP, JOKE_TYPES, Joke

log = logging.getLogger("preprocess")


def load_typed_csv(csv_path: Path) -> pd.DataFrame:
    """Read CSV and add a `_type` column (canonical joke type), dropping
    unknown-source rows. Centralized so `make_explanations_from_csv.py` and
    this module agree on the row set."""
    df = pd.read_csv(csv_path)
    required_cols = {"source", "Index", "joke", "gold_explanation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.assign(_type=df["source"].map(JOKE_TYPE_MAP))
    unknown = df["_type"].isna()
    if unknown.any():
        log.warning("dropping %d rows with unknown source values: %s",
                    unknown.sum(), df.loc[unknown, "source"].unique().tolist())
        df = df.loc[~unknown].copy()
    return df


def assign_joke_ids(df: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by the CSV row index, with values like
    'homographic_007'. IDs are assigned in CSV row order within each type
    so they're stable across re-runs."""
    out = pd.Series(index=df.index, dtype="object")
    for jt in JOKE_TYPES:
        sub_idx = df.index[df["_type"] == jt]
        for i, row_idx in enumerate(sub_idx):
            out.loc[row_idx] = f"{jt}_{i:03d}"
    return out


def build_jokes(csv_path: Path) -> list[Joke]:
    df = load_typed_csv(csv_path)
    ids = assign_joke_ids(df)
    jokes: list[Joke] = []
    for row_idx, row in df.iterrows():
        # `Index` is only populated for topical (the others were resampled and
        # the SemEval/r-Jokes id wasn't preserved). Keep it when present.
        raw_idx = row["Index"]
        source_index = int(raw_idx) if pd.notna(raw_idx) else None
        jokes.append(
            Joke(
                id=ids.loc[row_idx],
                type=row["_type"],
                joke=str(row["joke"]).strip(),
                reference_explanation=str(row["gold_explanation"]).strip(),
                source_index=source_index,
            )
        )
    # Sort: type order, then numeric suffix, so output file is grouped & monotone.
    type_order = {jt: i for i, jt in enumerate(JOKE_TYPES)}
    jokes.sort(key=lambda j: (type_order[j.type], int(j.id.rsplit("_", 1)[1])))
    return jokes


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess new_jokes.csv -> jokes.jsonl")
    ap.add_argument("--input", default="new_jokes.csv", type=Path)
    ap.add_argument("--output", default="data/jokes.jsonl", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (debug)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="if output exists, only append jokes whose id isn't already present",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    jokes = build_jokes(args.input)
    if args.limit is not None:
        jokes = jokes[: args.limit]

    if args.resume and args.output.exists():
        existing = read_jsonl_keys(args.output, key_fn=lambda r: r["id"])
        before = len(jokes)
        jokes = [j for j in jokes if j.id not in existing]
        log.info("resume: skipping %d already-present rows", before - len(jokes))
        # In resume mode we append; otherwise we overwrite.
        from io_jsonl import append_jsonl
        for j in jokes:
            append_jsonl(args.output, dc.asdict(j))
        n = len(jokes)
    else:
        n = write_jsonl(args.output, (dc.asdict(j) for j in jokes))

    log.info("wrote %d rows to %s", n, args.output)


if __name__ == "__main__":
    main()
