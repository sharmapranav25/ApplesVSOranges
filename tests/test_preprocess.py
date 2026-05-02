"""Preprocessor: 600 jokes, 150 each, well-formed ids, no empties."""

from collections import Counter
from pathlib import Path

import pytest

from preprocess import build_jokes
from schema import JOKE_TYPES

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "new_jokes.csv"


@pytest.mark.skipif(not CSV.exists(), reason="new_jokes.csv not present")
def test_six_hundred_jokes_balanced():
    jokes = build_jokes(CSV)
    assert len(jokes) == 600
    counts = Counter(j.type for j in jokes)
    for jt in JOKE_TYPES:
        assert counts[jt] == 150, f"{jt} count = {counts[jt]}"


@pytest.mark.skipif(not CSV.exists(), reason="new_jokes.csv not present")
def test_ids_unique_and_well_formed():
    jokes = build_jokes(CSV)
    ids = [j.id for j in jokes]
    assert len(set(ids)) == len(ids)  # unique
    for j in jokes:
        assert j.id == f"{j.type}_{int(j.id.rsplit('_', 1)[1]):03d}"
        assert j.id.startswith(j.type + "_")
        # 0..149 zero-padded
        assert j.id.split("_")[-1].isdigit() and len(j.id.split("_")[-1]) == 3


@pytest.mark.skipif(not CSV.exists(), reason="new_jokes.csv not present")
def test_no_empty_required_fields():
    jokes = build_jokes(CSV)
    for j in jokes:
        assert j.joke and j.joke.strip(), f"empty joke for {j.id}"
        assert j.reference_explanation and j.reference_explanation.strip(), \
            f"empty reference_explanation for {j.id}"
        assert j.type in JOKE_TYPES
