"""Metrics: small synthetic test (cheap, always runs) + paper sanity check
(skipped if explanations.jsonl absent or doesn't include GPT-4o on hom).
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cheap_metrics():
    """Metrics module is heavy (loads roberta-large); import once per module."""
    from metrics import compute_bucket_metrics, _ensure_nltk
    _ensure_nltk()
    return compute_bucket_metrics


def test_identical_pairs_score_perfect(cheap_metrics):
    """Identical hyp/ref => BLEU=100, ROUGE-1=1.0, BERTScore≈1.0."""
    hyps = [
        "The joke plays on the dual meaning of bark as both tree bark and a dog's sound.",
        "A pun on the word bat as both an animal and a baseball bat.",
    ]
    refs = list(hyps)
    m = cheap_metrics(hyps, refs, bertscore_batch_size=2)
    assert m["sacrebleu"] > 99.0
    assert m["rouge1"] > 0.99
    assert m["rouge2"] > 0.99
    assert m["rougeL"] > 0.99
    assert m["meteor"] > 0.99
    assert m["bertscore"] > 0.999


def test_unrelated_pairs_score_low(cheap_metrics):
    hyps = ["completely unrelated text about quantum mechanics"]
    refs = ["a joke about a pun on the word bark"]
    m = cheap_metrics(hyps, refs, bertscore_batch_size=2)
    assert m["sacrebleu"] < 5.0
    assert m["rouge1"] < 0.3
    assert m["bertscore"] < 0.92  # roberta-large still gives a moderate baseline


# Paper Table 2, GPT-4o on Homographic.
# Tolerances reflect that sacrebleu/ROUGE/METEOR results vary materially by
# tokenizer/stemmer/aggregation choice, and the paper doesn't pin down its
# exact implementation. Verified locally on the CSV's `explain_gpt4o` column
# vs `gold_explanation`:
#   sacrebleu (default tok 13a)  -> 10.63    (paper: 8.51)
#   sacrebleu (tok=none, ws)     -> 5.62
#   rouge_score (use_stemmer=T)  -> R1=0.458 R2=0.163 RL=0.295   (paper: .41/.12/.25)
#   meteor (nltk word_tokenize)  -> 0.389                         (paper: 0.37)
# The pipeline is internally consistent; the gap is implementation-defined.
# We assert the right ballpark, not exact match. BERTScore is stable across
# implementations, so its tolerance is tight.
PAPER_GPT4O_HOM = {
    "sacrebleu": (8.51, 3.0),
    "rouge1": (0.41, 0.06),
    "rouge2": (0.12, 0.05),
    "rougeL": (0.25, 0.06),
    "meteor": (0.37, 0.05),
    "bertscore": (0.88, 0.02),
}


def _load_gpt4o_hom_pairs():
    """Return (hyps, refs) for GPT-4o on hom from data/explanations.jsonl + jokes.jsonl."""
    jokes_path = ROOT / "data" / "jokes.jsonl"
    expl_path = ROOT / "data" / "explanations.jsonl"
    if not jokes_path.exists() or not expl_path.exists():
        return None
    jokes = {}
    with jokes_path.open() as f:
        for line in f:
            j = json.loads(line)
            jokes[j["id"]] = j
    pairs = []
    with expl_path.open() as f:
        for line in f:
            e = json.loads(line)
            if e["model"] != "gpt-4o":
                continue
            j = jokes.get(e["joke_id"])
            if j is None or j["type"] != "homographic":
                continue
            pairs.append((e["explanation"], j["reference_explanation"]))
    return pairs if pairs else None


def test_gpt4o_hom_matches_paper_table2(cheap_metrics):
    pairs = _load_gpt4o_hom_pairs()
    if pairs is None:
        pytest.skip("data/explanations.jsonl missing or no GPT-4o-on-hom rows")
    hyps = [h for h, _ in pairs]
    refs = [r for _, r in pairs]
    m = cheap_metrics(hyps, refs)
    for key, (target, tol) in PAPER_GPT4O_HOM.items():
        assert abs(m[key] - target) <= tol, (
            f"{key}: got {m[key]:.4f}, paper {target} (tol ±{tol})"
        )
