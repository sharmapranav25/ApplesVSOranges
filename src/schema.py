"""Shared dataclasses + name-mapping tables used across the pipeline.

If a teammate hands over `data/explanations.jsonl` using different model slugs,
edit MODEL_SLUG_MAP — that's the single point of canonicalization.
"""

from __future__ import annotations

from dataclasses import dataclass


JOKE_TYPES: tuple[str, ...] = (
    "homographic",
    "heterographic",
    "non_topical",
    "topical",
)

# CSV `source` column values → our canonical type names.
JOKE_TYPE_MAP: dict[str, str] = {
    "hom": "homographic",
    "het": "heterographic",
    "non-topical": "non_topical",
    "topical": "topical",
}

# CSV `explain_<x>` column → our canonical model slug used in explanations.jsonl.
MODEL_SLUG_MAP: dict[str, str] = {
    "explain_gpt4o": "gpt-4o",
    "explain_gpt4o_mini": "gpt-4o-mini",
    "explain_gemini_pro": "gemini-1.5-pro",
    "explain_gemini_flash": "gemini-1.5-flash",
    "explain_llama31_8b": "llama-3.1-8b",
    "explain_llama31_70b": "llama-3.1-70b",
    "explain_r1_8b": "r1-distill-llama-8b",
    "explain_r1_70b": "r1-distill-llama-70b",
}

# CSV `<x>_qwen25_72b_judge` column → (model_slug, criterion). Used only by
# make_explanations_from_csv.py --extract-paper-ratings to dump a smoke baseline.
PAPER_QWEN_JUDGE_COLS: dict[str, tuple[str, str]] = {
    "gpt4o_accuracy_qwen25_72b_judge": ("gpt-4o", "accuracy"),
    "gpt4o_completeness_qwen25_72b_judge": ("gpt-4o", "completeness"),
    "gpt4o_mini_accuracy_qwen25_72b_judge": ("gpt-4o-mini", "accuracy"),
    "gpt4o_mini_completeness_qwen25_72b_judge": ("gpt-4o-mini", "completeness"),
    "gemini_pro_accuracy_qwen25_72b_judge": ("gemini-1.5-pro", "accuracy"),
    "gemini_pro_completeness_qwen25_72b_judge": ("gemini-1.5-pro", "completeness"),
    "gemini_flash_accuracy_qwen25_72b_judge": ("gemini-1.5-flash", "accuracy"),
    "gemini_flash_completeness_qwen25_72b_judge": ("gemini-1.5-flash", "completeness"),
    "llama31_8b_accuracy_qwen25_72b_judge": ("llama-3.1-8b", "accuracy"),
    "llama31_8b_completeness_qwen25_72b_judge": ("llama-3.1-8b", "completeness"),
    "llama31_70b_accuracy_qwen25_72b_judge": ("llama-3.1-70b", "accuracy"),
    "llama31_70b_completeness_qwen25_72b_judge": ("llama-3.1-70b", "completeness"),
    "r1_8b_accuracy_qwen25_72b_judge": ("r1-distill-llama-8b", "accuracy"),
    "r1_8b_completeness_qwen25_72b_judge": ("r1-distill-llama-8b", "completeness"),
    "r1_70b_accuracy_qwen25_72b_judge": ("r1-distill-llama-70b", "accuracy"),
    "r1_70b_completeness_qwen25_72b_judge": ("r1-distill-llama-70b", "completeness"),
}

CRITERIA: tuple[str, ...] = ("accuracy", "completeness")

JUDGE_MODEL_NAME = "qwen2.5-7b-instruct"


@dataclass(frozen=True)
class Joke:
    id: str
    type: str
    joke: str
    reference_explanation: str
    source_index: int | None = None  # original SemEval/r/Jokes index, kept for traceability


@dataclass(frozen=True)
class Explanation:
    joke_id: str
    model: str
    explanation: str


@dataclass(frozen=True)
class Rating:
    joke_id: str
    model: str
    criterion: str  # "accuracy" or "completeness"
    score: int      # 0..5
    annotator: str  # e.g. "qwen2.5-7b-instruct" (our judge) or "qwen2.5-72b-instruct" (paper baseline)


@dataclass(frozen=True)
class MetricRow:
    model: str
    joke_type: str
    n: int
    sacrebleu: float
    rouge1: float
    rouge2: float
    rougeL: float
    meteor: float
    bertscore: float
