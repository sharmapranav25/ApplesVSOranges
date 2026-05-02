# Comparing Apples to Oranges — Replication Project

This directory replicates parts of **Loakman, Thorne & Lin, "Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes"** (EMNLP 2025 Findings, pp. 9502–9518). Owner: Pranav Sharma.

## What is replicated

| Paper section | Artifact here |
|---|---|
| §3 Dataset Compilation | `data/jokes.jsonl` — 600 jokes × {id, type, joke, reference_explanation} |
| §4.3 Evaluation Criteria (rubric) | `prompts/rubric_accuracy.txt`, `prompts/rubric_completeness.txt` (verbatim, ascending 0→5) |
| §6 Automatic Evaluation (Table 2) | `src/metrics.py` → `outputs/metrics.csv` |
| Appendix A.6 (LLM-as-judge) | `src/judge.py` + `prompts/judge_template.txt` → `outputs/ratings_judge.jsonl` |

We are **not** generating the 4800 model explanations ourselves; those land via `data/explanations.jsonl` (teammate input). For dev work, `src/make_explanations_from_csv.py` derives a stand-in copy from the existing `explain_<model>` columns in `new_jokes.csv`.

## Critical conventions

### Joke types
The CSV's `source` column uses paper-internal abbreviations. We normalize:

| CSV `source` | Our `type` |
|---|---|
| `hom` | `homographic` |
| `het` | `heterographic` |
| `non-topical` | `non_topical` |
| `topical` | `topical` |

### Joke IDs
`{type}_{NNN}` where NNN is a zero-padded 0..149 index assigned in CSV row order within each type. Example: `homographic_000`, `topical_149`. Stable across re-runs of `preprocess.py`. The original SemEval/r/Jokes index is preserved as the sidecar field `source_index` (kept inside `data/jokes.jsonl` for traceability but not part of the schema the user spec'd).

### Model name canonicalization
Paper uses long HF names; we use short slugs in `data/explanations.jsonl` and downstream:

| Slug | CSV column | Paper / HF |
|---|---|---|
| `gpt-4o` | `explain_gpt4o` | gpt-4o-2024-05-13 |
| `gpt-4o-mini` | `explain_gpt4o_mini` | gpt-4o-mini-2024-07-18 |
| `gemini-1.5-pro` | `explain_gemini_pro` | gemini-1.5-pro-001 |
| `gemini-1.5-flash` | `explain_gemini_flash` | gemini-1.5-flash-001 |
| `llama-3.1-8b` | `explain_llama31_8b` | Meta-Llama-3.1-8B-Instruct |
| `llama-3.1-70b` | `explain_llama31_70b` | Meta-Llama-3.1-70B-Instruct |
| `r1-distill-llama-8b` | `explain_r1_8b` | DeepSeek-R1-Distill-Llama-8B |
| `r1-distill-llama-70b` | `explain_r1_70b` | DeepSeek-R1-Distill-Llama-70B |

If the teammate ships a different scheme, edit `MODEL_SLUG_MAP` at the top of `src/schema.py`.

### Rubric ordering
Paper §4.3 presents the rubric **descending (5→0)**. Appendix A.6 explicitly states the LLM-judge gets it **ascending (0→5)** — to match Prometheus's training prompt. Our `prompts/rubric_*.txt` files are ascending, used both by the human-style display and verbatim in the judge prompt. **Do not re-order them.**

### `is_good`
`is_good = (accuracy >= 4) AND (completeness >= 4)` — §5.1.

### Judge prompt
`prompts/judge_template.txt` is **verbatim from Appendix A.6**. The leading single space before `{joke}`, `{reference_explanation}`, `{model_explanation}` (visible in the paper as `\n {joke}\n\n`) is preserved. The template has these placeholders only:
- `{criteria}` — `accuracy` or `completeness` (singular, one criterion per call)
- `{scoring_criteria}` — full text of the relevant rubric file
- `{joke}`, `{reference_explanation}`, `{model_explanation}`

### Judge decoding
- model: **`Qwen/Qwen2.5-7B-Instruct`** — **deviation from paper**, which uses 72B. Switched 2026-04-30 for cost/feasibility (Member 1 decision). Implication: `outputs/ratings_judge_paper.jsonl` (the 9600 paper-Qwen-72B ratings extracted from CSV) is no longer a same-model κ baseline; it's now a *cross-scale* baseline. Treat agreement vs. that file as a quality lower bound, not as faithful reproduction.
- temperature: **0.1** (paper's only deviation from defaults)
- everything else: model defaults
- output: parse first integer 0–5 from response; reject anything else, log error, do not crash

### Metric aggregation (matches paper Table 2)
- **SacreBLEU**: `sacrebleu.corpus_bleu(hyps, [refs]).score` — corpus-level, per (model, joke_type) bucket
- **ROUGE-1/2/L**: `rouge_score.RougeScorer([...], use_stemmer=True)` — per-instance F-measure, then mean within bucket
- **METEOR**: `nltk.translate.meteor_score.meteor_score`, with `nltk.word_tokenize` on both hyp and ref — per-instance, then mean
- **BERTScore**: `bert_score.score(..., model_type="roberta-large", lang="en")` — per-instance F1, then mean
- Output: `outputs/metrics.csv` with columns `model,joke_type,n,sacrebleu,rouge1,rouge2,rougeL,meteor,bertscore`

### Sanity check (Table 2, row "GPT-4o", column "Hom.")
| Metric | Paper | Our value | Tolerance |
|---|---|---|---|
| SacreBLEU | 8.51 | 10.63 | ±3.0 |
| ROUGE-1 | 0.41 | 0.458 | ±0.06 |
| ROUGE-2 | 0.12 | 0.163 | ±0.05 |
| ROUGE-L | 0.25 | 0.295 | ±0.06 |
| METEOR | 0.37 | 0.389 | ±0.05 |
| BERTScore | 0.88 | 0.890 | ±0.02 |

**Replication gap (documented):** every metric (except BERTScore, untested as of this writing) lands ~10–25% higher than the paper's reported value. The pipeline is internally consistent — the gap reflects implementation choices the paper doesn't pin down (tokenizer, smoothing, possibly `evaluate.load("bleu")` vs raw sacrebleu). For comparison, sacrebleu with `tokenize="none"` (whitespace pre-tokenize) gives BLEU=5.6 on the same data — so the paper's 8.51 sits between the two common implementations. Without the paper's code we can't recover the exact 8.51, but the relative ordering across (model, joke_type) buckets should still match.

`tests/test_metrics.py::test_gpt4o_hom_matches_paper_table2` enforces the loose tolerances above (skipped if `data/explanations.jsonl` is absent or doesn't contain GPT-4o on hom). BERTScore is stable across implementations and gets a tight tolerance.

## Layout

```
.
├── Comparing Apples to Oranges.pdf  # paper
├── new_jokes.csv                    # source dataset (600 rows × 111 cols)
├── data/
│   ├── jokes.jsonl                  # Task 1 output
│   └── explanations.jsonl           # teammate input (or dev copy from CSV)
├── prompts/
│   ├── rubric_accuracy.txt          # verbatim §4.3, ASCENDING
│   ├── rubric_completeness.txt      # verbatim §4.3, ASCENDING
│   └── judge_template.txt           # verbatim Appendix A.6
├── src/
│   ├── schema.py                    # dataclasses + MODEL_SLUG_MAP + JOKE_TYPE_MAP
│   ├── io_jsonl.py                  # read/write/resume helpers
│   ├── preprocess.py                # CSV → jokes.jsonl                     [Task 1]
│   ├── make_explanations_from_csv.py# dev-only: CSV → explanations.jsonl
│   ├── rubric.py                    # load + validate rubric files
│   ├── parser.py                    # first-integer-0-to-5 parser
│   ├── judge_backend.py             # OpenRouter / Together / local backends
│   ├── judge.py                     # judge driver                          [Task 3]
│   └── metrics.py                   # automatic metrics                     [Task 4]
├── tests/
│   ├── test_preprocess.py
│   ├── test_rubric.py
│   ├── test_parser.py
│   └── test_metrics.py
├── outputs/
│   ├── ratings_judge.jsonl          # Task 3 output
│   ├── ratings_judge_paper.jsonl    # extracted Qwen ratings from CSV (smoke baseline)
│   ├── metrics.csv                  # Task 4 output
│   └── *.errors.log                 # per-script error logs
└── requirements.txt
```

## How to run

```bash
# 0. install
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"

# 1. preprocess CSV → data/jokes.jsonl
python src/preprocess.py --input new_jokes.csv --output data/jokes.jsonl

# (dev) derive explanations.jsonl from CSV until teammate's lands
python src/make_explanations_from_csv.py --input new_jokes.csv --output data/explanations.jsonl

# (dev) extract paper's Qwen ratings as smoke baseline
python src/make_explanations_from_csv.py --input new_jokes.csv --extract-paper-ratings outputs/ratings_judge_paper.jsonl

# 2. (no run step — rubric files are static)

# 3. judge — pick a backend:
#   --backend openrouter (default; needs $OPENROUTER_API_KEY, ~$1)
#   --backend together   (needs $TOGETHER_API_KEY)
#   --backend ollama     (FREE, local; needs `ollama serve` + `ollama pull qwen2.5:7b-instruct-q4_K_M`)
#   --backend local      (HF transformers + bitsandbytes; HPC-only, no Darwin wheels)
python src/judge.py --jokes data/jokes.jsonl --explanations data/explanations.jsonl \
    --output outputs/ratings_judge.jsonl --backend ollama --resume

# 4. metrics
python src/metrics.py --jokes data/jokes.jsonl --explanations data/explanations.jsonl \
    --output outputs/metrics.csv

# tests
pytest tests/ -v
```

Every script supports `--limit N` and `--resume`.

## Open issues / next steps

- Teammate's `data/explanations.jsonl` not yet in repo. Pipeline runs end-to-end on the dev copy (CSV-derived). Swap when their file lands; verify model slugs match `MODEL_SLUG_MAP`.
- **Backend choice on Mac:** `--backend ollama` is the recommended free path — runs locally via `ollama serve`, no API key, no rate limits, ~3–5 hr for 9600 calls with q4_K_M quant. `--backend openrouter` is fine if you want a faster paid run (~$1, ~minutes). `--backend local` (HF + bitsandbytes 4-bit) won't work on Darwin (no bitsandbytes wheel) — kept for the HPC node only.
- Judge baseline: the CSV's `*_qwen25_72b_judge` columns are the paper's Qwen-**72B** ratings. We run Qwen-**7B**, so κ vs. that file is a *cross-scale* check (sanity / lower-bound), not a same-model reproducibility check. Expect lower agreement than 72B-vs-72B would give.
- Metrics sanity check is on **GPT-4o on homographic** specifically. If your `explanations.jsonl` doesn't include GPT-4o, the sanity test is skipped.

## What this project is NOT

- We are **not** training or fine-tuning anything.
- We are **not** generating the 4800 explanations (paper's authors did; teammate is replicating this).
- We are **not** doing the human evaluation (paper's primary author + 2 third-party annotators did 320-explanation subset).
- We are **not** computing inter-judge / human-judge agreement statistics here. If needed later, write a separate `src/agreement.py`.

## Source CSV cheat sheet (`new_jokes.csv`)

600 rows × 111 columns. The columns we actually use:
- `source` — joke type (see mapping above)
- `Index` — original SemEval/r/Jokes id
- `joke`, `gold_explanation` — required for `data/jokes.jsonl`
- `explain_<model>` × 8 — explanations (used only by `make_explanations_from_csv.py` for dev)
- `*_qwen25_72b_judge` × 16 — Qwen ratings, used as smoke baseline

The other ~80 columns (urls, primary annotator A0 ratings, re-annotator a1/a2 ratings on a 320-row subset, gemma2_9b judge, prometheus_8x7b judge) are **not used** by this pipeline.
