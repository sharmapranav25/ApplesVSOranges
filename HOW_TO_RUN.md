# How to run — Apples vs Oranges replication (Member 1 components)

End-to-end run of the four Member-1 build tasks. Every step is idempotent and resumable.

## 0. One-time setup

```bash
cd "/path/to/apples vs oranges/"

# Python deps
pip install -r requirements.txt

# NLTK data (METEOR needs these)
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','wordnet','omw-1.4')]"
```

## 1. Preprocess CSV → `data/jokes.jsonl`  (Task 1)

Already produced and committed. Re-run only if `new_jokes.csv` changes.

```bash
python src/preprocess.py --input new_jokes.csv --output data/jokes.jsonl
```

Output: 600 lines, 150 per type (`homographic`/`heterographic`/`non_topical`/`topical`), schema `{id, type, joke, reference_explanation, source_index}`.

## 2. (dev) Build `data/explanations.jsonl` from CSV  (placeholder until teammate's file lands)

```bash
python src/make_explanations_from_csv.py \
    --input new_jokes.csv \
    --output data/explanations.jsonl
```

Output: 4800 lines (600 jokes × 8 models). Swap with teammate's deliverable when ready; verify model slugs match `MODEL_SLUG_MAP` in `src/schema.py`.

Optional: dump the paper's own Qwen-72B ratings as a cross-scale baseline:

```bash
python src/make_explanations_from_csv.py \
    --input new_jokes.csv \
    --extract-paper-ratings outputs/ratings_judge_paper.jsonl
```

## 3. Rubric files  (Task 2)

Static — no run step. `prompts/rubric_accuracy.txt`, `prompts/rubric_completeness.txt`, `prompts/judge_template.txt` ship verbatim. **Do not re-order** (paper §4.3 prints descending; A.6 says judge gets ascending — files are ascending; `tests/test_rubric.py` enforces).

## 4. Run the LLM-as-a-judge  (Task 3)

**Model: Qwen2.5-7B-Instruct** (deviation from paper's 72B — see `status_task3_judge.md`).

Four backends. Pick one:

### 4a. Ollama — local, free  ⭐ recommended on Mac

```bash
brew install ollama                                 # one-time
ollama serve &                                      # background daemon (port 11434)
ollama pull qwen2.5:7b-instruct-q4_K_M              # ~4.4 GB, one-time

python src/judge.py --backend ollama --resume
# → outputs/ratings_judge.jsonl
```

No API key, no rate limits, ~3–5 hrs for 9600 calls on Apple Silicon. Override defaults via env: `OLLAMA_MODEL` (default `qwen2.5:7b-instruct-q4_K_M`), `OLLAMA_URL` (default `http://localhost:11434/v1/chat/completions`).

### 4b. OpenRouter — paid, fast

```bash
export OPENROUTER_API_KEY=sk-or-...
python src/judge.py --resume                        # default --backend openrouter
```

Cost: ~$0.50–2 at 7B prices. Minutes, not hours.

### 4c. Together — paid alternative (free signup credit ~$5)

```bash
export TOGETHER_API_KEY=...
python src/judge.py --resume --backend together
```

### 4d. Local (HPC node only)

```bash
python src/judge.py --resume --backend local
```

`transformers` + `bitsandbytes` 4-bit. **Will not run on macOS** — bitsandbytes has no Darwin wheel. Use Ollama instead on Mac.

### Smoke run (any backend)

```bash
python src/judge.py --backend ollama --limit 4      # 4 calls, ~10s
```

Errors logged to `outputs/judge.errors.log`; never crashes mid-run.

## 5. Compute automatic metrics  (Task 4)

```bash
python src/metrics.py --resume
# → outputs/metrics.csv  (32 rows: 8 models × 4 joke types)
```

Local-only, free. Time on Mac CPU is dominated by BERTScore (`roberta-large`); estimate 15–30 min for the full 31-bucket backfill.

Restrict for debugging:

```bash
python src/metrics.py --models gpt-4o --types homographic   # single bucket
python src/metrics.py --limit 50                            # cap explanations
```

`--bertscore-batch-size 64` (or higher) on a GPU node.

## 6. Tests

```bash
pytest tests/ -v        # 27 tests; the heaviest is the BERTScore sanity check on GPT-4o/hom
```

Skips automatically if dependencies (e.g. `data/explanations.jsonl`) are absent. Tolerances on the paper sanity check are intentionally loose — see CLAUDE.md "Replication gap (documented)".

## 7. (Optional) Cross-scale κ vs paper's 72B baseline

Once `outputs/ratings_judge.jsonl` exists, compute Cohen's κ vs `outputs/ratings_judge_paper.jsonl` per (model, criterion) bucket. We use Qwen-7B; the paper-baseline is Qwen-72B → expect modest agreement, not high. This is a quality lower bound, not a same-model reproducibility check. One-off notebook is fine; not yet automated.

## Common flags (all scripts)

| Flag | Effect |
|---|---|
| `--limit N` | cap rows / calls processed (debug) |
| `--resume` | skip rows already present in `--output` |
| `--input`, `--output` | override default paths |

## Quick end-to-end

### Free path (Mac, ~5 hrs total — judge is the long pole)
```bash
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M
python src/preprocess.py
python src/make_explanations_from_csv.py
python src/judge.py --backend ollama --resume     # ~3–5 hrs
python src/metrics.py --resume                    # ~10 min
pytest tests/ -v
```

### Paid path (~10 min total)
```bash
export OPENROUTER_API_KEY=sk-or-...
python src/preprocess.py
python src/make_explanations_from_csv.py
python src/judge.py --resume                      # ~minutes
python src/metrics.py --resume                    # ~10 min
pytest tests/ -v
```

Outputs land in `outputs/`. Logs land alongside (`outputs/*.errors.log`, `outputs/metrics_run.log`).
