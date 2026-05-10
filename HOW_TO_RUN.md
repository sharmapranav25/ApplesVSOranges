# How to run — Apples vs Oranges replication

End-to-end run of all tasks. Every step is idempotent and resumable.

## 0. One-time setup

```bash
cd /mnt/d/ApplesVSOranges
conda create -n applesVsOranges python=3.10 -y  # create a new conda environment (recommended) (optional)
conda activate applesVsOranges #  (optional)

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

## 2. Build `data/explanations.jsonl`

```bash
python src/make_explanations_from_csv.py \
    --input new_jokes.csv \
    --extract-paper-ratings outputs/ratings_judge_paper.jsonl
```

## 3. Rubric files  (Task 2)

## 3. Run the judge → `outputs/ratings_judge.jsonl`

**Model: Qwen2.5-7B-Instruct** (deviation from paper's 72B — see `status_task3_judge.md`).

Pick one option:

### 3a. Notebook — recommended (Windows WSL + GPU or Google Colab)

```bash
jupyter notebook run_judge.ipynb
```

Works through setup, GPU verification, smoke test, and full 9,600-call run interactively.
Resumable — re-run the full-run cell to pick up after any interruption.
Expected time: ~2–3 hrs with GPU.

### 3b. Command line — alternative

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
python src/judge.py --backend ollama --limit 4
```

## 4. Compute automatic metrics → `outputs/metrics.csv`

```bash
python src/metrics.py --resume
```

Output: 32 rows (8 models × 4 joke types). Takes ~15–30 min (dominated by BERTScore).

## 5. Produce figures, tables, and summary

```bash
python src/analyze.py
```

Output:
- `outputs/figures/` — 4 figures (Fig 3b, 3c, 4, judge comparison)
- `outputs/tables/` — 6 tables (avg scores, success rates, gap, logistic regression, judge agreement, hypothesis checks)
- `outputs/results_summary.txt` — full results printed to file

## 6. Tests

```bash
pytest tests/ -v    # all 27 should pass
```

## Quick end-to-end

```bash
## Optional: setup conda env
# conda create -n applesVsOranges python=3.10 -y
# conda activate applesVsOranges
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M
python src/preprocess.py
python src/make_explanations_from_csv.py
jupyter notebook run_judge.ipynb                         # ~2–3 hrs with GPU
# OR:  python src/judge.py --backend ollama --resume     # ~3–5 hrs
python src/metrics.py --resume                           # ~10 min
python src/analyze.py                                    # ~30 sec
pytest tests/ -v
```

## Common flags

| Flag | Effect |
|---|---|
| `--limit N` | cap rows processed (useful for debugging) |
| `--resume` | skip rows already in the output file |
| `--input`, `--output` | override default paths |
