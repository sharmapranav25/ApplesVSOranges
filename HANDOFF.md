# Handoff — Apples vs Oranges replication

EMNLP 2025 Findings paper (Loakman et al., humour explanation). All 4 build tasks shipped; **27 tests passing**. This file is the at-a-glance briefing — see `CLAUDE.md` for the full guide.

## Status

| Component | State |
|---|---|
| `data/jokes.jsonl` | ✅ 600 rows, balanced 150/type |
| `prompts/rubric_*.txt`, `judge_template.txt` | ✅ verbatim from §4.3 / A.6, ascending 0→5 |
| `src/judge.py` (**Qwen2.5-7B**, deviation from paper's 72B) | ✅ implemented, resumable, switchable backend — **not yet run** against the real API |
| `src/metrics.py` (BLEU/ROUGE/METEOR/BERTScore) | ✅ implemented, sanity-checked against Table 2 |
| `data/explanations.jsonl` | ⚠️ dev copy derived from CSV; **swap when teammate delivers** |
| `outputs/ratings_judge.jsonl` | ❌ not yet produced — run the judge |
| `outputs/ratings_judge_paper.jsonl` | ✅ 9600 paper-Qwen-**72B** ratings extracted from CSV (cross-scale baseline only — we judge with 7B) |
| `outputs/metrics.csv` | partial (1 row from a smoke run) — re-run for the full 32 |

## Next steps (in order)

1. **When teammate's `data/explanations.jsonl` arrives**, replace the dev copy and verify model slugs match `MODEL_SLUG_MAP` in `src/schema.py`.
2. Run the judge → `outputs/ratings_judge.jsonl`. ~9600 calls. Pick one:
   - **Free, local (recommended on Mac):** `ollama serve & ; ollama pull qwen2.5:7b-instruct-q4_K_M ; python src/judge.py --backend ollama --resume`. ~3–5 hrs.
   - **Paid, fast:** `export OPENROUTER_API_KEY=... ; python src/judge.py --resume`. **~$0.50–2 at 7B** (would have been ~$5–15 at 72B); minutes.
3. `python src/metrics.py` → `outputs/metrics.csv` (8 models × 4 types = 32 rows).
4. *(Optional)* Compute Cohen's κ between `ratings_judge.jsonl` (our 7B) and `ratings_judge_paper.jsonl` (paper's 72B). This is a **cross-scale** check — expect modest agreement, not high; treat it as a lower bound on judge quality.

## Things you'd otherwise re-discover

- **CSV `Index` column is null for hom/het/non-topical** (only `topical` has it). `source_index` is therefore optional. Don't join on it — use `assign_joke_ids()` in `src/preprocess.py`, which assigns `{type}_{NNN}` by within-type CSV row order.
- **Rubric ordering trap:** paper §4.3 presents the rubric DESCENDING; A.6 says the judge gets it ASCENDING. Files are ascending. `tests/test_rubric.py` enforces — don't "fix" it.
- **Replication gap (documented):** my BLEU/ROUGE/METEOR runs ~10–25% above paper's Table 2 numbers; BERTScore is essentially exact (0.890 vs 0.88). Paper doesn't publish its metric code; the gap reflects unspecified tokenizer/aggregation choices. Sanity-test tolerances are loose accordingly. **Don't chase this further** unless the paper's code drops.
- **macOS judge backend:** `bitsandbytes` has no Darwin wheel → `--backend local` won't run on this laptop. **Use `--backend ollama`** (free, local, ~3–5 hrs) or `--backend openrouter` (paid, ~$1, minutes). Ollama is the recommended free path; setup is `brew install ollama && ollama serve & && ollama pull qwen2.5:7b-instruct-q4_K_M`.

## Layout

| Path | Purpose |
|---|---|
| `CLAUDE.md` | full guide — conventions, mapping tables, run details, sanity-check table |
| `prompts/` | rubric + judge template (verbatim, do not reorder) |
| `src/schema.py` | dataclasses + `MODEL_SLUG_MAP` + `JOKE_TYPE_MAP` + `PAPER_QWEN_JUDGE_COLS` |
| `src/preprocess.py` | CSV → `data/jokes.jsonl` |
| `src/make_explanations_from_csv.py` | dev: CSV → `data/explanations.jsonl` + extract paper Qwen ratings |
| `src/judge.py`, `src/judge_backend.py` | Qwen judge driver + 4 backends (openrouter / together / ollama / local) |
| `src/parser.py` | first-int-0-to-5 score parser |
| `src/metrics.py` | automatic metrics → `outputs/metrics.csv` |
| `tests/` | 27 pytest tests (rubric, parser, preprocess, metrics — last is heavy) |

## Quick run

```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','wordnet','omw-1.4')]"

python src/preprocess.py                                  # CSV → data/jokes.jsonl
python src/make_explanations_from_csv.py \
    --extract-paper-ratings outputs/ratings_judge_paper.jsonl   # dev explanations + κ baseline

export OPENROUTER_API_KEY=sk-or-...
python src/judge.py --resume                              # → outputs/ratings_judge.jsonl
python src/metrics.py                                     # → outputs/metrics.csv

pytest tests/ -v                                          # all 27 should pass
```

Every script supports `--limit N` and `--resume`.
