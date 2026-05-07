# Handoff — Apples vs Oranges replication

All tasks complete; **27 tests passing**. This file is the at-a-glance briefing — see `CLAUDE.md` for the full guide.

## Status

| Component | State |
|---|---|
| `data/jokes.jsonl` | ✅ 600 rows, balanced 150/type |
| `prompts/rubric_*.txt`, `judge_template.txt` | ✅ ascending 0→5 |
| `src/judge.py` (**Qwen2.5-7B**) | ✅ implemented, resumable, switchable backend |
| `src/metrics.py` (BLEU/ROUGE/METEOR/BERTScore) | ✅ implemented |
| `src/analyze.py` — figures + tables + summary | ✅ implemented |
| `src/extension.py` — Not yet created | ⚠️ pending |
| `data/explanations.jsonl` | ✅ 4,800 rows — model explanations extracted from paper's released CSV |
| `outputs/ratings_judge.jsonl` | ✅ 9,600 rows — our Qwen-7B judge, via `run_judge.ipynb` |
| `outputs/ratings_judge_paper.jsonl` | ✅ 9,600 Qwen-72B baseline ratings extracted from CSV |
| `outputs/metrics.csv` | ✅ 32 rows (8 models × 4 types) |
| `outputs/figures/` (4 figures) | ✅ fig3b, fig3c, fig4, judge comparison |
| `outputs/tables/` (6 tables) | ✅ avg scores, success rates, gap, logreg, agreement, hypothesis checks |
| `outputs/results_summary.txt` | ✅ full console output saved on every `src/analyze.py` run |
| `pytest tests/ -v` | ✅ 27/27 passing |

---

## Next steps — Member 3

### Pick and implement an extension (could be one of the options below, or your own idea)

All options below work entirely from data that **already exists** — no new model calls, no new data collection. Pick whichever is most interesting. Negative results written up rigorously are fully acceptable.

**Option A — Accuracy vs Completeness gap analysis**
Across all models and joke types, completeness scores are consistently lower than accuracy. Dig into *why*: which joke types show the largest gap? Which models hallucinate vs omit? Do larger models close the gap more than smaller ones? All data is in `outputs/ratings_judge.jsonl` and `outputs/tables/avg_scores.csv`. Pure pandas — no new runs needed.

**Option B — Per-joke difficulty analysis**
Some jokes are hard for all models; others only trip up smaller models. Using `outputs/ratings_judge.jsonl`, compute a per-joke "difficulty score" (average judge score across all 8 models) and identify the hardest and easiest jokes in each category. Do the hardest topical jokes share any features (named entities, recency, obscurity)? Cross-reference with `new_jokes.csv` for joke text.

**Option C — Correlation between automatic metrics and judge scores**
`outputs/metrics.csv` has BLEU/ROUGE/METEOR/BERTScore per (model, joke type). `outputs/ratings_judge.jsonl` has human-proxy scores for the same cells. Compute Spearman correlation between each automatic metric and judge score per joke type. Test whether automatic metrics and judge scores agree per-model too.

**Option D — Model family comparison**
Group models into families (GPT, Gemini, Llama, R1) and compare within-family consistency vs between-family differences. Do models within the same family make similar errors? Use `outputs/ratings_judge.jsonl` to compute pairwise Spearman correlation between model score vectors. This tells us whether GPT-4o and GPT-4o Mini "understand" jokes in the same way.

**Option E — Score distribution shift analysis**
The score distribution in `outputs/figures/fig_judge_comparison.png` shows our 7B judge clusters at 4. Extend this: plot score distributions per joke type and per model. Does the distribution shift more for topical jokes? Do larger models get more 5s or just fewer 0-2s? All data is in `outputs/ratings_judge.jsonl` — add a few matplotlib plots to `src/analyze.py`.

**Option F — Few-shot prompting**
The paper uses zero-shot prompting for all models. Test whether providing 1–2 example jokes with their gold explanations in the prompt improves completeness scores — which were consistently lower than accuracy across all models. Only the prompt template changes; the rest of the pipeline stays the same. Run on a subset (e.g. 50 jokes per type) to keep it manageable.

**Option F — Few-shot prompting**
Instead of zero-shot, provide 1–2 example jokes with gold explanations in the prompt. The gold explanations are already written for all 600 jokes — pick one representative example per joke type and prepend it to the prompt. Only the prompt template changes; run on a small subset (e.g. 50 jokes × 2–3 models) to keep it feasible. Hypothesis: few-shot examples improve completeness scores, which were consistently the weakest metric across all models.


---

## Things you'd otherwise re-discover

- **CSV `Index` column is null for hom/het/non-topical** (only `topical` has it). `source_index` is therefore optional. Don't join on it — use `assign_joke_ids()` in `src/preprocess.py`, which assigns `{type}_{NNN}` by within-type CSV row order.
- **Rubric ordering trap:** rubric files are ASCENDING (0→5). `tests/test_rubric.py` enforces this — don't "fix" it.
- **Replication gap (documented):** BLEU/ROUGE/METEOR runs 4–15% above baseline; BERTScore matches within ±0.5%. Gap reflects tokenizer choices. Don't chase this further — see `outputs/tables/replication_gap.csv`.
- **H1/H3 near-ties:** our Qwen-7B judge concentrates scores at 4, reducing sensitivity to joke difficulty differences. Directional findings hold in success-rate analysis. See `status_task3_judge.md`.
- **WSL judge setup:** Ollama daemon must be started manually each WSL session (`ollama serve`). GPU passthrough requires Windows NVIDIA driver ≥ 527.41. Set `OLLAMA_NUM_GPU=-1` for full GPU utilisation. See `run_judge.ipynb` for full setup.

---

## Layout

| Path | Purpose |
|---|---|
| `CLAUDE.md` | full guide — conventions, mapping tables, run details, sanity-check table |
| `run_judge.ipynb` | interactive judge runner (Windows WSL + GPU) |
| `prompts/` | rubric + judge template (verbatim, do not reorder) |
| `src/schema.py` | dataclasses + `MODEL_SLUG_MAP` + `JOKE_TYPE_MAP` |
| `src/preprocess.py` | CSV → `data/jokes.jsonl` |
| `src/make_explanations_from_csv.py` | dev: CSV → `data/explanations.jsonl` + extract baseline ratings |
| `src/judge.py`, `src/judge_backend.py` | Qwen judge driver + 4 backends (openrouter / together / ollama / local) |
| `src/parser.py` | first-int-0-to-5 score parser |
| `src/metrics.py` | automatic metrics → `outputs/metrics.csv` |
| `src/analyze.py` | figures + tables + summary from judge ratings |
| `src/extension.py` | ⚠️ pending |
| `tests/` | 27 pytest tests (rubric, parser, preprocess, metrics) |

---

## Quick run (fresh machine)

```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','wordnet','omw-1.4')]"

python src/preprocess.py                                  # CSV → data/jokes.jsonl
python src/make_explanations_from_csv.py \
    --extract-paper-ratings outputs/ratings_judge_paper.jsonl   # dev explanations + baseline ratings

# Run judge (GPU setup is optional but speeds up 7B judge dramatically; see `run_judge.ipynb` for details)
jupyter notebook run_judge.ipynb
# Alternative (paid): export OPENROUTER_API_KEY=sk-or-... && python src/judge.py --resume

python src/metrics.py                                     # → outputs/metrics.csv
python src/analyze.py                                     # → outputs/figures/, tables/, results_summary.txt

pytest tests/ -v                                          # all 27 should pass
```

Every script supports `--limit N` and `--resume`.
