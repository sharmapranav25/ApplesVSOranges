# Task 3 — LLM-as-a-judge pipeline — ✅ DONE

**Implementation:** Pranav (Member 1) — 2026-04-30  
**Run:** Khoi (Member 2) — 2026-05-05

---

## ⚠ Deviation from paper (decided 2026-04-30)
Paper uses **Qwen2.5-72B-Instruct**. We use **Qwen2.5-7B-Instruct**.

**Why:** cost + feasibility. 72B run costs ~$5–15 on OpenRouter and needs ~40GB VRAM locally; 7B is ~$0.50–2 paid, or **free via Ollama** on Apple Silicon (~5GB RAM at q4_K_M).

**Implications to be honest about in any writeup:**
- Judge quality was *expected* to be lower than paper's 72B, but post-run results show our 7B achieved r=0.642/0.721 vs the paper's 72B ratings — comparable to the paper's own reported 72B-vs-human agreement (r=0.641/0.602). However, the result is better than anticipated.
- `outputs/ratings_judge_paper.jsonl` (the 9600 paper-Qwen-72B ratings extracted from CSV) is no longer a same-model κ baseline. Cohen's κ vs. that file becomes a *cross-scale* check — a quality lower bound, not a reproducibility check. Expect modest agreement.
- If reviewers ask, the deviation is documented here, in CLAUDE.md, and in HANDOFF.md.

## Goal
Run Qwen2.5-7B-Instruct as the judge over (joke, reference_explanation, model_explanation) tuples for each of the two criteria (accuracy, completeness), producing `outputs/ratings_judge.jsonl` (~9600 ratings: 600 jokes × 8 models × 2 criteria).

---

## Code (Member 1 -- Pranav)
| Path | Notes |
|---|---|
| `prompts/judge_template.txt` | Verbatim Appendix A.6 (preserves the leading-space-before-`{joke}` formatting) |
| `src/judge.py` | Driver — resumable, per-row error log, `--limit`, `--retries` |
| `src/judge_backend.py` | 4 backends: `openrouter` (default → `qwen/qwen-2.5-7b-instruct`), `together` (→ `Qwen/Qwen2.5-7B-Instruct-Turbo`), **`ollama`** (→ `qwen2.5:7b-instruct-q4_K_M`, free local), `local` (→ `Qwen/Qwen2.5-7B-Instruct`, HF + bitsandbytes, HPC-only) |
| `src/parser.py` | First-int-0..5 score parser |
| `src/rubric.py` | Loads + validates rubric (ascending) |
| `src/schema.py` | `JUDGE_MODEL_NAME = "qwen2.5-7b-instruct"` (annotator tag in output rows) |
| `tests/test_parser.py` | 18 tests — all passing |

## Data prep (Member 1 -- Pranav)
- **Temperature:** 0.1 (paper's Appendix A.6), `max_tokens=16` (digit + whitespace headroom)
- **Score parser:** first integer in response in `[0, 5]`; failures logged to `outputs/judge.errors.log`, never crash, never clamp
- **Resume key:** `(joke_id, model, criterion)` — re-running is a no-op once complete
- **Annotator tag:** `qwen2.5-7b-instruct` (constant from `schema.JUDGE_MODEL_NAME`)
- **One criterion per call:** 2 calls per (joke, model) pair — template `{criteria}` is singular

---

## Run (Member 2 -- Khoi)
### Environment
- Platform: Windows WSL2 (Ubuntu), NVIDIA GPU, CUDA passthrough
- Conda env: `applesVsOranges` (Python 3.10)
- Model: `qwen2.5:7b-instruct-q4_K_M` served via Ollama
- GPU env vars: `OLLAMA_NUM_GPU=-1`, `CUDA_VISIBLE_DEVICES=0`

### How it was run
```bash
conda activate applesVsOranges
cd /mnt/d/ApplesVSOranges
jupyter notebook run_judge.ipynb
```

The notebook (`run_judge.ipynb`) imports `src/judge.py` via `sys.path` and calls `main()` with `sys.argv` — no code duplication. GPU env vars are set before the Ollama daemon starts so all layers load onto VRAM.

Smoke test (4 calls) confirmed before full run:
```
topical_000  gpt-4o        accuracy      score=4
topical_000  gpt-4o        completeness  score=4
topical_000  gpt-4o-mini   accuracy      score=4
topical_000  gpt-4o-mini   completeness  score=4
```

Full run completed in a single session with no interruptions.

### How to re-run (if needed)
**Free, local (WSL + GPU):**
```bash
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M
jupyter notebook run_judge.ipynb   # or:    python src/judge.py --backend ollama --resume
```
No key, no rate limits. ~3–5 hrs for ~9600 calls on RTX 3090

**Paid, fast:**
```bash
export OPENROUTER_API_KEY=sk-or-...
python src/judge.py --resume                  # default --backend openrouter
```
~9600 calls, budget **~$0.50–2** at 7B prices, runs in minutes.

---

## Output

| File | Rows | Notes |
|---|---|---|
| `outputs/ratings_judge.jsonl` | **9,600** | 600 jokes × 8 models × 2 criteria — 100% complete |
| `outputs/ratings_judge_smoketest.jsonl` | 4 | smoke run only, not used downstream |
| `outputs/judge_errors.log` | 0 errors | clean run |

Output row schema:
```json
{"joke_id": "topical_000", "model": "gpt-4o",
 "criterion": "accuracy", "score": 4,
 "annotator": "qwen2.5-7b-instruct"}
```

---

## Results

### Average scores by joke type (all models)

| Joke type     | Accuracy | Completeness |
|---|---|---|
| Homographic   | 3.866    | 3.578        |
| Heterographic | 3.656    | 3.368        |
| Non-Topical   | 3.784    | 3.424        |
| Topical       | 3.784    | 3.398        |

### Hypothesis checks

| Hypothesis | Result | Notes |
|---|---|---|
| H1: Puns easier than Reddit | **SIMILAR** (3.761 vs 3.784) | See note below |
| H2: Homographic > Heterographic | CONFIRMED (3.866 vs 3.656) | |
| H3: Topical hardest | **SIMILAR** (3.784 == 3.784) | See note below |
| H4: Larger > Smaller | ✅ CONFIRMED (3/4 pairs) | Gemini Pro vs Flash reversed by <0.05 — near-tied |

**H1/H3 note:** near-identical scores (Δ < 0.02) rather than clear separations. Expected consequence of the 7B judge concentrating ~80% of scores at 4, losing the distributional spread needed to rank joke types. The directional finding holds in success-rate analysis (see `status_task5_analyze.md`).

### Judge agreement vs paper's Qwen-72B baseline

| Criterion    | Pearson r | MAE   | Cohen κ | Within ±1 |
|---|---|---|---|---|
| Accuracy     | 0.642     | 0.438 | 0.570   | 98.3%     |
| Completeness | 0.721     | 0.434 | 0.578   | 99.2%     |

Paper's own 72B-vs-human: r=0.641/0.602, κ=0.565/0.519. Our 7B matches or exceeds this, which is a stronger result than expected given the model size difference. Full numbers in `outputs/tables/judge_agreement.csv`.

Score distribution comparison: `outputs/figures/fig_judge_comparison.png`.

---

## WSL-specific notes for reproducibility (Window User with VSCode Remote WSL setup)

1. Ollama does not start automatically in WSL — run `ollama serve` in a separate terminal first, or let `run_judge.ipynb` Section 3 start it.
2. GPU passthrough requires Windows NVIDIA driver ≥ 527.41 on the host. `nvidia-smi` must be visible inside WSL.
3. `OLLAMA_NUM_GPU=-1` forces all layers onto GPU — without it Ollama may split CPU/GPU.
4. `libtinfo.so.6` / `libcurl.so.4` warnings during Ollama install are harmless (conda library conflict with WSL).

---

## Status
| Item | State |
|---|---|
| Code (judge.py, backends, parser) | ✅ Done |
| Prompts / rubric files | ✅ Done |
| Tests (18 parser tests) | ✅ Done |
| Ollama setup (WSL + GPU) — `run_judge.ipynb` | ✅ Done |
| Smoke test | ✅ 4/4 parsed |
| Full run | ✅ 9,600 / 9,600 rows — 0 errors |
| Judge agreement vs 72B — `outputs/tables/judge_agreement.csv` | ✅ Done |
| Re-judge extension explanations (`outputs/extension/ratings_rag.jsonl`) | ⏳ Pending — WIP on `src/extension.py` (Member 3 -- Jahdon) |
