# Task 3.5 — LLM-as-a-judge pipeline — ✅ DONE

**Implementation:** Pranav (Member 1)
**Run:** Khoi (Member 2)
**Last verified:** 2026-05-11
**Depends on:** Task 3 (inference outputs in `data/explanations.jsonl`)

---

## ⚠ Deviation from paper (decided 2026-04-30)
Paper uses **Qwen2.5-72B-Instruct**. We use **Qwen2.5-7B-Instruct**.

**Why:** cost + feasibility. 72B run costs ~$5–15 on OpenRouter and needs ~40 GB VRAM locally; 7B is ~$0.50–2 paid, or **free via Ollama** on a consumer GPU (~5 GB VRAM at q4_K_M).

**Implications to be honest about in any writeup:**
- Judge quality is expected to be lower than the paper's 72B. The 7B judge concentrates scores around 4, compressing some joke-type differences (visible as H1's near-tie verdict).
- `outputs/tables/judge_agreement.csv` currently compares "our 7B × our explanations" vs "paper's 72B × paper's explanations". Only 2 model slugs overlap (Llama 3.1 8B, R1-Llama 8B) and the explanations being scored differ in each file, so the agreement values (r=0.245 / 0.288) are a weak floor rather than a true 7B-vs-72B calibration.
- For a proper apples-to-apples calibration, re-run `src/judge.py --explanations data/explanations_paper.jsonl` and use that file as `--ratings` to `analyze.py`. ~3–4 hrs of extra judge time. Not blocking for the replication writeup.
- The deviation is documented here, in CLAUDE.md, and in HANDOFF.md.

## Goal
Run Qwen2.5-7B-Instruct as the judge over (joke, reference_explanation, model_explanation) tuples for each of the two criteria (accuracy, completeness), producing `outputs/ratings_judge.jsonl` — 9,600 ratings: 600 jokes × 8 models × 2 criteria.

---

## Code (Member 1 — Pranav)
| Path | Notes |
|---|---|
| `prompts/judge_template.txt` | Verbatim Appendix A.6 (preserves the leading-space-before-`{joke}` formatting) |
| `src/judge.py` | Driver — resumable, per-row error log, `--limit`, `--retries` |
| `src/judge_backend.py` | 4 backends: `openrouter` (default → `qwen/qwen-2.5-7b-instruct`), `together` (→ `Qwen/Qwen2.5-7B-Instruct-Turbo`), **`ollama`** (→ `qwen2.5:7b-instruct-q4_K_M`, free local), `local` (→ `Qwen/Qwen2.5-7B-Instruct`, HF + bitsandbytes, HPC-only) |
| `src/parser.py` | First-int-0..5 score parser |
| `src/rubric.py` | Loads + validates rubric (ascending) |
| `src/schema.py` | `JUDGE_MODEL_NAME = "qwen2.5-7b-instruct"` (annotator tag in output rows) |
| `tests/test_parser.py` | 18 tests — all passing |

## Data prep (Member 1 — Pranav)
- **Temperature:** 0.1 (paper's Appendix A.6), `max_tokens=16` (digit + whitespace headroom)
- **Score parser:** first integer in response in `[0, 5]`; failures logged to `outputs/judge_errors.log`, never crash, never clamp
- **Resume key:** `(joke_id, model, criterion)` — re-running is a no-op once complete
- **Annotator tag:** `qwen2.5-7b-instruct` (constant from `schema.JUDGE_MODEL_NAME`)
- **One criterion per call:** 2 calls per (joke, model) pair — template `{criteria}` is singular

---

## Run (Member 2 — Khoi)
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

The notebook imports `src/judge.py` via `sys.path` and calls `main()` with `sys.argv` — no code duplication. GPU env vars are set before the Ollama daemon starts so all layers load onto VRAM.

Full run completed in a single session with no interruptions.

### How to re-run (if needed)
**Free, local (WSL + GPU):**
```bash
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M
jupyter notebook run_judge.ipynb        # or: python src/judge.py --backend ollama --resume
```
No key, no rate limits. ~3–5 hrs for 9,600 calls on RTX 3060/4060 class.

**Paid, fast:**
```bash
export OPENROUTER_API_KEY=sk-or-...
python src/judge.py --resume            # default --backend openrouter
```
~9,600 calls, budget **~$0.50–2** at 7B prices, runs in minutes.

---

## Output

| File | Rows | Notes |
|---|---|---|
| `outputs/ratings_judge.jsonl` | **9,600** | 600 jokes × 8 models × 2 criteria — 100% complete |
| `outputs/ratings_judge_smoketest.jsonl` | 4 | smoke run only, not used downstream |
| `outputs/judge_errors.log` | 0 errors | clean run |

Output row schema:
```json
{"joke_id": "topical_000", "model": "r1-distill-llama-8b",
 "criterion": "accuracy", "score": 4,
 "annotator": "qwen2.5-7b-instruct"}
```

---

## Results

### Average scores by joke type (across our 8 models)

| Joke type     | Accuracy | Completeness |
|---|---:|---:|
| Homographic   | 3.706    | 3.470        |
| Heterographic | 3.476    | 3.248        |
| Non-Topical   | 3.669    | 3.277        |
| Topical       | 3.553    | 3.193        |

Completeness is consistently lower than accuracy across every cell — matches the paper's observation that models more often omit details than hallucinate incorrect ones.

### Hypothesis checks (from `outputs/tables/hypothesis_checks.csv`)

| Hypothesis | Δ | Verdict |
|---|---:|---|
| H1: Puns > Reddit jokes | -0.020 (accuracy) | NEAR-TIE |
| H2: Homographic > Heterographic | 0.230 (accuracy) | CONFIRMED |
| H3: Topical hardest | 0.117 (accuracy) | CONFIRMED |
| H4 (Gemma 2 9B vs 2B, clean pair) | 0.077 (success rate) | CONFIRMED |
| H4 (Phi-3 Medium 14B vs Mini 3.8B, clean pair) | -0.065 (success rate) | FAILED |
| H4 (Llama 3.1 8B vs 3.2 3B, cross-gen) | 0.022 (success rate) | NEAR-TIE |

**H1 note:** the puns vs Reddit difference is tiny (Δ=-0.020) and lands on the wrong side of the paper's hypothesis. The 7B judge concentrates ~80% of scores at 4, compressing this distinction. The directional finding still holds in the success-rate analysis — see `status_task5_analyze.md`.

**H4 (Phi) failed result:** the Phi-3 family is the standout outlier — Phi-3 Mini outperforms Phi-3 Medium in every joke type. This is a clean within-family pair (same generation, only size differs) so the result is a real counter-example to H4, not an artefact. Worth discussing in the writeup as evidence that the H4 "larger > smaller" pattern is family-specific rather than universal.

### Judge agreement vs paper's Qwen-72B (`judge_agreement.csv`)

| Criterion    | n_paired | n_models_overlap | Pearson r | MAE   | Cohen κ | Within ±1 |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy     | 1,200 | 2 | 0.245 | 0.724 | 0.223 | 86.7% |
| Completeness | 1,200 | 2 | 0.288 | 0.660 | 0.269 | 90.2% |

This is a **weak baseline**: only 2 model slugs overlap (Llama 3.1 8B, R1-Llama 8B), and the two ratings files score different explanation texts even for shared model slugs. For a proper calibration, judge `data/explanations_paper.jsonl` with our 7B judge and use that file as `--ratings` to `analyze.py`. See `outputs/figures/fig_judge_comparison.png` for the score-distribution overlap.

---

## WSL-specific notes for reproducibility (Windows user with VS Code Remote WSL setup)

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
| Judge agreement vs 72B — `outputs/tables/judge_agreement.csv` | ✅ Done (weak baseline; proper calibration is optional follow-up) |
| Re-judge extension explanations (`outputs/extension/ratings_rag.jsonl`) | ⏳ Pending — WIP on `src/extension.py` (Member 3 — Jahdon) |
