# Task 3 — LLM-as-a-judge pipeline (Qwen2.5-**7B**-Instruct) — ⚠ IMPLEMENTED, NOT YET RUN

**Owner:** Pranav (Member 1)
**Last verified:** 2026-04-30

## ⚠ Deviation from paper (decided 2026-04-30)
Paper uses **Qwen2.5-72B-Instruct**. We use **Qwen2.5-7B-Instruct**.

**Why:** cost + feasibility. 72B run costs ~$5–15 on OpenRouter and needs ~40GB VRAM locally; 7B is ~$0.50–2 paid, or **free via Ollama** on Apple Silicon (~5GB RAM at q4_K_M).

**Implications to be honest about in any writeup:**
- Judge quality is *expected* to be lower than paper's 72B — Qwen2.5-7B is a much weaker instruction-follower and rater.
- `outputs/ratings_judge_paper.jsonl` (the 9600 paper-Qwen-72B ratings extracted from CSV) is no longer a same-model κ baseline. Cohen's κ vs. that file becomes a *cross-scale* check — a quality lower bound, not a reproducibility check. Expect modest agreement.
- If reviewers ask, the deviation is documented here, in CLAUDE.md, and in HANDOFF.md.

## Goal
Run Qwen2.5-7B-Instruct as the judge over (joke, reference_explanation, model_explanation) tuples for each of the two criteria (accuracy, completeness), producing `outputs/ratings_judge.jsonl` (~9600 ratings: 600 jokes × 8 models × 2 criteria).

## What exists (code)
| Path | Notes |
|---|---|
| `prompts/judge_template.txt` | Verbatim Appendix A.6 (preserves the leading-space-before-`{joke}` formatting) |
| `src/judge.py` | Driver — resumable, per-row error log, `--limit`, `--retries` |
| `src/judge_backend.py` | 4 backends: `openrouter` (default → `qwen/qwen-2.5-7b-instruct`), `together` (→ `Qwen/Qwen2.5-7B-Instruct-Turbo`), **`ollama`** (→ `qwen2.5:7b-instruct-q4_K_M`, free local), `local` (→ `Qwen/Qwen2.5-7B-Instruct`, HF + bitsandbytes, HPC-only) |
| `src/parser.py` | First-int-0..5 score parser |
| `src/rubric.py` | Loads + validates rubric (ascending) |
| `src/schema.py` | `JUDGE_MODEL_NAME = "qwen2.5-7b-instruct"` (annotator tag in output rows) |
| `tests/test_parser.py` | 18 tests — all passing |

## What's missing (output)
- ❌ `outputs/ratings_judge.jsonl` — **never produced**.

## How to run (when ready)

**Free, local (recommended on Mac):**
```bash
ollama serve &                                # daemon on :11434
ollama pull qwen2.5:7b-instruct-q4_K_M        # ~4.4 GB, one-time
python src/judge.py --backend ollama --resume
# → outputs/ratings_judge.jsonl
```
No key, no rate limits. ~3–5 hrs for ~9600 calls on Apple Silicon.

**Paid, fast:**
```bash
export OPENROUTER_API_KEY=sk-or-...
python src/judge.py --resume                  # default --backend openrouter
```
~9600 calls, budget **~$0.50–2** at 7B prices, runs in minutes.

`--backend together` (paid alternative) and `--backend local` (HPC-only — bitsandbytes blocked on Darwin) also available.

## Conventions captured (don't re-derive)
- **Decoding:** temperature **0.1** (paper's only deviation from defaults), `max_tokens=16` (digit + headroom for whitespace), everything else default.
- **Score parser:** first integer in response. If not in `[0, 5]`, log to `outputs/judge.errors.log` and continue — never crash, never clamp.
- **Resume key:** `(joke_id, model, criterion)`. Re-running the same command is a no-op once complete.
- **Annotator tag** in output rows: `qwen2.5-7b-instruct` (constant from `schema.JUDGE_MODEL_NAME`).
- **One criterion per call.** The template's `{criteria}` placeholder is singular; we issue 2 calls per (joke, model) pair, not 1 combined call.

## Output schema (one row)
```json
{"joke_id": "topical_000", "model": "gpt-4o",
 "criterion": "accuracy", "score": 5,
 "annotator": "qwen2.5-7b-instruct"}
```
Same shape as `outputs/ratings_judge_paper.jsonl` (which carries `"annotator": "qwen2.5-72b-instruct"`); downstream code keys on `(joke_id, model, criterion)` and treats `annotator` as metadata.

## Smoke check (after running)
After producing `ratings_judge.jsonl`, compute Cohen's κ vs. `ratings_judge_paper.jsonl` per (model, criterion) bucket. Caveat above: this is cross-scale (7B vs 72B), so high agreement isn't expected. Useful as a quality floor — if κ is essentially zero, our pipeline is broken.

## Status
| Item | State |
|---|---|
| Code | ✅ shipped (now points at 7B) |
| Tests | ✅ 18 parser tests passing |
| Run on real API | ❌ pending — needs `OPENROUTER_API_KEY` and ~$1 budget |
| κ vs paper Qwen-72B baseline | ❌ pending (depends on the run above) |

## Why not yet run
1. Teammate's `data/explanations.jsonl` hasn't landed — currently using the dev copy derived from CSV. Re-judging is cheap (free with Ollama, ~$1 with OpenRouter), so this isn't really blocking.
2. Pulling `qwen2.5:7b-instruct-q4_K_M` (~4.4 GB) hadn't been done as of 2026-04-30; Ollama daemon is running but model isn't pulled yet.

**Decision point:** with Ollama wired in, the run is now $0 + a few hours of background compute. Pull the model and kick it off whenever convenient.
