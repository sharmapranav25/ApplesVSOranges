# Task 3 — Inference (generate model explanations) — ✅ DONE

**Owner:** Khoi (Member 2)
**Last verified:** 2026-05-11

---

## ⚠ Deviation from paper
Paper tests 8 models (GPT-4o, GPT-4o Mini, Gemini 1.5 Pro/Flash, Llama 3.1 70B/8B, DeepSeek-R1-Distill 70B/8B). We use **8 locally-served open-source models via Ollama**: 2 paper-match + 6 extension models spanning four model families.

**Why:** all-local fully-reproducible stack — no API costs, no rate limits, no closed-source dependencies. The paper's closed-source models (GPT, Gemini) require paid API access; the 70B models require ~40 GB VRAM. The 6 extension models enable:
- Clean within-family H4 size tests for two families (Gemma 2 2B vs 9B; Phi-3 Mini 3.8B vs Medium 14B).
- Newer-generation cross-validation (Llama 3.2 3B vs Llama 3.1 8B).
- Cross-family coverage (Gemma, Mistral, Phi-3) testing whether paper findings generalise beyond Llama-family models.

**Implications:**
- Direct paper-comparison via Table 2 is available for two models: Llama 3.1 8B and R1-Llama 8B. See `outputs/tables/replication_gap.csv` for per-model gaps.
- No judge family-bias: none of the 8 inference models share a family with the Qwen 7B judge.

## Goal
Generate `data/explanations.jsonl` containing 4,800 joke explanations — 8 models × 600 jokes, one explanation per (model, joke) pair. Output is the input for Task 3.5 (judge) and Task 4 (metrics).

---

## Final model lineup

| # | Model | Ollama tag | Slug | Paper match | H4 role |
|---|---|---|---|---|---|
| 1 | DeepSeek-R1-Distill-Llama-8B | `deepseek-r1:8b` | `r1-distill-llama-8b` | exact | — |
| 2 | Llama 3.1 8B Instruct | `llama3.1:8b` | `llama-3.1-8b` | exact | — |
| 3 | Llama 3.2 3B Instruct | `llama3.2:3b` | `llama-3.2-3b` | extension | small Llama (cross-gen pair) |
| 4 | Gemma 2 2B Instruct | `gemma2:2b` | `gemma-2-2b` | extension | small Gemma (clean H4 pair) |
| 5 | Gemma 2 9B Instruct | `gemma2:9b` | `gemma-2-9b` | extension | large Gemma (clean H4 pair) |
| 6 | Mistral 7B Instruct v0.3 | `mistral:7b` | `mistral-7b` | extension | cross-family coverage |
| 7 | Phi-3 Mini 3.8B Instruct | `phi3:mini` | `phi-3-mini` | extension | small Phi (clean H4 pair) |
| 8 | Phi-3 Medium 14B Instruct | `phi3:medium` | `phi-3-medium` | extension | large Phi (clean H4 pair) |

---

## Code

| Path | Notes |
|---|---|
| `src/inference.py` | Driver — mirrors `src/judge.py`'s structure; resumable; per-row error log; `--limit`, `--retries` flags. Default `--output` is `data/explanations.jsonl`. |
| `src/inference_backend.py` | Adds `GeminiBackend`; reuses `OllamaBackend` from `judge_backend.py` |
| `run_inference.ipynb` | Notebook orchestrating the eight runs |
| `smoke_test_inference.py` | Standalone smoke test (no Ollama required) — confirms argument parsing, schema, and resume logic |

### Data layout
| File | Contents |
|---|---|
| `data/jokes.jsonl` | The 600 jokes (from Task 1) |
| `data/explanations_paper.jsonl` | Paper authors' released explanations (renamed from original, kept as reference) |
| `data/explanations.jsonl` | Our 8-model outputs — produced by `inference.py` |

Renaming the paper file means `judge.py`, `metrics.py`, and `analyze.py` pick up our explanations via their default `--explanations` flag without overrides.

---

## Implementation choices
- **max_tokens:** 4096 — needed for R1's reasoning prefix; sufficient for all 8 models
- **Temperature:** model defaults except R1-distill which uses `temperature=0.5` to avoid output truncation
- **Resume key:** `(joke_id, model)` — re-running is a no-op once complete
- **Storage:** Ollama models redirected to D: drive (~36 GB total) via `OLLAMA_MODELS=/mnt/d/ollama-models`

---

## Run

### Environment
- WSL2 (Ubuntu) with NVIDIA GPU CUDA passthrough
- Conda env: `applesVsOranges` (Python 3.10)
- Ollama daemon with `OLLAMA_NUM_GPU=-1` for full GPU utilisation

### One-time setup — point Ollama at D: drive
By default Ollama stores models in `~/.ollama/models` on the WSL filesystem (C:). The lineup is ~36 GB; redirect to D: first.

```bash
# 1. Stop any running ollama daemon
pkill ollama 2>/dev/null
sleep 2

# 2. Move existing models
if [ -d ~/.ollama/models ] && [ "$(ls -A ~/.ollama/models 2>/dev/null)" ]; then
    mkdir -p /mnt/d/ollama-models
    mv ~/.ollama/models/* /mnt/d/ollama-models/
fi

# 3. Set OLLAMA_MODELS for this session and start the daemon
export OLLAMA_MODELS=/mnt/d/ollama-models
ollama serve &
sleep 3

# 4. Make it permanent
echo 'export OLLAMA_MODELS=/mnt/d/ollama-models' >> ~/.bashrc
```

Verify:
```bash
cat /proc/$(pgrep -f 'ollama serve' | head -1)/environ | tr '\0' '\n' | grep OLLAMA
```

### Pull all eight models (~36 GB total)
```bash
ollama pull deepseek-r1:8b      # ~5 GB — paper-exact R1
ollama pull llama3.1:8b         # ~5 GB — paper-exact Llama 8B
ollama pull llama3.2:3b         # ~2 GB
ollama pull gemma2:2b           # ~1.5 GB
ollama pull gemma2:9b           # ~6 GB
ollama pull mistral:7b          # ~4.4 GB
ollama pull phi3:mini           # ~2.2 GB
ollama pull phi3:medium         # ~7.9 GB

du -sh /mnt/d/ollama-models     # confirm storage on D:
ollama list
```

### How it was run

Notebook (recommended):
```bash
conda activate applesVsOranges
jupyter notebook run_inference.ipynb
```
Smoke test cell first, then the eight model cells in any order.

CLI (alternative — one command per model):
```bash
python src/inference.py --backend ollama --model-id deepseek-r1:8b --model-slug r1-distill-llama-8b
python src/inference.py --backend ollama --model-id llama3.1:8b    --model-slug llama-3.1-8b
python src/inference.py --backend ollama --model-id llama3.2:3b    --model-slug llama-3.2-3b
python src/inference.py --backend ollama --model-id gemma2:2b      --model-slug gemma-2-2b
python src/inference.py --backend ollama --model-id gemma2:9b      --model-slug gemma-2-9b
python src/inference.py --backend ollama --model-id mistral:7b     --model-slug mistral-7b
python src/inference.py --backend ollama --model-id phi3:mini      --model-slug phi-3-mini
python src/inference.py --backend ollama --model-id phi3:medium    --model-slug phi-3-medium
```

All eight append to `data/explanations.jsonl`. Approximate runtimes on a consumer GPU (RTX 3060/4060 class):

| Model | Runtime for 600 jokes |
|---|---|
| DeepSeek-R1-Distill-Llama-8B | ~60 min |
| Llama 3.1 8B | ~45 min |
| Llama 3.2 3B | ~20 min |
| Gemma 2 2B | ~15 min |
| Gemma 2 9B | ~50 min |
| Mistral 7B | ~40 min |
| Phi-3 Mini | ~25 min |
| Phi-3 Medium | ~65 min |
| **Total** | **~5.3 hours** |

---

## Output

| File | Rows | Notes |
|---|---|---|
| `data/explanations.jsonl` | **4,800** | 8 models × 600 jokes — 100% complete |
| `outputs/inference.errors.log` | 0 errors | clean run after `max_tokens=4096` + R1 `temperature=0.5` |

Output row schema:
```json
{"joke_id": "topical_000", "model": "r1-distill-llama-8b",
 "explanation": "...", "annotator": "r1-distill-llama-8b"}
```

---

## Caveats for the writeup

1. **Two paper matches, six extension models** — frame in two parts:
   - For `r1-distill-llama-8b` and `llama-3.1-8b`: direct comparison to paper's per-model numbers in Figure 3 and Appendix A.4 Table 2.
   - For the six extension models: framed as *"do the paper's findings generalize across families (Gemma, Mistral, Phi-3) and to newer generations (Llama 3.2)?"*

2. **Three H4 tests, with varying cleanliness:**
   - **Clean H4 (Gemma 2 2B vs 9B)** — same generation, only size differs. H4 CONFIRMED (Δ=0.077 success rate).
   - **Clean H4 (Phi-3 Mini 3.8B vs Medium 14B)** — same generation, only size differs. H4 FAILED — Phi-3 Mini actually outperforms Phi-3 Medium (Δ=-0.065). Worth flagging in the writeup as a counter-example to the paper's H4 claim within this family.
   - **Cross-gen H4 (Llama 3.1 8B vs 3.2 3B)** — different generations, so size + generation are conflated. NEAR-TIE (Δ=0.022) — useful as cross-validation but flag the confound.

3. **No judge family-bias** — none of the 8 inference models is Qwen-derived; the Qwen 7B judge has no shared-family inflation risk for any of them.

4. **All-local stack, fully reproducible** — no API calls, no closed-source dependencies. Anyone with Ollama can rerun the whole experiment.

---

## Status

| Item | State |
|---|---|
| Code (`src/inference.py`, backend) | ✅ shipped |
| Smoke test | ✅ passing |
| Models pulled (8 × ~36 GB on D:) | ✅ |
| Full run | ✅ 4,800 / 4,800 rows — 0 errors |
| Notebook (`run_inference.ipynb`) | ✅ |
