# Runbook — inference step

Generating fresh joke explanations from five local Ollama models, then
feeding them into the existing judge + analyze pipeline. No API keys, no
billing, no rate limits. Models are stored on the **D: drive** to avoid
filling up C:.

The paper explanations are renamed to `data/explanations_paper.jsonl`

## File layout

| File | Contents |
|---|---|
| `data/jokes.jsonl` | The 600 jokes (already produced) |
| `data/explanations_paper.jsonl` | The paper authors' released explanations (renamed from the original) |
| `data/explanations.jsonl` | **Our** model outputs — produced by `inference.py` |

This rename means downstream scripts (`judge.py`, `metrics.py`, `analyze.py`)
work without flag changes — their default `--explanations data/explanations.jsonl`
now picks up outputs automatically.

## Files added

- `src/inference.py` — main script (mirrors `src/judge.py`'s structure).
  Default `--output` is `data/explanations.jsonl`.
- `src/inference_backend.py` — adds `GeminiBackend`; reuses `OllamaBackend`
  from `judge_backend.py`.
- `run_inference.ipynb` — Jupyter notebook orchestrating the five runs

## Final model lineup

| # | Model | Ollama tag | Slug | Paper match | H4 role |
|---|---|---|---|---|---|
| 1 | DeepSeek-R1-Distill-**Llama**-8B | `deepseek-r1:8b` | `r1-distill-llama-8b` | exact | — |
| 2 | Llama 3.1 8B Instruct | `llama3.1:8b` | `llama-3.1-8b` | exact | — |
| 3 | Llama 3.2 3B Instruct | `llama3.2:3b` | `llama-3.2-3b` | extension | small Llama (cross-gen pair) |
| 4 | Gemma 2 **2B** Instruct | `gemma2:2b` | `gemma-2-2b` | extension | small Gemma (clean H4 pair) |
| 5 | Gemma 2 **9B** Instruct | `gemma2:9b` | `gemma-2-9b` | extension | large Gemma (clean H4 pair) |

## One-time setup — point Ollama at D: drive

By default Ollama stores models in `~/.ollama/models` (on the WSL filesystem,
which lives on C:). The lineup above is ~21 GB; redirect it to D: first.

**Run in a WSL terminal (not in the notebook):**

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

## Pull all five models (~21 GB total)

```bash
ollama pull deepseek-r1:8b      # ~5 GB — paper-exact R1
ollama pull llama3.1:8b         # ~5 GB — paper-exact Llama 8B
ollama pull llama3.2:3b         # ~2 GB
ollama pull gemma2:2b           # ~1.5 GB
ollama pull gemma2:9b           # ~6 GB

du -sh /mnt/d/ollama-models     # confirm storage on D:
ollama list
```

## Run the five models

The notebook is the easiest path:

```bash
conda activate applesVsOranges
jupyter notebook run_inference.ipynb
```

Section 3 (smoke test) first, then Sections 4-8 in any order.

Or via terminal:

```bash
python src/inference.py --backend ollama --model-id deepseek-r1:8b --model-slug r1-distill-llama-8b
python src/inference.py --backend ollama --model-id llama3.1:8b    --model-slug llama-3.1-8b
python src/inference.py --backend ollama --model-id llama3.2:3b    --model-slug llama-3.2-3b
python src/inference.py --backend ollama --model-id gemma2:2b      --model-slug gemma-2-2b
python src/inference.py --backend ollama --model-id gemma2:9b      --model-slug gemma-2-9b
```

All five append to `data/explanations.jsonl`. Total runtime depends on your
GPU. Rough estimates on a consumer GPU like RTX 3060/4060:

| Model | Approx runtime for 600 jokes |
|---|---|
| DeepSeek-R1-Distill-Llama-8B | ~60 min |
| Llama 3.1 8B | ~45 min |
| Llama 3.2 3B | ~20 min |
| Gemma 2 2B | ~15 min |
| Gemma 2 9B | ~50 min |
| **Total** | **~3.2 hours** |

## Once inference is done

```bash
# Judge explanations (Qwen 7B via the same Ollama daemon)
python src/judge.py \
    --output outputs/ratings_judge_ours.jsonl \
    --backend ollama

# Automatic metrics
python src/metrics.py \
    --output outputs/metrics_ours.csv

# Figures + hypothesis tests
python src/analyze.py \
    --ratings outputs/ratings_judge_ours.jsonl \
    --metrics outputs/metrics_ours.csv \
    --outdir outputs/ours/
```

If you want to re-run the analysis on the paper's released explanations as a
sanity check, point the scripts at the renamed file:

```bash
python src/judge.py \
    --explanations data/explanations_paper.jsonl \
    --output outputs/ratings_judge_paper_rerun.jsonl \
    --backend ollama
```

If `metrics.py` or `analyze.py` don't have an `--explanations` / `--metrics`
flag yet, that's a small argparse change to add — they're standard scripts.

## Caveats for the writeup

1. **Two paper matches, three extension models** — frame in two parts:
   - For `r1-distill-llama-8b` and `llama-3.1-8b`: direct comparison to the
     paper's reported per-model numbers in their Figure 3.
   - For `llama-3.2-3b`, `gemma-2-2b`, and `gemma-2-9b`: framed as *"do the
     paper's findings generalize to newer / different-family models?"*

2. **Two H4 tests, one cleaner than the other.**
   - **Clean H4 (Gemma 2B vs 9B)** — same generation, only size differs.
     Strongest evidence.
   - **Confounded H4 (Llama 3.1 8B vs 3.2 3B)** — different generations,
     so size + generation are conflated. Useful as cross-validation but
     flag the confound.

3. **No judge family-bias** — all five inference models are Llama-derived or
   Gemma; the Qwen 7B judge has no shared-family inflation risk.

4. **All-local stack, fully reproducible** — no API calls, no closed-source
   dependencies. Anyone with Ollama can rerun the whole experiment.

## Smoke test (no model needed)

```bash
python smoke_test_inference.py
```

Should print four checks ending in `ALL SMOKE TESTS PASSED`. This is the
standalone test that doesn't require Ollama or any models — just confirms
the script's argument parsing, schema, and resume logic work correctly.
