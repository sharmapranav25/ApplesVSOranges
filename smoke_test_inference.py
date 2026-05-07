"""Smoke test: run src/inference.py end-to-end with a mock backend.

Verifies, without needing a real Ollama daemon or any pulled models:
  1. The script reads jokes.jsonl correctly
  2. The output schema matches Explanation (joke_id, model, explanation)
  3. Resume skips already-done (joke_id, model_slug) pairs
  4. R1-style <think> blocks are stripped by default
  5. Multi-model writes coexist in the same file
  6. The pre-flight Ollama-model check passes when the model is "pulled"
     (this version stubs requests.get so the check finds our mock model)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


class MockBackend:
    """Returns deterministic fake explanations. Includes a <think> block so
    we can verify the strip_reasoning step works."""
    name = "mock"

    def __init__(self, model: str | None = None, **kw):
        self.model = model or "mock-model"
        self.calls = 0

    def complete(self, messages, temperature, max_tokens):
        self.calls += 1
        return (
            "<think>The joke seems to involve wordplay, let me think about "
            "the double meaning here.</think>\n\n"
            f"This is mock explanation #{self.calls} for testing purposes. "
            f"It exercises the {self.model} backend through the full pipeline."
        )


class _FakeOllamaTagsResponse:
    """Stand-in for the requests.get() response from /api/tags. Reports any
    model-id we ask about as already pulled, so the pre-flight check passes
    in the smoke test without requiring a real daemon."""
    _models = ["mock-model", "fake-tag", "fake-tag-2", "deepseek-r1:8b",
               "llama3.1:8b", "llama3.2:3b", "gemma2:2b", "gemma2:9b"]
    def raise_for_status(self): pass
    def json(self): return {"models": [{"name": n} for n in self._models]}


def _patches(model_name):
    """Return the two patches every smoke run needs:
       1. Replace get_inference_backend with one that returns MockBackend
       2. Replace requests.get with one that returns the fake daemon response
    """
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("inference.get_inference_backend",
                              return_value=MockBackend(model=model_name)))
    stack.enter_context(patch("requests.get",
                              return_value=_FakeOllamaTagsResponse()))
    return stack


def run_smoke():
    from inference import main as inference_main

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        jokes_path = td / "jokes.jsonl"
        out_path = td / "explanations.jsonl"
        err_path = td / "errors.log"

        # Tiny 3-joke fixture (one of each from the real dataset, abbreviated)
        jokes = [
            {"id": "homographic_000", "type": "homographic",
             "joke": "They hid in a sauna where they could sweat it out.",
             "reference_explanation": "...", "source_index": None},
            {"id": "heterographic_000", "type": "heterographic",
             "joke": "I used to be a banker but I lost interest.",
             "reference_explanation": "...", "source_index": None},
            {"id": "topical_000", "type": "topical",
             "joke": "Tide ad budget grew after cornering the teenage snack market.",
             "reference_explanation": "...", "source_index": None},
        ]
        with jokes_path.open("w") as f:
            for j in jokes:
                f.write(json.dumps(j) + "\n")

        common_argv = [
            "inference.py",
            "--jokes", str(jokes_path),
            "--output", str(out_path),
            "--errors", str(err_path),
            "--backend", "ollama",
        ]

        # ---- Run 1: model A on all 3 jokes ----
        with _patches("model-A"):
            sys.argv = common_argv + ["--model-id", "fake-tag", "--model-slug", "model-A"]
            inference_main()

        rows = [json.loads(l) for l in out_path.open()]
        assert len(rows) == 3, f"expected 3 rows after run 1, got {len(rows)}"
        for r in rows:
            assert set(r.keys()) == {"joke_id", "model", "explanation"}, f"bad schema: {r}"
            assert r["model"] == "model-A"
            assert "<think>" not in r["explanation"], "reasoning trace was not stripped"
            assert r["explanation"].startswith("This is mock explanation"), \
                f"wrong content: {r['explanation'][:80]}"
        print("✓ Run 1: model-A wrote 3 rows, schema OK, <think> stripped")

        # ---- Run 2: model A again — should resume to nothing ----
        with _patches("model-A"):
            sys.argv = common_argv + ["--model-id", "fake-tag", "--model-slug", "model-A"]
            inference_main()
        rows = [json.loads(l) for l in out_path.open()]
        assert len(rows) == 3, f"resume failed — expected 3 rows still, got {len(rows)}"
        print("✓ Run 2: resume correctly skipped all 3 already-done jokes")

        # ---- Run 3: model B on the same jokes — should add 3 new rows ----
        with _patches("model-B"):
            sys.argv = common_argv + ["--model-id", "fake-tag-2", "--model-slug", "model-B"]
            inference_main()
        rows = [json.loads(l) for l in out_path.open()]
        assert len(rows) == 6, f"multi-model failed — expected 6 rows, got {len(rows)}"
        models_written = {r["model"] for r in rows}
        assert models_written == {"model-A", "model-B"}, f"unexpected models: {models_written}"
        per_model = {m: sum(1 for r in rows if r["model"] == m) for m in models_written}
        assert per_model == {"model-A": 3, "model-B": 3}, per_model
        print("✓ Run 3: model-B added 3 rows; total 6, both models coexist")

        # ---- Run 4: --no-resume + --limit 1 → appends 1 duplicate ----
        with _patches("model-B"):
            sys.argv = common_argv + ["--model-id", "fake-tag-2", "--model-slug", "model-B",
                                      "--no-resume", "--limit", "1"]
            inference_main()
        rows = [json.loads(l) for l in out_path.open()]
        assert len(rows) == 7, f"--no-resume + --limit 1 expected 7 rows, got {len(rows)}"
        print("✓ Run 4: --no-resume + --limit honored (7 rows; 1 duplicate as expected)")


if __name__ == "__main__":
    run_smoke()
    print("\nALL SMOKE TESTS PASSED")
