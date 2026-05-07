"""Generate joke explanations from an LLM (Member 2's inference step).

For each joke in `data/jokes.jsonl`, prompt the configured model with the
paper's verbatim prompt (Appendix A.1) and append an Explanation row to
`data/explanations.jsonl`.

One model per run — re-run with different `--model-slug` values to add more
models. The downstream `judge.py` and `analyze.py` already group rows by the
`model` field, so writing to a single shared file is intentional.

Resume: skips (joke_id, model_slug) keys already present in --output.
Errors logged to outputs/inference.errors.log; the run continues.

Examples:
    # R1-Distill-Qwen-7B via Ollama (after `ollama pull deepseek-r1:7b`)
    python src/inference.py --backend ollama \\
        --model-id deepseek-r1:7b \\
        --model-slug r1-distill-qwen-7b

    # Gemini 1.5 Flash via Google AI Studio's free tier
    export GEMINI_API_KEY=...
    python src/inference.py --backend gemini \\
        --model-id gemini-1.5-flash \\
        --model-slug gemini-1.5-flash \\
        --sleep 4.0          # 15 req/min free-tier cap

    # Gemini 1.5 Pro — slow free tier (50 req/day)
    python src/inference.py --backend gemini \\
        --model-id gemini-1.5-pro \\
        --model-slug gemini-1.5-pro \\
        --sleep 31.0
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from inference_backend import get_inference_backend
from io_jsonl import append_jsonl, read_jsonl, read_jsonl_keys
from schema import Explanation


log = logging.getLogger("inference")


# Verbatim from paper Appendix A.1, "Prompt" subsection.
PROMPT_TEMPLATE = (
    "Explain the following joke (presented in square brackets) "
    "in approximately 100 words.\n\n[{joke}]\n\n"
)

# R1 distillations emit a <think>...</think> block before the final answer.
# The paper's R1 explanations don't include the trace — only the final answer.
# We strip the block to match that convention.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def _check_ollama_model(model_id: str,
                       ollama_url: str = "http://localhost:11434") -> tuple[bool, str]:
    """Return (available, message). Used as a pre-flight before the main loop.

    Catches the most common silent failure: the inference model wasn't pulled
    on the Ollama daemon. Without this check, the script burns the full retry
    budget on every single joke (3 retries × 2s backoff = 6s wasted per call,
    × 600 jokes = 1 hour of pure error before the user notices).
    """
    import requests
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as ex:
        return False, f"Ollama daemon unreachable at {ollama_url}: {ex!r}"

    pulled = [m["name"] for m in resp.json().get("models", [])]
    if model_id in pulled:
        return True, f"OK — {model_id} is pulled"

    # Quantization variants: user pulled `deepseek-r1:8b`, daemon may have
    # labeled it `deepseek-r1:8b-llama-distill-q4_K_M`. Accept that as a hit.
    similar = [p for p in pulled if p.startswith(model_id + "-")
                                  or p == f"{model_id}-latest"]
    if similar:
        return True, f"OK — {model_id} matches pulled tag {similar[0]}"

    return False, (
        f"Model {model_id!r} is NOT pulled on the Ollama daemon.\n"
        f"  Currently pulled: {pulled}\n"
        f"  Run in a WSL terminal: ollama pull {model_id}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate joke explanations from an LLM")
    ap.add_argument("--jokes", default="data/jokes.jsonl", type=Path)
    ap.add_argument("--output", default="data/explanations.jsonl", type=Path)
    ap.add_argument("--errors", default="outputs/inference.errors.log", type=Path)
    ap.add_argument("--backend", required=True,
                    choices=["ollama", "gemini", "openrouter", "together"])
    ap.add_argument("--model-id", required=True,
                    help="Backend-specific model identifier (Ollama tag like "
                         "'deepseek-r1:7b' or Gemini model name like "
                         "'gemini-1.5-flash')")
    ap.add_argument("--model-slug", required=True,
                    help="Canonical name written into the explanation row "
                         "(e.g. 'r1-distill-qwen-7b'). Downstream judge.py "
                         "and analyze.py group by this.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 keeps runs reproducible; paper uses model defaults")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="paper requested ~100 words; 400 leaves headroom for "
                         "verbose models (esp. R1's <think> trace)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between requests (set for rate-limited APIs: "
                         "Gemini Flash free tier ~4s, Pro ~31s)")
    ap.add_argument("--limit", type=int, default=None, help="cap total calls (debug)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip jokes already explained for this model_slug")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--keep-reasoning", action="store_true",
                    help="keep <think>...</think> blocks in R1 outputs (default: strip)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)

    jokes = list(read_jsonl(args.jokes))
    if not jokes:
        log.error("no jokes in %s — run preprocess.py first", args.jokes)
        sys.exit(1)

    done: set[tuple[str, str]] = set()
    if args.resume and args.output.exists():
        done = read_jsonl_keys(
            args.output,
            key_fn=lambda r: (r["joke_id"], r["model"]),
        )
        n_for_this_model = sum(1 for k in done if k[1] == args.model_slug)
        log.info("resume: %d explanations already in %s (%d for this model)",
                 len(done), args.output, n_for_this_model)

    todo = [j for j in jokes if (j["id"], args.model_slug) not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    if not todo:
        log.info("nothing to do — all %d jokes already explained for model=%s",
                 len(jokes), args.model_slug)
        return

    backend = get_inference_backend(args.backend, model=args.model_id)

    # Pre-flight: for Ollama, fail fast if the model isn't pulled. Otherwise
    # the script silently retries 3× per joke and burns hours on a bad config.
    if args.backend == "ollama":
        ok, msg = _check_ollama_model(args.model_id)
        if not ok:
            log.error("Pre-flight check failed: %s", msg)
            sys.exit(2)

    log.info(
        "backend=%s model_id=%s slug=%s — %d jokes to do (temp=%g, sleep=%gs)",
        backend.name, args.model_id, args.model_slug,
        len(todo), args.temperature, args.sleep,
    )

    n_ok, n_err = 0, defaultdict(int)
    with args.errors.open("a") as err_f:
        for i, joke in enumerate(tqdm(todo, desc=f"infer[{args.model_slug}]")):
            joke_id = joke["id"]
            prompt = PROMPT_TEMPLATE.format(joke=joke["joke"])
            messages = [{"role": "user", "content": prompt}]

            response: str | None = None
            for attempt in range(args.retries):
                try:
                    response = backend.complete(
                        messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    break
                except Exception as ex:
                    if attempt == args.retries - 1:
                        err_f.write(f"backend ({joke_id}, {args.model_slug}): {ex!r}\n")
                        n_err["backend"] += 1
                    else:
                        time.sleep(args.retry_sleep * (attempt + 1))

            if response is None:
                if args.sleep > 0 and i < len(todo) - 1:
                    time.sleep(args.sleep)
                continue

            if not args.keep_reasoning:
                response = strip_reasoning(response)
            response = response.strip()

            if not response:
                err_f.write(f"empty ({joke_id}, {args.model_slug})\n")
                n_err["empty"] += 1
            else:
                explanation = Explanation(
                    joke_id=joke_id,
                    model=args.model_slug,
                    explanation=response,
                )
                append_jsonl(args.output, dc.asdict(explanation))
                n_ok += 1

            if args.sleep > 0 and i < len(todo) - 1:
                time.sleep(args.sleep)

    log.info("done: ok=%d errors=%s", n_ok, dict(n_err))


if __name__ == "__main__":
    main()
