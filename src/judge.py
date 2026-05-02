"""Task 3: Qwen2.5-7B-Instruct LLM-as-a-judge.

(Deviation from paper: paper uses Qwen2.5-72B-Instruct. We use 7B for cost/feasibility.
See status_task3_judge.md for rationale and implications.)

For each (joke, model_explanation, criterion in {accuracy, completeness}),
call Qwen with the verbatim Appendix A.6 prompt at temperature 0.1, parse
the first integer 0-5 from the response, and append a row to
outputs/ratings_judge.jsonl.

Resumable: skips (joke_id, model, criterion) keys already present in the output.
Errors (HTTP failure, unparseable response, etc.) are logged to
outputs/judge.errors.log; the run continues.
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from io_jsonl import append_jsonl, read_jsonl, read_jsonl_keys
from judge_backend import get_backend
from parser import parse_score
from rubric import load_judge_template, load_rubric
from schema import CRITERIA, JUDGE_MODEL_NAME, Rating

log = logging.getLogger("judge")


def build_prompt(template: str, criterion: str, scoring_criteria: str,
                 joke: str, reference: str, model_explanation: str) -> str:
    return template.format(
        criteria=criterion,
        scoring_criteria=scoring_criteria.rstrip("\n"),
        joke=joke,
        reference_explanation=reference,
        model_explanation=model_explanation,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen2.5-7B LLM-as-a-judge")
    ap.add_argument("--jokes", default="data/jokes.jsonl", type=Path)
    ap.add_argument("--explanations", default="data/explanations.jsonl", type=Path)
    ap.add_argument("--output", default="outputs/ratings_judge.jsonl", type=Path)
    ap.add_argument("--errors", default="outputs/judge.errors.log", type=Path)
    ap.add_argument("--backend", choices=["openrouter", "together", "ollama", "local"],
                    default="openrouter",
                    help="ollama = free local via `ollama serve` (recommended on Mac)")
    ap.add_argument("--temperature", type=float, default=0.1, help="paper: 0.1")
    ap.add_argument("--max-tokens", type=int, default=16,
                    help="response is just a digit; 16 leaves headroom for stray whitespace")
    ap.add_argument("--limit", type=int, default=None, help="cap total calls (debug)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip rows already in --output (default on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)

    template = load_judge_template()
    rubrics = {c: load_rubric(c) for c in CRITERIA}

    jokes = {j["id"]: j for j in read_jsonl(args.jokes)}
    if not jokes:
        log.error("no jokes in %s — run preprocess.py first", args.jokes)
        sys.exit(1)

    explanations = list(read_jsonl(args.explanations))
    if not explanations:
        log.error("no explanations in %s — wait for teammate or run "
                  "make_explanations_from_csv.py", args.explanations)
        sys.exit(1)

    # Build the full work list, then subtract anything already done.
    todo: list[tuple[str, str, str]] = []  # (joke_id, model, criterion)
    for e in explanations:
        for c in CRITERIA:
            todo.append((e["joke_id"], e["model"], c))

    done: set[tuple[str, str, str]] = set()
    if args.resume and args.output.exists():
        done = read_jsonl_keys(
            args.output,
            key_fn=lambda r: (r["joke_id"], r["model"], r["criterion"]),
        )
        log.info("resume: %d ratings already present", len(done))

    todo = [k for k in todo if k not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    if not todo:
        log.info("nothing to do")
        return

    backend = get_backend(args.backend)
    log.info("backend=%s, %d calls remaining (temp=%g)", backend.name, len(todo), args.temperature)

    explanations_by_key = {(e["joke_id"], e["model"]): e["explanation"] for e in explanations}

    n_ok, n_err = 0, defaultdict(int)
    with args.errors.open("a") as err_f:
        for joke_id, model, criterion in tqdm(todo, desc="judge"):
            joke = jokes.get(joke_id)
            if joke is None:
                err_f.write(f"unknown joke_id={joke_id!r}\n")
                n_err["unknown_joke"] += 1
                continue
            explanation = explanations_by_key.get((joke_id, model))
            if explanation is None:
                err_f.write(f"missing explanation for ({joke_id}, {model})\n")
                n_err["missing_expl"] += 1
                continue

            prompt = build_prompt(
                template,
                criterion=criterion,
                scoring_criteria=rubrics[criterion],
                joke=joke["joke"],
                reference=joke["reference_explanation"],
                model_explanation=explanation,
            )
            messages = [{"role": "user", "content": prompt}]

            response: str | None = None
            for attempt in range(args.retries):
                try:
                    response = backend.complete(messages, temperature=args.temperature,
                                                max_tokens=args.max_tokens)
                    break
                except Exception as ex:
                    if attempt == args.retries - 1:
                        err_f.write(f"backend error ({joke_id}, {model}, {criterion}): {ex!r}\n")
                        n_err["backend"] += 1
                    else:
                        time.sleep(args.retry_sleep * (attempt + 1))

            if response is None:
                continue

            score = parse_score(response)
            if score is None:
                err_f.write(
                    f"unparseable ({joke_id}, {model}, {criterion}): {response!r}\n"
                )
                n_err["parse"] += 1
                continue

            rating = Rating(
                joke_id=joke_id,
                model=model,
                criterion=criterion,
                score=score,
                annotator=JUDGE_MODEL_NAME,
            )
            append_jsonl(args.output, dc.asdict(rating))
            n_ok += 1

    log.info("done: ok=%d errors=%s", n_ok, dict(n_err))


if __name__ == "__main__":
    main()
