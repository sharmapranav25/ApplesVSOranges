# Task 1 — Dataset preprocessing — ✅ DONE

**Owner:** Pranav (Member 1)
**Last verified:** 2026-04-30

## Goal
Download/preprocess the source dataset (`new_jokes.csv`, 600 rows × 111 cols, derived from the paper's repo) into the canonical `data/jokes.jsonl` schema used by the rest of the pipeline.

## What exists
| Path | Notes |
|---|---|
| `new_jokes.csv` | Source — 600 rows, copied from paper repo |
| `src/preprocess.py` | CSV → `data/jokes.jsonl` driver, supports `--limit`, `--resume` |
| `src/schema.py` | `Joke` dataclass + `JOKE_TYPE_MAP` + `JOKE_TYPES` |
| `data/jokes.jsonl` | 600 lines, balanced 150 per type |
| `tests/test_preprocess.py` | 3 tests — all passing |

## Verification
- `wc -l data/jokes.jsonl` → **600**
- Type counts: `{homographic: 150, heterographic: 150, non_topical: 150, topical: 150}` ✅
- `pytest tests/test_preprocess.py` → **3 passed**
  - `test_six_hundred_jokes_balanced`
  - `test_ids_unique_and_well_formed`
  - `test_no_empty_required_fields`

## Schema (one row)
```json
{"id": "homographic_000", "type": "homographic",
 "joke": "...", "reference_explanation": "...",
 "source_index": null}
```

## Conventions captured (don't re-derive)
- **Joke type slugs:** CSV `hom`/`het`/`non-topical`/`topical` → `homographic`/`heterographic`/`non_topical`/`topical` (`JOKE_TYPE_MAP` in `schema.py`).
- **IDs:** `{type}_{NNN}` zero-padded 0..149, assigned in CSV row order *within each type bucket* by `assign_joke_ids()`. Stable across re-runs.
- **`source_index` is null for hom/het/non-topical** — only `topical` carries the original SemEval/r-Jokes index. Don't try to join on it.
- Output is sorted: type order, then numeric suffix → grouped & monotone.

## Nothing left to do here
Re-run only if the CSV is replaced.

```bash
python src/preprocess.py --input new_jokes.csv --output data/jokes.jsonl
```
