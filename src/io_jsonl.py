"""Tiny JSONL helpers shared by every script.

`read_jsonl_keys` enables --resume: read an existing output file and return
the set of keys already produced so the driver can skip them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_keys(path: str | Path, key_fn: Callable[[dict[str, Any]], Any]) -> set:
    """Return the set of keys already present in `path` for --resume."""
    return {key_fn(row) for row in read_jsonl(path)}
