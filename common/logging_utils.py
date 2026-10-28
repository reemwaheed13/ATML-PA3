"""Append-only JSONL metric logging.

Handout asks to log SFT loss and learning rate against optimizer update (Task 1 Step 1),
and training loss / dev metrics throughout Task 2 (Step 1). JSONL is append-only so a resumed
run keeps extending the same file, and the logs are machine-readable for the plots/tables that
go under results/. One JSON object per line; no prose, no raw notebook output (handout p15).
"""

from __future__ import annotations

import json
import os
from typing import Any


class JsonlLogger:
    """Append-only JSONL writer. Each call writes one flat record (a dict) as one line."""

    def __init__(self, path: str, *, append: bool = True) -> None:
        """Open `path` for appending (append=True on resume so prior rows survive)."""
        self.path = os.fspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._f = open(self.path, "a" if append else "w", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        """Serialize one record as a JSON line and flush.

        Convention: include a monotonically increasing key (e.g. {"update": ...}) so rows can be
        de-duplicated if a resume replays a step.
        """
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f and not self._f.closed:
            self._f.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file back into a list of dicts (for building result tables)."""
    records: list[dict[str, Any]] = []
    with open(os.fspath(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
