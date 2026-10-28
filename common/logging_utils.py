"""Append-only JSONL metric logging.

Handout asks to log SFT loss and learning rate against optimizer update (Task 1 Step 1),
and training loss / dev metrics throughout Task 2 (Step 1). JSONL is append-only so a resumed
run keeps extending the same file, and the logs are machine-readable for the plots/tables that
go under results/. One JSON object per line; no prose, no raw notebook output (handout p15).
"""

from __future__ import annotations

from typing import Any


class JsonlLogger:
    """Append-only JSONL writer. Each call writes one flat record (a dict) as one line."""

    def __init__(self, path: str, *, append: bool = True) -> None:
        """Open `path` for appending (append=True on resume so prior rows survive).

        TODO: open file handle; create parent dir via common.paths if needed.
        """
        raise NotImplementedError

    def log(self, record: dict[str, Any]) -> None:
        """Serialize one record as a JSON line and flush.

        Convention: include a monotonically increasing key (e.g. {"update": ...}) so rows can be
        de-duplicated if a resume replays a step.

        TODO: json.dumps(record) + "\n", write, flush.
        """
        raise NotImplementedError

    def close(self) -> None:
        """TODO: close the file handle."""
        raise NotImplementedError


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file back into a list of dicts (for building result tables).

    TODO: parse each line with json.loads.
    """
    raise NotImplementedError
