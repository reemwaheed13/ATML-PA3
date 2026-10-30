"""Append-only JSONL metric logging, crash-safe against a Colab/Drive disconnect.

Handout asks to log SFT loss and learning rate against optimizer update (Task 1 Step 1), and
training loss / dev metrics throughout Task 2 (Step 1). JSONL is append-only so a resumed run
keeps extending the same file, and the logs are machine-readable for the plots/tables that go
under results/. One JSON object per line; no prose, no raw notebook output (handout p15).

Durability on Colab
-------------------
Google Drive's FUSE mount only *uploads* a file to Drive when it is closed — a file left open
and merely flushed is not persisted, so a runtime that dies mid-run loses everything written
since the file was opened. (This cost us the first half of three train.jsonl logs.)

So we never keep the Drive file open. Each record is written to a LOCAL working copy (on Colab's
local disk, which `fsync` actually persists), and that file is copied to its real Drive path
*atomically* (copy to a temp file in the same directory, then `os.replace`) every `flush_every`
records and on `close()`. A reader on Drive therefore sees a complete prefix of the log — never a
half-written line — and at most the last `flush_every` records are at risk if the runtime dies
between syncs.

On resume (`append=True`) the local copy is seeded from the existing Drive file first, so earlier
rows are preserved and the appended run extends them.

The public API (`JsonlLogger(path, append=...)`, `.log`, `.close`, context manager, `read_jsonl`)
is unchanged; `flush_every` / `local_path` are optional knobs with sensible defaults.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Optional

DEFAULT_FLUSH_EVERY = 25  # sync the Drive copy at least this often (records)


class JsonlLogger:
    """Append-only JSONL writer that persists to Drive atomically every N records + on close."""

    def __init__(self, path: str, *, append: bool = True,
                 flush_every: int = DEFAULT_FLUSH_EVERY, local_path: Optional[str] = None) -> None:
        """Open `path` (the Drive target) for append-logging via a local working copy.

        Args:
            append: on resume, seed the local copy from the existing Drive file so prior rows
                survive and new rows extend them. append=False starts a fresh log.
            flush_every: copy the local file to Drive after this many records (also on close()).
            local_path: override the local working-copy path (defaults to Colab-local disk).
        """
        self.path = os.fspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self._since_sync = 0

        self._local_path = os.fspath(local_path) if local_path else self._default_local_path(self.path)
        os.makedirs(os.path.dirname(self._local_path) or ".", exist_ok=True)

        if append and os.path.exists(self.path):
            # Resume: start the local copy from what Drive already has, then append.
            shutil.copyfile(self.path, self._local_path)
            self._f = open(self._local_path, "a", encoding="utf-8")
        else:
            # Fresh run (or append with nothing on Drive yet): start the local copy empty.
            self._f = open(self._local_path, "w", encoding="utf-8")

    @staticmethod
    def _default_local_path(drive_path: str) -> str:
        """A stable local working-copy path for `drive_path` (same across a run and its resumes)."""
        base = "/content" if os.path.isdir("/content") else tempfile.gettempdir()
        digest = hashlib.sha1(os.path.abspath(drive_path).encode("utf-8")).hexdigest()[:8]
        return os.path.join(base, "atml_jsonl_logs", f"{digest}_{os.path.basename(drive_path)}")

    def log(self, record: dict[str, Any]) -> None:
        """Serialize one record as a JSON line locally; sync to Drive every `flush_every` records.

        Convention: include a monotonically increasing key (e.g. {"update": ...}) so rows can be
        de-duplicated if a resume replays a step.
        """
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        self._since_sync += 1
        if self._since_sync >= self.flush_every:
            self._sync()

    def _sync(self) -> None:
        """Atomically copy the local working copy onto the Drive path (temp file + os.replace)."""
        # Make sure buffered writes have reached local disk before we copy them to Drive.
        if self._f is not None and not self._f.closed:
            self._f.flush()
            try:
                os.fsync(self._f.fileno())
            except (OSError, AttributeError):
                pass

        directory = os.path.dirname(self.path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".jsonl.", suffix=".tmp", dir=directory)
        os.close(fd)
        try:
            shutil.copyfile(self._local_path, tmp_path)
            os.replace(tmp_path, self.path)  # atomic; a reader never sees a partial line
        except BaseException:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        self._since_sync = 0

    def close(self) -> None:
        """Flush, close the local file, and do a final atomic sync to Drive."""
        if self._f is not None and not self._f.closed:
            self._f.close()
        self._sync()

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
