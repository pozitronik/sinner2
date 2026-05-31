"""File-backed store for BatchTasks — one JSON file per task.

A flat folder of `<task_id>.json` files. The store has no in-memory
index — every operation re-reads the folder. Trades a tiny perf hit
(O(N) listings against a folder you're unlikely to fill with thousands
of tasks) for the property that external edits to a task file just
land naturally on the next list/load.

Resilience:
  - Corrupt task files (bad JSON, missing required fields) are silently
    skipped by list() so one bad file doesn't break the Batch tab.
  - Missing files on load() raise FileNotFoundError — callers (queue,
    GUI) should check exists() first or catch.
  - save() atomically replaces the target file (tmp + rename) so a
    crashed write doesn't leave a half-written corrupt file.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from sinner2.batch.task import BatchTask

# Windows: a transient AV/indexer handle on tmp or dest can fail os.replace
# with PermissionError. Retry a handful of times (~1.1s total worst case).
_REPLACE_RETRIES = 10


class BatchTaskStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, task_id: str) -> Path:
        # task_id is a uuid4 hex prefix — safe filename without escaping.
        # Defensive validation: reject ids that could escape the root
        # (path traversal). They shouldn't occur via _new_id() but
        # external callers could pass anything.
        if "/" in task_id or "\\" in task_id or ".." in task_id:
            raise ValueError(f"invalid task id: {task_id!r}")
        return self._root / f"{task_id}.json"

    def exists(self, task_id: str) -> bool:
        return self._path_for(task_id).is_file()

    def load(self, task_id: str) -> BatchTask:
        path = self._path_for(task_id)
        return BatchTask.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, task: BatchTask) -> None:
        path = self._path_for(task.id)
        # Unique tmp per write (mkstemp) so two writes can never share a
        # staging file, and a retry always re-stages cleanly. Same dir as
        # the target, so os.replace stays a same-filesystem atomic rename.
        fd, tmp_name = tempfile.mkstemp(
            dir=self._root, prefix=f"{task.id}.", suffix=".json.tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(task.model_dump_json(indent=2))
            # os.replace is atomic, but on Windows Defender's real-time
            # scan / the search indexer can briefly hold a handle on the
            # freshly-written tmp or the destination, making the rename
            # fail with PermissionError (WinError 5). It clears in a few
            # ms — retry with backoff before giving up.
            for attempt in range(_REPLACE_RETRIES):
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    if attempt == _REPLACE_RETRIES - 1:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            # Clean up the staging file if we raised before the replace
            # consumed it (replace removes tmp on success).
            try:
                tmp.unlink()
            except OSError:
                pass

    def delete(self, task_id: str) -> bool:
        path = self._path_for(task_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def list(self) -> list[BatchTask]:
        """Return all valid tasks in the store. Corrupt files are
        silently skipped — better to surface N-1 tasks than to error
        the whole Batch tab over one bad file."""
        out: list[BatchTask] = []
        try:
            entries = sorted(self._root.glob("*.json"))
        except OSError:
            return out
        for path in entries:
            try:
                out.append(
                    BatchTask.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return out
