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
from pathlib import Path

from sinner2.batch.task import BatchTask


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
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(task.model_dump_json(indent=2), encoding="utf-8")
        # os.replace is atomic on POSIX and on Windows when source +
        # dest are on the same filesystem (always true here — same
        # store root). Beats two-step delete+rename which could lose
        # the file on crash.
        os.replace(tmp, path)

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
