"""Persisted Batch Defaults — the template every new batch task is born from.

Batch processing is deliberately decoupled from the live preview: clicking
"Add to batch" carries over ONLY the source + target, and every other field
(the whole chain look, the per-processor execution profiles, output policy,
processing scale, …) comes from this editable template instead of being
inherited from the preview or hardcoded. The template is just a BatchTask with
sentinel paths — reusing BatchTask as the single schema means there is no
parallel field list to drift out of sync (cf. processor_snapshot.py).

Persisted as its own JSON file next to settings.json (app-level config, not a
queued task), so the BatchTaskStore folder stays a clean set of real tasks.

`mint_task()` is the one place that turns the template into a runnable task:
fresh id, the chosen source/target, auto output, and all runtime state reset.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from sinner2.batch.task import BatchTask, BatchTaskStatus
from sinner2.config.settings import settings_path

_log = logging.getLogger(__name__)

# Sentinel paths for the template. A template has no real source/target; these
# are placeholders that mint_task() always overwrites. "." is the only Path
# that is always valid and never mistaken for a real media file.
_SENTINEL = Path(".")


def batch_defaults_path() -> Path:
    """Resolve the batch-defaults file location.

    SINNER2_BATCH_DEFAULTS_PATH wins; otherwise it sits beside settings.json
    so the same SINNER2_SETTINGS_PATH redirect (used by tests + portable
    installs) isolates both files together.
    """
    env = os.environ.get("SINNER2_BATCH_DEFAULTS_PATH")
    if env:
        return Path(env)
    return settings_path().parent / "batch_defaults.json"


def default_template() -> BatchTask:
    """A fresh template carrying BatchTask's own field defaults + sentinel
    paths. Used when no defaults file exists yet (first run)."""
    return BatchTask(source_path=_SENTINEL, target_path=_SENTINEL)


def load_defaults(path: Path) -> BatchTask:
    """Load the template, or return `default_template()` when the file is
    absent or unreadable. Never raises — a corrupt defaults file must not
    block the GUI; the user just gets stock defaults and can re-save."""
    if not path.is_file():
        return default_template()
    try:
        return BatchTask.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        _log.warning("batch defaults unreadable (%s); using stock defaults", exc)
        return default_template()


def save_defaults(path: Path, template: BatchTask) -> None:
    """Atomically persist the template (tmp + os.replace), mirroring
    settings.save so a crash mid-write can't truncate the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(template.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)


def mint_task(template: BatchTask, source: Path, target: Path) -> BatchTask:
    """Build a runnable BatchTask from the template for a given source/target.

    Everything except identity, paths, and runtime state is copied verbatim
    from the template. A fresh id is generated (BatchTask's default_factory),
    output_path is reset to None (auto-derive), and all progress/timing markers
    are cleared so the new task starts clean regardless of what the template
    last held.
    """
    return template.model_copy(
        update={
            "id": BatchTask.model_fields["id"].get_default(
                call_default_factory=True
            ),
            "source_path": source,
            "target_path": target,
            "output_path": None,
            # Runtime state reset — a template should never seed progress.
            "status": BatchTaskStatus.PENDING,
            "last_completed_frame": -1,
            "total_frames": -1,
            "completed_stages": 0,
            "cache_fingerprint": "",
            "error_message": None,
            "started_at": None,
            "finished_at": None,
        }
    )
