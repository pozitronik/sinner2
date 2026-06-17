"""Per-target FaceMap persistence — a sidecar JSON keyed by the target path.

A target's identity catalog is expensive to build (a full analysis scan) and
worth reusing across launches, so it's saved under the cache, keyed by a hash of
the target path. Mirrors the atomic-write / corrupt-safe pattern of settings +
batch defaults.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from sinner2.pipeline.face_map import FaceMap

_log = logging.getLogger(__name__)


def face_map_path(target: Path, root: Path) -> Path:
    """Sidecar path for ``target``'s catalog under ``root`` (the face-maps dir),
    keyed by a hash of the target path so a different target gets its own file."""
    digest = hashlib.sha1(str(target).encode()).hexdigest()[:16]
    return root / f"{digest}.json"


def load_face_map(path: Path) -> FaceMap | None:
    """Load a catalog, or None when absent / unreadable (never raises — a corrupt
    sidecar must not block loading a target)."""
    if not path.is_file():
        return None
    try:
        return FaceMap.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, ValueError, KeyError) as exc:
        _log.warning("face map unreadable (%s); ignoring", exc)
        return None


def save_face_map(path: Path, face_map: FaceMap) -> None:
    """Atomically persist a catalog (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(face_map.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def delete_face_map(path: Path) -> bool:
    """Remove a catalog sidecar; False when it wasn't there."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


# ---- "Use this map for playback" preference (per target) ----

def use_map_path(target: Path, root: Path) -> Path:
    """Marker sidecar recording that the user chose to ROUTE playback through
    this target's map (independent of the editor panel being open)."""
    digest = hashlib.sha1(str(target).encode()).hexdigest()[:16]
    return root / f"{digest}.usemap"


def save_use_map(path: Path, on: bool) -> None:
    """Persist the per-target 'use the map for playback' preference: the marker
    exists when on, is removed when off."""
    if on:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1", encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def load_use_map(path: Path) -> bool:
    """Whether playback routing through the map was last left ON for this target."""
    return path.is_file()


# ---- Scan progress (resume) ----

def progress_path(target: Path, root: Path) -> Path:
    """Sidecar holding how far the last scan got (separate from the catalog so
    the catalog stays a clean value object)."""
    digest = hashlib.sha1(str(target).encode()).hexdigest()[:16]
    return root / f"{digest}.progress.json"


def load_progress(path: Path) -> dict | None:
    """``{signature, scanned, total}`` of the last scan, or None when absent /
    unreadable. Never raises."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def save_progress(path: Path, signature: str, scanned: int, total: int) -> None:
    """Atomically record scan progress for resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({"signature": signature, "scanned": scanned, "total": total}),
        encoding="utf-8",
    )
    os.replace(tmp, path)
