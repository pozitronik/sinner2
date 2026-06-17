"""Per-frame face geometry — the precomputed table that lets face-mapping skip
detection at runtime.

When face-mapping mode is on, the runtime must NOT detect faces every frame.
Instead it reads this table: for every covered frame, each face's bounding box +
5 keypoints, tagged with the catalog identity it matched. The swapper rebuilds a
face from that (no detection) and routes it to that identity's assigned source.

Geometry holds identity-tagged POSITIONS only — source assignments live in the
catalog (FaceMap), so re-assigning a source needs no re-precompute. Stored as a
compact NPZ sidecar keyed by the target path (parallel arrays of frame index,
identity index, bbox, kps + the distinct identity ids), mirroring the
corrupt-safe / atomic-write pattern of the JSON catalog store.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

_log = logging.getLogger(__name__)

_KPS = 5  # insightface 5-point keypoints per face


@dataclass(frozen=True)
class GeomFace:
    """A face's drawable + swappable geometry at one frame, tagged with the
    catalog identity it matched.

    ``embedding`` is the face's baked ArcFace embedding — when present the runtime
    routes by matching it against the LIVE catalog (so merges / reassignments /
    threshold changes take effect with no re-precompute, and detection-free
    routing is identical to live). Empty () = an older sidecar with no embeddings;
    the runtime then falls back to routing by ``identity_id``'s centroid."""

    identity_id: str
    bbox: tuple[float, float, float, float]
    kps: tuple[tuple[float, float], ...]  # 5 keypoints
    embedding: tuple[float, ...] = ()
    # In-plane roll (degrees) measured at bake time from the steadiest source
    # available (2dfan4 eye-line). None = not baked → the runtime measures roll
    # live. Lets rotation compensation use a good angle in detection-free mode,
    # where a rebuilt face has no pose estimate (POSE would fall back to kps).
    roll: float | None = None


@dataclass(frozen=True)
class FrameGeometry:
    """Per-frame geometry: ``faces[frame]`` is the tuple of mapped faces present
    at ``frame``. ``frame_count`` is the total covered (bookkeeping). ``refined``
    records whether the stored keypoints were 2dfan4-refined during the
    precompute (landmark-refine was on) — the runtime uses them as-is when its
    setting still matches, else re-refines the cheap landmarker per frame."""

    faces: dict[int, tuple[GeomFace, ...]] = field(default_factory=dict)
    frame_count: int = 0
    refined: bool = False
    # The frame resolution (width, height) the bboxes/kps were baked at — the
    # scan reads at native (processing_scale=1.0). The runtime rescales to the
    # frame it actually processes (which a processing_scale < 1 downsizes), so
    # detection-free swaps land correctly at any scale. None = old sidecar →
    # assume the geometry already matches the live frame (no rescale).
    bake_size: tuple[int, int] | None = None

    def faces_at(self, frame: int) -> tuple[GeomFace, ...]:
        return self.faces.get(int(frame), ())

    def is_empty(self) -> bool:
        return not self.faces

    def face_count(self) -> int:
        return sum(len(v) for v in self.faces.values())

    @staticmethod
    def empty() -> "FrameGeometry":
        return FrameGeometry()


def geometry_path(target: Path, root: Path) -> Path:
    """Sidecar path for ``target``'s per-frame geometry under ``root``, keyed by
    a hash of the target path (distinct from the catalog + progress sidecars)."""
    digest = hashlib.sha1(str(target).encode()).hexdigest()[:16]
    return root / f"{digest}.geometry.npz"


def save_geometry(path: Path, geometry: FrameGeometry) -> None:
    """Atomically persist the geometry as a compact NPZ (tmp + os.replace)."""
    frames: list[int] = []
    ident_idx: list[int] = []
    bboxes: list[tuple[float, float, float, float]] = []
    kps: list[tuple[tuple[float, float], ...]] = []
    embeddings: list[tuple[float, ...]] = []
    rolls: list[float] = []
    ids: list[str] = []
    id_to_idx: dict[str, int] = {}
    have_embeddings = True
    have_rolls = True
    for frame in sorted(geometry.faces):
        for gf in geometry.faces[frame]:
            idx = id_to_idx.get(gf.identity_id)
            if idx is None:
                idx = id_to_idx[gf.identity_id] = len(ids)
                ids.append(gf.identity_id)
            frames.append(int(frame))
            ident_idx.append(idx)
            bboxes.append(gf.bbox)
            kps.append(gf.kps)
            if gf.embedding:
                embeddings.append(gf.embedding)
            else:
                have_embeddings = False
            if gf.roll is not None:
                rolls.append(float(gf.roll))
            else:
                have_rolls = False
    arrays: dict[str, np.ndarray] = dict(
        frames=np.asarray(frames, dtype=np.int32),
        identity_idx=np.asarray(ident_idx, dtype=np.int32),
        bboxes=np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
        kps=np.asarray(kps, dtype=np.float32).reshape(-1, _KPS, 2),
        # Unicode (not object) array → no pickle needed on load.
        identity_ids=np.asarray(ids, dtype=np.str_),
        frame_count=np.asarray(geometry.frame_count, dtype=np.int64),
        refined=np.asarray(geometry.refined, dtype=bool),
    )
    # Store embeddings only when EVERY face carries one of a consistent length
    # (a ragged array would be object-dtype → can't save without pickle). float16
    # halves the (potentially large) array; cosine matching tolerates it easily.
    if have_embeddings and embeddings:
        emb_arr = np.asarray(embeddings, dtype=np.float16)
        if emb_arr.ndim == 2 and emb_arr.shape[0] == len(frames):
            arrays["embeddings"] = emb_arr
    # Baked roll (one float/face) — stored only when every face has one (the whole
    # scan either baked angles or didn't).
    if have_rolls and rolls and len(rolls) == len(frames):
        arrays["roll"] = np.asarray(rolls, dtype=np.float32)
    # The bake resolution, so the runtime can rescale to a processing-scaled frame.
    if geometry.bake_size is not None:
        arrays["bake_size"] = np.asarray(geometry.bake_size, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Write to the temp file object so np.savez doesn't append its own ".npz",
    # then atomically swap it into place.
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def load_geometry(path: Path) -> FrameGeometry | None:
    """Load the geometry, or None when absent / unreadable (never raises — a
    corrupt sidecar must not block loading a target)."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"]
            ident_idx = data["identity_idx"]
            bboxes = data["bboxes"]
            kps = data["kps"]
            ids = [str(x) for x in data["identity_ids"]]
            frame_count = int(data["frame_count"])
            # Older sidecars predate the flag → treat as un-refined (raw kps).
            refined = bool(data["refined"]) if "refined" in data else False
            # Older sidecars have no baked embeddings → route by id at runtime.
            embeddings = data["embeddings"] if "embeddings" in data else None
            rolls = data["roll"] if "roll" in data else None
            # Older sidecars have no bake size → assume native (no rescale).
            bake_size = (
                (int(data["bake_size"][0]), int(data["bake_size"][1]))
                if "bake_size" in data else None
            )
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
        _log.warning("face geometry unreadable (%s); ignoring", exc)
        return None
    faces: dict[int, list[GeomFace]] = {}
    for i in range(len(frames)):
        emb = (
            tuple(float(x) for x in embeddings[i])
            if embeddings is not None else ()
        )
        roll = float(rolls[i]) if rolls is not None else None
        gf = GeomFace(
            ids[int(ident_idx[i])],
            (
                float(bboxes[i][0]), float(bboxes[i][1]),
                float(bboxes[i][2]), float(bboxes[i][3]),
            ),
            tuple((float(p[0]), float(p[1])) for p in kps[i]),
            emb,
            roll,
        )
        faces.setdefault(int(frames[i]), []).append(gf)
    return FrameGeometry(
        {k: tuple(v) for k, v in faces.items()}, frame_count, refined, bake_size
    )


def delete_geometry(path: Path) -> bool:
    """Remove a geometry sidecar; False when it wasn't there."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
