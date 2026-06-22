"""Temporal stabilization for face-mapped swaps — smooth the precomputed
per-frame keypoint timeline so the swapped face stops swimming/jittering.

Independent per-frame detection makes a swapped face wobble frame-to-frame (the
dominant deepfake-flicker source). When face-mapping is active the runtime
already holds a COMPLETE, identity-tagged, stride-1 keypoint timeline
(``FrameGeometry``) precomputed up front — so stabilization is not an ordering
problem that fights the parallel swap workers, but an offline post-process on
that table: smooth each identity's keypoint track once, then the workers consume
the smoothed positions unchanged.

Three properties fall out of operating on the precomputed geometry:

* **Centered (non-causal)** — the whole timeline is known, so smoothing has no
  lag (unlike a causal EMA that can only look backwards).
* **Gap-aware** — an identity's track is smoothed only within contiguous runs of
  frames. A gap (a frame where that identity is absent — occluded, off-screen, a
  hard cut) is a discontinuity we never smooth across, so a cut can't smear.
* **Tracking-free** — every ``GeomFace`` already carries the ``identity_id`` it
  matched during precompute, so grouping a track is a dict lookup; there is no
  association step and therefore no track-swap risk.

Only the keypoints are smoothed (they drive the swap's affine alignment); bbox /
embedding / roll / identity are carried through untouched.
"""
from __future__ import annotations

import numpy as np

from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace

_KPS = 5  # insightface 5-point keypoints per face


def _gaussian_kernel(window: int) -> np.ndarray:
    """Normalized, centered Gaussian kernel of odd length ``window`` (>= 3).

    ``sigma = radius / 3`` so the window spans roughly ±3σ — the weights have
    decayed to near-zero at the edges, giving a smooth taper without a hard cut.
    """
    radius = window // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(radius / 3.0, 1e-3)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _smooth_columns(seg: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Edge-correct centered smoothing of each column of ``seg`` (shape ``(L, C)``).

    Uses normalized convolution — dividing by the kernel's own running sum — so
    the kernel is renormalized to its available support at the run's ends and
    endpoints are not dragged toward zero. ``kernel`` is sized so its length
    never exceeds ``L`` (np.convolve 'same' returns ``max(len(sig), len(ker))``).
    """
    norm = np.convolve(np.ones(seg.shape[0]), kernel, mode="same")
    out = np.empty_like(seg)
    for col in range(seg.shape[1]):
        out[:, col] = np.convolve(seg[:, col], kernel, mode="same") / norm
    return out


def _smooth_track(karr: np.ndarray, frames: list[int], window: int) -> np.ndarray:
    """Smooth one identity's keypoint track ``karr`` (shape ``(M, _KPS, 2)``).

    ``frames`` is the ascending frame index of each row. The track is split into
    maximal contiguous-frame runs (a jump > 1 starts a new run); each run is
    smoothed independently with a kernel shrunk to fit short runs. Runs shorter
    than 3 frames are left as-is (nothing to center a kernel on).
    """
    out = karr.copy()
    total = karr.shape[0]
    start = 0
    for i in range(1, total + 1):
        if i == total or frames[i] != frames[i - 1] + 1:
            length = i - start
            eff = min(window, length)
            eff -= 1 - (eff % 2)  # largest odd <= eff
            if eff >= 3:
                kernel = _gaussian_kernel(eff)
                seg = karr[start:i].reshape(length, -1)
                out[start:i] = _smooth_columns(seg, kernel).reshape(length, _KPS, 2)
            start = i
    return out


def smooth_geometry(
    geometry: FrameGeometry, *, window: int, strength: float
) -> FrameGeometry:
    """Return a copy of ``geometry`` with each identity's keypoint track
    temporally smoothed.

    ``window`` is the smoothing span in frames (forced odd; <= 1 disables).
    ``strength`` in [0, 1] linearly blends raw → smoothed (0 disables, 1 is fully
    smoothed). A no-op input (empty geometry, ``window <= 1`` or ``strength <=
    0``) returns the original object unchanged.
    """
    if geometry.is_empty() or window <= 1 or strength <= 0.0:
        return geometry
    if window % 2 == 0:
        window += 1
    strength = min(float(strength), 1.0)

    # Group every face into its identity's track (frame, position-in-frame, face).
    tracks: dict[str, list[tuple[int, int, GeomFace]]] = {}
    for frame, faces in geometry.faces.items():
        for pos, face in enumerate(faces):
            tracks.setdefault(face.identity_id, []).append((frame, pos, face))

    smoothed: dict[tuple[int, int], tuple[tuple[float, float], ...]] = {}
    for entries in tracks.values():
        entries.sort(key=lambda entry: entry[0])
        frames = [entry[0] for entry in entries]
        # An identity appearing twice in one frame would interleave two physical
        # faces into one track — leave such a track untouched rather than mix them.
        if len(set(frames)) != len(frames):
            continue
        karr = np.asarray([entry[2].kps for entry in entries], dtype=np.float64)
        out = _smooth_track(karr, frames, window)
        if strength < 1.0:
            out = karr + strength * (out - karr)
        for (frame, pos, _), kps in zip(entries, out, strict=True):
            smoothed[(frame, pos)] = tuple(
                (float(point[0]), float(point[1])) for point in kps
            )

    new_faces: dict[int, tuple[GeomFace, ...]] = {}
    for frame, faces in geometry.faces.items():
        rebuilt: list[GeomFace] = []
        for pos, face in enumerate(faces):
            kps = smoothed.get((frame, pos))
            if kps is None or kps == face.kps:
                rebuilt.append(face)
            else:
                rebuilt.append(
                    GeomFace(face.identity_id, face.bbox, kps, face.embedding, face.roll)
                )
        new_faces[frame] = tuple(rebuilt)

    return FrameGeometry(
        new_faces, geometry.frame_count, geometry.refined, geometry.bake_size
    )
