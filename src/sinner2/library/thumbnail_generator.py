"""Background generation of thumbnails for library entries.

Submits work to a ThreadPoolExecutor; each task loads the source file,
produces a resized JPEG, persists it via ThumbnailCache, and invokes
the on_ready callback with (path, jpeg_path, meta). Callbacks run on
worker threads — the GUI integration layer is responsible for marshalling
back to the main thread (e.g. via QMetaObject.invokeMethod or a Qt
signal).

Image loads use PIL/Pillow (already pulled in by GFPGAN); video first-
frame extraction uses cv2 (already a hard dependency of the pipeline).
Both error paths log to the on_ready callback as `error: str` instead
of raising — one bad file shouldn't poison the rest of the library.
"""
from __future__ import annotations

import io
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
from PIL import Image

from sinner2.library.media_kind import MediaKind, detect_kind
from sinner2.library.thumbnail_cache import ThumbnailCache, ThumbnailMeta

_DEFAULT_QUALITY = 80
_DEFAULT_VIDEO_FRAME_FRACTION = 0.5


@dataclass(frozen=True)
class ThumbnailResult:
    """Successful generation: paths to the cached JPEG + meta."""

    source: Path
    jpeg_path: Path
    meta: ThumbnailMeta


@dataclass(frozen=True)
class ThumbnailError:
    """Failed generation: the source path + a human-readable reason."""

    source: Path
    reason: str


ThumbnailOutcome = Union[ThumbnailResult, ThumbnailError]
OnReady = Callable[[ThumbnailOutcome], None]


class ThumbnailGenerator:
    """Produces cached thumbnails off the main thread."""

    def __init__(
        self,
        cache: ThumbnailCache,
        *,
        thumb_dim: int = 384,
        max_workers: int | None = None,
        video_frame_fraction: float = _DEFAULT_VIDEO_FRAME_FRACTION,
    ) -> None:
        # thumb_dim is the EXTRACTION size — what gets cached on disk.
        # The view scales the pixmap down (or, at most, to this size)
        # on every paint, so set this generously: 384 covers display
        # sizes up to 384px without upscale blur, costs ~30-50KB per
        # JPEG on disk. Don't set it tiny "to save space" — the view's
        # display_dim is independent and any future resize past the
        # cached size would force a full re-extraction.
        #
        # max_workers=None auto-scales to os.cpu_count() so big libraries
        # land in the background pool quickly. PIL.thumbnail/save and
        # cv2.imdecode/VideoCapture all release the GIL during their
        # C-level work, so threads (not processes) parallelise across
        # cores effectively — sinner1's pattern, kept here.
        if not 0.0 <= video_frame_fraction <= 1.0:
            raise ValueError(
                f"video_frame_fraction must be in [0,1]; got {video_frame_fraction}"
            )
        self._cache = cache
        self._thumb_dim = thumb_dim
        self._video_frame_fraction = video_frame_fraction
        workers = max_workers if max_workers is not None else (os.cpu_count() or 4)
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="sinner2-thumb"
        )
        # In-flight jobs: path -> coalesced callbacks. We don't double-submit a
        # path already being generated, but a path requested by BOTH libraries
        # (used as source AND target) while in flight must deliver the outcome
        # to ALL callers — dropping the second left its tile blank forever.
        self._inflight_lock = threading.Lock()
        self._inflight: dict[Path, list[OnReady]] = {}

    @property
    def thumb_dim(self) -> int:
        return self._thumb_dim

    def submit(self, source: Path, on_ready: OnReady) -> Future | None:
        """Submit a generation job. Returns None if already in flight,
        or if the executor has been shut down (post-shutdown submits
        are common during app close: a pending folder-scan batch signal
        fires after closeEvent has already torn the pool down).

        on_ready is called from a worker thread with either ThumbnailResult
        or ThumbnailError. The GUI layer must marshal back to the main
        thread before touching widgets.
        """
        with self._inflight_lock:
            waiters = self._inflight.get(source)
            if waiters is not None:
                waiters.append(on_ready)  # coalesce onto the in-flight job
                return None
            self._inflight[source] = [on_ready]
        try:
            return self._executor.submit(self._run, source)
        except RuntimeError:
            # The executor was shut down between our inflight check and
            # this submit. The caller (model.add_path → view.add_paths)
            # treats None as a no-op, so don't leak the inflight entry.
            with self._inflight_lock:
                self._inflight.pop(source, None)
            return None

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Stop accepting new work and (optionally) drop pending tasks.

        cancel_futures=True is the right call on app exit: the
        ThreadPoolExecutor's threads otherwise process every queued
        thumbnail before letting Python's atexit join them, which
        keeps the process alive for minutes when a large folder was
        just added (the scan submitted thousands of jobs).

        Also signals the cache so an in-flight put() landing during
        shutdown skips its (potentially slow) prune step — that was
        the root cause of an observed multi-minute hang on exit when
        the cache had thousands of entries."""
        self._cache.shutdown()
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _run(self, source: Path) -> None:
        try:
            outcome = self._produce(source)
        except Exception as e:
            outcome = ThumbnailError(source=source, reason=f"unexpected: {e}")
        finally:
            with self._inflight_lock:
                callbacks = self._inflight.pop(source, [])
        for on_ready in callbacks:
            try:
                on_ready(outcome)
            except Exception:
                # The callback is GUI-side; swallowing here keeps a buggy slot
                # from killing the worker thread (and thus all subsequent jobs).
                pass

    def _produce(self, source: Path) -> ThumbnailOutcome:
        if not source.is_file():
            return ThumbnailError(source=source, reason="file not found")
        kind = detect_kind(source)
        if kind is None:
            return ThumbnailError(source=source, reason="unsupported media type")

        cached = self._cache.get(source, self._thumb_dim)
        if cached is not None:
            jpeg_path, meta = cached
            return ThumbnailResult(source=source, jpeg_path=jpeg_path, meta=meta)

        if kind is MediaKind.IMAGE:
            jpeg_bytes, meta = self._render_image(source)
        else:
            jpeg_bytes, meta = self._render_video(source)
        jpeg_path = self._cache.put(source, self._thumb_dim, jpeg_bytes, meta)
        return ThumbnailResult(source=source, jpeg_path=jpeg_path, meta=meta)

    def _render_image(self, source: Path) -> tuple[bytes, ThumbnailMeta]:
        with Image.open(source) as img:
            # Decode now while the file handle is still open; thumbnail()
            # operates on the loaded data so we can close cleanly.
            img.load()
            full_w, full_h = img.size
            thumb = img.copy()
            thumb.thumbnail(
                (self._thumb_dim, self._thumb_dim), Image.Resampling.LANCZOS
            )
            if thumb.mode not in ("RGB", "L"):
                thumb = thumb.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=_DEFAULT_QUALITY)
        caption = f"{source.name} [{full_w}x{full_h}]"
        return buf.getvalue(), ThumbnailMeta(caption=caption, pixel_count=full_w * full_h)

    def _render_video(self, source: Path) -> tuple[bytes, ThumbnailMeta]:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"cv2 could not open video: {source}")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            target_index = max(0, int(frame_count * self._video_frame_fraction))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                # Fallback to first frame if seek failed (some containers
                # don't support precise seeking, or fraction was past end).
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f"could not read any frame from: {source}")
        finally:
            cap.release()

        # cv2 returns BGR; PIL expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((self._thumb_dim, self._thumb_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_DEFAULT_QUALITY)
        caption = f"{source.name} [{width}x{height}]"
        return buf.getvalue(), ThumbnailMeta(caption=caption, pixel_count=width * height)
