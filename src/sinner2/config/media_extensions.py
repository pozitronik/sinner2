"""Configurable image/video file extensions, used app-wide.

Replaces `mimetypes` for deciding what counts as an image or a video. mimetypes
is registry-driven on Windows and unreliable (e.g. `.wmv` often resolves to
None), which made the libraries silently ignore valid files. Extension lists
are deterministic and user-configurable via settings (no UI — just the file).

The active sets are module-level so a single `configure()` at startup applies
everywhere (library accept filters, file-dialog filters, Target.kind). With no
configuration, comprehensive sensible defaults apply.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Comprehensive defaults — bias toward accepting anything plausibly decodable;
# the reader fails loudly later if a specific file can't actually be opened.
DEFAULT_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    "png", "jpg", "jpeg", "jfif", "bmp", "tiff", "tif", "webp", "gif",
    "ppm", "pgm", "pbm", "tga",
})
DEFAULT_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    "mp4", "m4v", "mov", "avi", "mkv", "webm", "wmv", "flv", "f4v",
    "mpg", "mpeg", "mpe", "m2v", "ts", "m2ts", "mts", "3gp", "3g2",
    "ogv", "vob", "asf", "divx", "rm", "rmvb",
})

_image_exts: set[str] = set(DEFAULT_IMAGE_EXTENSIONS)
_video_exts: set[str] = set(DEFAULT_VIDEO_EXTENSIONS)


def _normalize(exts: Iterable[str]) -> set[str]:
    return {e.lower().lstrip(".") for e in exts if e}


def configure(
    image_exts: Iterable[str] | None = None,
    video_exts: Iterable[str] | None = None,
) -> None:
    """Set the active extension sets. None → reset that kind to its defaults
    (so calling this at startup is deterministic regardless of prior state)."""
    global _image_exts, _video_exts
    _image_exts = _normalize(image_exts) if image_exts else set(DEFAULT_IMAGE_EXTENSIONS)
    _video_exts = _normalize(video_exts) if video_exts else set(DEFAULT_VIDEO_EXTENSIONS)


def image_extensions() -> set[str]:
    return set(_image_exts)


def video_extensions() -> set[str]:
    return set(_video_exts)


def _ext(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def is_image_ext(path: Path) -> bool:
    return _ext(path) in _image_exts


def is_video_ext(path: Path) -> bool:
    return _ext(path) in _video_exts


def is_media_ext(path: Path) -> bool:
    e = _ext(path)
    return e in _image_exts or e in _video_exts


# ---- Qt file-dialog filter strings (kept in sync with the active sets) ----

def _glob(exts: set[str]) -> str:
    return " ".join(f"*.{e}" for e in sorted(exts))


def images_filter() -> str:
    return f"Images ({_glob(_image_exts)});;All files (*)"


def media_filter() -> str:
    both = _image_exts | _video_exts
    return (
        f"Media ({_glob(both)});;"
        f"Images ({_glob(_image_exts)});;"
        f"Videos ({_glob(_video_exts)});;"
        "All files (*)"
    )
