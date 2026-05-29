"""File-kind detection for library entries.

The library accepts images and videos; the rest are silently filtered out
when scanning folders. Detection uses mimetypes (path/extension based) —
same approach as Target.kind, so library and Target agree on what counts
as what.
"""
import mimetypes
from enum import Enum
from pathlib import Path


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


def detect_kind(path: Path) -> MediaKind | None:
    """Return the kind, or None for non-media files (silently filtered)."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        return None
    if mime.startswith("image/"):
        return MediaKind.IMAGE
    if mime.startswith("video/"):
        return MediaKind.VIDEO
    return None


def is_image(path: Path) -> bool:
    return detect_kind(path) is MediaKind.IMAGE


def is_video(path: Path) -> bool:
    return detect_kind(path) is MediaKind.VIDEO


def is_media(path: Path) -> bool:
    return detect_kind(path) is not None
