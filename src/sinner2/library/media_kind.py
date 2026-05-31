"""File-kind detection for library entries.

The library accepts images and videos; the rest are silently filtered out when
scanning folders. Detection is extension-based via the configurable sets in
config.media_extensions — the same source Target.kind uses, so library and
Target always agree (and `.wmv` & friends aren't dropped by an unreliable
mimetypes registry).
"""
from enum import Enum
from pathlib import Path

from sinner2.config.media_extensions import is_image_ext, is_media_ext, is_video_ext


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


def detect_kind(path: Path) -> MediaKind | None:
    """Return the kind, or None for non-media files (silently filtered)."""
    if is_video_ext(path):
        return MediaKind.VIDEO
    if is_image_ext(path):
        return MediaKind.IMAGE
    return None


def is_image(path: Path) -> bool:
    return is_image_ext(path)


def is_video(path: Path) -> bool:
    return is_video_ext(path)


def is_media(path: Path) -> bool:
    return is_media_ext(path)
