"""Image-encoding strategies for the persistent frame cache.

Separates the "what extension / what cv2 params" decision from the store
implementations so callers can swap formats without changing store code.
The discriminator string from each writer also feeds the cache hash so
switching formats lands frames in a fresh directory (old PNG cache from
a previous run survives untouched if the user switches back).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2

from sinner2.io.cv2_unicode import imread_unicode, imwrite_unicode
from sinner2.types import Frame


class ImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"


@runtime_checkable
class ImageWriter(Protocol):
    """Reads/writes a single frame as a single image file."""

    @property
    def extension(self) -> str: ...

    @property
    def cache_key(self) -> str:
        """A short, stable identifier for this writer's encode settings.

        Goes into the persistent-cache directory hash so different formats
        (or different quality settings) don't share a directory and so
        switching back-and-forth doesn't trigger reprocessing.
        """
        ...

    def write(self, path: Path, frame: Frame) -> None: ...

    def read(self, path: Path) -> Frame | None: ...


class PNGImageWriter:
    extension = "png"

    def __init__(self, compression: int = 1) -> None:
        if not 0 <= compression <= 9:
            raise ValueError(f"PNG compression must be 0-9; got {compression}")
        self._compression = compression

    @property
    def compression(self) -> int:
        return self._compression

    @property
    def cache_key(self) -> str:
        return f"png-c{self._compression}"

    def write(self, path: Path, frame: Frame) -> None:
        if not imwrite_unicode(
            path, frame, [cv2.IMWRITE_PNG_COMPRESSION, self._compression]
        ):
            raise OSError(f"PNG write failed: {path}")

    def read(self, path: Path) -> Frame | None:
        if not path.is_file():
            return None
        return imread_unicode(path)


class JPEGImageWriter:
    extension = "jpg"

    def __init__(self, quality: int = 95) -> None:
        if not 1 <= quality <= 100:
            raise ValueError(f"JPEG quality must be 1-100; got {quality}")
        self._quality = quality

    @property
    def quality(self) -> int:
        return self._quality

    @property
    def cache_key(self) -> str:
        return f"jpg-q{self._quality}"

    def write(self, path: Path, frame: Frame) -> None:
        if not imwrite_unicode(
            path, frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        ):
            raise OSError(f"JPEG write failed: {path}")

    def read(self, path: Path) -> Frame | None:
        if not path.is_file():
            return None
        return imread_unicode(path)


def build_image_writer(image_format: ImageFormat, quality: int) -> ImageWriter:
    """Factory used by callers that have settings values but not writer instances.

    `quality` is interpreted per-format: 0-9 for PNG (compression level),
    1-100 for JPEG (encode quality). Clamping is the writers' job; this
    factory just dispatches.
    """
    if image_format is ImageFormat.PNG:
        return PNGImageWriter(compression=quality)
    if image_format is ImageFormat.JPEG:
        return JPEGImageWriter(quality=quality)
    raise ValueError(f"unsupported image format: {image_format}")
