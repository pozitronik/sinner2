from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator

from sinner2.config.base import SinnerBaseModel
from sinner2.config.media_extensions import is_image_ext, is_video_ext


class TargetKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class Target(SinnerBaseModel):
    path: Path = Field(description="Path to target media file (image or video)")

    @field_validator("path")
    @classmethod
    def _path_must_be_existing_file(cls, v: Path) -> Path:
        if not v.is_file():
            raise ValueError(f"target must be an existing file: {v}")
        return v

    @property
    def kind(self) -> TargetKind:
        # Extension-based (config.media_extensions) so it agrees with the
        # library and isn't at the mercy of the OS mimetypes registry.
        if is_video_ext(self.path):
            return TargetKind.VIDEO
        if is_image_ext(self.path):
            return TargetKind.IMAGE
        raise ValueError(f"unsupported media type for: {self.path}")
