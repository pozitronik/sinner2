import mimetypes
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator

from sinner2.config.base import SinnerBaseModel


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
        mime, _ = mimetypes.guess_type(str(self.path))
        if mime is None:
            raise ValueError(f"unknown media type for: {self.path}")
        if mime.startswith("image/"):
            return TargetKind.IMAGE
        if mime.startswith("video/"):
            return TargetKind.VIDEO
        raise ValueError(f"unsupported media type {mime!r} for: {self.path}")
