from pathlib import Path

from pydantic import Field, field_validator

from sinner2.config.base import SinnerBaseModel


class Source(SinnerBaseModel):
    path: Path = Field(description="Path to source face image")

    @field_validator("path")
    @classmethod
    def _path_must_be_existing_file(cls, v: Path) -> Path:
        if not v.is_file():
            raise ValueError(f"source must be an existing file: {v}")
        return v
