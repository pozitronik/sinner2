"""Save / restore the live working state as a portable ``.sinner`` project file.

A project captures the source + target media (ABSOLUTE paths), the timeline
section selection, and the full processor/chain configuration (the same flat
field map that settings persistence consumes). Opening one re-drives the normal
load path — pickers + transport + the settings restore — so a project can't
diverge from how the app loads anything else.

The file is plain JSON with a version tag for forward-compat. Enum-valued
processor fields are stored as their stable string tokens; the GUI coerces them
back through the Settings model (which already round-trips them) on open.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

PROJECT_VERSION = 1
PROJECT_SUFFIX = ".sinner"


def _jsonable(value: Any) -> Any:
    """str-Enums serialize as their token; everything else passes through."""
    return value.value if isinstance(value, Enum) else value


@dataclass(frozen=True)
class Project:
    """A saved working session: media + selection + chain config."""

    source_path: Path | None
    target_path: Path | None
    sections: list[list[int]] | None  # SectionSet.to_pairs(); None = whole video
    processor: dict[str, Any] = field(default_factory=dict)  # snapshot kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROJECT_VERSION,
            "source_path": str(self.source_path) if self.source_path else None,
            "target_path": str(self.target_path) if self.target_path else None,
            "sections": self.sections,
            "processor": {k: _jsonable(v) for k, v in self.processor.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        src = data.get("source_path")
        tgt = data.get("target_path")
        return cls(
            source_path=Path(src) if src else None,
            target_path=Path(tgt) if tgt else None,
            sections=data.get("sections"),
            processor=dict(data.get("processor") or {}),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Project":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
