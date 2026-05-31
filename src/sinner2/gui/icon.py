"""Application icon, built from the bundled PNGs.

A multi-size QIcon: Qt picks the closest source for each context (title bar /
taskbar / alt-tab / dialogs), so we ship both the small and large renders. No
.ico is needed at runtime — that only matters for a packaged .exe later.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon

_ASSETS = Path(__file__).resolve().parent / "assets"
# Smallest first; QIcon keeps every size it's given and scales from the nearest.
_ICON_FILES = ("sinner_32.png", "sinner_265.png")


def app_icon() -> QIcon:
    """The sinner2 window/taskbar icon. Returns an empty (but valid) QIcon if
    the asset files are somehow missing, so callers never need to guard."""
    icon = QIcon()
    for name in _ICON_FILES:
        path = _ASSETS / name
        if path.is_file():
            icon.addFile(str(path))
    return icon
