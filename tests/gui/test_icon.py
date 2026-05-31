"""The bundled app icon loads and carries multiple sizes."""
from __future__ import annotations

from sinner2.gui.icon import _ASSETS, _ICON_FILES, app_icon


def test_asset_files_are_bundled():
    # The PNGs must live inside the package (not just repo-root images/) so they
    # ship with an install.
    for name in _ICON_FILES:
        assert (_ASSETS / name).is_file(), f"missing bundled icon asset: {name}"


def test_app_icon_loads_with_sizes(qtbot):
    icon = app_icon()
    assert not icon.isNull()
    assert icon.availableSizes(), "icon should expose at least one concrete size"
