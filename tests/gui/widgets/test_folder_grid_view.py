"""Tests for the folder-mirror view (grid grouped into collapsible folders)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSortFilterProxyModel
from PySide6.QtGui import QStandardItem, QStandardItemModel

from sinner2.gui.widgets.folder_grid_view import QFolderGridView, _section_label
from sinner2.library.library_model import ROLE_PATH


def _source_with(paths):
    """A minimal source proxy carrying ROLE_PATH — all the folder view reads."""
    model = QStandardItemModel()
    for p in paths:
        item = QStandardItem(p.name)
        item.setData(str(p), ROLE_PATH)
        model.appendRow(item)
    proxy = QSortFilterProxyModel()
    proxy.setSourceModel(model)
    return proxy, model


class TestSectionLabel:
    def test_root_relative_nested(self, tmp_path):
        root = tmp_path / "lib"
        assert _section_label(root / "footage" / "b-roll", [root]) == str(
            Path("footage") / "b-roll"
        )

    def test_root_itself_uses_name(self, tmp_path):
        root = tmp_path / "lib"
        assert _section_label(root, [root]) == "lib"

    def test_no_matching_root_uses_name(self, tmp_path):
        assert _section_label(tmp_path / "other", []) == "other"


class TestFolderGridView:
    def test_rebuild_groups_by_folder(self, qtbot, tmp_path):
        a, b = tmp_path / "footage", tmp_path / "sources"
        proxy, _ = _source_with([a / "x.mp4", a / "y.jpg", b / "hero.png"])
        view = QFolderGridView(proxy, display_dim=96)
        qtbot.addWidget(view)
        view.set_roots([tmp_path])
        view.rebuild()
        sections = view._sections  # noqa: SLF001
        assert [s._label for s in sections] == ["footage", "sources"]  # noqa: SLF001
        assert sections[0]._proxy.rowCount() == 2  # noqa: SLF001 — footage's files
        assert sections[1]._proxy.rowCount() == 1  # noqa: SLF001

    def test_tile_click_emits_path_selected(self, qtbot, tmp_path):
        f = tmp_path / "footage"
        proxy, _ = _source_with([f / "clip.mp4"])
        view = QFolderGridView(proxy, display_dim=96)
        qtbot.addWidget(view)
        view.set_roots([tmp_path])
        view.rebuild()
        got: list = []
        view.pathSelected.connect(got.append)
        section = view._sections[0]  # noqa: SLF001
        idx = section.grid().model().index(0, 0)
        section._emit(idx)  # noqa: SLF001 — simulate a tile activation
        assert got == [f / "clip.mp4"]

    def test_section_folds(self, qtbot, tmp_path):
        f = tmp_path / "footage"
        proxy, _ = _source_with([f / "clip.mp4"])
        view = QFolderGridView(proxy, display_dim=96)
        qtbot.addWidget(view)
        view.set_roots([tmp_path])
        view.rebuild()
        section = view._sections[0]  # noqa: SLF001
        assert section._grid.isHidden() is False  # noqa: SLF001 — expanded default
        section._header.setChecked(False)  # noqa: SLF001 — collapse
        assert section._grid.isHidden() is True  # noqa: SLF001

    def test_inactive_view_does_not_auto_rebuild(self, qtbot, tmp_path):
        proxy, model = _source_with([tmp_path / "a" / "x.mp4"])
        view = QFolderGridView(proxy, display_dim=96)
        qtbot.addWidget(view)
        # Not active → a model change schedules nothing.
        model.appendRow(QStandardItem("y"))
        assert not view._rebuild_timer.isActive()  # noqa: SLF001
        view.set_active(True)
        model.appendRow(QStandardItem("z"))
        assert view._rebuild_timer.isActive()  # noqa: SLF001
