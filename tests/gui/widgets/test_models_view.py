"""Tests for QModelsView — population, status/size cells, delete, and the
background download queue (with a fake download_model, no network)."""
from __future__ import annotations

import pytest

from sinner2.pipeline import model_cache
from sinner2.pipeline.models_catalog import MODEL_CATALOG
from sinner2.gui.widgets.models_view import _COL_SIZE, _COL_STATUS, QModelsView


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def view(qtbot, models_dir):
    v = QModelsView()
    qtbot.addWidget(v)
    yield v
    v.shutdown()


def _row_for(view, name):
    return view._row_for(name)  # noqa: SLF001


def _status(view, name):
    return view._model.item(_row_for(view, name), _COL_STATUS).text()  # noqa: SLF001


def _size(view, name):
    return view._model.item(_row_for(view, name), _COL_SIZE).text()  # noqa: SLF001


class TestPopulation:
    def test_lists_every_catalog_model(self, view):
        assert view._model.rowCount() == len(MODEL_CATALOG)  # noqa: SLF001

    def test_absent_model_shows_not_installed_and_approx_size(self, view):
        assert _status(view, "span_kendata_x4.onnx") == "Not installed"
        assert _size(view, "span_kendata_x4.onnx") == "~2 MB"

    def test_required_absent_model_flagged(self, view):
        assert _status(view, "inswapper_128.onnx") == "Required — missing"

    def test_present_model_shows_installed_and_disk_size(self, view, models_dir):
        (models_dir / "span_kendata_x4.onnx").write_bytes(b"x" * (3 * 1024 * 1024))
        view.refresh()
        assert _status(view, "span_kendata_x4.onnx") == "✓ Installed"
        assert _size(view, "span_kendata_x4.onnx") == "3 MB"

    def test_summary_counts_installed(self, view, models_dir):
        (models_dir / "span_kendata_x4.onnx").write_bytes(b"weights")
        view.refresh()
        assert "Installed 1/" in view._summary.text()  # noqa: SLF001


class TestSortingAndDetails:
    def test_size_column_sorts_numerically(self, view):
        from PySide6.QtCore import Qt

        from sinner2.gui.widgets.models_view import _COL_SIZE as C

        view._table.sortByColumn(C, Qt.SortOrder.AscendingOrder)  # noqa: SLF001
        # Read the sort key (bytes) down the column in view order — must be
        # ascending, proving it's not a lexical "816 MB" < "2 MB" sort.
        from sinner2.gui.widgets.models_view import _ROLE_SORT

        keys = [
            view._model.item(r, C).data(_ROLE_SORT)  # noqa: SLF001
            for r in range(view._model.rowCount())  # noqa: SLF001
        ]
        assert keys == sorted(keys)

    def test_table_is_sortable_and_resizable(self, view):
        from PySide6.QtWidgets import QHeaderView

        assert view._table.isSortingEnabled() is True  # noqa: SLF001
        hh = view._table.horizontalHeader()  # noqa: SLF001
        assert hh.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive

    def test_details_html_has_license_and_source(self, view):
        html = view._details_html("simswap_256.onnx")  # noqa: SLF001
        assert "non-commercial" in html.lower()
        assert model_cache.MODEL_SOURCES["simswap_256.onnx"] in html
        assert "<a href=" in html

    def test_details_html_shows_install_location_when_present(
        self, view, models_dir
    ):
        (models_dir / "span_kendata_x4.onnx").write_bytes(b"w")
        html = view._details_html("span_kendata_x4.onnx")  # noqa: SLF001
        assert "Installed" in html
        assert "span_kendata_x4.onnx" in html


class TestDelete:
    def test_delete_confirmed_removes_and_updates(self, view, models_dir, monkeypatch):
        f = models_dir / "span_kendata_x4.onnx"
        f.write_bytes(b"weights")
        view.refresh()
        monkeypatch.setattr(
            "sinner2.gui.widgets.models_view.confirm", lambda *a, **k: True
        )
        view._delete("span_kendata_x4.onnx")  # noqa: SLF001
        assert not f.exists()
        assert _status(view, "span_kendata_x4.onnx") == "Not installed"

    def test_delete_declined_keeps_file(self, view, models_dir, monkeypatch):
        f = models_dir / "span_kendata_x4.onnx"
        f.write_bytes(b"weights")
        monkeypatch.setattr(
            "sinner2.gui.widgets.models_view.confirm", lambda *a, **k: False
        )
        view._delete("span_kendata_x4.onnx")  # noqa: SLF001
        assert f.exists()

    def test_required_model_delete_is_not_suppressible(self, view, monkeypatch):
        # A REQUIRED model's delete confirm must pass suppressible=False so it
        # can never be silently auto-confirmed (that would brick the app).
        captured: dict = {}
        monkeypatch.setattr(
            "sinner2.gui.widgets.models_view.confirm",
            lambda *a, **k: captured.update(k) or False,
        )
        view._delete("inswapper_128.onnx")  # noqa: SLF001  (REQUIRED)
        assert captured.get("suppressible") is False


class TestDownloadQueue:
    @staticmethod
    def _fake_download(models_dir):
        def _dl(name, on_progress=None, should_cancel=None):
            if on_progress:
                on_progress(50, 100)
            (models_dir / name).write_bytes(b"weights")
        return _dl

    def test_download_installs_and_updates_row(
        self, view, models_dir, monkeypatch, qtbot
    ):
        monkeypatch.setattr(
            model_cache, "download_model", self._fake_download(models_dir)
        )
        view._enqueue(["span_kendata_x4.onnx"])  # noqa: SLF001
        qtbot.waitUntil(
            lambda: model_cache.model_present("span_kendata_x4.onnx")
            and view._current is None,  # noqa: SLF001
            timeout=3000,
        )
        assert _status(view, "span_kendata_x4.onnx") == "✓ Installed"

    def test_queue_runs_sequentially(self, view, models_dir, monkeypatch, qtbot):
        monkeypatch.setattr(
            model_cache, "download_model", self._fake_download(models_dir)
        )
        names = ["span_kendata_x4.onnx", "scrfd_2.5g.onnx", "yoloface_8n.onnx"]
        view._enqueue(names)  # noqa: SLF001
        qtbot.waitUntil(
            lambda: all(model_cache.model_present(n) for n in names)
            and view._current is None,  # noqa: SLF001
            timeout=4000,
        )
        for n in names:
            assert _status(view, n) == "✓ Installed"

    def test_download_failure_marks_row_failed(
        self, view, models_dir, monkeypatch, qtbot
    ):
        def _boom(name, on_progress=None, should_cancel=None):
            raise RuntimeError("network down")

        monkeypatch.setattr(model_cache, "download_model", _boom)
        view._enqueue(["span_kendata_x4.onnx"])  # noqa: SLF001
        qtbot.waitUntil(lambda: view._current is None, timeout=3000)  # noqa: SLF001
        assert _status(view, "span_kendata_x4.onnx") == "Failed"

    def test_download_all_missing_confirmed(
        self, view, models_dir, monkeypatch, qtbot
    ):
        # Mark everything present except one, so the bulk action queues just it.
        for name in MODEL_CATALOG:
            if name != "span_kendata_x4.onnx":
                (models_dir / name).write_bytes(b"w")
        view.refresh()
        monkeypatch.setattr(
            "sinner2.gui.widgets.models_view.confirm", lambda *a, **k: True
        )
        monkeypatch.setattr(
            model_cache, "download_model", self._fake_download(models_dir)
        )
        view._on_download_all_missing()  # noqa: SLF001
        qtbot.waitUntil(
            lambda: model_cache.model_present("span_kendata_x4.onnx"),
            timeout=3000,
        )
        assert _status(view, "span_kendata_x4.onnx") == "✓ Installed"


def _memory(view, name):
    from sinner2.gui.widgets.models_view import _COL_MEMORY
    return view._model.item(_row_for(view, name), _COL_MEMORY).text()  # noqa: SLF001


_KNOWN = "GFPGANv1.4.pth"  # a catalog filename used for the footprint matching


class TestMemoryColumn:
    def test_blank_until_a_model_is_measured(self, view):
        from sinner2.pipeline import memory_probe

        memory_probe.reset_footprints()
        view._refresh_memory_cells()  # noqa: SLF001
        assert _memory(view, _KNOWN) == ""

    def test_shows_measured_vram_footprint_by_filename(self, view):
        from sinner2.pipeline import memory_probe
        from sinner2.pipeline.memory_probe import ModelFootprint

        memory_probe.reset_footprints()
        memory_probe._footprints[_KNOWN] = ModelFootprint(  # noqa: SLF001
            _KNOWN, vram_bytes=int(0.30 * 1024 ** 3), ram_bytes=0, first_load=False
        )
        view._refresh_memory_cells()  # noqa: SLF001
        assert _memory(view, _KNOWN) == "+0.30 GB"
        memory_probe.reset_footprints()

    def test_first_load_marked_with_star(self, view):
        from sinner2.pipeline import memory_probe
        from sinner2.pipeline.memory_probe import ModelFootprint

        memory_probe.reset_footprints()
        memory_probe._footprints[_KNOWN] = ModelFootprint(  # noqa: SLF001
            _KNOWN, vram_bytes=1024 ** 3, ram_bytes=0, first_load=True
        )
        view._refresh_memory_cells()  # noqa: SLF001
        assert _memory(view, _KNOWN) == "+1.00 GB *"
        memory_probe.reset_footprints()


class TestFmtFootprint:
    def test_vram_delta(self):
        from sinner2.gui.widgets.models_view import _fmt_footprint
        from sinner2.pipeline.memory_probe import ModelFootprint

        text, sort_b, tip = _fmt_footprint(
            ModelFootprint("m", 2 * 1024 ** 3, 0, False)
        )
        assert text == "+2.00 GB" and sort_b == 2 * 1024 ** 3 and tip

    def test_ram_only_when_no_vram(self):
        from sinner2.gui.widgets.models_view import _fmt_footprint
        from sinner2.pipeline.memory_probe import ModelFootprint

        text, _s, _t = _fmt_footprint(
            ModelFootprint("m", None, 512 * 1024 ** 2, False)
        )
        assert text == "+0.50 GB RAM"

    def test_nothing_measured_is_blank(self):
        from sinner2.gui.widgets.models_view import _fmt_footprint
        from sinner2.pipeline.memory_probe import ModelFootprint

        assert _fmt_footprint(ModelFootprint("m", 0, 0, False)) == ("", 0, "")
