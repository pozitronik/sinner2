"""Tests for QLibraryView.

Verify: add_paths / set_paths / clear API, accept-filter, folder
ingestion, click-activates-pathSelected, drag-drop URL handling,
sort/filter UI plumbing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent

from sinner2.gui.widgets.library_view import QLibraryView
from sinner2.library.media_kind import is_image
from sinner2.library.thumbnail_cache import ThumbnailCache
from sinner2.library.thumbnail_generator import ThumbnailGenerator


@pytest.fixture
def generator(tmp_path: Path):
    cache = ThumbnailCache(tmp_path / "cache")
    g = ThumbnailGenerator(cache, thumb_dim=32, max_workers=2)
    yield g
    g.shutdown(wait=True)


class TestSort:
    def test_default_sort_getters(self, view):
        assert view.sort_field() == "name"
        assert view.sort_order() == "asc"

    def test_set_sort_restores_field_and_order(self, view):
        view.set_sort("date", "desc")
        assert view.sort_field() == "date"
        assert view.sort_order() == "desc"

    def test_set_sort_is_silent(self, view):
        fired: list = []
        view.sortChanged.connect(lambda: fired.append(1))
        view.set_sort("size", "asc")
        assert fired == []  # restore must not re-emit (would re-persist)

    def test_toggle_direction_emits_sort_changed(self, view, qtbot):
        with qtbot.waitSignal(view.sortChanged, timeout=1000):
            view._toggle_sort_direction()  # noqa: SLF001


@pytest.fixture
def view(qtbot, generator):
    v = QLibraryView(generator)
    qtbot.addWidget(v)
    yield v
    # Cancel + join any folder scans the test left running. Without
    # this, QThread destruction at widget cleanup hangs ("QThread:
    # Destroyed while thread is still running") and the next test's
    # collection blocks.
    v.shutdown()


def _make_image(path: Path) -> Path:
    arr = np.full((10, 10, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


class TestAddAndPaths:
    def test_initially_empty(self, view):
        assert view.paths() == []

    def test_add_paths_filters_via_accept(self, qtbot, generator, tmp_path):
        # Sources accept images only — a video URL should be silently
        # rejected.
        v = QLibraryView(generator, accept=is_image)
        qtbot.addWidget(v)
        img = _make_image(tmp_path / "a.png")
        vid = tmp_path / "b.mp4"
        vid.write_bytes(b"")  # fake video — passes mime check via extension
        added = v.add_paths([img, vid])
        assert added == 1
        assert v.paths() == [img]

    def test_add_paths_emits_paths_changed(self, view, qtbot, tmp_path):
        img = _make_image(tmp_path / "a.png")
        with qtbot.waitSignal(view.pathsChanged, timeout=1000) as blocker:
            view.add_paths([img])
        assert blocker.args == [[img]]

    def test_set_paths_replaces_silently(self, view, qtbot, tmp_path):
        # set_paths is for settings restore — must NOT emit pathsChanged
        # (would round-trip through main_window and re-save).
        img = _make_image(tmp_path / "a.png")
        with qtbot.assertNotEmitted(view.pathsChanged, wait=100):
            view.set_paths([img])
        assert view.paths() == [img]

    def test_remove_path(self, view, qtbot, tmp_path):
        img = _make_image(tmp_path / "a.png")
        view.add_paths([img])
        with qtbot.waitSignal(view.pathsChanged, timeout=1000) as blocker:
            assert view.remove_path(img) is True
        assert blocker.args == [[]]

    def test_clear(self, view, qtbot, tmp_path):
        for n in ("a.png", "b.png"):
            view.add_paths([_make_image(tmp_path / n)])
        with qtbot.waitSignal(view.pathsChanged, timeout=1000):
            view.clear()
        assert view.paths() == []

    def test_clear_when_empty_is_silent(self, view, qtbot):
        with qtbot.assertNotEmitted(view.pathsChanged, wait=100):
            view.clear()


class TestFolderIngestion:
    def test_ingest_folder_scans_recursively_in_background(
        self, view, tmp_path, qtbot
    ):
        # Mixed folder: media files in nested subdirs + a noise file.
        # Scan runs on a QThread; wait for paths to arrive via the
        # pathsChanged signal rather than asserting on a return value
        # (ingest_files_and_folders is now fire-and-forget).
        (tmp_path / "sub" / "deep").mkdir(parents=True)
        _make_image(tmp_path / "a.png")
        _make_image(tmp_path / "sub" / "b.png")
        _make_image(tmp_path / "sub" / "deep" / "c.png")
        (tmp_path / "notes.txt").write_text("hi")
        view.ingest_files_and_folders([tmp_path])
        # Three image files should land via streamed scan batches.
        # Use _wait_until-style polling on view.paths() since one folder
        # may produce multiple batch emissions.
        import time as _time
        end = _time.monotonic() + 5.0
        while len(view.paths()) < 3 and _time.monotonic() < end:
            qtbot.wait(20)
        assert len(view.paths()) == 3

    def test_ingest_skips_unreadable_folder(self, view, tmp_path, qtbot):
        # Non-existent path — should not raise. Nothing lands.
        view.ingest_files_and_folders([tmp_path / "nope"])
        qtbot.wait(100)
        assert view.paths() == []

    def test_ingest_immediate_files_appear_synchronously(
        self, view, tmp_path
    ):
        # Bare files in the input list are added synchronously (no
        # folder walk needed). pathsChanged fires before
        # ingest_files_and_folders returns for that part.
        a = _make_image(tmp_path / "a.png")
        b = _make_image(tmp_path / "b.png")
        view.ingest_files_and_folders([a, b])
        assert view.paths() == [a, b]


class TestActivation:
    def test_click_emits_path_selected(self, view, qtbot, tmp_path):
        img = _make_image(tmp_path / "a.png")
        view.add_paths([img])
        proxy_index = view._list.model().index(0, 0)  # noqa: SLF001
        with qtbot.waitSignal(view.pathSelected, timeout=1000) as blocker:
            view._list.clicked.emit(proxy_index)  # noqa: SLF001
        assert blocker.args == [img]


class TestDragDrop:
    def test_drop_event_ingests_files(self, view, qtbot, tmp_path):
        # Build a synthetic QDropEvent payload with file URLs.
        img = _make_image(tmp_path / "a.png")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(img))])
        evt = QDropEvent(
            QPointF(0, 0),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        with qtbot.waitSignal(view.pathsChanged, timeout=1000):
            view.dropEvent(evt)
        assert view.paths() == [img]


class TestDisplayDim:
    """Display dim is decoupled from extraction: the generator caches at
    a constant size and the view scales the cached pixmap at paint time,
    so the user can resize tiles live without re-extracting."""

    @pytest.fixture
    def big_generator(self, tmp_path):
        # Bigger extract size so display-dim tests have a meaningful
        # range (the default generator fixture uses 32 to keep
        # thumbnail-render tests fast, but that's below the 64px
        # display minimum).
        cache = ThumbnailCache(tmp_path / "cache")
        g = ThumbnailGenerator(cache, thumb_dim=256, max_workers=2)
        yield g
        g.shutdown(wait=True)

    @pytest.fixture
    def view(self, qtbot, big_generator):
        v = QLibraryView(big_generator, display_dim=128)
        qtbot.addWidget(v)
        return v

    def test_initial_display_dim_propagates_to_view(self, qtbot, big_generator):
        v = QLibraryView(big_generator, display_dim=96)
        qtbot.addWidget(v)
        # Snap-to-step + min check: 96 is already on the grid (multiple
        # of 32 and >= MIN 64), so it's preserved verbatim.
        assert v.display_dim() == 96
        from PySide6.QtCore import QSize

        assert v._list.iconSize() == QSize(96, 96)  # noqa: SLF001

    def test_display_dim_clamped_to_generator_thumb_dim(
        self, qtbot, big_generator
    ):
        # Asking for a display larger than what we extracted means
        # there's nothing to scale UP to without blurring — cap at the
        # generator's thumb_dim.
        oversize = big_generator.thumb_dim + 100
        v = QLibraryView(big_generator, display_dim=oversize)
        qtbot.addWidget(v)
        assert v.display_dim() == big_generator.thumb_dim

    def test_display_dim_clamped_to_minimum(self, qtbot, big_generator):
        v = QLibraryView(big_generator, display_dim=8)
        qtbot.addWidget(v)
        # The min is 64 (anything smaller is unusable).
        assert v.display_dim() == 64

    def test_set_display_dim_updates_view_and_emits(
        self, view, qtbot
    ):
        from PySide6.QtCore import QSize

        starting = view.display_dim()
        new_dim = starting - 32 if starting > 64 else starting + 32
        with qtbot.waitSignal(view.displayDimChanged, timeout=1000) as blocker:
            view.set_display_dim(new_dim)
        assert blocker.args == [new_dim]
        assert view.display_dim() == new_dim
        assert view._list.iconSize() == QSize(new_dim, new_dim)  # noqa: SLF001

    def test_set_display_dim_no_op_at_same_value(self, view, qtbot):
        current = view.display_dim()
        with qtbot.assertNotEmitted(view.displayDimChanged, wait=100):
            view.set_display_dim(current)

    def test_set_display_dim_snaps_to_step(self, view):
        # 100 isn't on the 32-step grid; expect snap to 96.
        view.set_display_dim(100)
        assert view.display_dim() % 32 == 0

    def test_minus_button_shrinks_one_step(self, view, qtbot):
        starting = view.display_dim()
        view._shrink_button.click()  # noqa: SLF001
        if starting > 64:
            assert view.display_dim() == starting - 32
        else:
            # Already at min — no change.
            assert view.display_dim() == 64

    def test_plus_button_grows_one_step(self, view, qtbot):
        starting = view.display_dim()
        view._grow_button.click()  # noqa: SLF001
        upper = view._generator.thumb_dim  # noqa: SLF001
        if starting + 32 <= upper:
            assert view.display_dim() == starting + 32
        else:
            assert view.display_dim() == upper

    def test_ctrl_wheel_resizes(self, view, qtbot):
        # Send a Wheel event with the Ctrl modifier directly to the
        # event filter. Cheaper than driving the actual input system,
        # and verifies the wiring we care about.
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent

        starting = view.display_dim()
        # angleDelta() positive y = wheel up = grow.
        evt_grow = QWheelEvent(
            QPointF(10, 10),  # position in widget
            QPointF(10, 10),  # global position
            QPoint(0, 0),  # pixelDelta
            QPoint(0, 120),  # angleDelta — one notch up
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        view.eventFilter(view._list.viewport(), evt_grow)  # noqa: SLF001
        upper = view._generator.thumb_dim  # noqa: SLF001
        expected_after_grow = min(starting + 32, upper)
        assert view.display_dim() == expected_after_grow
        # Shrink: negative angleDelta y.
        evt_shrink = QWheelEvent(
            QPointF(10, 10),
            QPointF(10, 10),
            QPoint(0, 0),
            QPoint(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        view.eventFilter(view._list.viewport(), evt_shrink)  # noqa: SLF001
        assert view.display_dim() == max(expected_after_grow - 32, 64)

    def test_wheel_without_ctrl_does_not_resize(self, view):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent

        starting = view.display_dim()
        evt = QWheelEvent(
            QPointF(10, 10),
            QPointF(10, 10),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,  # no ctrl — must pass through
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        view.eventFilter(view._list.viewport(), evt)  # noqa: SLF001
        # Display dim unchanged; the wheel just scrolls the grid.
        assert view.display_dim() == starting


class TestShutdownCancelsScans:
    """Regression: a large network folder added before close kept the
    process alive while the folder scan completed. Shutdown must cancel
    all live scans + drop pending thumbnail jobs so the pool exits
    promptly."""

    def test_shutdown_cancels_live_scan_worker(
        self, qtbot, generator, tmp_path
    ):
        # Build a folder with many files so the scan stays busy for a
        # measurable window. Then start a scan and call shutdown
        # immediately — the worker's cancel flag should fire before it
        # finishes enumerating.
        import time as _time

        folder = tmp_path / "lib"
        folder.mkdir()
        for i in range(200):
            (folder / f"f{i:03d}.png").write_bytes(b"x")
        view = QLibraryView(generator)
        qtbot.addWidget(view)
        view.ingest_files_and_folders([folder])
        # Verify a scan is actually live before shutdown — otherwise
        # the test would pass trivially.
        assert view._scan_jobs, "scan didn't start"  # noqa: SLF001
        t0 = _time.monotonic()
        view.shutdown()
        elapsed = _time.monotonic() - t0
        # 5s is the per-thread wait cap inside shutdown. Anything close
        # to that means the cancel didn't fire and we ran to natural
        # completion — bug. Under normal conditions cancel + return
        # should land within ~0.3s.
        assert elapsed < 2.0, (
            f"shutdown took {elapsed:.2f}s — scan didn't honor cancel"
        )

    def test_shutdown_clears_scan_registry(self, qtbot, generator, tmp_path):
        folder = tmp_path / "lib"
        folder.mkdir()
        for i in range(5):
            (folder / f"f{i}.png").write_bytes(b"x")
        view = QLibraryView(generator)
        qtbot.addWidget(view)
        view.ingest_files_and_folders([folder])
        view.shutdown()
        # After shutdown, the registry of (thread, worker) pairs must
        # be empty — leaked entries would leak threads and keep the
        # process alive.
        assert view._scan_jobs == []  # noqa: SLF001


class TestRoots:
    """Roots = user-added entries (file OR folder, as added). The
    persistence unit. A folder root stays one entry no matter how
    many files it expands to."""

    def test_initial_roots_empty(self, view):
        assert view.roots() == []

    def test_file_ingest_records_file_root(self, view, qtbot, tmp_path):
        f = _make_image(tmp_path / "a.png")
        with qtbot.waitSignal(view.rootsChanged, timeout=1000) as blocker:
            view.ingest_files_and_folders([f])
        assert blocker.args == [[f]]
        assert view.roots() == [f]

    def test_folder_ingest_records_folder_as_single_root(
        self, view, qtbot, tmp_path
    ):
        # Folder with three images expands to three grid items but
        # ONE root entry — the folder itself.
        folder = tmp_path / "lib"
        folder.mkdir()
        for n in ("a.png", "b.png", "c.png"):
            _make_image(folder / n)
        with qtbot.waitSignal(view.rootsChanged, timeout=1000) as blocker:
            view.ingest_files_and_folders([folder])
        assert blocker.args == [[folder]]
        assert view.roots() == [folder]
        # Wait for the background scan to add the three children.
        import time as _time
        end = _time.monotonic() + 5.0
        while len(view.paths()) < 3 and _time.monotonic() < end:
            qtbot.wait(20)
        assert len(view.paths()) == 3

    def test_duplicate_root_ingest_is_silent(self, view, qtbot, tmp_path):
        f = _make_image(tmp_path / "a.png")
        view.ingest_files_and_folders([f])
        with qtbot.assertNotEmitted(view.rootsChanged, wait=100):
            view.ingest_files_and_folders([f])

    def test_set_roots_restores_silently(self, view, qtbot, tmp_path):
        # Restore mustn't round-trip back into persist — rootsChanged
        # is suppressed by set_roots.
        f = _make_image(tmp_path / "a.png")
        with qtbot.assertNotEmitted(view.rootsChanged, wait=100):
            view.set_roots([f])
        assert view.roots() == [f]

    def test_set_roots_drops_missing_file_roots(self, view, tmp_path):
        # Persisted state may reference files since deleted. Silently
        # filter them out instead of polluting the next save.
        present = _make_image(tmp_path / "alive.png")
        missing = tmp_path / "deleted.png"
        view.set_roots([present, missing])
        assert view.roots() == [present]

    def test_set_roots_keeps_folder_roots_even_if_empty(self, view, tmp_path):
        # An empty folder (or one whose contents are all rejected) is
        # still a valid root — the user might fill it later.
        folder = tmp_path / "empty"
        folder.mkdir()
        view.set_roots([folder])
        assert view.roots() == [folder]

    def test_remove_file_root_drops_from_roots(self, view, qtbot, tmp_path):
        f = _make_image(tmp_path / "a.png")
        view.ingest_files_and_folders([f])
        with qtbot.waitSignal(view.rootsChanged, timeout=1000) as blocker:
            view.remove_path(f)
        assert blocker.args == [[]]
        assert view.roots() == []

    def test_remove_folder_child_does_not_touch_roots(
        self, view, qtbot, tmp_path
    ):
        # Removing a single file that came from a folder expansion
        # does NOT modify roots — the folder root stays, and the file
        # will reappear on the next restart's re-scan. Surface bug if
        # this changes (would make per-file removal "stick" but in a
        # confusing way that diverges from the persisted state).
        folder = tmp_path / "lib"
        folder.mkdir()
        children = [_make_image(folder / n) for n in ("a.png", "b.png")]
        view.ingest_files_and_folders([folder])
        import time as _time
        end = _time.monotonic() + 5.0
        while len(view.paths()) < 2 and _time.monotonic() < end:
            qtbot.wait(20)
        # Now remove one of the children — must NOT emit rootsChanged.
        with qtbot.assertNotEmitted(view.rootsChanged, wait=200):
            view.remove_path(children[0])
        assert view.roots() == [folder]

    def test_clear_resets_both_paths_and_roots(self, view, qtbot, tmp_path):
        f = _make_image(tmp_path / "a.png")
        view.ingest_files_and_folders([f])
        with qtbot.waitSignal(view.rootsChanged, timeout=1000) as blocker:
            view.clear()
        assert blocker.args == [[]]
        assert view.roots() == []
        assert view.paths() == []


class TestClearDuringScan:
    """Regressions for the two bugs reported when clearing a library
    mid-scan: (1) confirm dialog showed root count instead of visible
    grid count, (2) clear didn't cancel the scan so late batches kept
    repopulating the wiped model."""

    def test_clear_cancels_live_scan(self, view, qtbot, tmp_path):
        # Stage a folder with enough files that the scan stays busy
        # for a measurable window.
        import time as _time

        folder = tmp_path / "lib"
        folder.mkdir()
        for i in range(100):
            (folder / f"f{i:03d}.png").write_bytes(b"x")
        view.ingest_files_and_folders([folder])
        # Verify a scan is live.
        assert view._scan_jobs, "scan didn't start"  # noqa: SLF001
        # Call clear() while the scan is in flight. Cancel epoch bumps,
        # worker's cancel flag fires — late batches are discarded by
        # _on_scan_batch_if_current.
        view.clear()
        # Pump events briefly to let any in-flight queued batches arrive.
        # With the epoch check, they MUST be discarded — view.paths()
        # stays empty.
        end = _time.monotonic() + 1.0
        while _time.monotonic() < end:
            qtbot.wait(50)
        assert view.paths() == [], (
            f"late scan batches repopulated the wiped library: {view.paths()}"
        )

    def test_late_batch_discarded_after_clear(self, view, qtbot, tmp_path):
        # Direct unit test of the epoch filter. Build a fake "sender"
        # carrying the stale start_epoch and stash it via Qt's
        # QObject.sender() machinery. The QThread is real but never
        # started — keeps shutdown's wait() call cheap at teardown.
        from PySide6.QtCore import QThread

        from sinner2.gui.widgets.library_view import _FolderScanWorker

        view._scan_cancel_epoch = 5  # noqa: SLF001
        stale_thread = QThread()
        stale_worker = _FolderScanWorker([], view._accept)  # noqa: SLF001
        stale_worker.start_epoch = 2  # type: ignore[attr-defined]
        view._scan_jobs.append((stale_thread, stale_worker))  # noqa: SLF001
        stale_worker.batch.connect(view._on_scan_batch)  # noqa: SLF001
        stale_worker.batch.emit([_make_image(tmp_path / "stale.png")])
        assert view.paths() == []

    def test_current_batch_lands_normally(self, view, qtbot, tmp_path):
        # Sanity: a batch tagged with the current epoch still works.
        from PySide6.QtCore import QThread

        from sinner2.gui.widgets.library_view import _FolderScanWorker

        view._scan_cancel_epoch = 5  # noqa: SLF001
        fresh_thread = QThread()
        fresh_worker = _FolderScanWorker([], view._accept)  # noqa: SLF001
        fresh_worker.start_epoch = 5  # type: ignore[attr-defined]
        view._scan_jobs.append((fresh_thread, fresh_worker))  # noqa: SLF001
        fresh_worker.batch.connect(view._on_scan_batch)  # noqa: SLF001
        p = _make_image(tmp_path / "fresh.png")
        fresh_worker.batch.emit([p])
        assert view.paths() == [p]


class TestClearButton:
    def test_clear_button_present(self, view):
        # Surfaced in the controls bar so the user can find it without
        # a menu bar. Sinner1 parity.
        assert view._clear_button.text() == "Clear"  # noqa: SLF001

    def test_confirm_clear_on_empty_is_silent_no_op(
        self, view, qtbot, monkeypatch
    ):
        # Empty library: pressing Clear shouldn't pop the confirmation
        # at all (and certainly shouldn't fire rootsChanged).
        prompted: list[object] = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.question",
            lambda *_a, **_k: (prompted.append(True), None)[1],
        )
        with qtbot.assertNotEmitted(view.rootsChanged, wait=100):
            view._confirm_clear()  # noqa: SLF001
        assert prompted == []

    def test_confirm_clear_yes_wipes_library(
        self, view, qtbot, tmp_path, monkeypatch
    ):
        from PySide6.QtWidgets import QMessageBox

        view.ingest_files_and_folders([_make_image(tmp_path / "a.png")])
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.question",
            lambda *_a, **_k: QMessageBox.StandardButton.Yes,
        )
        with qtbot.waitSignal(view.rootsChanged, timeout=1000):
            view._confirm_clear()  # noqa: SLF001
        assert view.roots() == []
        assert view.paths() == []

    def test_confirm_message_uses_visible_grid_count(
        self, view, qtbot, tmp_path, monkeypatch
    ):
        # Regression: when the library has a single folder root that
        # expanded to many files, the confirm dialog showed "1 entry"
        # (root count) instead of the visible count the user is
        # looking at. Capture the message body and assert it reflects
        # the grid count.
        import time as _time
        from PySide6.QtWidgets import QMessageBox

        folder = tmp_path / "lib"
        folder.mkdir()
        for i in range(5):
            _make_image(folder / f"f{i}.png")
        view.ingest_files_and_folders([folder])
        # Let the scan finish so the grid has all 5.
        end = _time.monotonic() + 3.0
        while len(view.paths()) < 5 and _time.monotonic() < end:
            qtbot.wait(50)
        captured: list[str] = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.question",
            lambda _self, _title, msg, *_a, **_k: (
                captured.append(msg),
                QMessageBox.StandardButton.No,
            )[1],
        )
        view._confirm_clear()  # noqa: SLF001
        assert captured, "confirm dialog wasn't shown"
        assert "5" in captured[0], (
            f"expected grid count (5) in message, got: {captured[0]}"
        )

    def test_confirm_message_notes_scan_in_progress(
        self, view, qtbot, tmp_path, monkeypatch
    ):
        from PySide6.QtWidgets import QMessageBox

        folder = tmp_path / "lib"
        folder.mkdir()
        # Many files so the scan is still running when we click clear.
        for i in range(200):
            _make_image(folder / f"f{i:03d}.png")
        view.ingest_files_and_folders([folder])
        # Don't wait for completion — click clear mid-scan.
        captured: list[str] = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.question",
            lambda _self, _title, msg, *_a, **_k: (
                captured.append(msg),
                QMessageBox.StandardButton.No,
            )[1],
        )
        view._confirm_clear()  # noqa: SLF001
        # Message should mention the scan so the user knows it'll also
        # be cancelled.
        assert captured, "confirm dialog wasn't shown"
        assert "scan" in captured[0].lower(), (
            f"expected scan mention in mid-scan confirm, got: {captured[0]}"
        )

    def test_confirm_clear_no_keeps_library(
        self, view, qtbot, tmp_path, monkeypatch
    ):
        from PySide6.QtWidgets import QMessageBox

        f = _make_image(tmp_path / "a.png")
        view.ingest_files_and_folders([f])
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.question",
            lambda *_a, **_k: QMessageBox.StandardButton.No,
        )
        with qtbot.assertNotEmitted(view.rootsChanged, wait=100):
            view._confirm_clear()  # noqa: SLF001
        assert view.roots() == [f]


class TestAddButton:
    def test_main_click_targets_files_dialog(self, view):
        # The primary action on the split-button is "add files" (the
        # common case). Folder is reached through the dropdown menu.
        assert view._add_button.popupMode() == view._add_button.popupMode().MenuButtonPopup  # noqa: SLF001

    def test_menu_has_files_and_folder_actions(self, view):
        menu = view._add_button.menu()  # noqa: SLF001
        labels = [a.text() for a in menu.actions()]
        assert labels == ["Files…", "Folder…"]


class TestSortAndFilterUi:
    def test_filter_edit_updates_proxy(self, view, qtbot, tmp_path):
        for n in ("alpha.png", "beta.png"):
            view.add_paths([_make_image(tmp_path / n)])
        # The grid initially shows both.
        assert view._list.model().rowCount() == 2  # noqa: SLF001
        view._filter_edit.setText("alpha")  # noqa: SLF001
        assert view._list.model().rowCount() == 1  # noqa: SLF001

    def test_sort_combo_change_actually_resorts_grid(
        self, view, tmp_path
    ):
        # End-to-end regression: previously _on_sort_field_changed
        # checked isinstance(SortField) on the value from itemData,
        # but Qt strips the str-Enum type on QVariant round-trip, so
        # the check always failed and the grid never re-sorted on
        # dropdown change. This is the failure my unit test missed
        # because the unit test called set_sort_field directly.
        from PIL import Image
        import numpy as np

        # Three images where name and size sort orders differ:
        #   alphabetic:  apple, mango, zebra
        #   by size asc: zebra (10), apple (50), mango (100)
        def _img_sized(path, w, h):
            arr = np.full((h, w, 3), 200, dtype=np.uint8)
            Image.fromarray(arr).save(path)
            return path

        z = _img_sized(tmp_path / "zebra.png", 10, 10)
        a = _img_sized(tmp_path / "apple.png", 50, 50)
        m = _img_sized(tmp_path / "mango.png", 100, 100)
        view.add_paths([z, a, m])

        from sinner2.library.library_model import ROLE_PATH

        proxy = view._list.model()  # noqa: SLF001

        def current_order() -> list[Path]:
            return [
                Path(proxy.data(proxy.index(i, 0), ROLE_PATH))
                for i in range(proxy.rowCount())
            ]

        # Default sort is by NAME, ascending.
        assert current_order() == [a, m, z]

        # Find the dropdown index whose userData is the Size field's
        # string value. Click-equivalent: setCurrentIndex emits the
        # currentIndexChanged signal we wired to _on_sort_field_changed.
        size_index = next(
            i
            for i in range(view._sort_combo.count())  # noqa: SLF001
            if view._sort_combo.itemData(i) == "size"  # noqa: SLF001
        )
        view._sort_combo.setCurrentIndex(size_index)  # noqa: SLF001
        assert current_order() == [z, a, m]
