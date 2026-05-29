"""Tests for LibraryItemModel + LibrarySortFilterProxy.

Uses real ThumbnailGenerator with tiny PIL images so the queued signal
path through the model gets exercised end-to-end. qtbot.waitSignal
synchronises the worker-thread callback with the main thread.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt

from sinner2.library.library_model import (
    ROLE_ERROR,
    ROLE_FILE_SIZE,
    ROLE_MOD_DATE,
    ROLE_PATH,
    ROLE_PIXEL_COUNT,
    LibraryItemModel,
    LibrarySortFilterProxy,
    SortField,
    item_cell_size,
)
from sinner2.library.thumbnail_cache import ThumbnailCache
from sinner2.library.thumbnail_generator import ThumbnailGenerator


@pytest.fixture
def generator(tmp_path: Path):
    cache = ThumbnailCache(tmp_path / "cache")
    gen = ThumbnailGenerator(cache, thumb_dim=32, max_workers=2)
    yield gen
    gen.shutdown(wait=True)


@pytest.fixture
def model(qtbot, generator):
    m = LibraryItemModel(generator)
    return m


def _make_image(path: Path, w: int = 80, h: int = 60) -> Path:
    arr = np.full((h, w, 3), 200, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


class TestPathManagement:
    def test_initially_empty(self, model):
        assert model.rowCount() == 0
        assert model.paths() == []

    def test_add_path_appends_row(self, model, tmp_path):
        p = _make_image(tmp_path / "a.png")
        assert model.add_path(p) is True
        assert model.rowCount() == 1
        assert model.paths() == [p]

    def test_add_dedupes(self, model, tmp_path):
        p = _make_image(tmp_path / "a.png")
        model.add_path(p)
        assert model.add_path(p) is False
        assert model.rowCount() == 1

    def test_add_paths_bulk(self, model, tmp_path):
        paths = [_make_image(tmp_path / f"f{i}.png") for i in range(3)]
        # Bulk add includes a duplicate — should still report 3 added.
        added = model.add_paths(paths + [paths[0]])
        assert added == 3
        assert model.rowCount() == 3

    def test_remove_path(self, model, tmp_path):
        a = _make_image(tmp_path / "a.png")
        b = _make_image(tmp_path / "b.png")
        model.add_path(a)
        model.add_path(b)
        assert model.remove_path(a) is True
        assert model.paths() == [b]

    def test_remove_unknown_is_noop(self, model, tmp_path):
        assert model.remove_path(tmp_path / "nope.png") is False

    def test_clear_paths(self, model, tmp_path):
        for i in range(3):
            model.add_path(_make_image(tmp_path / f"f{i}.png"))
        model.clear_paths()
        assert model.rowCount() == 0
        assert model.paths() == []

    def test_stat_metadata_present_immediately(self, model, tmp_path):
        # Sort must work even before the thumbnail arrives — so size
        # and mtime are populated on add, not in the thumbnail callback.
        p = _make_image(tmp_path / "a.png", 40, 40)
        model.add_path(p)
        item = model.item(0)
        assert item.data(ROLE_FILE_SIZE) == p.stat().st_size
        assert item.data(ROLE_MOD_DATE) == pytest.approx(p.stat().st_mtime)
        assert item.data(ROLE_PATH) == str(p)


class TestThumbnailApplication:
    def test_thumbnail_arrives_via_queued_signal(self, model, tmp_path, qtbot):
        p = _make_image(tmp_path / "src.png", 100, 75)
        model.add_path(p)
        item = model.item(0)
        # Wait for the generator's worker callback to round-trip back
        # to the main thread via the model's internal queued signals.
        # The PIXEL_COUNT gets populated only by the thumbnail callback,
        # so it's the cleanest waitable side-effect.
        end = time.monotonic() + 3.0
        while item.data(ROLE_PIXEL_COUNT) == 0 and time.monotonic() < end:
            qtbot.wait(20)
        assert item.data(ROLE_PIXEL_COUNT) == 100 * 75

    def test_failure_path_marks_item_with_error(self, model, tmp_path, qtbot):
        # Non-media file — generator returns ThumbnailError, model
        # populates ROLE_ERROR and rewrites the caption to show the
        # reason so the user notices.
        p = tmp_path / "notes.txt"
        p.write_text("hello")
        model.add_path(p)
        item = model.item(0)
        end = time.monotonic() + 3.0
        while item.data(ROLE_ERROR) is None and time.monotonic() < end:
            qtbot.wait(20)
        assert item.data(ROLE_ERROR) is not None
        assert "notes.txt" in item.text()


class TestCaptionArea:
    """Caption area scales with display_dim so big tiles get proportional
    text room and small tiles don't waste height. Tooltip carries the
    full info regardless of how aggressively the visible caption
    clips."""

    def test_caption_height_floor_at_small_dims(self):
        from sinner2.library.library_model import _caption_height

        # Below the dim threshold, caption stays at the 48px floor —
        # anything smaller would clip to less than two lines at the
        # default font.
        assert _caption_height(64) >= 48
        assert _caption_height(96) >= 48

    def test_caption_height_grows_with_dim(self):
        from sinner2.library.library_model import _caption_height

        # Larger tiles get more caption room — both for proportional
        # appearance and for the longer wrap that a wider line accepts.
        assert _caption_height(384) > _caption_height(128)
        assert _caption_height(256) >= _caption_height(128)

    def test_item_cell_size_uses_scaled_caption(self):
        # cell.height should grow MORE than just display_dim once we
        # cross the scaling threshold; verifies the helper actually
        # plumbs through.
        from sinner2.library.library_model import item_cell_size

        small = item_cell_size(128)
        large = item_cell_size(384)
        small_caption_extra = small.height() - 128
        large_caption_extra = large.height() - 384
        assert large_caption_extra > small_caption_extra

    @pytest.fixture
    def gen(self, tmp_path):
        from sinner2.library.thumbnail_cache import ThumbnailCache
        from sinner2.library.thumbnail_generator import ThumbnailGenerator

        cache = ThumbnailCache(tmp_path / "cache")
        g = ThumbnailGenerator(cache, thumb_dim=128, max_workers=2)
        yield g
        g.shutdown(wait=True)

    def test_tooltip_set_on_add_with_path(self, qtbot, gen, tmp_path):
        # Before the thumbnail callback runs, the tooltip is the path
        # itself — so a clipped placeholder tile still tells the user
        # what file it is on hover.
        import numpy as np
        from PIL import Image
        from PySide6.QtCore import Qt

        model = LibraryItemModel(gen)
        p = tmp_path / "Buлановa.png"
        Image.fromarray(np.full((40, 40, 3), 200, dtype=np.uint8)).save(p)
        model.add_path(p)
        tooltip = model.item(0).data(Qt.ItemDataRole.ToolTipRole)
        assert tooltip is not None
        assert str(p) in tooltip

    def test_tooltip_updated_with_caption_and_path_after_thumbnail(
        self, qtbot, gen, tmp_path
    ):
        # After the thumbnail callback fires, the tooltip is caption +
        # path on two lines. Wait for the queued signal to apply.
        import time as _time
        import numpy as np
        from PIL import Image
        from PySide6.QtCore import Qt

        model = LibraryItemModel(gen)
        p = tmp_path / "src.png"
        Image.fromarray(np.full((40, 40, 3), 200, dtype=np.uint8)).save(p)
        model.add_path(p)
        end = _time.monotonic() + 3.0
        while model.item(0).data(ROLE_PIXEL_COUNT) == 0 and _time.monotonic() < end:
            qtbot.wait(20)
        tooltip = model.item(0).data(Qt.ItemDataRole.ToolTipRole)
        assert "src.png" in tooltip
        assert str(p) in tooltip


class TestItemSizing:
    """Regression: cells overlapped when extraction size (cached
    pixmap = 384px) didn't match display dim (icon = 128px). Fix is
    pre-scale-on-apply plus an explicit per-item sizeHint that agrees
    with the view's gridSize."""

    @pytest.fixture
    def big_generator(self, tmp_path):
        from sinner2.library.thumbnail_cache import ThumbnailCache
        from sinner2.library.thumbnail_generator import ThumbnailGenerator

        cache = ThumbnailCache(tmp_path / "cache")
        g = ThumbnailGenerator(cache, thumb_dim=256, max_workers=2)
        yield g
        g.shutdown(wait=True)

    def test_item_size_hint_matches_cell_for_display_dim(
        self, qtbot, big_generator, tmp_path
    ):
        model = LibraryItemModel(big_generator, display_dim=128)
        f = tmp_path / "a.png"
        import numpy as np
        from PIL import Image

        Image.fromarray(np.full((40, 40, 3), 200, dtype=np.uint8)).save(f)
        model.add_path(f)
        item = model.item(0)
        # Per-item sizeHint must match the cell footprint at the
        # current display_dim. Without this, IconMode falls back to
        # the pixmap's intrinsic size and tiles overlap.
        assert item.sizeHint() == item_cell_size(128)

    def test_set_display_dim_updates_all_size_hints(
        self, qtbot, big_generator, tmp_path
    ):
        model = LibraryItemModel(big_generator, display_dim=128)
        import numpy as np
        from PIL import Image

        for n in ("a.png", "b.png"):
            p = tmp_path / n
            Image.fromarray(np.full((40, 40, 3), 200, dtype=np.uint8)).save(p)
            model.add_path(p)
        model.set_display_dim(192)
        for row in range(model.rowCount()):
            assert model.item(row).sizeHint() == item_cell_size(192)


class TestProxySort:
    def _build_proxy_with_three(self, model, tmp_path):
        # Build three distinguishable entries in a deterministic order.
        # Names interleaved so sort-by-name reorders; sizes distinct so
        # sort-by-size reorders too.
        a = _make_image(tmp_path / "banana.png", 10, 10)  # smallest
        b = _make_image(tmp_path / "apple.png", 50, 50)
        c = _make_image(tmp_path / "cherry.png", 100, 100)  # largest
        for p in (a, b, c):
            model.add_path(p)
        proxy = LibrarySortFilterProxy()
        proxy.setSourceModel(model)
        return proxy, a, b, c

    def _proxy_paths(self, proxy):
        return [
            Path(proxy.data(proxy.index(row, 0), ROLE_PATH))
            for row in range(proxy.rowCount())
        ]

    def test_default_sort_is_by_name(self, model, tmp_path):
        proxy, a, b, c = self._build_proxy_with_three(model, tmp_path)
        proxy.sort(0)
        ordered = self._proxy_paths(proxy)
        # Alphabetic: apple, banana, cherry.
        assert ordered == [b, a, c]

    def test_sort_by_size(self, model, tmp_path, qtbot):
        proxy, a, b, c = self._build_proxy_with_three(model, tmp_path)
        proxy.set_sort_field(SortField.SIZE)
        proxy.sort(0)
        ordered = self._proxy_paths(proxy)
        # File size correlates with image dimensions for PNGs of identical
        # solid content: small → large = a, b, c.
        assert ordered == [a, b, c]

    def test_set_sort_field_alone_resorts_without_explicit_sort_call(
        self, model, tmp_path
    ):
        # Regression: set_sort_field used to set the role but rely on
        # Qt's dynamicSortFilter to re-sort — which it doesn't do on
        # role change. The user-visible symptom was "dropdown changes
        # but order doesn't." invalidate() inside set_sort_field is the
        # fix; this test passes only if it's in place.
        proxy, a, b, c = self._build_proxy_with_three(model, tmp_path)
        proxy.sort(0, Qt.SortOrder.AscendingOrder)  # initial sort by name
        # After sorting by name, the order is [apple, banana, cherry]
        # i.e. [b, a, c]. Switching the role to SIZE without an
        # explicit re-sort must still produce [a, b, c].
        assert self._proxy_paths(proxy) == [b, a, c]
        proxy.set_sort_field(SortField.SIZE)
        # No proxy.sort() call here — set_sort_field must internally
        # invalidate + re-apply for the visible order to update.
        assert self._proxy_paths(proxy) == [a, b, c]

    def test_sort_by_pixels_uses_thumbnail_meta(
        self, model, tmp_path, qtbot
    ):
        # PIXELS sort needs the thumbnail callback to have run (only it
        # sets pixel_count). Wait for all three to complete.
        proxy, a, b, c = self._build_proxy_with_three(model, tmp_path)
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            if all(
                model.item(r).data(ROLE_PIXEL_COUNT) > 0
                for r in range(model.rowCount())
            ):
                break
            qtbot.wait(20)
        proxy.set_sort_field(SortField.PIXELS)
        proxy.sort(0)
        ordered = self._proxy_paths(proxy)
        # Small to large by pixel count.
        assert ordered == [a, b, c]


class TestProxyFilter:
    def test_filter_matches_caption_substring(self, model, tmp_path):
        # Caption starts as the filename; filter is case-insensitive.
        for name in ("alpha.png", "beta.png", "alpine.png"):
            model.add_path(_make_image(tmp_path / name))
        proxy = LibrarySortFilterProxy()
        proxy.setSourceModel(model)
        proxy.set_filter_text("alp")
        # Both 'alpha.png' and 'alpine.png' match; 'beta.png' filtered out.
        paths = [
            Path(proxy.data(proxy.index(row, 0), ROLE_PATH))
            for row in range(proxy.rowCount())
        ]
        names = sorted(p.name for p in paths)
        assert names == ["alpha.png", "alpine.png"]

    def test_filter_clears_with_empty_string(self, model, tmp_path):
        for name in ("x.png", "y.png"):
            model.add_path(_make_image(tmp_path / name))
        proxy = LibrarySortFilterProxy()
        proxy.setSourceModel(model)
        proxy.set_filter_text("x")
        assert proxy.rowCount() == 1
        proxy.set_filter_text("")
        assert proxy.rowCount() == 2
