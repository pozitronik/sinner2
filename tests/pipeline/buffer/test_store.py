from pathlib import Path

import numpy as np
import pytest

from sinner2.pipeline.buffer.store import (
    DiskFrameStore,
    FrameStore,
    PersistentFrameStore,
    SessionFrameStore,
)
from sinner2.types import Frame


def _frame() -> Frame:
    return np.full((20, 20, 3), 128, dtype=np.uint8)


class TestDiskFrameStore:
    def test_compliant_with_protocol(self, tmp_path: Path):
        assert isinstance(DiskFrameStore(tmp_path), FrameStore)

    def test_write_then_read_roundtrip(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        f = _frame()
        store.write(42, f)
        out = store.read(42)
        assert out is not None
        assert np.array_equal(out, f)

    def test_has_reflects_write(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        assert not store.has(0)
        store.write(0, _frame())
        assert store.has(0)

    def test_read_missing_returns_none(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        assert store.read(99) is None

    def test_cached_indices_lists_written_frames(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        assert store.cached_indices() == []
        store.write(0, _frame())
        store.write(5, _frame())
        store.write(42, _frame())
        assert sorted(store.cached_indices()) == [0, 5, 42]

    def test_clear_from_drops_only_at_or_above(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        for i in [10, 20, 30, 40]:
            store.write(i, _frame())
        store.clear_from(25)
        assert store.has(10)
        assert store.has(20)
        assert not store.has(30)
        assert not store.has(40)

    def test_clear_from_zero_drops_everything(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        store.write(0, _frame())
        store.write(5, _frame())
        store.clear_from(0)
        assert not store.has(0)
        assert not store.has(5)
        # The directory is recreated empty — the store stays usable afterwards.
        store.write(1, _frame())
        assert store.has(1)

    def test_creates_nested_directory_if_missing(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        store = DiskFrameStore(nested)
        assert nested.exists()
        store.write(0, _frame())
        assert store.has(0)

    def test_extension_from_writer(self, tmp_path: Path):
        from sinner2.pipeline.image_writer import JPEGImageWriter

        store = DiskFrameStore(tmp_path, writer=JPEGImageWriter())
        store.write(0, _frame())
        assert (tmp_path / "00000000.jpg").exists()

    def test_filename_uses_zero_padding(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        store.write(7, _frame())
        assert (tmp_path / "00000007.png").exists()

    def test_write_failure_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # The PNG writer now goes through imwrite_unicode (cv2.imencode +
        # Path.write_bytes) rather than cv2.imwrite directly. Force the
        # encode step to fail so write() surfaces as an OSError.
        import cv2

        store = DiskFrameStore(tmp_path)
        monkeypatch.setattr(
            cv2, "imencode", lambda *_a, **_k: (False, None)
        )
        with pytest.raises(OSError):
            store.write(0, _frame())


class TestPersistentFrameStore:
    def test_compliant_with_protocol(self, tmp_path: Path):
        store = PersistentFrameStore(tmp_path / "cache")
        assert isinstance(store, FrameStore)

    def test_creates_directory_lazily(self, tmp_path: Path):
        path = tmp_path / "deep" / "cache"
        store = PersistentFrameStore(path)
        store.write(0, _frame())
        assert path.is_dir()

    def test_write_read_roundtrip(self, tmp_path: Path):
        store = PersistentFrameStore(tmp_path)
        f = _frame()
        store.write(5, f)
        assert np.array_equal(store.read(5), f)

    def test_close_does_not_remove_directory(self, tmp_path: Path):
        store = PersistentFrameStore(tmp_path)
        store.write(0, _frame())
        store.close()
        assert tmp_path.is_dir()
        assert (tmp_path / "00000000.png").is_file()

    def test_reopen_finds_previous_frames(self, tmp_path: Path):
        store1 = PersistentFrameStore(tmp_path)
        store1.write(3, _frame())
        store1.close()
        store2 = PersistentFrameStore(tmp_path)
        assert store2.has(3)
        assert store2.read(3) is not None


class TestSessionFrameStore:
    def test_compliant_with_protocol(self):
        store = SessionFrameStore()
        try:
            assert isinstance(store, FrameStore)
        finally:
            store.close()

    def test_creates_fresh_scratch_dir(self):
        a = SessionFrameStore()
        b = SessionFrameStore()
        try:
            assert a.scratch_dir != b.scratch_dir
            assert a.scratch_dir.is_dir()
            assert b.scratch_dir.is_dir()
        finally:
            a.close()
            b.close()

    def test_write_read_roundtrip(self):
        store = SessionFrameStore()
        try:
            f = _frame()
            store.write(0, f)
            out = store.read(0)
            assert out is not None
            assert np.array_equal(out, f)
        finally:
            store.close()

    def test_close_removes_scratch_dir(self):
        store = SessionFrameStore()
        path = store.scratch_dir
        store.write(0, _frame())
        store.close()
        assert not path.exists()

    def test_close_is_idempotent(self):
        store = SessionFrameStore()
        store.close()
        store.close()  # must not raise
