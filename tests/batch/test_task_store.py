"""Tests for the file-backed BatchTaskStore."""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.batch.task import BatchTask, BatchTaskStatus
from sinner2.batch.task_store import BatchTaskStore


def _task(tmp_path: Path, **overrides) -> BatchTask:
    kwargs = {
        "source_path": tmp_path / "src.png",
        "target_path": tmp_path / "tgt.mp4",
    }
    kwargs.update(overrides)
    return BatchTask(**kwargs)


@pytest.fixture
def store(tmp_path: Path) -> BatchTaskStore:
    return BatchTaskStore(tmp_path / "batch")


class TestSaveLoad:
    def test_save_then_load_roundtrip(self, store, tmp_path):
        t = _task(tmp_path, status=BatchTaskStatus.RUNNING)
        store.save(t)
        loaded = store.load(t.id)
        assert loaded == t

    def test_load_unknown_id_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_exists_returns_false_before_save(self, store, tmp_path):
        t = _task(tmp_path)
        assert store.exists(t.id) is False
        store.save(t)
        assert store.exists(t.id) is True

    def test_save_overwrites_existing(self, store, tmp_path):
        t = _task(tmp_path)
        store.save(t)
        t2 = t.model_copy(update={"status": BatchTaskStatus.COMPLETED})
        store.save(t2)
        assert store.load(t.id).status is BatchTaskStatus.COMPLETED

    def test_save_creates_root_dir_if_missing(self, tmp_path):
        # Constructor creates the root; deleting it then saving must
        # NOT fail (we want defensive recreation? Actually no — we
        # only create on construction. Test the simpler contract:
        # constructor creates root.)
        store = BatchTaskStore(tmp_path / "deeply" / "nested" / "root")
        assert store.root.is_dir()


class TestDelete:
    def test_delete_existing_returns_true(self, store, tmp_path):
        t = _task(tmp_path)
        store.save(t)
        assert store.delete(t.id) is True
        assert not store.exists(t.id)

    def test_delete_unknown_returns_false(self, store):
        assert store.delete("nonexistent") is False


class TestList:
    def test_empty_store_returns_empty_list(self, store):
        assert store.list() == []

    def test_lists_all_saved_tasks(self, store, tmp_path):
        ids = []
        for _ in range(3):
            t = _task(tmp_path)
            store.save(t)
            ids.append(t.id)
        listed = [t.id for t in store.list()]
        assert set(listed) == set(ids)
        assert len(listed) == 3

    def test_corrupt_file_silently_skipped(self, store, tmp_path):
        # Save one valid task, then drop a corrupt JSON file next to
        # it. list() must return the valid one and skip the bad one.
        good = _task(tmp_path)
        store.save(good)
        (store.root / "broken.json").write_text("not json {", encoding="utf-8")
        listed = store.list()
        assert len(listed) == 1
        assert listed[0].id == good.id

    def test_missing_required_field_silently_skipped(
        self, store, tmp_path
    ):
        # A file with missing required fields fails Pydantic validation.
        # Like a corrupt file, must be silently skipped — surfaces the
        # remaining valid tasks instead of erroring.
        good = _task(tmp_path)
        store.save(good)
        (store.root / "incomplete.json").write_text(
            '{"id": "bad"}', encoding="utf-8"
        )
        assert len(store.list()) == 1


class TestPathTraversal:
    def test_rejects_traversal_via_slash(self, store):
        with pytest.raises(ValueError):
            store.load("../etc/passwd")

    def test_rejects_traversal_via_dotdot(self, store):
        with pytest.raises(ValueError):
            store.load("..")

    def test_rejects_backslash(self, store):
        with pytest.raises(ValueError):
            store.load(r"foo\bar")


class TestAtomicSave:
    def test_no_tmp_file_left_after_save(self, store, tmp_path):
        # tmp file is renamed atomically — must not remain in the
        # store on the happy path.
        t = _task(tmp_path)
        store.save(t)
        tmps = list(store.root.glob("*.tmp"))
        assert tmps == []


class TestOrdering:
    """list() honours the explicit `order` field (stable — ties keep their
    filename order); set_order renumbers tasks densely by position."""

    def test_list_sorted_by_order_field(self, store, tmp_path):
        a = _task(tmp_path, order=2)
        b = _task(tmp_path, order=0)
        c = _task(tmp_path, order=1)
        for t in (a, b, c):
            store.save(t)
        assert [t.id for t in store.list()] == [b.id, c.id, a.id]

    def test_legacy_zero_order_keeps_filename_order(self, store, tmp_path):
        # All-default (order 0) tasks must stay in their pre-existing order.
        ids = []
        for _ in range(3):
            t = _task(tmp_path)  # order defaults to 0
            store.save(t)
            ids.append(t.id)
        assert [t.id for t in store.list()] == sorted(ids)  # filename = id order

    def test_set_order_renumbers_densely(self, store, tmp_path):
        a, b, c = (_task(tmp_path) for _ in range(3))
        for t in (a, b, c):
            store.save(t)
        store.set_order([c.id, a.id, b.id])
        assert store.load(c.id).order == 0
        assert store.load(a.id).order == 1
        assert store.load(b.id).order == 2
        assert [t.id for t in store.list()] == [c.id, a.id, b.id]
