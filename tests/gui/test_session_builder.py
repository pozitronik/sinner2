"""Tests for the session-build value objects + helpers (Phase 3.1).

_build_session / _install_session open real files + load models, so they aren't
unit-testable; what's pinned here is the relocated, Qt-free surface — the
reader-factory thunk and the _SessionBundle warnings channel (build collects
non-fatal problems; the GUI caller emits them).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from sinner2.config.source import Source
from sinner2.config.target import Target
from sinner2.gui.session_builder import (
    CacheSettings,
    SessionBuilder,
    SessionBuildSpec,
    _SessionBundle,
    _make_reader,
    _make_reader_factory,
)
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.buffer.store import PersistentFrameStore
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import ImageFormat


def _bundle(**over) -> _SessionBundle:
    base = dict(
        executor=SimpleNamespace(),
        write_executor=SimpleNamespace(),
        session_store=SimpleNamespace(),
        cache_dir=SimpleNamespace(),
        source=SimpleNamespace(),
        source_path=SimpleNamespace(),
        target_path=SimpleNamespace(),
        target_fps=30.0,
        frame_count=10,
        native_size=(64, 64),
    )
    base.update(over)
    return _SessionBundle(**base)


def test_session_bundle_warnings_default_empty():
    assert _bundle().warnings == []


def test_session_bundle_carries_warnings():
    b = _bundle(warnings=["cache root unavailable; running with cache OFF"])
    assert b.warnings == ["cache root unavailable; running with cache OFF"]


def test_make_reader_factory_is_a_fresh_thunk(tmp_path, monkeypatch):
    # The factory curries args and builds a NEW reader per call (ReaderPool needs
    # N independent readers). We stub _make_reader to avoid opening real media.
    built = []
    monkeypatch.setattr(
        "sinner2.gui.session_builder._make_reader",
        lambda target, backend, scale: built.append((target, scale)) or object(),
    )
    tgt = tmp_path / "t.mp4"
    tgt.write_bytes(b"x")
    target = Target(path=tgt)
    factory = _make_reader_factory(target, VideoBackend.FFMPEG, 0.5)
    r1, r2 = factory(), factory()
    assert r1 is not r2
    assert built == [(target, 0.5), (target, 0.5)]


class TestBuildStoreForChain:
    """build_store_for_chain re-keys the disk cache to a new chain's dir so a
    chain change reuses a prior run's cache instead of wiping the old dir."""

    def _builder(self, *, available=True, cache_dir=None):
        cache = MagicMock()
        cache.cache_dir_for.return_value = cache_dir
        manager = MagicMock()
        manager.is_available.return_value = available
        cache.cache_manager.return_value = manager
        return SessionBuilder(cache, MagicMock()), cache, manager

    def _spec(self, mode=CacheMode.WRITE_READ):
        return SessionBuildSpec(
            strategy=MagicMock(),
            playback_mode=MagicMock(),
            cache_settings=CacheSettings(
                mode=mode, image_format=ImageFormat.JPEG, image_quality=90,
                memory_max_bytes=1, write_workers=1, write_queue_size=1,
            ),
            video_backend=VideoBackend.FFMPEG,
            reader_pool_size=1,
            processing_scale=1.0,
        )

    def _src_tgt(self, tmp_path):
        s = tmp_path / "src.jpg"
        s.write_bytes(b"x")
        t = tmp_path / "tgt.mp4"
        t.write_bytes(b"x")
        return Source(path=s), Target(path=t)

    def test_off_mode_returns_none_without_touching_cache(self, tmp_path):
        builder, cache, _ = self._builder()
        src, tgt = self._src_tgt(tmp_path)
        out = builder.build_store_for_chain(
            src, tgt, [SimpleNamespace(name="swap")],
            self._spec(mode=CacheMode.OFF), 10,
        )
        assert out is None
        cache.cache_dir_for.assert_not_called()  # short-circuit, no key compute

    def test_unavailable_root_returns_none(self, tmp_path):
        cache_dir = tmp_path / "key"
        builder, cache, _ = self._builder(available=False, cache_dir=cache_dir)
        src, tgt = self._src_tgt(tmp_path)
        out = builder.build_store_for_chain(
            src, tgt, [SimpleNamespace(name="swap")], self._spec(), 10,
        )
        assert out is None  # caller falls back to invalidate-in-place
        cache.enforce_cap.assert_not_called()

    def test_builds_rekeyed_store_and_writes_meta(self, tmp_path):
        cache_dir = tmp_path / "key"
        builder, cache, manager = self._builder(cache_dir=cache_dir)
        src, tgt = self._src_tgt(tmp_path)
        out = builder.build_store_for_chain(
            src, tgt, [SimpleNamespace(name="swap")], self._spec(), 7,
        )
        assert out is not None
        store, returned_dir = out
        assert isinstance(store, PersistentFrameStore)
        assert returned_dir == cache_dir
        assert store.directory == cache_dir
        # cap enforced sparing the NEW dir; session meta written for it
        cache.enforce_cap.assert_called_once_with(manager, cache_dir)
        manager.write_meta.assert_called_once()
        assert manager.write_meta.call_args.args[0] == cache_dir


def test_make_reader_rejects_unknown_kind():
    bad = SimpleNamespace(kind="bogus")
    try:
        _make_reader(bad, VideoBackend.FFMPEG)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "unsupported target kind" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
