"""Tests for the session-build value objects + helpers (Phase 3.1).

_build_session / _install_session open real files + load models, so they aren't
unit-testable; what's pinned here is the relocated, Qt-free surface — the
reader-factory thunk and the _SessionBundle warnings channel (build collects
non-fatal problems; the GUI caller emits them).
"""
from __future__ import annotations

from types import SimpleNamespace

from sinner2.config.target import Target
from sinner2.gui.session_builder import (
    _SessionBundle,
    _make_reader,
    _make_reader_factory,
)
from sinner2.io.video_backend import VideoBackend


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


def test_make_reader_rejects_unknown_kind():
    bad = SimpleNamespace(kind="bogus")
    try:
        _make_reader(bad, VideoBackend.FFMPEG)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "unsupported target kind" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
