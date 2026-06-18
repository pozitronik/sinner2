"""Tests for chain-refresh + cache reconfiguration on apply_session_config.

When the user toggles a chain-affecting param (enhancer on/off, swapper
detection interval, etc.), apply_session_config rebuilds the chain and calls
executor.set_chain(). set_chain itself now invalidates the whole frame cache
and re-renders the current frame (paused or playing), so the controller no
longer issues a compensating seek — and crucially every cached frame refreshes,
not just the visible one, so a tweak applies across the clip even with a large
memory cache. A memory-cache-size change is hot-applied to the live buffer
(no restart / session rebuild needed)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.config.source import Source
from sinner2.gui.player_controller import PlayerController, _DEFAULT_CACHE_SETTINGS
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processors.face_enhancer import FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapperParams
from sinner2.pipeline.skip_strategy import BestEffortStrategy


@pytest.fixture
def widgets(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return display, transport


@pytest.fixture
def fake_source_path(tmp_path):
    """Source validates path existence; touch a real file so it accepts it."""
    p = tmp_path / "source.jpg"
    p.write_bytes(b"")
    return p


def _make_controller(widgets) -> PlayerController:
    display, transport = widgets
    return PlayerController(frame_display=display, transport=transport)


def _attach_fake_session(
    ctrl: PlayerController,
    *,
    playing: bool,
    current_frame: int,
    source_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    ctrl._current_source = Source(path=source_path)  # noqa: SLF001
    fake_executor = MagicMock()
    fake_executor.is_playing.get.return_value = playing
    fake_executor.current_frame.get.return_value = current_frame
    ctrl._executor = fake_executor  # noqa: SLF001
    # Bypass real chain construction — the test only cares that the
    # controller asks the executor for a set_chain + (conditionally) seek.
    monkeypatch.setattr(ctrl, "_build_chain", lambda src: [object()])
    return fake_executor


def _apply(ctrl: PlayerController, *, enhancer_enabled: bool) -> None:
    """Call apply_session_config with the given enhancer_enabled. Flip the
    value relative to the controller's stored state to force
    chain_changed=True; pass the current value to leave the chain alone."""
    ctrl.apply_session_config(
        swapper_params=FaceSwapperParams(),
        enhancer_params=FaceEnhancerParams(),
        enhancer_enabled=enhancer_enabled,
        strategy=BestEffortStrategy(),
        worker_count=1,
        playback_mode=PlaybackMode.FIXED_30,
        cache_settings=_DEFAULT_CACHE_SETTINGS,
    )


class TestDetectionSizeResetsSharedDetector:
    """det_size is baked into the shared insightface detector at build time,
    so changing it must drop the singleton (like a providers change) — else
    the rebuilt chain reuses a detector still prepared at the old size."""

    def test_det_size_change_resets_shared_detector(
        self, widgets, fake_source_path, monkeypatch
    ):
        from sinner2.pipeline import face_analyser

        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=True,
            current_frame=10,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        # Isolate det_size as the trigger: stored providers match the default
        # arg so providers_changed stays False.
        ctrl._swapper_providers = ()  # noqa: SLF001
        ctrl._swapper_params = FaceSwapperParams(detection_size=640)  # noqa: SLF001
        calls: list[int] = []
        monkeypatch.setattr(
            face_analyser, "reset_shared_face_analysis", lambda: calls.append(1)
        )
        ctrl.apply_session_config(
            swapper_params=FaceSwapperParams(detection_size=320),
            enhancer_params=FaceEnhancerParams(),
            enhancer_enabled=ctrl._enhancer_enabled,  # noqa: SLF001
            strategy=BestEffortStrategy(),
            worker_count=1,
            playback_mode=PlaybackMode.FIXED_30,
            cache_settings=_DEFAULT_CACHE_SETTINGS,
        )
        assert calls == [1]
        fake.set_chain.assert_called_once()
        ctrl.shutdown()

    def test_same_det_size_does_not_reset(
        self, widgets, fake_source_path, monkeypatch
    ):
        from sinner2.pipeline import face_analyser

        ctrl = _make_controller(widgets)
        _attach_fake_session(
            ctrl,
            playing=True,
            current_frame=10,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        ctrl._swapper_providers = ()  # noqa: SLF001
        ctrl._swapper_params = FaceSwapperParams(detection_size=640)  # noqa: SLF001
        calls: list[int] = []
        monkeypatch.setattr(
            face_analyser, "reset_shared_face_analysis", lambda: calls.append(1)
        )
        # Force a chain rebuild via enhancer toggle WITHOUT touching det_size —
        # the detector reset must NOT fire for an unrelated chain change.
        ctrl.apply_session_config(
            swapper_params=FaceSwapperParams(detection_size=640),
            enhancer_params=FaceEnhancerParams(),
            enhancer_enabled=not ctrl._enhancer_enabled,  # noqa: SLF001
            strategy=BestEffortStrategy(),
            worker_count=1,
            playback_mode=PlaybackMode.FIXED_30,
            cache_settings=_DEFAULT_CACHE_SETTINGS,
        )
        assert calls == []
        ctrl.shutdown()


class TestChainRefresh:
    """set_chain now owns the reprocess (it invalidates the whole cache and
    re-renders the current frame itself), so the controller just calls set_chain
    — no compensating seek, paused or playing."""

    def test_paused_chain_change_calls_set_chain_without_seek(
        self, widgets, fake_source_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=False,
            current_frame=42,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        # Controller default is enhancer_enabled=True; flip to False to
        # force chain_changed.
        _apply(ctrl, enhancer_enabled=False)
        fake.set_chain.assert_called_once()
        fake.seek.assert_not_called()
        ctrl.shutdown()

    def test_playing_chain_change_calls_set_chain_without_seek(
        self, widgets, fake_source_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=True,
            current_frame=42,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        _apply(ctrl, enhancer_enabled=False)
        fake.set_chain.assert_called_once()
        fake.seek.assert_not_called()
        ctrl.shutdown()

    def test_no_chain_change_does_not_set_chain_or_seek(
        self, widgets, fake_source_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=False,
            current_frame=42,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        # Apply the controller's current state — chain_changed=False.
        _apply(ctrl, enhancer_enabled=ctrl._enhancer_enabled)  # noqa: SLF001
        fake.set_chain.assert_not_called()
        fake.seek.assert_not_called()
        ctrl.shutdown()


class TestChainChangeRekeysCache:
    """A chain/param change RE-KEYS the disk cache to the new chain's dir (so a
    prior run's cache is reused) instead of wiping the current dir — except for
    provider/det-size changes, which aren't in the cache key and so keep the old
    invalidate-in-place behaviour (store=None) to avoid serving stale frames."""

    def _attach_with_target(self, ctrl, fake, tmp_path):
        tgt = tmp_path / "t.mp4"
        tgt.write_bytes(b"x")
        ctrl._current_target_path = tgt  # noqa: SLF001
        fake.frame_count.return_value = 100
        # Pin swapper state so the only change is the enhancer flip below —
        # providers/det-size stay equal, so the re-key gate is open.
        ctrl._swapper_providers = ()  # noqa: SLF001
        ctrl._swapper_params = FaceSwapperParams()  # noqa: SLF001
        return tgt

    def test_param_change_passes_rekeyed_store_to_set_chain(
        self, widgets, fake_source_path, tmp_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl, playing=False, current_frame=5,
            source_path=fake_source_path, monkeypatch=monkeypatch,
        )
        self._attach_with_target(ctrl, fake, tmp_path)
        new_store = MagicMock()
        new_dir = tmp_path / "newkey"
        ctrl._session_builder = MagicMock()  # noqa: SLF001
        ctrl._session_builder.build_store_for_chain.return_value = (  # noqa: SLF001
            new_store, new_dir,
        )
        old_store = MagicMock()
        ctrl._session_store = old_store  # noqa: SLF001
        scratch: list = []
        ctrl.sessionScratchDirChanged.connect(scratch.append)

        _apply(ctrl, enhancer_enabled=False)  # flip enhancer → chain_changed

        assert fake.set_chain.call_count == 1
        assert fake.set_chain.call_args.kwargs["store"] is new_store
        # builder asked for the NEW chain's store with the LIVE frame count
        assert ctrl._session_builder.build_store_for_chain.call_args.args[4] == 100  # noqa: SLF001
        # controller re-points at the new dir/store; old store closed (dir lives)
        assert ctrl._session_store is new_store  # noqa: SLF001
        assert ctrl._session_cache_dir == new_dir  # noqa: SLF001
        old_store.close.assert_called_once()
        assert scratch == [new_dir]
        ctrl.shutdown()

    def test_provider_change_invalidates_in_place_no_rekey(
        self, widgets, fake_source_path, tmp_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl, playing=False, current_frame=5,
            source_path=fake_source_path, monkeypatch=monkeypatch,
        )
        self._attach_with_target(ctrl, fake, tmp_path)
        ctrl._session_builder = MagicMock()  # noqa: SLF001
        old_store = MagicMock()
        ctrl._session_store = old_store  # noqa: SLF001
        # Avoid touching real model/detector singletons on a providers change.
        from sinner2.pipeline import face_analyser, model_cache
        monkeypatch.setattr(model_cache, "clear_session_cache", lambda: None)
        monkeypatch.setattr(
            face_analyser, "reset_shared_face_analysis", lambda: None
        )

        ctrl.apply_session_config(
            swapper_params=FaceSwapperParams(),
            enhancer_params=FaceEnhancerParams(),
            enhancer_enabled=ctrl._enhancer_enabled,  # noqa: SLF001
            strategy=BestEffortStrategy(),
            worker_count=1,
            playback_mode=PlaybackMode.FIXED_30,
            cache_settings=_DEFAULT_CACHE_SETTINGS,
            swapper_providers=("CUDAExecutionProvider",),  # providers_changed
        )

        # Provider change rebuilds the chain but must NOT re-key (stale risk).
        fake.set_chain.assert_called_once()
        assert fake.set_chain.call_args.kwargs["store"] is None
        ctrl._session_builder.build_store_for_chain.assert_not_called()  # noqa: SLF001
        assert ctrl._session_store is old_store  # noqa: SLF001 — unchanged
        old_store.close.assert_not_called()
        ctrl.shutdown()


class TestMemoryCacheHotResize:
    """A memory-cache-size change must hot-resize the LIVE buffer (so it takes
    effect without a restart / source change), and only when it actually
    changes."""

    def test_memory_size_change_hot_resizes_live_cache(
        self, widgets, fake_source_path, monkeypatch
    ):
        import dataclasses

        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=False,
            current_frame=10,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        bigger = dataclasses.replace(
            _DEFAULT_CACHE_SETTINGS, memory_max_bytes=256 * 1024 * 1024
        )
        ctrl.apply_session_config(
            swapper_params=FaceSwapperParams(),
            enhancer_params=FaceEnhancerParams(),
            enhancer_enabled=ctrl._enhancer_enabled,  # noqa: SLF001 — no chain change
            strategy=BestEffortStrategy(),
            worker_count=1,
            playback_mode=PlaybackMode.FIXED_30,
            cache_settings=bigger,
        )
        fake.set_memory_cache_bytes.assert_called_once_with(256 * 1024 * 1024)
        fake.set_chain.assert_not_called()
        ctrl.shutdown()

    def test_same_memory_size_does_not_resize(
        self, widgets, fake_source_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=False,
            current_frame=10,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        ctrl.apply_session_config(
            swapper_params=FaceSwapperParams(),
            enhancer_params=FaceEnhancerParams(),
            enhancer_enabled=ctrl._enhancer_enabled,  # noqa: SLF001
            strategy=BestEffortStrategy(),
            worker_count=1,
            playback_mode=PlaybackMode.FIXED_30,
            cache_settings=_DEFAULT_CACHE_SETTINGS,
        )
        fake.set_memory_cache_bytes.assert_not_called()
        ctrl.shutdown()
