"""Tests for the processing-visualiser heatmap bar (QFrameStateBar)."""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt

from sinner2.gui.widgets.frame_state_bar import QFrameStateBar
from sinner2.pipeline.realtime.frame_state import FrameState


def _states(*values: FrameState) -> bytes:
    return bytes(int(v) for v in values)


class TestFrameAt:
    def test_maps_x_to_frame(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(100, 16)
        bar.set_data(_states(*([FrameState.NOT_REACHED] * 200)), 200)
        assert bar._frame_at(0) == 0       # noqa: SLF001
        assert bar._frame_at(50) == 100    # noqa: SLF001 — halfway
        assert bar._frame_at(99) == 198    # noqa: SLF001

    def test_clamps_within_bounds(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(100, 16)
        bar.set_data(_states(*([FrameState.NOT_REACHED] * 10)), 10)
        assert bar._frame_at(99999) == 9   # noqa: SLF001 — last frame
        assert bar._frame_at(-5) == 0      # noqa: SLF001

    def test_zero_frames_is_zero(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        assert bar._frame_at(50) == 0      # noqa: SLF001


class TestClickToSeek:
    def test_left_click_emits_seek_for_frame(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(100, 16)
        bar.set_data(_states(*([FrameState.READY_MEM] * 200)), 200)
        with qtbot.waitSignal(bar.seekRequested, timeout=1000) as blocker:
            qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(50, 8))
        assert blocker.args[0] == 100

    def test_no_emit_without_frames(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(100, 16)
        fired: list[int] = []
        bar.seekRequested.connect(fired.append)
        qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(50, 8))
        assert fired == []  # nothing loaded → no seek


class TestPaint:
    def test_paints_mixed_states_without_crashing(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(120, 16)
        bar.set_data(
            _states(
                FrameState.READY_MEM, FrameState.READY_DISK, FrameState.PROCESSING,
                FrameState.QUEUED, FrameState.SKIPPED, FrameState.INVALID,
                FrameState.NOT_REACHED,
            ),
            7,
        )
        bar.set_playhead(2)
        assert not bar.grab().isNull()  # forces paintEvent; must not raise

    def test_paints_long_clip_binned_without_crashing(self, qtbot):
        # More frames than pixels → proportional binning path.
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(80, 16)
        pattern = [FrameState.READY_MEM, FrameState.SKIPPED, FrameState.PROCESSING]
        bar.set_data(_states(*[pattern[i % 3] for i in range(5000)]), 5000)
        assert not bar.grab().isNull()

    def test_paints_empty_without_crashing(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(80, 16)
        assert not bar.grab().isNull()  # no data → just background

    def test_ignores_out_of_range_state_bytes(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(40, 16)
        bar.set_data(bytes([99, 250, int(FrameState.READY_MEM)]), 3)  # bad bytes
        assert not bar.grab().isNull()


class TestClear:
    def test_clear_resets(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(40, 16)
        bar.set_data(_states(FrameState.READY_MEM), 1)
        bar.set_playhead(0)
        bar.clear()
        assert bar._frame_count == 0       # noqa: SLF001
        assert bar._states == b""          # noqa: SLF001
        assert bar._playhead == -1         # noqa: SLF001
        assert not bar.grab().isNull()


class TestProblemMarkers:
    """The bar overlays a marker on columns covering a no-face (ABSENT) frame,
    fed via the optional `faces` arg, and paints without error."""

    def test_set_data_stores_faces_and_paints(self, qtbot):
        from sinner2.pipeline.realtime.frame_state import FaceMark

        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(120, 16)
        states = _states(*([FrameState.READY_MEM] * 6))
        faces = bytes(
            int(m) for m in (
                FaceMark.PRESENT, FaceMark.PRESENT, FaceMark.ABSENT,
                FaceMark.PRESENT, FaceMark.ABSENT, FaceMark.UNKNOWN,
            )
        )
        bar.set_data(states, 6, faces)
        assert bar._faces == faces  # noqa: SLF001
        bar.grab()  # force a paint — must not raise with the overlay

    def test_faces_optional(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.set_data(_states(*([FrameState.READY_MEM] * 4)), 4)  # no faces arg
        assert bar._faces == b""  # noqa: SLF001
        bar.grab()


class TestContextMenu:
    """Right-click cache actions surfaced from the visualiser bar."""

    @staticmethod
    def _actions(bar):
        return {a.text(): a for a in bar._build_context_menu().actions()}  # noqa: SLF001

    def test_menu_offers_both_clear_actions(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.set_data(_states(*([FrameState.READY_DISK] * 10)), 10)
        actions = self._actions(bar)
        assert "Clear session cache" in actions
        assert "Clear all caches…" in actions

    def test_clear_session_action_emits(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.set_data(_states(*([FrameState.READY_DISK] * 10)), 10)
        act = self._actions(bar)["Clear session cache"]
        with qtbot.waitSignal(bar.clearSessionCacheRequested, timeout=1000):
            act.trigger()

    def test_clear_all_action_emits(self, qtbot):
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.set_data(_states(*([FrameState.READY_DISK] * 10)), 10)
        act = self._actions(bar)["Clear all caches…"]
        with qtbot.waitSignal(bar.clearAllCachesRequested, timeout=1000):
            act.trigger()

    def test_no_menu_without_a_session(self, qtbot):
        from PySide6.QtGui import QContextMenuEvent

        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        # frame_count == 0 (no data): the menu must NOT pop (it would otherwise
        # block on exec()) and no cache action fires.
        fired: list[int] = []
        bar.clearSessionCacheRequested.connect(lambda: fired.append(1))
        bar.clearAllCachesRequested.connect(lambda: fired.append(1))
        ev = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            QPoint(5, 5),
            bar.mapToGlobal(QPoint(5, 5)),
        )
        bar.contextMenuEvent(ev)  # frame_count == 0 → super(), no exec, no emit
        assert fired == []


class TestRenderCache:
    def test_playhead_move_reuses_cached_stack(self, qtbot):
        # A playhead move must NOT rebuild the per-column state stack — that was
        # the per-frame GUI-thread cost. set_data / clear invalidate it.
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(50, 16)
        bar.set_data(_states(*([FrameState.READY_MEM] * 100)), 100)
        assert bar._cache is None  # noqa: SLF001 — invalidated by set_data
        bar.grab()  # force a paint → builds the cached stack
        cached = bar._cache  # noqa: SLF001
        assert cached is not None
        bar.set_playhead(10)
        assert bar._cache is cached  # noqa: SLF001 — same pixmap, not rebuilt
        bar.set_data(_states(*([FrameState.QUEUED] * 100)), 100)
        assert bar._cache is None  # noqa: SLF001 — data change invalidates

    def test_set_data_skips_rerender_when_unchanged(self, qtbot):
        # The 20 Hz visualiser polls set_data every tick; an unchanged snapshot
        # must NOT invalidate the cache (no per-column re-render).
        bar = QFrameStateBar()
        qtbot.addWidget(bar)
        bar.resize(50, 16)
        states = _states(*([FrameState.READY_MEM] * 10))
        bar.set_data(states, 10)
        bar.grab()  # build the cache
        cached = bar._cache  # noqa: SLF001
        assert cached is not None
        bar.set_data(bytes(states), 10)  # value-identical snapshot → skip
        assert bar._cache is cached  # noqa: SLF001 — cache NOT invalidated
        bar.set_data(_states(*([FrameState.QUEUED] * 10)), 10)  # changed
        assert bar._cache is None  # noqa: SLF001 — invalidated
