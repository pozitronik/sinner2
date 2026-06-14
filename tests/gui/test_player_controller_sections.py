"""PlayerController section selection: set_sections pushes to the executor and
holds the authority; a playhead gap-skip re-seeks the audio backend."""
from __future__ import annotations

from unittest.mock import MagicMock

from sinner2.gui.player_controller import PlayerController
from sinner2.pipeline.sections import SectionSet


def test_set_sections_stores_and_pushes_to_executor():
    pc = PlayerController.__new__(PlayerController)
    pc._executor = MagicMock()  # noqa: SLF001
    s = SectionSet.of([(50, 120)])
    pc.set_sections(s)
    assert pc.sections() == s
    pc._executor.set_sections.assert_called_once_with(s)  # noqa: SLF001


def test_set_sections_without_executor_is_safe():
    pc = PlayerController.__new__(PlayerController)
    pc._executor = None  # noqa: SLF001
    s = SectionSet.of([(1, 5)])
    pc.set_sections(s)  # must not raise
    assert pc.sections() == s


def test_playhead_jump_reseeks_audio():
    pc = PlayerController.__new__(PlayerController)
    pc._audio = MagicMock()  # noqa: SLF001
    pc._target_fps = 25.0  # noqa: SLF001
    pc._on_playhead_jumped(100)  # noqa: SLF001
    # 100 / 25 fps = 4.0 s.
    pc._audio.seek_if_loaded.assert_called_once_with(4.0)  # noqa: SLF001


def test_playhead_jump_ignored_without_fps():
    pc = PlayerController.__new__(PlayerController)
    pc._audio = MagicMock()  # noqa: SLF001
    pc._target_fps = 0.0  # noqa: SLF001 — no timeline fps known yet
    pc._on_playhead_jumped(100)  # noqa: SLF001
    pc._audio.seek_if_loaded.assert_not_called()  # noqa: SLF001


def test_playhead_jump_ignores_sentinel():
    pc = PlayerController.__new__(PlayerController)
    pc._audio = MagicMock()  # noqa: SLF001
    pc._target_fps = 25.0  # noqa: SLF001
    pc._on_playhead_jumped(-1)  # noqa: SLF001 — initial observable value
    pc._audio.seek_if_loaded.assert_not_called()  # noqa: SLF001
