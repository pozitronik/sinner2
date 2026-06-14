"""PlayerController.set_face_map: holds the authority + hot-applies to the live
executor."""
from __future__ import annotations

from unittest.mock import MagicMock

from sinner2.gui.player_controller import PlayerController
from sinner2.pipeline.face_map import FaceMap, Identity, normalize


def _map():
    return FaceMap(identities=(Identity("a", normalize([1, 0]), source_path="/s.png"),))


def test_set_face_map_stores_and_pushes():
    pc = PlayerController.__new__(PlayerController)
    pc._executor = MagicMock()  # noqa: SLF001
    fm = _map()
    pc.set_face_map(fm)
    assert pc.face_map() == fm
    pc._executor.set_face_map.assert_called_once_with(fm)  # noqa: SLF001


def test_set_face_map_without_executor_is_safe():
    pc = PlayerController.__new__(PlayerController)
    pc._executor = None  # noqa: SLF001
    fm = _map()
    pc.set_face_map(fm)  # must not raise
    assert pc.face_map() is fm
