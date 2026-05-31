"""Tests for the CodeFormer backend — restore I/O, paste composite, and the
per-face enhance loop (stub session/analyser; no model/weights)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sinner2.pipeline.processors.codeformer import (
    CodeFormerBackend,
    _paste_face,
    _restore_aligned,
)


class _IdentitySession:
    """Returns the input as the 'restored' output (in the model's [-1,1] space).
    The input is already [-1,1] normalized, so this round-trips to the source."""

    def run(self, _names, feeds):
        return [feeds["input"]]


def test_restore_aligned_shape_and_dtype():
    aligned = np.random.randint(0, 255, (512, 512, 3), np.uint8)
    out = _restore_aligned(_IdentitySession(), aligned, 0.7)
    assert out.shape == (512, 512, 3)
    assert out.dtype == np.uint8


def test_paste_face_blends_center_keeps_corners():
    frame = np.full((100, 100, 3), 50, np.uint8)
    restored = np.full((512, 512, 3), 200, np.uint8)
    m = np.array([[5.12, 0, 0], [0, 5.12, 0]], np.float32)  # frame→512 scale
    out = _paste_face(frame, restored, m)
    assert out.shape == (100, 100, 3)
    assert out.max() > 50   # restored blended into the center
    assert out.min() == 50  # corners untouched (feathered mask is 0 there)


def test_enhance_restores_each_detected_face():
    backend = CodeFormerBackend()
    backend._session = _IdentitySession()  # noqa: SLF001
    backend._analyser = SimpleNamespace(  # noqa: SLF001
        analyse=lambda _img: [
            SimpleNamespace(
                kps=np.array(
                    [[40, 45], [60, 45], [50, 55], [42, 62], [58, 62]], np.float32
                )
            )
        ]
    )
    out = backend.enhance(np.full((100, 100, 3), 80, np.uint8))
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8


def test_enhance_before_setup_raises():
    import pytest

    with pytest.raises(RuntimeError, match="before setup"):
        CodeFormerBackend().enhance(np.zeros((4, 4, 3), np.uint8))


def test_release_evicts_session(monkeypatch):
    import sinner2.pipeline.processors.codeformer as cf

    evicted = []
    monkeypatch.setattr(cf, "release_onnx_session", evicted.append)
    backend = CodeFormerBackend()
    backend._session = _IdentitySession()  # noqa: SLF001
    backend._analyser = object()  # noqa: SLF001
    backend.release()
    assert backend._session is None  # noqa: SLF001
    assert backend._analyser is None  # noqa: SLF001
    assert evicted == [cf.MODEL_FILE]  # the exclusive CodeFormer session
