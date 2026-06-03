"""Tests for the generic plain BFR-ONNX enhancer backend (GPEN / RestoreFormer++).

Same align → restore → paste shape as CodeFormer, minus the fidelity scalar.
Stub session/analyser; no model/weights. The model I/O contract (verified
against gpen_bfr_*.onnx): single input 'input' (1,3,512,512) RGB in [-1,1],
single output 'output' (1,3,512,512) in [-1,1].
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sinner2.pipeline.processors.bfr_onnx import (
    PlainBfrBackend,
    _paste_face,
    _restore_aligned,
)


class _IdentitySession:
    """Echo the input as the 'restored' output (already in [-1,1]), so the
    round-trip returns the source crop."""

    def run(self, _names, feeds):
        return [feeds["input"]]


def test_restore_aligned_shape_and_dtype():
    aligned = np.random.randint(0, 255, (512, 512, 3), np.uint8)
    out = _restore_aligned(_IdentitySession(), aligned)
    assert out.shape == (512, 512, 3)
    assert out.dtype == np.uint8


def test_restore_aligned_roundtrips_identity():
    # An identity model in [-1,1] space should return the input crop ~exactly
    # (within uint8 rounding).
    aligned = np.random.randint(0, 255, (512, 512, 3), np.uint8)
    out = _restore_aligned(_IdentitySession(), aligned)
    assert np.abs(out.astype(int) - aligned.astype(int)).max() <= 1


def test_restore_aligned_uses_dynamic_io_names():
    captured = {}

    class _NamedSession:
        def run(self, names, feeds):
            captured["out_names"] = names
            captured["in_keys"] = list(feeds.keys())
            return [feeds["x"]]

    aligned = np.random.randint(0, 255, (512, 512, 3), np.uint8)
    _restore_aligned(_NamedSession(), aligned, in_name="x", out_name="y")
    assert captured["out_names"] == ["y"]
    assert captured["in_keys"] == ["x"]


def test_paste_face_blends_center_keeps_corners():
    frame = np.full((100, 100, 3), 50, np.uint8)
    restored = np.full((512, 512, 3), 200, np.uint8)
    m = np.array([[5.12, 0, 0], [0, 5.12, 0]], np.float32)  # frame→512 scale
    out = _paste_face(frame, restored, m)
    assert out.shape == (100, 100, 3)
    assert out.max() > 50   # restored blended into the center
    assert out.min() == 50  # corners untouched (feather mask is 0 there)


def test_enhance_restores_each_detected_face():
    backend = PlainBfrBackend("gpen_bfr_512.onnx")
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


def test_enhance_best_effort_on_face_error():
    # A face whose alignment blows up must leave the frame untouched, not raise.
    backend = PlainBfrBackend("gpen_bfr_512.onnx")
    backend._session = _IdentitySession()  # noqa: SLF001
    backend._analyser = SimpleNamespace(  # noqa: SLF001
        analyse=lambda _img: [SimpleNamespace(kps=None)]  # bad kps → estimate raises
    )
    src = np.full((100, 100, 3), 80, np.uint8)
    out = backend.enhance(src)
    assert np.array_equal(out, src)


def test_enhance_before_setup_raises():
    backend = PlainBfrBackend("gpen_bfr_512.onnx")
    with pytest.raises(RuntimeError):
        backend.enhance(np.zeros((10, 10, 3), np.uint8))
