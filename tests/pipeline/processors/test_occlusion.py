"""Tests for the occlusion-mask composite (model-agnostic, stub masker)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sinner2.pipeline.processors.occlusion import apply_occlusion


class TestHardenOccluderMask:
    """The boundary-hardening shared by the XSeg and depth occluders: feather
    then remap [0.5, 1.0] → [0.0, 1.0]. On a uniform field GaussianBlur is the
    identity, so the remap is exact and checkable pointwise."""

    def test_remaps_band_to_unit_range(self):
        from sinner2.pipeline.processors.occlusion import _harden_occluder_mask

        def at(v):
            return float(_harden_occluder_mask(np.full((8, 8), v, np.float32))[0, 0])

        assert at(1.0) == pytest.approx(1.0)   # fully visible stays opaque
        assert at(0.5) == pytest.approx(0.0)   # band floor → transparent
        assert at(0.0) == pytest.approx(0.0)   # below floor clips to transparent
        assert at(0.75) == pytest.approx(0.5)  # midpoint → 0.5


def _face():
    return SimpleNamespace(
        kps=np.array(
            [[40, 45], [60, 45], [50, 55], [42, 62], [58, 62]], np.float32
        )
    )


class _StubMasker:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def face_mask(self, _aligned) -> np.ndarray:
        return self._mask


class TestApplyOcclusion:
    def test_composites_swapped_in_face_region_only(self):
        before = np.full((100, 100, 3), 50, np.uint8)
        swapped = np.full((100, 100, 3), 200, np.uint8)
        mask = np.zeros((512, 512), np.float32)
        mask[180:330, 180:330] = 1.0  # central (face) region of the aligned crop
        out = apply_occlusion(before, swapped, _face(), _StubMasker(mask))
        assert out.shape == before.shape
        assert out.max() > 50   # swapped present where the mask mapped back
        assert out.min() == 50  # original kept in the corners (mask 0)

    def test_roi_blend_matches_legacy_full_frame_blend(self):
        # The ROI restriction must reproduce the original full-frame composite
        # (same warp + blend; outside the warped square alpha is 0 → `before`).
        # Tolerance contract as for paste_back: translated-matrix warps can
        # shift isolated pixels one interpolation tap (cv2 fixed-point math).
        import cv2

        from sinner2.pipeline.processors.occlusion import (
            _ALIGN_SIZE,
            _FEATHER_SIGMA,
            _align_matrix,
        )

        rng = np.random.default_rng(8)
        before = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)
        swapped = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)
        mask512 = np.zeros((512, 512), np.float32)
        mask512[100:400, 120:380] = 1.0
        face = SimpleNamespace(
            kps=np.array(
                [[200, 100], [240, 100], [220, 122], [205, 140], [236, 140]],
                np.float32,
            )
        )

        def legacy() -> np.ndarray:
            m = _align_matrix(face.kps)
            mask = cv2.GaussianBlur(mask512, (0, 0), sigmaX=_FEATHER_SIGMA)
            m_inv = cv2.invertAffineTransform(m)
            h, w = before.shape[:2]
            alpha = cv2.warpAffine(mask, m_inv, (w, h))[..., None]
            return (
                swapped.astype(np.float32) * alpha
                + before.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)

        out = apply_occlusion(before, swapped, face, _StubMasker(mask512))
        expected = legacy()
        diff = np.abs(out.astype(np.int16) - expected.astype(np.int16))
        assert (diff > 0).mean() < 0.005
        assert diff.max() <= 8
        # _ALIGN_SIZE pinned so the legacy reproduction can't silently drift.
        assert _ALIGN_SIZE == 512

    def test_inputs_not_mutated(self):
        before = np.full((100, 100, 3), 50, np.uint8)
        swapped = np.full((100, 100, 3), 200, np.uint8)
        b0, s0 = before.copy(), swapped.copy()
        mask = np.ones((512, 512), np.float32)
        apply_occlusion(before, swapped, _face(), _StubMasker(mask))
        assert np.array_equal(before, b0)
        assert np.array_equal(swapped, s0)

    def test_falls_back_to_swapped_on_error(self):
        before = np.full((10, 10, 3), 50, np.uint8)
        swapped = np.full((10, 10, 3), 200, np.uint8)

        class _Boom:
            def face_mask(self, _a):
                raise RuntimeError("parser down")

        out = apply_occlusion(before, swapped, _face(), _Boom())
        assert np.array_equal(out, swapped)


class TestOnnxParserMasker:
    """facefusion's BiSeNet ONNX parsers: contract pinned (512 RGB, /255 then
    ImageNet mean/std, NCHW in; 19-class logits out, argmax → _FACE_CLASSES),
    thread-safe (shared session → no swapper lock)."""

    class _SpySession:
        """Returns logits that put class 1 (skin) everywhere except a corner
        block of class 0 (background); records the input blob."""

        def __init__(self) -> None:
            self.blobs: list[np.ndarray] = []

        def get_inputs(self):
            from types import SimpleNamespace

            return [SimpleNamespace(name="in")]

        def get_outputs(self):
            from types import SimpleNamespace

            return [SimpleNamespace(name="out")]

        def run(self, _names, feeds):
            blob = feeds["in"]
            self.blobs.append(blob)
            logits = np.zeros((1, 19, 512, 512), np.float32)
            logits[0, 1] = 1.0       # skin wins everywhere...
            logits[0, 0, :64, :64] = 2.0  # ...except a background corner
            return [logits]

    def _masker(self, monkeypatch):
        from sinner2.pipeline import model_cache
        from sinner2.pipeline.processors.occlusion import OnnxParserMasker

        session = self._SpySession()
        monkeypatch.setattr(
            model_cache, "get_onnx_session", lambda *a, **k: session
        )
        # occlusion.py imports get_onnx_session lazily inside setup() from
        # model_cache, so patching the source module is enough.
        m = OnnxParserMasker("bisenet_resnet_34.onnx")
        m.setup()
        return m, session

    def test_thread_safe_flag(self):
        from sinner2.pipeline.processors.occlusion import (
            OcclusionMasker,
            OnnxParserMasker,
        )

        assert OnnxParserMasker.thread_safe is True
        assert OcclusionMasker.thread_safe is False

    def test_preprocessing_contract(self, monkeypatch):
        m, session = self._masker(monkeypatch)
        aligned = np.zeros((512, 512, 3), np.uint8)
        aligned[:, :, 2] = 255  # pure red in BGR
        m.face_mask(aligned)
        blob = session.blobs[0]
        assert blob.shape == (1, 3, 512, 512)
        assert blob.dtype == np.float32
        # BGR→RGB flip: the red channel must land in channel 0, normalized
        # (1.0 - 0.485) / 0.229; blue/green carry (0 - mean) / std.
        assert np.isclose(blob[0, 0, 0, 0], (1.0 - 0.485) / 0.229, atol=1e-5)
        assert np.isclose(blob[0, 2, 0, 0], (0.0 - 0.406) / 0.225, atol=1e-5)

    def test_mask_from_argmax_classes(self, monkeypatch):
        m, _ = self._masker(monkeypatch)
        mask = m.face_mask(np.zeros((512, 512, 3), np.uint8))
        assert mask.shape == (512, 512)
        assert mask[300, 300] == 1.0  # skin (class 1) → facial region
        assert mask[10, 10] == 0.0    # background corner → reverted

    def test_face_mask_before_setup_raises(self):
        from sinner2.pipeline.processors.occlusion import OnnxParserMasker

        with pytest.raises(RuntimeError, match="before setup"):
            OnnxParserMasker("x.onnx").face_mask(np.zeros((512, 512, 3), np.uint8))

    def test_release_evicts_session(self, monkeypatch):
        from sinner2.pipeline import model_cache

        m, _ = self._masker(monkeypatch)
        evicted: list[str] = []
        monkeypatch.setattr(
            model_cache, "release_onnx_session",
            lambda name, providers=None: evicted.append(name),
        )
        m.release()
        assert evicted == ["bisenet_resnet_34.onnx"]
        assert m._session is None  # noqa: SLF001


class TestBuildParserMasker:
    def test_torch_parsers_build_occlusion_masker(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            OcclusionMasker,
            build_parser_masker,
        )

        for p in (FaceParser.BISENET, FaceParser.PARSENET):
            assert isinstance(build_parser_masker(p), OcclusionMasker)

    def test_onnx_parsers_build_onnx_masker(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            OnnxParserMasker,
            build_parser_masker,
        )

        m = build_parser_masker(FaceParser.BISENET_ONNX_34)
        assert isinstance(m, OnnxParserMasker)
        assert m._model_file == "bisenet_resnet_34.onnx"  # noqa: SLF001
        m18 = build_parser_masker(FaceParser.BISENET_ONNX_18)
        assert m18._model_file == "bisenet_resnet_18.onnx"  # noqa: SLF001

    def test_parser_model_file_covers_all_parsers(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            parser_model_file,
        )

        files = {parser_model_file(p) for p in FaceParser}
        assert files == {
            "parsing_bisenet.pth", "parsing_parsenet.pth",
            "bisenet_resnet_34.onnx", "bisenet_resnet_18.onnx",
        }


class _XsegSpySession:
    """Records the input blob; returns a constant occlusion probability map
    with an 'occluded' low-probability corner. Output shaped (1, 256, 256, 1)
    — facefusion indexes it [0][0]... [0] here, then clips/resizes."""

    def __init__(self, value: float = 1.0, low_corner: bool = True) -> None:
        self.blobs: list[np.ndarray] = []
        self._value = value
        self._low_corner = low_corner

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, _names, feeds):
        blob = feeds["input"]
        self.blobs.append(blob)
        mask = np.full((256, 256, 1), self._value, np.float32)
        if self._low_corner:
            mask[:64, :64] = 0.0  # an occluder in the top-left corner
        return [mask[None]]


class TestXsegOccluderMasker:
    def _masker(self, monkeypatch, sessions):
        from sinner2.pipeline import model_cache
        from sinner2.pipeline.processors.occlusion import (
            OccluderModel,
            XsegOccluderMasker,
        )

        it = iter(sessions)
        monkeypatch.setattr(
            model_cache, "get_onnx_session", lambda *a, **k: next(it)
        )
        model = (
            OccluderModel.XSEG_MANY if len(sessions) > 1 else OccluderModel.XSEG_1
        )
        m = XsegOccluderMasker(model)
        m.setup()
        return m

    def test_thread_safe(self):
        from sinner2.pipeline.processors.occlusion import XsegOccluderMasker

        assert XsegOccluderMasker.thread_safe is True

    def test_preprocessing_contract(self, monkeypatch):
        # facefusion verbatim: resize to 256, NHWC, BGR as-is, float32/255.
        session = _XsegSpySession()
        m = self._masker(monkeypatch, [session])
        aligned = np.zeros((512, 512, 3), np.uint8)
        aligned[:, :, 0] = 255  # pure blue in BGR — must stay channel 0
        m.face_mask(aligned)
        blob = session.blobs[0]
        assert blob.shape == (1, 256, 256, 3)
        assert blob.dtype == np.float32
        assert np.isclose(blob[0, 0, 0, 0], 1.0)  # blue / 255 in place (no flip)
        assert np.isclose(blob[0, 0, 0, 2], 0.0)

    def test_mask_resized_and_hardened(self, monkeypatch):
        m = self._masker(monkeypatch, [_XsegSpySession()])
        mask = m.face_mask(np.zeros((512, 512, 3), np.uint8))
        assert mask.shape == (512, 512)
        # Unoccluded area → 1.0 after the (clip(0.5,1)-0.5)*2 hardening;
        # the occluded corner → 0.0 (its blurred tail clipped at 0.5).
        assert mask[400, 400] == pytest.approx(1.0)
        assert mask[10, 10] == pytest.approx(0.0)

    def test_many_min_combines_all_three(self, monkeypatch):
        # xseg_many runs all three sessions and takes the per-pixel minimum:
        # one model flagging an occluder is enough to keep the original.
        sessions = [
            _XsegSpySession(low_corner=False),
            _XsegSpySession(low_corner=True),   # only this one sees the hand
            _XsegSpySession(low_corner=False),
        ]
        m = self._masker(monkeypatch, sessions)
        mask = m.face_mask(np.zeros((512, 512, 3), np.uint8))
        assert all(len(s.blobs) == 1 for s in sessions)
        assert mask[10, 10] == pytest.approx(0.0)   # min wins
        assert mask[400, 400] == pytest.approx(1.0)

    def test_release_evicts_every_session(self, monkeypatch):
        from sinner2.pipeline import model_cache

        m = self._masker(
            monkeypatch,
            [_XsegSpySession(), _XsegSpySession(), _XsegSpySession()],
        )
        evicted: list[str] = []
        monkeypatch.setattr(
            model_cache, "release_onnx_session",
            lambda name, providers=None: evicted.append(name),
        )
        m.release()
        assert evicted == ["xseg_1.onnx", "xseg_2.onnx", "xseg_3.onnx"]

    def test_face_mask_before_setup_raises(self):
        from sinner2.pipeline.processors.occlusion import XsegOccluderMasker

        with pytest.raises(RuntimeError, match="before setup"):
            XsegOccluderMasker().face_mask(np.zeros((512, 512, 3), np.uint8))


class TestDepthOccluderMasker:
    class _DepthSpySession:
        """Inverse-depth map: face plane at 10.0, a closer blob (30.0) in the
        top-left corner, a farther background strip (1.0) on the right."""

        def __init__(self) -> None:
            self.blobs: list[np.ndarray] = []

        def get_inputs(self):
            return [SimpleNamespace(name="pixel_values")]

        def get_outputs(self):
            return [SimpleNamespace(name="predicted_depth")]

        def run(self, _names, feeds):
            blob = feeds["pixel_values"]
            self.blobs.append(blob)
            depth = np.full((518, 518), 10.0, np.float32)
            depth[:130, :130] = 30.0   # occluder: much closer than the face
            depth[:, 390:] = 1.0       # background: farther
            return [depth[None]]

    def _masker(self, monkeypatch):
        from sinner2.pipeline import model_cache
        from sinner2.pipeline.processors.occlusion import DepthOccluderMasker

        session = self._DepthSpySession()
        monkeypatch.setattr(
            model_cache, "get_onnx_session", lambda *a, **k: session
        )
        m = DepthOccluderMasker()
        m.setup()
        return m, session

    def test_preprocessing_contract(self, monkeypatch):
        # 518x518, BGR→RGB, /255 then ImageNet mean/std, NCHW (from the HF
        # preprocessor config).
        m, session = self._masker(monkeypatch)
        aligned = np.zeros((512, 512, 3), np.uint8)
        aligned[:, :, 2] = 255  # pure red in BGR
        m.face_mask(aligned)
        blob = session.blobs[0]
        assert blob.shape == (1, 3, 518, 518)
        assert np.isclose(blob[0, 0, 0, 0], (1.0 - 0.485) / 0.229, atol=1e-5)

    def test_closer_pixels_masked_face_and_background_kept(self, monkeypatch):
        m, _ = self._masker(monkeypatch)
        mask = m.face_mask(np.zeros((512, 512, 3), np.uint8))
        assert mask.shape == (512, 512)
        assert mask[10, 10] == pytest.approx(0.0)    # closer blob → occluder
        assert mask[300, 200] == pytest.approx(1.0)  # face plane kept
        assert mask[300, 480] == pytest.approx(1.0)  # farther background kept

    def test_flat_depth_yields_no_occlusion(self, monkeypatch):
        from sinner2.pipeline import model_cache
        from sinner2.pipeline.processors.occlusion import DepthOccluderMasker

        class _Flat(self._DepthSpySession):
            def run(self, _names, feeds):
                return [np.full((1, 518, 518), 5.0, np.float32)]

        monkeypatch.setattr(
            model_cache, "get_onnx_session", lambda *a, **k: _Flat()
        )
        m = DepthOccluderMasker()
        m.setup()
        mask = m.face_mask(np.zeros((512, 512, 3), np.uint8))
        assert mask.min() == pytest.approx(1.0)  # no spread → no thresholding

    def test_builder_dispatches_depth(self):
        from sinner2.pipeline.processors.occlusion import (
            DepthOccluderMasker,
            FaceParser,
            OccluderModel,
            OcclusionMaskMode,
            build_occlusion_masker,
        )

        m = build_occlusion_masker(
            OcclusionMaskMode.OCCLUDER, FaceParser.BISENET, OccluderModel.DEPTH,
        )
        assert isinstance(m, DepthOccluderMasker)

    def test_occluder_model_files_depth(self):
        from sinner2.pipeline.processors.occlusion import (
            OccluderModel,
            occluder_model_files,
        )

        assert occluder_model_files(OccluderModel.DEPTH) == [
            "depth_anything_v2_small.onnx"
        ]


class TestCombinedMasker:
    class _Const:
        def __init__(self, value: float, thread_safe: bool = True) -> None:
            self._v = value
            self.thread_safe = thread_safe
            self.setups = 0
            self.releases = 0

        def setup(self):
            self.setups += 1

        def face_mask(self, _a):
            return np.full((512, 512), self._v, np.float32)

        def release(self):
            self.releases += 1

    def test_min_combination_and_lifecycle(self):
        from sinner2.pipeline.processors.occlusion import CombinedMasker

        a, b = self._Const(0.8), self._Const(0.3)
        c = CombinedMasker([a, b])
        c.setup()
        mask = c.face_mask(np.zeros((512, 512, 3), np.uint8))
        assert mask[0, 0] == pytest.approx(0.3)  # min of the parts
        c.release()
        assert (a.setups, b.setups, a.releases, b.releases) == (1, 1, 1, 1)

    def test_thread_safe_only_when_all_parts_are(self):
        from sinner2.pipeline.processors.occlusion import CombinedMasker

        assert CombinedMasker([self._Const(1), self._Const(1)]).thread_safe
        assert not CombinedMasker(
            [self._Const(1), self._Const(1, thread_safe=False)]
        ).thread_safe


class TestBuildOcclusionMasker:
    def test_region_builds_parser_only(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            OccluderModel,
            OcclusionMaskMode,
            OnnxParserMasker,
            build_occlusion_masker,
        )

        m = build_occlusion_masker(
            OcclusionMaskMode.REGION, FaceParser.BISENET_ONNX_34,
            OccluderModel.XSEG_1,
        )
        assert isinstance(m, OnnxParserMasker)

    def test_cache_wraps_the_masker(self):
        from sinner2.pipeline.processors.occlusion import (
            CachingMasker,
            FaceParser,
            OccluderModel,
            OcclusionMaskMode,
            OnnxParserMasker,
            build_occlusion_masker,
        )

        args = (
            OcclusionMaskMode.REGION, FaceParser.BISENET_ONNX_18,
            OccluderModel.XSEG_1,
        )
        cached = build_occlusion_masker(*args, cache=True)
        assert isinstance(cached, CachingMasker)
        assert isinstance(cached._inner, OnnxParserMasker)  # noqa: SLF001
        # Default (cache=False) is the bare masker — no behaviour change.
        assert not isinstance(
            build_occlusion_masker(*args), CachingMasker
        )

    def test_occluder_builds_xseg_only(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            OccluderModel,
            OcclusionMaskMode,
            XsegOccluderMasker,
            build_occlusion_masker,
        )

        m = build_occlusion_masker(
            OcclusionMaskMode.OCCLUDER, FaceParser.BISENET,
            OccluderModel.XSEG_2,
        )
        assert isinstance(m, XsegOccluderMasker)
        assert m._files == ["xseg_2.onnx"]  # noqa: SLF001

    def test_both_builds_combined(self):
        from sinner2.pipeline.processors.occlusion import (
            CombinedMasker,
            FaceParser,
            OccluderModel,
            OcclusionMaskMode,
            build_occlusion_masker,
        )

        m = build_occlusion_masker(
            OcclusionMaskMode.BOTH, FaceParser.BISENET_ONNX_18,
            OccluderModel.XSEG_MANY,
        )
        assert isinstance(m, CombinedMasker)
        assert m.thread_safe is True  # ONNX parser + xseg → both lock-free

    def test_both_with_torch_parser_not_thread_safe(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            OccluderModel,
            OcclusionMaskMode,
            build_occlusion_masker,
        )

        m = build_occlusion_masker(
            OcclusionMaskMode.BOTH, FaceParser.BISENET, OccluderModel.XSEG_1,
        )
        assert m.thread_safe is False  # torch parser serializes the combo

    def test_occluder_model_files(self):
        from sinner2.pipeline.processors.occlusion import (
            OccluderModel,
            occluder_model_files,
        )

        assert occluder_model_files(OccluderModel.XSEG_1) == ["xseg_1.onnx"]
        assert occluder_model_files(OccluderModel.XSEG_MANY) == [
            "xseg_1.onnx", "xseg_2.onnx", "xseg_3.onnx",
        ]


class TestRelease:
    def test_release_frees_cuda_and_nulls_model(self, monkeypatch):
        from unittest.mock import MagicMock

        import torch

        from sinner2.pipeline.processors.occlusion import OcclusionMasker

        m = OcclusionMasker()
        m._model = MagicMock()  # noqa: SLF001
        m._device_is_cuda = True  # noqa: SLF001  (set in setup() after the fix)
        empties: list[int] = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empties.append(1))
        m.release()
        assert m._model is None  # noqa: SLF001
        assert empties == [1]  # VRAM handed back


class _CountingMasker:
    """Records how many real forwards ran + returns a distinct mask per call."""

    thread_safe = True

    def __init__(self) -> None:
        self.calls = 0
        self.released = False

    def setup(self) -> None:
        pass

    def face_mask(self, _aligned) -> np.ndarray:
        self.calls += 1
        return np.full((512, 512), float(self.calls), np.float32)

    def release(self) -> None:
        self.released = True


def _solid(luma: int) -> np.ndarray:
    return np.full((512, 512, 3), luma, np.uint8)


class TestCachingMasker:
    def test_identical_crop_hits_cache(self):
        from sinner2.pipeline.processors.occlusion import CachingMasker

        inner = _CountingMasker()
        cm = CachingMasker(inner)
        m1 = cm.face_mask(_solid(100))
        m2 = cm.face_mask(_solid(100))
        assert inner.calls == 1  # the second call was served from the cache
        assert m2 is m1

    def test_near_static_within_bucket_reuses(self):
        from sinner2.pipeline.processors.occlusion import CachingMasker

        inner = _CountingMasker()
        cm = CachingMasker(inner)
        cm.face_mask(_solid(96))   # 96 // 16 == 6
        cm.face_mask(_solid(100))  # 100 // 16 == 6 → same bucket → the lag/reuse
        assert inner.calls == 1

    def test_real_movement_misses(self):
        from sinner2.pipeline.processors.occlusion import CachingMasker

        inner = _CountingMasker()
        cm = CachingMasker(inner)
        cm.face_mask(_solid(50))   # bucket 3
        cm.face_mask(_solid(200))  # bucket 12 → distinct → recompute
        assert inner.calls == 2

    def test_lru_evicts_oldest(self):
        from sinner2.pipeline.processors.occlusion import CachingMasker

        inner = _CountingMasker()
        cm = CachingMasker(inner, max_entries=2)
        cm.face_mask(_solid(16))   # bucket 1
        cm.face_mask(_solid(48))   # bucket 3
        cm.face_mask(_solid(80))   # bucket 5 → evicts bucket 1
        cm.face_mask(_solid(16))   # bucket 1 was evicted → recompute
        assert inner.calls == 4

    def test_thread_safe_mirrors_inner(self):
        from sinner2.pipeline.processors.occlusion import CachingMasker

        assert CachingMasker(_CountingMasker()).thread_safe is True

        class _NotSafe(_CountingMasker):
            thread_safe = False

        assert CachingMasker(_NotSafe()).thread_safe is False

    def test_release_clears_cache_and_releases_inner(self):
        from sinner2.pipeline.processors.occlusion import CachingMasker

        inner = _CountingMasker()
        cm = CachingMasker(inner)
        cm.face_mask(_solid(100))
        cm.release()
        assert inner.released is True
        assert len(cm._cache) == 0  # noqa: SLF001

    def test_setup_delegates_to_inner(self):
        from sinner2.pipeline.processors.occlusion import CachingMasker

        seen = []

        class _Inner(_CountingMasker):
            def setup(self):
                seen.append(1)

        CachingMasker(_Inner()).setup()
        assert seen == [1]
