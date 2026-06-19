"""Single-source-of-truth guard for the shared combo catalogs.

`model_choices` is the one place the processor dropdowns are defined; the live
panel and the batch task form both consume it, so they can't drift. These tests
pin each catalog to its enum: a token added or renamed in the enum but not the
catalog (or vice versa) fails here — exactly the live↔batch drift the module
exists to prevent, now caught structurally instead of by manual review.
"""
from __future__ import annotations

import pytest

from sinner2.gui.widgets import model_choices as mc
from sinner2.pipeline.detectors import DetectorModel
from sinner2.pipeline.processors.face_enhancer import EnhancerModel
from sinner2.pipeline.processors.face_swapper import RotationAngleSource, SwapperModel
from sinner2.pipeline.processors.occlusion import (
    FaceParser,
    OccluderModel,
    OcclusionMaskMode,
)
from sinner2.pipeline.processors.upscaler import UpscalerModel

# (token, label) catalogs paired with the enum whose .value sequence they must equal.
_VALUE_FIRST = [
    (mc.SWAPPER_MODELS, SwapperModel),
    (mc.DETECTOR_MODELS, DetectorModel),
    (mc.ENHANCER_MODELS, EnhancerModel),
    (mc.UPSCALER_MODELS, UpscalerModel),
    (mc.OCCLUSION_PARSERS, FaceParser),
    (mc.OCCLUSION_MODES, OcclusionMaskMode),
    (mc.OCCLUDER_MODELS, OccluderModel),
]


class TestCatalogMatchesEnum:
    @pytest.mark.parametrize("catalog,enum", _VALUE_FIRST)
    def test_tokens_are_exactly_the_enum_values_in_order(self, catalog, enum):
        """Completeness + order: every enum member appears once, nothing extra."""
        assert [token for token, _label in catalog] == [m.value for m in enum]

    def test_rotation_sources_tokens_match_enum(self):
        # The rotation list is (label, token) — token is the second element.
        assert [token for _label, token in mc.ROTATION_SOURCES] == [
            m.value for m in RotationAngleSource
        ]


class TestCatalogSanity:
    @pytest.mark.parametrize("catalog,enum", _VALUE_FIRST)
    def test_labels_present_and_unique(self, catalog, enum):
        labels = [label for _token, label in catalog]
        assert all(labels), "every item needs a display label"
        assert len(set(labels)) == len(labels), "labels must be unique"

    @pytest.mark.parametrize("catalog,enum", _VALUE_FIRST)
    def test_tokens_unique(self, catalog, enum):
        tokens = [token for token, _label in catalog]
        assert len(set(tokens)) == len(tokens)
