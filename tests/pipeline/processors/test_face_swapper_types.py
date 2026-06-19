"""Value-contract tests for the swapper enums.

TargetSex and RotationAngleSource are (str, Enum) members whose *string values*
are persisted in settings.json — and TargetSex's single-letter tokens are shared
with sinner1 for cross-version round-tripping. A rename of a value would silently
break every stored settings file, so these tests pin the values and the
value→member deserialization that settings loading relies on.
"""
from __future__ import annotations

import pytest

from sinner2.pipeline.processors.face_swapper_types import RotationAngleSource, TargetSex


class TestTargetSex:
    def test_values_are_the_persisted_tokens(self):
        assert {m.name: m.value for m in TargetSex} == {
            "BOTH": "B",
            "MALE": "M",
            "FEMALE": "F",
            "AS_SOURCE": "I",
        }

    @pytest.mark.parametrize(
        "token,member",
        [("B", TargetSex.BOTH), ("M", TargetSex.MALE), ("F", TargetSex.FEMALE), ("I", TargetSex.AS_SOURCE)],
    )
    def test_round_trips_from_stored_token(self, token, member):
        """A token read back from settings.json resolves to its member."""
        assert TargetSex(token) is member


class TestRotationAngleSource:
    def test_values_are_the_persisted_strings(self):
        assert {m.name: m.value for m in RotationAngleSource} == {
            "KEYPOINTS": "keypoints",
            "POSE": "pose",
            "LANDMARK_68": "landmark_68",
        }

    @pytest.mark.parametrize(
        "value,member",
        [
            ("keypoints", RotationAngleSource.KEYPOINTS),
            ("pose", RotationAngleSource.POSE),
            ("landmark_68", RotationAngleSource.LANDMARK_68),
        ],
    )
    def test_round_trips_from_stored_value(self, value, member):
        assert RotationAngleSource(value) is member
