"""Enums shared between the face swapper and its rotation-compensation helper.

Kept in their own module so `rotation_compensation` can use them without a
circular import back into `face_swapper` (which imports the helper).
"""
from enum import Enum


class TargetSex(str, Enum):
    """Which detected faces to swap based on insightface's sex
    classification. Single-letter values match sinner1's CLI tokens
    so settings files round-trip between versions."""

    BOTH = "B"          # Swap every detected face regardless of sex.
    MALE = "M"          # Only swap faces classified male.
    FEMALE = "F"        # Only swap faces classified female.
    AS_SOURCE = "I"     # Match the source face's sex ("as input").


class RotationAngleSource(str, Enum):
    """How rotation compensation measures a face's in-plane roll."""

    KEYPOINTS = "keypoints"  # from the eye landmarks (robust, always present)
    POSE = "pose"            # from insightface's 3D pose estimate (face.pose[2])
    LANDMARK_68 = "landmark_68"  # from the 2dfan4 eye-centre line (steadiest)
