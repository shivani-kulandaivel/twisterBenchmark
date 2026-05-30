"""Joint schema and simulation constants for the Twister humanoid."""

from __future__ import annotations

# Human-readable joint names mapped to MJCF joint names.
JOINT_SCHEMA: dict[str, dict[str, float]] = {
    "abdomen_pitch": {"low": -30.0, "high": 30.0, "neutral": 0.0},
    "left_shoulder_pitch": {"low": -120.0, "high": 120.0, "neutral": 0.0},
    "left_shoulder_roll": {"low": -90.0, "high": 90.0, "neutral": -10.0},
    "left_elbow": {"low": 0.0, "high": 150.0, "neutral": 10.0},
    "right_shoulder_pitch": {"low": -120.0, "high": 120.0, "neutral": 0.0},
    "right_shoulder_roll": {"low": -90.0, "high": 90.0, "neutral": 10.0},
    "right_elbow": {"low": 0.0, "high": 150.0, "neutral": 10.0},
    "left_hip_pitch": {"low": -120.0, "high": 45.0, "neutral": -5.0},
    "left_hip_roll": {"low": -45.0, "high": 45.0, "neutral": 0.0},
    "left_knee": {"low": 0.0, "high": 150.0, "neutral": 20.0},
    "left_ankle": {"low": -45.0, "high": 45.0, "neutral": -10.0},
    "right_hip_pitch": {"low": -120.0, "high": 45.0, "neutral": -5.0},
    "right_hip_roll": {"low": -45.0, "high": 45.0, "neutral": 0.0},
    "right_knee": {"low": 0.0, "high": 150.0, "neutral": 20.0},
    "right_ankle": {"low": -45.0, "high": 45.0, "neutral": -10.0},
}

MJCF_JOINT_NAMES: dict[str, str] = {name: name for name in JOINT_SCHEMA}

LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")

END_EFFECTOR_SITES: dict[str, str] = {
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "left_foot": "left_foot",
    "right_foot": "right_foot",
}

TORSO_BODY = "torso"

PHYSICS_SUBSTEPS = 20
PHYSICS_TIMESTEP = 0.01

# Fall detection thresholds (tilt-based; pelvis is fixed in v1 MJCF).
MIN_TORSO_HEIGHT = 0.85
MAX_TORSO_TILT_DEG = 55.0

# Placement validation.
PLACEMENT_RADIUS = 0.08
MAT_ORIGIN_X = 0.0
MAT_ORIGIN_Y = 0.0
CIRCLE_SPACING = 0.35

MAX_DELTA_DEG = 15.0
