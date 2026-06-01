"""Joint schema and simulation constants for the Twister humanoid."""

from __future__ import annotations

# Human-readable joint names mapped to MJCF joint names.
JOINT_SCHEMA: dict[str, dict[str, float]] = {
    "lumbar_pitch": {"low": -35.0, "high": 35.0, "neutral": 0.0},
    "lumbar_roll": {"low": -25.0, "high": 25.0, "neutral": 0.0},
    "thoracic_yaw": {"low": -45.0, "high": 45.0, "neutral": 0.0},
    "thoracic_pitch": {"low": -35.0, "high": 35.0, "neutral": 0.0},
    "left_shoulder_pitch": {"low": -140.0, "high": 140.0, "neutral": 0.0},
    "left_shoulder_roll": {"low": -110.0, "high": 30.0, "neutral": 0.0},
    "left_elbow": {"low": -5.0, "high": 145.0, "neutral": 5.0},
    "right_shoulder_pitch": {"low": -140.0, "high": 140.0, "neutral": 0.0},
    "right_shoulder_roll": {"low": -30.0, "high": 110.0, "neutral": 0.0},
    "right_elbow": {"low": -5.0, "high": 145.0, "neutral": 5.0},
    "left_hip_pitch": {"low": -120.0, "high": 45.0, "neutral": 0.0},
    "left_hip_roll": {"low": -95.0, "high": 95.0, "neutral": 0.0},
    "left_hip_yaw": {"low": -45.0, "high": 45.0, "neutral": 0.0},
    "left_knee": {"low": -5.0, "high": 135.0, "neutral": 0.0},
    "left_ankle": {"low": -35.0, "high": 35.0, "neutral": 0.0},
    "right_hip_pitch": {"low": -120.0, "high": 45.0, "neutral": 0.0},
    "right_hip_roll": {"low": -95.0, "high": 95.0, "neutral": 0.0},
    "right_hip_yaw": {"low": -45.0, "high": 45.0, "neutral": 0.0},
    "right_knee": {"low": -5.0, "high": 135.0, "neutral": 0.0},
    "right_ankle": {"low": -35.0, "high": 35.0, "neutral": 0.0},
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

# Per-physics-substep cap on joint target change (degrees) for fluid motion.
SMOOTH_MAX_DELTA_DEG = 4.0

# Balance-assist (virtual model control) gains: PD on pelvis to keep the
# center of mass over the support polygon and the trunk upright.
BALANCE_COM_KP = 1500.0
BALANCE_COM_KD = 240.0
BALANCE_ROT_KP = 450.0
BALANCE_ROT_KD = 110.0
BALANCE_F_MAX = 700.0
BALANCE_T_MAX = 260.0

# Fall detection thresholds (tilt-based; floating pelvis with free joint).
MIN_TORSO_HEIGHT = 0.55
MAX_TORSO_TILT_DEG = 55.0

# Placement validation.
PLACEMENT_RADIUS = 0.12
MAT_ORIGIN_X = 0.0
MAT_ORIGIN_Y = 0.0
CIRCLE_SPACING = 0.20

MAX_DELTA_DEG = 8.0
