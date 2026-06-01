"""Whole-body reach controller."""

from __future__ import annotations

from sim.constants import JOINT_SCHEMA
from sim.humanoid import HumanoidSim

CONTACT_Z: dict[str, float] = {
    "left_hand": 0.06,
    "right_hand": 0.06,
    "left_foot": 0.028,
    "right_foot": 0.028,
}


class ReachController:
    def __init__(self, sim: HumanoidSim) -> None:
        self._sim = sim

    def contact_z(self, limb: str) -> float:
        return CONTACT_Z.get(limb, 0.06)

    def target_xyz(self, limb: str, x: float, y: float, z_override: float | None = None) -> tuple[float, float, float]:
        return (float(x), float(y), float(self.contact_z(limb) if z_override is None else z_override))

    def reach_vector(self, limb: str, end_effector: dict[str, float], target_x: float, target_y: float) -> dict[str, float]:
        tz = self.contact_z(limb)
        return {
            "dx": round(target_x - end_effector["x"], 4),
            "dy": round(target_y - end_effector["y"], 4),
            "dz": round(tz - end_effector["z"], 4),
            "target_z": tz,
        }

    def primary_joints(self, limb: str) -> list[str]:
        side = "left" if limb.startswith("left") else "right"
        if limb.endswith("hand"):
            return [f"{side}_shoulder_pitch", f"{side}_shoulder_roll", f"{side}_elbow", "thoracic_pitch", "thoracic_yaw"]
        return [f"{side}_hip_pitch", f"{side}_hip_roll", f"{side}_hip_yaw", f"{side}_knee", f"{side}_ankle"]

    def suggest_targets(
        self,
        limb: str,
        target_x: float,
        target_y: float,
        *,
        ik_iterations: int = 12,
        anchor_limbs: list[str] | None = None,
        target_z: float | None = None,
    ) -> dict[str, float]:
        del ik_iterations, anchor_limbs
        targets = self._sim.get_joint_targets_deg()
        side = "left" if limb.startswith("left") else "right"
        dx = max(-0.6, min(0.6, target_x))
        dy = max(-0.6, min(0.6, target_y))
        if limb.endswith("hand"):
            targets[f"{side}_shoulder_pitch"] = float(max(-140.0, min(140.0, 110.0 * dy)))
            targets[f"{side}_shoulder_roll"] = float(max(-110.0, min(110.0, -100.0 * dx)))
            targets[f"{side}_elbow"] = float(max(-5.0, min(145.0, 60.0 + 60.0 * abs(dx))))
            targets["thoracic_pitch"] = float(max(-35.0, min(35.0, 45.0 * dy)))
            targets["thoracic_yaw"] = float(max(-45.0, min(45.0, -40.0 * dx)))
        else:
            tz = self.contact_z(limb) if target_z is None else target_z
            lift = 1.0 if tz > 0.05 else 0.0
            targets[f"{side}_hip_pitch"] = float(max(-120.0, min(45.0, -80.0 * dy + 15.0 * lift)))
            targets[f"{side}_hip_roll"] = float(max(-95.0, min(95.0, -95.0 * dx)))
            targets[f"{side}_knee"] = float(max(-5.0, min(135.0, 40.0 + 45.0 * lift)))
            targets[f"{side}_ankle"] = float(max(-35.0, min(35.0, -20.0 * dy)))
        return {
            name: float(max(spec["low"], min(spec["high"], targets.get(name, spec["neutral"]))))
            for name, spec in JOINT_SCHEMA.items()
        }

    def rehold_locked(self, locked: list[tuple[str, float, float]], *, skip_limb: str | None = None) -> None:
        del locked, skip_limb
