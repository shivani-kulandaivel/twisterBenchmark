"""Motion planner to smoothly track target joint poses."""

from __future__ import annotations

from dataclasses import dataclass

from sim.humanoid import HumanoidSim


@dataclass
class MotionPlan:
    steps: list[dict[str, float]]


class MotionPlanner:
    def __init__(self, sim: HumanoidSim) -> None:
        self._sim = sim

    def interpolate_to(self, target_pose: dict[str, float], steps: int = 8) -> MotionPlan:
        current = self._sim.get_joint_targets_deg()
        seq: list[dict[str, float]] = []
        for i in range(1, max(2, steps) + 1):
            alpha = i / max(2, steps)
            seq.append({k: (1 - alpha) * current[k] + alpha * v for k, v in target_pose.items() if k in current})
        return MotionPlan(steps=seq)
