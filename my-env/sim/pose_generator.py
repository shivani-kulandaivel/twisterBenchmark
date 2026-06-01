"""Pose generation helpers for planning."""

from __future__ import annotations

import random

from sim.constants import JOINT_SCHEMA


class PoseGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def neutral_pose(self) -> dict[str, float]:
        return {name: spec["neutral"] for name, spec in JOINT_SCHEMA.items()}

    def sampled_pose(self, jitter_deg: float = 6.0) -> dict[str, float]:
        pose: dict[str, float] = {}
        for name, spec in JOINT_SCHEMA.items():
            v = spec["neutral"] + self._rng.uniform(-jitter_deg, jitter_deg)
            pose[name] = float(max(spec["low"], min(spec["high"], v)))
        return pose

    def generate(self, n: int = 8) -> list[dict[str, float]]:
        return [self.sampled_pose() for _ in range(max(1, n))]
