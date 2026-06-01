"""Balance metrics and assist management."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sim.constants import DEFAULT_BALANCE_ASSIST_STRENGTH
from sim.humanoid import HumanoidSim


@dataclass
class BalanceMetrics:
    com: tuple[float, float, float]
    support_polygon: list[tuple[float, float]]
    support_center: tuple[float, float]
    margin: float
    zmp_proxy: tuple[float, float]
    upright: bool
    fallen: bool
    assist_strength: float


class BalanceController:
    """Thin controller around HumanoidSim support and CoM estimation."""

    def __init__(self, sim: HumanoidSim, assist_strength: float = DEFAULT_BALANCE_ASSIST_STRENGTH) -> None:
        self._sim = sim
        self._assist_strength = float(max(0.0, min(1.0, assist_strength)))
        self._sim.set_balance_assist_strength(self._assist_strength)

    def set_assist_strength(self, strength: float) -> None:
        self._assist_strength = float(max(0.0, min(1.0, strength)))
        self._sim.set_balance_assist_strength(self._assist_strength)

    def compute(self) -> BalanceMetrics:
        com = self._sim.com_xyz()
        support_center = self._sim.support_center_xy()
        polygon = self._estimate_support_polygon()
        margin = self._support_margin(com[0], com[1], polygon)
        zmp_proxy = support_center
        return BalanceMetrics(
            com=com,
            support_polygon=polygon,
            support_center=support_center,
            margin=margin,
            zmp_proxy=zmp_proxy,
            upright=self._sim.is_upright(),
            fallen=self._sim.has_fallen(),
            assist_strength=self._assist_strength,
        )

    def _estimate_support_polygon(self) -> list[tuple[float, float]]:
        feet = self._sim.get_end_effector_positions()
        pts = [
            (feet["left_foot"]["x"], feet["left_foot"]["y"]),
            (feet["right_foot"]["x"], feet["right_foot"]["y"]),
        ]
        # Basic rectangular support around feet.
        min_x = min(p[0] for p in pts) - 0.05
        max_x = max(p[0] for p in pts) + 0.05
        min_y = min(p[1] for p in pts) - 0.07
        max_y = max(p[1] for p in pts) + 0.07
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    @staticmethod
    def _support_margin(x: float, y: float, polygon: list[tuple[float, float]]) -> float:
        if not polygon:
            return -1.0
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = min(x - min_x, max_x - x)
        dy = min(y - min_y, max_y - y)
        return float(min(dx, dy))

    def stability_score(self) -> float:
        m = self.compute()
        if m.fallen:
            return 0.0
        return max(0.0, min(1.0, 0.5 + 4.0 * m.margin))
