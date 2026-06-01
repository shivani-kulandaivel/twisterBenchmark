"""Twister game rules, mat layout, and validation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from sim.constants import (
    CIRCLE_SPACING,
    LIMBS,
    MAT_ORIGIN_X,
    MAT_ORIGIN_Y,
    PLACEMENT_RADIUS,
)

MAT_ROWS = ("red", "yellow", "green", "blue")
MAT_COLS = 6

SPINNER_OPTIONS: list[tuple[str, str]] = [
    (limb, color)
    for limb in LIMBS
    for color in MAT_ROWS
]


@dataclass
class Circle:
    row: int
    col: int
    color: str
    x: float
    y: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "color": self.color,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
        }


@dataclass
class TwisterCommand:
    limb: str
    color: str
    row: int
    col: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "limb": self.limb,
            "color": self.color,
            "circle": [self.row, self.col],
            "instruction": f"Place {self.limb.replace('_', ' ')} on {self.color}",
        }


@dataclass
class LockedLimb:
    limb: str
    row: int
    col: int
    color: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "limb": self.limb,
            "circle": [self.row, self.col],
            "color": self.color,
        }


class TwisterMat:
    def __init__(
        self,
        origin_x: float = MAT_ORIGIN_X,
        origin_y: float = MAT_ORIGIN_Y,
        spacing: float = CIRCLE_SPACING,
    ) -> None:
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.spacing = spacing
        self.circles = self._build_circles()

    def _build_circles(self) -> list[Circle]:
        circles: list[Circle] = []
        for row, color in enumerate(MAT_ROWS):
            for col in range(MAT_COLS):
                x = self.origin_x + (col - (MAT_COLS - 1) / 2) * self.spacing
                y = self.origin_y + (row - (len(MAT_ROWS) - 1) / 2) * self.spacing
                circles.append(Circle(row=row, col=col, color=color, x=x, y=y))
        return circles

    def circle_at(self, row: int, col: int) -> Circle:
        for circle in self.circles:
            if circle.row == row and circle.col == col:
                return circle
        raise ValueError(f"Invalid circle: row={row}, col={col}")

    def circles_by_color(self, color: str) -> list[Circle]:
        return [c for c in self.circles if c.color == color]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": len(MAT_ROWS),
            "cols": MAT_COLS,
            "row_colors": list(MAT_ROWS),
            "circles": [c.to_dict() for c in self.circles],
        }


class Spinner:
    def __init__(self, rng: random.Random, reachability_checker: Any | None = None) -> None:
        self._rng = rng
        self._reachability_checker = reachability_checker

    def spin(
        self,
        mat: TwisterMat,
        *,
        limb_weights: dict[str, float] | None = None,
        forbidden_circles: set[tuple[int, int]] | None = None,
        limb_positions: dict[str, dict[str, float]] | None = None,
        locked_limbs: list[str] | None = None,
    ) -> TwisterCommand:
        """Sample a command with reachability-aware rejection.

        The previous fully-uniform spinner generated many physically impossible
        circles for the current humanoid. This keeps randomness but avoids
        pathological commands outside the practical workspace.
        """
        options = SPINNER_OPTIONS
        for _ in range(40):
            limb, color = self._weighted_option(options, limb_weights)
            candidates = [
                c for c in mat.circles_by_color(color)
                if self._is_reasonably_reachable(limb, c)
                and self._ik_reachable(limb, c)
                and (forbidden_circles is None or (c.row, c.col) not in forbidden_circles)
            ]
            candidates = self._filter_crossbody(limb, candidates, locked_limbs)
            if candidates:
                circle = self._pick_nearest_candidate(limb, candidates, limb_positions)
                return TwisterCommand(limb=limb, color=color, row=circle.row, col=circle.col)

        # Fallback: pick from globally easiest cells.
        feasible: list[tuple[str, str, Circle]] = []
        for limb, color in options:
            for circle in mat.circles_by_color(color):
                if (
                    self._is_reasonably_reachable(limb, circle)
                    and self._ik_reachable(limb, circle)
                    and (forbidden_circles is None or (circle.row, circle.col) not in forbidden_circles)
                ):
                    filtered = self._filter_crossbody(limb, [circle], locked_limbs)
                    if filtered:
                        feasible.append((limb, color, circle))
        if feasible:
            limb, color, circle = self._rng.choice(feasible)
            return TwisterCommand(limb=limb, color=color, row=circle.row, col=circle.col)

        # Last resort: preserve old behavior.
        limb, color = self._weighted_option(options, limb_weights)
        circle = self._rng.choice(mat.circles_by_color(color))
        return TwisterCommand(limb=limb, color=color, row=circle.row, col=circle.col)

    def _pick_nearest_candidate(
        self,
        limb: str,
        candidates: list[Circle],
        limb_positions: dict[str, dict[str, float]] | None,
    ) -> Circle:
        if not limb_positions or limb not in limb_positions:
            return self._rng.choice(candidates)
        pos = limb_positions[limb]
        ranked = sorted(
            candidates,
            key=lambda c: math.hypot(c.x - float(pos["x"]), c.y - float(pos["y"])),
        )
        pool = ranked[: min(3, len(ranked))]
        return self._rng.choice(pool)

    @staticmethod
    def _filter_crossbody(
        limb: str,
        candidates: list[Circle],
        locked_limbs: list[str] | None,
    ) -> list[Circle]:
        if not locked_limbs or not candidates:
            return candidates
        locked = set(locked_limbs)
        filtered: list[Circle] = []
        for circle in candidates:
            if limb == "left_hand" and "right_foot" in locked and circle.x > 0.15:
                continue
            if limb == "right_hand" and "left_foot" in locked and circle.x < -0.15:
                continue
            if limb == "left_hand" and "left_foot" in locked and circle.x < -0.28:
                continue
            if limb == "right_hand" and "right_foot" in locked and circle.x > 0.28:
                continue
            filtered.append(circle)
        return filtered or candidates

    def _weighted_option(
        self,
        options: list[tuple[str, str]],
        limb_weights: dict[str, float] | None,
    ) -> tuple[str, str]:
        if not limb_weights:
            return self._rng.choice(options)
        weights = [max(0.0, float(limb_weights.get(limb, 1.0))) for limb, _ in options]
        if sum(weights) <= 0.0:
            return self._rng.choice(options)
        return self._rng.choices(options, weights=weights, k=1)[0]

    def _ik_reachable(self, limb: str, circle: Circle) -> bool:
        if self._reachability_checker is None:
            return True
        try:
            result = self._reachability_checker.can_reach(limb, circle.x, circle.y)
            return bool(result.feasible)
        except Exception:
            return False

    @staticmethod
    def _is_reasonably_reachable(limb: str, circle: Circle) -> bool:
        x, y = circle.x, circle.y
        r = math.hypot(x, y)
        if limb.endswith("foot"):
            # Feet are the current bottleneck; keep commands in a tighter
            # reachable envelope so the controller can produce clear motion.
            if r > 0.34 or abs(y) > 0.22:
                return False
            if limb == "left_foot" and x < -0.18:
                return False
            if limb == "right_foot" and x > 0.18:
                return False
            return True
        # Hands can reach forward/lateral circles but behind-the-body commands
        # are unstable with the current two-foot balance regime.
        if r > 0.44 or y < -0.10:
            return False
        # Keep hands mostly on their natural side to avoid impossible crossover.
        if limb == "left_hand" and x < -0.12:
            return False
        if limb == "left_hand" and x > 0.42:
            return False
        if limb == "right_hand" and x > 0.12:
            return False
        if limb == "right_hand" and x < -0.42:
            return False
        return True


def horizontal_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def placement_error(
    end_effector: dict[str, float],
    target: Circle,
) -> float:
    return horizontal_distance(end_effector["x"], end_effector["y"], target.x, target.y)


def is_placed(
    end_effector: dict[str, float],
    target: Circle,
    radius: float = PLACEMENT_RADIUS,
) -> bool:
    return placement_error(end_effector, target) <= radius


def end_effector_on_mat(end_effector: dict[str, float], z_threshold: float = 0.25) -> bool:
    return end_effector["z"] <= z_threshold


@dataclass
class ConstraintTracker:
    locked: list[LockedLimb] = field(default_factory=list)
    manager: Any | None = None

    def lock(self, command: TwisterCommand) -> None:
        for item in self.locked:
            if item.limb == command.limb:
                item.row = command.row
                item.col = command.col
                item.color = command.color
                return
        self.locked.append(
            LockedLimb(
                limb=command.limb,
                row=command.row,
                col=command.col,
                color=command.color,
            )
        )
        if self.manager is not None:
            self.manager.lock(
                command.limb,
                mat_circle_x(command.row, command.col),
                mat_circle_y(command.row, command.col),
                f"{command.color}:{command.row}:{command.col}",
            )

    def validate_locked(
        self,
        end_effectors: dict[str, dict[str, float]],
        mat: TwisterMat,
        radius: float = PLACEMENT_RADIUS,
        ignore_limb: str | None = None,
    ) -> list[str]:
        violations: list[str] = []
        for item in self.locked:
            if ignore_limb is not None and item.limb == ignore_limb:
                continue
            circle = mat.circle_at(item.row, item.col)
            limb_pos = end_effectors[item.limb]
            if not is_placed(limb_pos, circle, radius):
                violations.append(item.limb)
        return violations

    def to_dict(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.locked]


def mat_circle_x(row: int, col: int) -> float:
    del row
    return MAT_ORIGIN_X + (col - (MAT_COLS - 1) / 2) * CIRCLE_SPACING


def mat_circle_y(row: int, col: int) -> float:
    del col
    return MAT_ORIGIN_Y + (row - (len(MAT_ROWS) - 1) / 2) * CIRCLE_SPACING
