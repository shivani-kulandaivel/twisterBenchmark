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
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def spin(self, mat: TwisterMat) -> TwisterCommand:
        limb, color = self._rng.choice(SPINNER_OPTIONS)
        circle = self._rng.choice(mat.circles_by_color(color))
        return TwisterCommand(limb=limb, color=color, row=circle.row, col=circle.col)


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

    def validate_locked(
        self,
        end_effectors: dict[str, dict[str, float]],
        mat: TwisterMat,
        radius: float = PLACEMENT_RADIUS,
    ) -> list[str]:
        violations: list[str] = []
        for item in self.locked:
            circle = mat.circle_at(item.row, item.col)
            limb_pos = end_effectors[item.limb]
            if not is_placed(limb_pos, circle, radius):
                violations.append(item.limb)
        return violations

    def to_dict(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.locked]
