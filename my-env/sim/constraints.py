"""Constraint management for locked Twister limbs."""

from __future__ import annotations

from dataclasses import dataclass, field

from sim.constants import PLACEMENT_RADIUS
from sim.humanoid import HumanoidSim


@dataclass
class LockedConstraint:
    limb: str
    x: float
    y: float
    label: str


@dataclass
class ConstraintStatus:
    violations: list[str] = field(default_factory=list)


class ConstraintManager:
    """Physics-backed interface; currently enforces via monitoring + slip detection."""

    def __init__(self, sim: HumanoidSim) -> None:
        self._sim = sim
        self._locked: dict[str, LockedConstraint] = {}

    def lock(self, limb: str, x: float, y: float, label: str) -> None:
        self._locked[limb] = LockedConstraint(limb=limb, x=float(x), y=float(y), label=label)

    def unlock(self, limb: str) -> None:
        self._locked.pop(limb, None)

    def unlock_all(self) -> None:
        self._locked.clear()

    def active_constraints(self) -> list[str]:
        return [f"{c.limb}:{c.label}" for c in self._locked.values()]

    def locked_as_dicts(self) -> list[dict[str, float | str]]:
        return [
            {"limb": c.limb, "x": c.x, "y": c.y, "label": c.label}
            for c in self._locked.values()
        ]

    def validate(self, ignore_limb: str | None = None) -> ConstraintStatus:
        ee = self._sim.get_end_effector_positions()
        violations: list[str] = []
        for limb, c in self._locked.items():
            if limb == ignore_limb:
                continue
            err = ((ee[limb]["x"] - c.x) ** 2 + (ee[limb]["y"] - c.y) ** 2) ** 0.5
            if err > PLACEMENT_RADIUS:
                violations.append(limb)
        return ConstraintStatus(violations=violations)
