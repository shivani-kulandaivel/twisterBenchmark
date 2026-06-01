"""World state schema for Twister humanoid benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CircleState:
    row: int
    col: int
    color: str
    x: float
    y: float
    occupied: bool
    available: bool


@dataclass
class JointState:
    angles_deg: dict[str, float]
    targets_deg: dict[str, float]
    limits_deg: dict[str, dict[str, float]]


@dataclass
class EEState:
    x: float
    y: float
    z: float
    on_mat: bool


@dataclass
class BalanceState:
    upright: bool
    fallen: bool
    com: tuple[float, float, float]
    support_polygon: list[tuple[float, float]]
    support_center: tuple[float, float]
    margin: float
    zmp_proxy: tuple[float, float]
    assist_strength: float


@dataclass
class ContactEvent:
    geom1: str
    geom2: str
    distance: float


@dataclass
class PerfSnapshot:
    ik_ms: float = 0.0
    balance_ms: float = 0.0
    physics_ms: float = 0.0
    decision_ms: float = 0.0


@dataclass
class WorldState:
    circles: list[CircleState]
    joints: JointState
    end_effectors: dict[str, EEState]
    balance: BalanceState
    contacts: list[ContactEvent]
    command: dict[str, Any] | None
    target_circle: dict[str, Any] | None
    move_history: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 1
    locked_limbs: list[dict[str, Any]] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)
    failure_status: str | None = None
    physics_timestep: float = 0.01
    sim_time: float = 0.0
    perf: PerfSnapshot = field(default_factory=PerfSnapshot)

    def to_observation(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["end_effectors"] = {
            k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
            for k, v in self.end_effectors.items()
        }
        return payload
