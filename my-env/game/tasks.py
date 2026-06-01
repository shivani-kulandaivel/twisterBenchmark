"""Episode task logic for phased Twister benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.rewards import reward_breakdown
from game.twister import ConstraintTracker, Spinner, TwisterCommand, TwisterMat, is_placed, placement_error
from sim.balance_controller import BalanceController
from sim.constants import JOINT_SCHEMA, LIMBS, PHYSICS_TIMESTEP, PLACEMENT_RADIUS
from sim.humanoid import HumanoidSim
from sim.reach_controller import ReachController
from state.world_state import (
    BalanceState,
    CircleState,
    ContactEvent,
    EEState,
    JointState,
    PerfSnapshot,
    WorldState,
)


@dataclass
class Phase1State:
    command: TwisterCommand
    step_count: int = 0
    placement_errors: list[float] = field(default_factory=list)


@dataclass
class Phase2State:
    mat: TwisterMat
    spinner: Spinner
    command: TwisterCommand
    constraints: ConstraintTracker = field(default_factory=ConstraintTracker)
    turn: int = 1
    turns_completed: int = 0
    step_count: int = 0
    placement_errors: list[float] = field(default_factory=list)
    max_turns: int = 10
    limb_weights: dict[str, float] | None = None

    def advance_turn(self, end_effectors: dict[str, dict[str, float]] | None = None) -> None:
        self.turns_completed += 1
        self.turn += 1
        forbidden = {(item.row, item.col) for item in self.constraints.locked if item.limb != self.command.limb}
        locked_limbs = [item.limb for item in self.constraints.locked]
        self.command = self.spinner.spin(
            self.mat,
            limb_weights=self.limb_weights,
            forbidden_circles=forbidden,
            limb_positions=end_effectors,
            locked_limbs=locked_limbs,
        )


def build_world_state(
    *,
    phase: int,
    sim: HumanoidSim,
    mat: TwisterMat,
    command: TwisterCommand,
    turn: int,
    locked_limbs: list[dict[str, Any]],
    target_circle: dict[str, Any],
    balance: BalanceController,
    perf: PerfSnapshot,
    failure_status: str | None = None,
) -> WorldState:
    ee = sim.get_end_effector_positions()
    ee_state = {
        limb: EEState(x=v["x"], y=v["y"], z=v["z"], on_mat=v["z"] <= 0.25)
        for limb, v in ee.items()
    }
    b = balance.compute()
    occupied = {(item["circle"][0], item["circle"][1]) for item in locked_limbs if "circle" in item}
    circles = [
        CircleState(
            row=c.row,
            col=c.col,
            color=c.color,
            x=c.x,
            y=c.y,
            occupied=(c.row, c.col) in occupied,
            available=(c.row, c.col) not in occupied,
        )
        for c in mat.circles
    ]
    ws = WorldState(
        circles=circles,
        joints=JointState(
            angles_deg=sim.get_joint_angles_deg(),
            targets_deg=sim.get_joint_targets_deg(),
            limits_deg=JOINT_SCHEMA,
        ),
        end_effectors=ee_state,
        balance=BalanceState(
            upright=b.upright,
            fallen=b.fallen,
            com=b.com,
            support_polygon=b.support_polygon,
            support_center=b.support_center,
            margin=b.margin,
            zmp_proxy=b.zmp_proxy,
            assist_strength=b.assist_strength,
        ),
        contacts=[ContactEvent(**c) for c in sim.get_contacts()],
        command=command.to_dict(),
        target_circle=target_circle,
        turn=turn,
        locked_limbs=locked_limbs,
        active_constraints=[],
        failure_status=failure_status,
        physics_timestep=PHYSICS_TIMESTEP,
        sim_time=float(sim.data.time),
        perf=perf,
    )
    obs = ws.to_observation()
    obs["phase"] = phase
    obs["mat"] = mat.to_dict()
    reach = ReachController(sim)
    obs["primary_joints"] = reach.primary_joints(command.limb)
    obs["ik_suggestion"] = reach.suggest_targets(command.limb, target_circle["x"], target_circle["y"], ik_iterations=8)
    obs["reach_vector"] = reach.reach_vector(command.limb, ee[command.limb], target_circle["x"], target_circle["y"])
    return ws


def evaluate_phase1_step(
    state: Phase1State,
    sim: HumanoidSim,
    mat: TwisterMat,
    balance: BalanceController,
    *,
    radius: float = PLACEMENT_RADIUS,
) -> tuple[bool, bool, float, str | None, dict[str, float]]:
    target = mat.circle_at(state.command.row, state.command.col)
    limb_pos = sim.get_end_effector_positions()[state.command.limb]
    error = placement_error(limb_pos, target)
    state.placement_errors.append(error)
    if sim.has_fallen():
        return False, True, 0.0, "fall", {"fall": 1.0}
    if is_placed(limb_pos, target, radius):
        rb = reward_breakdown(True, error, balance.stability_score(), False, False)
        return True, True, rb["total"], "success", rb
    rb = reward_breakdown(False, error, balance.stability_score(), False, False)
    return False, False, rb["total"], None, rb


def evaluate_phase2_step(
    state: Phase2State,
    sim: HumanoidSim,
    balance: BalanceController,
    *,
    radius: float = PLACEMENT_RADIUS,
) -> tuple[bool, bool, float, str | None, dict[str, float], list[str]]:
    target = state.mat.circle_at(state.command.row, state.command.col)
    limb_pos = sim.get_end_effector_positions()[state.command.limb]
    error = placement_error(limb_pos, target)
    state.placement_errors.append(error)
    if sim.has_fallen():
        return False, True, 0.0, "fall", {"fall": 1.0}, []
    violations = state.constraints.validate_locked(sim.get_end_effector_positions(), state.mat, ignore_limb=state.command.limb)
    if violations:
        return False, True, 0.0, "locked_limb_slipped", {"locked_slip": 1.0}, violations
    placed = is_placed(limb_pos, target, radius)
    rb = reward_breakdown(placed, error, balance.stability_score(), False, bool(violations))
    if placed:
        state.constraints.lock(state.command)
        if state.turns_completed + 1 >= state.max_turns:
            return True, True, rb["total"] + 0.5, "max_turns", rb, []
        state.advance_turn(sim.get_end_effector_positions())
        return True, False, rb["total"] + 0.5, None, rb, []
    return False, False, rb["total"], None, rb, []
