"""Episode task logic for phased Twister benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.twister import (
    ConstraintTracker,
    Spinner,
    TwisterCommand,
    TwisterMat,
    is_placed,
    placement_error,
)
from sim.constants import LIMBS, PLACEMENT_RADIUS
from sim.humanoid import HumanoidSim
from sim.reach_controller import ReachController


@dataclass
class Phase1State:
    command: TwisterCommand
    step_count: int = 0
    placement_errors: list[float] = field(default_factory=list)

    def record_error(self, error: float) -> None:
        self.placement_errors.append(error)


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
        forbidden = {
            (item.row, item.col)
            for item in self.constraints.locked
            if item.limb != self.command.limb
        }
        locked_limbs = [item.limb for item in self.constraints.locked]
        self.command = self.spinner.spin(
            self.mat,
            limb_weights=self.limb_weights,
            forbidden_circles=forbidden,
            limb_positions=end_effectors,
            locked_limbs=locked_limbs,
        )

    def record_error(self, error: float) -> None:
        self.placement_errors.append(error)


def build_observation(
    *,
    phase: int,
    sim: HumanoidSim,
    mat: TwisterMat,
    command: TwisterCommand,
    turn: int,
    locked_limbs: list[dict[str, Any]],
    target_circle: dict[str, Any],
    placement_error_m: float | None,
    reach: ReachController | None = None,
    placement_radius: float = PLACEMENT_RADIUS,
) -> dict[str, Any]:
    end_effectors = sim.get_end_effector_positions()
    limb = command.limb
    for limb_name in LIMBS:
        pos = end_effectors[limb_name]
        pos["on_mat"] = pos["z"] <= 0.25

    reach_ctrl = reach or ReachController(sim)
    ee = end_effectors[limb]
    reach_vector = reach_ctrl.reach_vector(limb, ee, target_circle["x"], target_circle["y"])
    target_z = float(reach_vector["target_z"])
    err_x = float(target_circle["x"] - ee["x"])
    err_y = float(target_circle["y"] - ee["y"])
    err_z = float(target_z - ee["z"])
    ik_suggestion = reach_ctrl.suggest_targets(
        limb, target_circle["x"], target_circle["y"], ik_iterations=8
    )

    obs: dict[str, Any] = {
        "phase": phase,
        "command": command.to_dict(),
        "target_circle": target_circle,
        "joints": sim.get_joint_angles_deg(),
        "joint_targets": sim.get_joint_targets_deg(),
        "end_effectors": end_effectors,
        "mat": mat.to_dict(),
        "locked_limbs": locked_limbs,
        "turn": turn,
        "upright": sim.is_upright(),
        "torso": sim.get_torso_state(),
        "reach_vector": reach_vector,
        "primary_joints": reach_ctrl.primary_joints(limb),
        "ik_suggestion": ik_suggestion,
        "active_limb_goal": {
            "limb": limb,
            "control_point": f"end_effectors.{limb}",
            "target_xyz": {
                "x": round(float(target_circle["x"]), 4),
                "y": round(float(target_circle["y"]), 4),
                "z": round(target_z, 4),
            },
            "current_xyz": {
                "x": round(float(ee["x"]), 4),
                "y": round(float(ee["y"]), 4),
                "z": round(float(ee["z"]), 4),
            },
            "error_xyz": {
                "x": round(err_x, 4),
                "y": round(err_y, 4),
                "z": round(err_z, 4),
            },
        },
        "placement_contract": {
            "success_when": "horizontal distance(control_point.xy, target_circle.xy) <= placement_radius",
            "control_point": f"end_effectors.{limb}",
            "target_center": {
                "x": round(float(target_circle["x"]), 4),
                "y": round(float(target_circle["y"]), 4),
            },
        },
    }
    if placement_error_m is not None:
        obs["placement_error"] = round(placement_error_m, 4)
        obs["placement_radius"] = round(float(placement_radius), 4)
    return obs


def evaluate_phase1_step(
    state: Phase1State,
    sim: HumanoidSim,
    mat: TwisterMat,
    *,
    radius: float = PLACEMENT_RADIUS,
) -> tuple[bool, bool, float, str | None]:
    """Returns (success, terminated, reward, termination_reason)."""
    target = mat.circle_at(state.command.row, state.command.col)
    limb_pos = sim.get_end_effector_positions()[state.command.limb]
    error = placement_error(limb_pos, target)
    state.record_error(error)

    if sim.has_fallen():
        return False, True, 0.0, "fall"

    if is_placed(limb_pos, target, radius):
        return True, True, 1.0, "success"

    return False, False, max(0.0, 1.0 - error / 0.5), None


def evaluate_phase2_step(
    state: Phase2State,
    sim: HumanoidSim,
    *,
    radius: float = PLACEMENT_RADIUS,
) -> tuple[bool, bool, float, str | None]:
    """Returns (turn_completed, terminated, step_reward, termination_reason)."""
    target = state.mat.circle_at(state.command.row, state.command.col)
    limb_pos = sim.get_end_effector_positions()[state.command.limb]
    error = placement_error(limb_pos, target)
    state.record_error(error)

    if sim.has_fallen():
        return False, True, 0.0, "fall"

    violations = state.constraints.validate_locked(
        sim.get_end_effector_positions(),
        state.mat,
        ignore_limb=state.command.limb,
    )
    # #region agent log
    from debug_log import dbg

    locked_errs: dict[str, float] = {}
    for item in state.constraints.locked:
        if item.limb == state.command.limb:
            continue
        circle = state.mat.circle_at(item.row, item.col)
        locked_errs[item.limb] = round(
            placement_error(sim.get_end_effector_positions()[item.limb], circle), 4
        )
    dbg(
        "tasks.py:evaluate_phase2_step",
        "phase2_eval",
        {
            "turn": state.turn,
            "active_limb": state.command.limb,
            "active_error": round(error, 4),
            "active_placed": is_placed(limb_pos, target, radius),
            "locked_errors": locked_errs,
            "violations": violations,
            "fall": sim.has_fallen(),
        },
        "H3",
    )
    # #endregion
    if violations:
        return False, True, 0.0, "locked_limb_slipped"

    if is_placed(limb_pos, target, radius):
        state.constraints.lock(state.command)
        if state.turns_completed + 1 >= state.max_turns:
            return True, True, 1.5, "max_turns"
        state.advance_turn(sim.get_end_effector_positions())
        return True, False, 1.5, None

    return False, False, max(0.0, 0.2 - error), None
