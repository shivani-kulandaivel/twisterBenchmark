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


@dataclass
class Phase1State:
    command: TwisterCommand
    step_count: int = 0
    placement_errors: list[float] = field(default_factory=list)

    def record_error(self, error: float) -> None:
        self.placement_errors.append(error)


@dataclass
class MultiPlacementState:
    """Phase-1 variant: several limbs must be placed on their targets at once."""
    commands: list[TwisterCommand]
    step_count: int = 0
    mean_errors: list[float] = field(default_factory=list)

    def record_error(self, error: float) -> None:
        self.mean_errors.append(error)


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

    def advance_turn(self) -> None:
        self.turns_completed += 1
        self.turn += 1
        self.command = self.spinner.spin(self.mat)

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
) -> dict[str, Any]:
    end_effectors = sim.get_end_effector_positions()
    for limb in LIMBS:
        pos = end_effectors[limb]
        pos["on_mat"] = pos["z"] <= 0.25

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
        "physics_state": {"qpos": sim.get_qpos()},
    }
    if placement_error_m is not None:
        obs["placement_error"] = round(placement_error_m, 4)
        obs["placement_radius"] = PLACEMENT_RADIUS
    return obs


def build_multi_observation(
    *,
    sim: HumanoidSim,
    mat: TwisterMat,
    commands: list[TwisterCommand],
    placement_errors: list[float] | None,
) -> dict[str, Any]:
    end_effectors = sim.get_end_effector_positions()
    for limb in LIMBS:
        pos = end_effectors[limb]
        pos["on_mat"] = pos["z"] <= 0.25

    targets = [mat.circle_at(c.row, c.col) for c in commands]
    obs: dict[str, Any] = {
        "phase": 1,
        "multi": True,
        "num_targets": len(commands),
        "commands": [c.to_dict() for c in commands],
        "targets": [t.to_dict() for t in targets],
        "joints": sim.get_joint_angles_deg(),
        "joint_targets": sim.get_joint_targets_deg(),
        "end_effectors": end_effectors,
        "mat": mat.to_dict(),
        "upright": sim.is_upright(),
        "torso": sim.get_torso_state(),
        "physics_state": {"qpos": sim.get_qpos()},
    }
    if placement_errors is not None:
        obs["placement_errors"] = [round(e, 4) for e in placement_errors]
        obs["placement_radius"] = PLACEMENT_RADIUS
    return obs


def evaluate_multi_step(
    state: MultiPlacementState,
    sim: HumanoidSim,
    mat: TwisterMat,
) -> tuple[bool, bool, float, str | None, list[float]]:
    """Returns (all_placed, terminated, reward, reason, per_limb_errors)."""
    end_effectors = sim.get_end_effector_positions()
    errors: list[float] = []
    placed_flags: list[bool] = []
    for cmd in state.commands:
        target = mat.circle_at(cmd.row, cmd.col)
        limb_pos = end_effectors[cmd.limb]
        err = placement_error(limb_pos, target)
        errors.append(err)
        placed_flags.append(is_placed(limb_pos, target))

    mean_err = sum(errors) / len(errors)
    state.record_error(mean_err)

    if sim.has_fallen():
        return False, True, 0.0, "fall", errors

    if all(placed_flags):
        return True, True, 1.0, "success", errors

    # Partial credit: fraction placed + proximity of the rest.
    proximity = sum(max(0.0, 1.0 - e / 0.5) for e in errors) / len(errors)
    return False, False, proximity, None, errors


def evaluate_phase1_step(
    state: Phase1State,
    sim: HumanoidSim,
    mat: TwisterMat,
) -> tuple[bool, bool, float, str | None]:
    """Returns (success, terminated, reward, termination_reason)."""
    target = mat.circle_at(state.command.row, state.command.col)
    limb_pos = sim.get_end_effector_positions()[state.command.limb]
    error = placement_error(limb_pos, target)
    state.record_error(error)

    if sim.has_fallen():
        return False, True, 0.0, "fall"

    if is_placed(limb_pos, target):
        return True, True, 1.0, "success"

    return False, False, max(0.0, 1.0 - error / 0.5), None


def evaluate_phase2_step(
    state: Phase2State,
    sim: HumanoidSim,
) -> tuple[bool, bool, float, str | None]:
    """Returns (turn_completed, terminated, step_reward, termination_reason)."""
    target = state.mat.circle_at(state.command.row, state.command.col)
    limb_pos = sim.get_end_effector_positions()[state.command.limb]
    error = placement_error(limb_pos, target)
    state.record_error(error)

    if sim.has_fallen():
        return False, True, 0.0, "fall"

    violations = state.constraints.validate_locked(sim.get_end_effector_positions(), state.mat)
    if violations:
        return False, True, 0.0, "locked_limb_slipped"

    if is_placed(limb_pos, target):
        state.constraints.lock(state.command)
        if state.turns_completed + 1 >= state.max_turns:
            return True, True, 1.5, "max_turns"
        state.advance_turn()
        return True, False, 1.5, None

    return False, False, max(0.0, 0.2 - error), None
