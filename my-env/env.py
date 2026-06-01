"""Twister human-body control benchmark for Mesocosm."""

from __future__ import annotations

import json
import random
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

from game.tasks import (
    MultiPlacementState,
    Phase1State,
    Phase2State,
    build_multi_observation,
    build_observation,
    evaluate_multi_step,
    evaluate_phase1_step,
    evaluate_phase2_step,
)
from game.twister import Spinner, TwisterMat, placement_error
from sim.constants import JOINT_SCHEMA, MAX_DELTA_DEG
from sim.humanoid import HumanoidSim

PHASE1_MAX_STEPS = 50
PHASE2_MAX_STEPS = 200
PHASE2_MAX_TURNS = 10


class TwisterEnv(BaseEnv):
    def __init__(self) -> None:
        self._sim = HumanoidSim()
        self._rng = random.Random()
        self._phase = 1
        self._mat = TwisterMat()
        self._spinner: Spinner | None = None
        self._phase1: Phase1State | None = None
        self._phase2: Phase2State | None = None
        self._multi: MultiPlacementState | None = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._seed: int | None = None

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._seed = seed
        self._rng = random.Random(seed)
        self._phase = int(params.get("phase", 1))
        self._mat = TwisterMat()
        self._spinner = Spinner(self._rng)
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._multi = None

        self._sim.reset(seed=seed)

        if self._phase == 2:
            command = self._spinner.spin(self._mat)
            self._phase2 = Phase2State(
                mat=self._mat,
                spinner=self._spinner,
                command=command,
                max_turns=int(params.get("max_turns", PHASE2_MAX_TURNS)),
            )
            self._phase1 = None
            target = self._mat.circle_at(command.row, command.col)
            return build_observation(
                phase=2,
                sim=self._sim,
                mat=self._mat,
                command=command,
                turn=1,
                locked_limbs=[],
                target_circle=target.to_dict(),
                placement_error_m=None,
            )

        num_targets = int(params.get("num_targets", 1))
        if num_targets > 1:
            commands = self._spinner.spin_reachable_multi(self._mat, num_targets)
            self._multi = MultiPlacementState(commands=commands)
            self._phase1 = None
            self._phase2 = None
            return build_multi_observation(
                sim=self._sim,
                mat=self._mat,
                commands=commands,
                placement_errors=None,
            )

        command = self._spinner.spin_reachable(self._mat)
        self._phase1 = Phase1State(command=command)
        self._phase2 = None
        target = self._mat.circle_at(command.row, command.col)
        return build_observation(
            phase=1,
            sim=self._sim,
            mat=self._mat,
            command=command,
            turn=1,
            locked_limbs=[],
            target_circle=target.to_dict(),
            placement_error_m=None,
        )

    def step(self, action: Any) -> StepResult:
        if self._phase1 is None and self._phase2 is None and self._multi is None:
            raise RuntimeError("Call reset() before step()")

        parsed = self._parse_action(action)
        self._sim.set_targets(
            parsed.get("joint_targets", {}),
            delta=bool(parsed.get("delta", False)),
            max_delta=float(parsed.get("max_delta", MAX_DELTA_DEG)),
        )
        self._sim.step_physics()
        self._step_count += 1

        if self._multi is not None:
            return self._step_multi()
        if self._phase == 1:
            return self._step_phase1()
        return self._step_phase2()

    def _step_multi(self) -> StepResult:
        assert self._multi is not None
        state = self._multi
        all_placed, terminated, reward, reason, errors = evaluate_multi_step(
            state, self._sim, self._mat
        )
        truncated = not terminated and self._step_count >= PHASE1_MAX_STEPS
        if truncated:
            terminated = True
            reason = reason or "timeout"

        self._cumulative_reward += reward
        obs = build_multi_observation(
            sim=self._sim,
            mat=self._mat,
            commands=state.commands,
            placement_errors=errors,
        )
        mean_err = sum(errors) / len(errors)
        info = {
            "phase": "1-multi",
            "success": str(all_placed),
            "num_targets": str(len(state.commands)),
            "termination_reason": reason or "",
            "steps_used": str(self._step_count),
            "placement_error": str(round(mean_err, 4)),
            "per_limb_errors": ",".join(f"{c.limb}={e:.3f}" for c, e in zip(state.commands, errors)),
            "cumulative_reward": str(round(self._cumulative_reward, 4)),
        }
        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _step_phase1(self) -> StepResult:
        assert self._phase1 is not None
        state = self._phase1
        target = self._mat.circle_at(state.command.row, state.command.col)
        limb_pos = self._sim.get_end_effector_positions()[state.command.limb]
        current_error = placement_error(limb_pos, target)

        success, terminated, reward, reason = evaluate_phase1_step(state, self._sim, self._mat)
        truncated = not terminated and self._step_count >= PHASE1_MAX_STEPS
        if truncated:
            terminated = True
            reason = reason or "timeout"

        self._cumulative_reward += reward
        obs = build_observation(
            phase=1,
            sim=self._sim,
            mat=self._mat,
            command=state.command,
            turn=1,
            locked_limbs=[],
            target_circle=target.to_dict(),
            placement_error_m=current_error,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=self._info_phase1(success, reason, current_error),
        )

    def _step_phase2(self) -> StepResult:
        assert self._phase2 is not None
        state = self._phase2
        target = self._mat.circle_at(state.command.row, state.command.col)
        limb_pos = self._sim.get_end_effector_positions()[state.command.limb]
        current_error = placement_error(limb_pos, target)

        turn_done, terminated, reward, reason = evaluate_phase2_step(state, self._sim)
        truncated = not terminated and self._step_count >= PHASE2_MAX_STEPS
        if truncated:
            terminated = True
            reason = reason or "timeout"

        self._cumulative_reward += reward
        obs = build_observation(
            phase=2,
            sim=self._sim,
            mat=self._mat,
            command=state.command,
            turn=state.turn,
            locked_limbs=state.constraints.to_dict(),
            target_circle=target.to_dict(),
            placement_error_m=current_error,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=self._info_phase2(turn_done, reason, current_error),
        )

    def _parse_action(self, action: Any) -> dict[str, Any]:
        if isinstance(action, dict):
            return action
        if isinstance(action, str):
            text = action.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            if text.startswith("{"):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return {"joint_targets": {}}

    def _info_phase1(self, success: bool, reason: str | None, error: float) -> dict[str, str]:
        avg_error = (
            sum(self._phase1.placement_errors) / len(self._phase1.placement_errors)
            if self._phase1 and self._phase1.placement_errors
            else error
        )
        return {
            "phase": "1",
            "success": str(success),
            "termination_reason": reason or "",
            "steps_used": str(self._step_count),
            "placement_error": str(round(error, 4)),
            "avg_placement_error": str(round(avg_error, 4)),
            "cumulative_reward": str(round(self._cumulative_reward, 4)),
        }

    def _info_phase2(
        self, turn_completed: bool, reason: str | None, error: float
    ) -> dict[str, str]:
        assert self._phase2 is not None
        avg_error = (
            sum(self._phase2.placement_errors) / len(self._phase2.placement_errors)
            if self._phase2.placement_errors
            else error
        )
        return {
            "phase": "2",
            "turn_completed": str(turn_completed),
            "turns_survived": str(self._phase2.turns_completed),
            "current_turn": str(self._phase2.turn),
            "termination_reason": reason or "",
            "steps_used": str(self._step_count),
            "placement_error": str(round(error, 4)),
            "avg_placement_error": str(round(avg_error, 4)),
            "cumulative_reward": str(round(self._cumulative_reward, 4)),
        }


# Adapter imports MyEnv by name.
MyEnv = TwisterEnv


def joint_schema_for_prompt() -> dict[str, Any]:
    return {
        name: {"low": spec["low"], "high": spec["high"], "neutral": spec["neutral"]}
        for name, spec in JOINT_SCHEMA.items()
    }
