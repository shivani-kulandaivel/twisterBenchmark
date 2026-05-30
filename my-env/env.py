"""Twister human-body control benchmark for Mesocosm."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

from game.tasks import (
    Phase1State,
    Phase2State,
    build_observation,
    evaluate_phase1_step,
    evaluate_phase2_step,
)
from game.twister import Spinner, TwisterMat, placement_error
from sim.constants import JOINT_SCHEMA, MAX_DELTA_DEG, PLACEMENT_RADIUS
from sim.humanoid import HumanoidSim
from sim.reach_controller import ReachController

PHASE1_MAX_STEPS = 160
PHASE2_MAX_STEPS = 200
PHASE2_MAX_TURNS = 10

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
_CONTROLLER_DIR = Path(__file__).parent / "generated_controllers"


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return ""


class TwisterEnv(BaseEnv):
    def __init__(self) -> None:
        _CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)
        self._sim = HumanoidSim()
        self._reach = ReachController(self._sim)
        self._rng = random.Random()
        self._phase = 1
        self._mat = TwisterMat()
        self._spinner: Spinner | None = None
        self._phase1: Phase1State | None = None
        self._phase2: Phase2State | None = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._seed: int | None = None
        self._placement_radius_override: float | None = None

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._seed = seed
        self._rng = random.Random(seed)
        self._phase = int(params.get("phase", 1))
        self._mat = TwisterMat()
        self._spinner = Spinner(self._rng)
        self._step_count = 0
        self._cumulative_reward = 0.0

        self._sim.reset(seed=seed)

        if self._phase == 2:
            phase2_limb_weights = {
                "left_foot": 1.6,
                "right_foot": 1.6,
                "left_hand": 0.9,
                "right_hand": 0.9,
            }
            command = self._spinner.spin(
                self._mat,
                limb_weights=phase2_limb_weights,
            )
            self._phase2 = Phase2State(
                mat=self._mat,
                spinner=self._spinner,
                command=command,
                max_turns=int(params.get("max_turns", PHASE2_MAX_TURNS)),
                limb_weights=phase2_limb_weights,
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
                reach=self._reach,
            )

        command = self._spinner.spin(
            self._mat,
            limb_weights={
                "left_foot": 2.4,
                "right_foot": 2.4,
                "left_hand": 0.8,
                "right_hand": 0.8,
            },
        )
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
            reach=self._reach,
        )

    def step(self, action: Any) -> StepResult:
        if self._phase1 is None and self._phase2 is None:
            raise RuntimeError("Call reset() before step()")

        parsed = self._parse_action(action)
        if "controller" in parsed:
            return self._step_with_controller(parsed["controller"])
        self._apply_action(parsed)
        self._sim.step_physics()
        self._step_count += 1

        if self._phase == 1:
            return self._step_phase1()
        return self._step_phase2()

    def _step_with_controller(self, controller: Any) -> StepResult:
        action, repeat, max_steps, tolerance_m, script_path = self._resolve_controller_action(controller)
        max_steps = max(1, min(220, int(max_steps)))
        repeat = bool(repeat)
        tolerance_m = max(0.001, min(0.25, float(tolerance_m)))
        last: StepResult | None = None
        prior_override = self._placement_radius_override
        self._placement_radius_override = tolerance_m
        try:
            for _ in range(max_steps):
                self._apply_action(action)
                self._sim.step_physics()
                self._step_count += 1
                last = self._step_phase1() if self._phase == 1 else self._step_phase2()
                info = dict(last.info)
                info["controller_script"] = str(script_path)
                info["controller_tolerance_m"] = f"{tolerance_m:.4f}"
                last = StepResult(
                    observation=last.observation,
                    reward=last.reward,
                    terminated=last.terminated,
                    truncated=last.truncated,
                    info=info,
                    system_prompt=last.system_prompt,
                )
                if last.terminated:
                    break
                if not repeat:
                    break
        finally:
            self._placement_radius_override = prior_override
        assert last is not None
        return last

    def _resolve_controller_action(
        self, controller: Any
    ) -> tuple[dict[str, Any], bool, int, float, Path]:
        if not isinstance(controller, dict):
            return {"use_ik": True}, True, 60, 0.01, Path("")
        language = str(controller.get("language", "javascript")).strip().lower()
        code = str(controller.get("code", ""))
        repeat = bool(controller.get("repeat_until_placed", True))
        max_steps = int(controller.get("max_steps", 60))
        tolerance_m = float(controller.get("tolerance_m", 0.01))

        ext = ".js" if language in ("javascript", "js") else ".py"
        script_path = _CONTROLLER_DIR / f"controller_step{self._step_count:05d}{ext}"
        script_path.write_text(code, encoding="utf-8")
        if not code.strip():
            return {"use_ik": True}, repeat, max_steps, tolerance_m, script_path

        obs = self._controller_context()
        cmd = ["node", str(script_path)] if ext == ".js" else ["python3", str(script_path)]
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(obs),
                text=True,
                capture_output=True,
                timeout=3.0,
                check=False,
            )
            out = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            payload = json.loads(out) if out else {}
            if isinstance(payload, dict) and "action" in payload:
                action = payload.get("action", {})
                repeat = bool(payload.get("repeat_until_placed", repeat))
                max_steps = int(payload.get("max_steps", max_steps))
                tolerance_m = float(payload.get("tolerance_m", tolerance_m))
            else:
                action = payload if isinstance(payload, dict) else {}
            if not action:
                action = {"use_ik": True}
            return action, repeat, max_steps, tolerance_m, script_path
        except Exception:
            return {"use_ik": True}, repeat, max_steps, tolerance_m, script_path

    def _controller_context(self) -> dict[str, Any]:
        cmd = self._current_command()
        target = self._mat.circle_at(cmd.row, cmd.col)
        return {
            "command": cmd.to_dict(),
            "target_circle": target.to_dict(),
            "joints": self._sim.get_joint_angles_deg(),
            "joint_targets": self._sim.get_joint_targets_deg(),
            "end_effectors": self._sim.get_end_effector_positions(),
            "torso": self._sim.get_torso_state(),
            "placement_radius": self._active_placement_radius(),
        }

    def _active_placement_radius(self) -> float:
        if self._placement_radius_override is not None:
            return float(self._placement_radius_override)
        return float(PLACEMENT_RADIUS)

    def _apply_action(self, parsed: dict[str, Any]) -> None:
        """Apply agent action with optional IK assist and smooth target tracking."""
        if parsed.get("use_ik"):
            cmd = self._current_command()
            target = self._mat.circle_at(cmd.row, cmd.col)
            anchor_limbs: list[str] | None = None
            if self._phase2 is not None:
                anchor_limbs = [
                    item.limb for item in self._phase2.constraints.locked if item.limb != cmd.limb
                ]
            self._reach.apply_step(
                cmd.limb,
                target.x,
                target.y,
                anchor_limbs=anchor_limbs,
            )
            return

        self._sim.set_active_reach(None)

        joint_targets = parsed.get("joint_targets", {})
        if not joint_targets:
            return

        if parsed.get("follow_ik") and self._last_ik_suggestion():
            blend = float(parsed.get("ik_blend", 0.65))
            blend = max(0.0, min(1.0, blend))
            suggestion = self._last_ik_suggestion()
            merged = {}
            for name in JOINT_SCHEMA:
                if name in joint_targets:
                    agent = float(joint_targets[name])
                else:
                    agent = self._sim.get_joint_targets_deg().get(name, JOINT_SCHEMA[name]["neutral"])
                ref = suggestion.get(name, agent)
                merged[name] = agent * (1.0 - blend) + ref * blend
            joint_targets = merged

        if parsed.get("delta", False):
            self._sim.set_targets(
                joint_targets,
                delta=True,
                max_delta=float(parsed.get("max_delta", MAX_DELTA_DEG)),
            )
        else:
            self._sim.set_targets_smooth(joint_targets)

    def _current_command(self):
        if self._phase1 is not None:
            return self._phase1.command
        assert self._phase2 is not None
        return self._phase2.command

    def _last_ik_suggestion(self) -> dict[str, float] | None:
        cmd = self._current_command()
        target = self._mat.circle_at(cmd.row, cmd.col)
        return self._reach.suggest_targets(cmd.limb, target.x, target.y, ik_iterations=8)

    def _step_phase1(self) -> StepResult:
        assert self._phase1 is not None
        state = self._phase1
        target = self._mat.circle_at(state.command.row, state.command.col)
        limb_pos = self._sim.get_end_effector_positions()[state.command.limb]
        current_error = placement_error(limb_pos, target)

        success, terminated, reward, reason = evaluate_phase1_step(
            state,
            self._sim,
            self._mat,
            radius=self._active_placement_radius(),
        )
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
            reach=self._reach,
            placement_radius=self._active_placement_radius(),
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

        turn_done, terminated, reward, reason = evaluate_phase2_step(
            state,
            self._sim,
            radius=self._active_placement_radius(),
        )
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
            reach=self._reach,
            placement_radius=self._active_placement_radius(),
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
            action = action.strip()
            if action.startswith("{"):
                return json.loads(action)
        return {"joint_targets": {}}

    def render(self, mode: str = "text") -> Any:
        if self._phase1 is None and self._phase2 is None:
            return super().render(mode)
        if mode != "text":
            return super().render(mode)
        cmd = self._current_command()
        target = self._mat.circle_at(cmd.row, cmd.col)
        ee = self._sim.get_end_effector_positions()[cmd.limb]
        err = placement_error(ee, target)
        return (
            f"{cmd.limb} -> ({target.x:.2f},{target.y:.2f}) "
            f"err={err:.3f}m upright={self._sim.is_upright()}"
        )

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


MyEnv = TwisterEnv


def joint_schema_for_prompt() -> dict[str, Any]:
    return {
        name: {"low": spec["low"], "high": spec["high"], "neutral": spec["neutral"]}
        for name, spec in JOINT_SCHEMA.items()
    }


def action_schema_for_prompt() -> dict[str, Any]:
    return {
        "joint_targets": "dict of joint_name -> angle in degrees",
        "delta": "optional bool; if true, values are per-step changes (default false)",
        "max_delta": f"optional max change per step when delta=true (default {MAX_DELTA_DEG})",
        "follow_ik": "optional bool; blend your targets with observation.ik_suggestion",
        "ik_blend": "optional 0-1 blend weight when follow_ik=true (default 0.65)",
        "use_ik": "optional bool; let the built-in whole-body IK controller move this step",
        "controller": {
            "language": "javascript|python",
            "code": "controller source code written each prompt to generated_controllers/",
            "repeat_until_placed": "bool, default true",
            "max_steps": "internal loop cap while pursuing current target",
            "tolerance_m": "controller-mode success radius, default 0.01m",
        },
    }
