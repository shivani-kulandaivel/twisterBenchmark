"""Twister humanoid control environment."""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

from game.tasks import (
    Phase1State,
    Phase2State,
    build_world_state,
    evaluate_phase1_step,
    evaluate_phase2_step,
)
from game.twister import Spinner, TwisterMat
from sim.balance_controller import BalanceController
from sim.constants import DEFAULT_BALANCE_ASSIST_STRENGTH, PHYSICS_SUBSTEPS, PLACEMENT_RADIUS
from sim.constraints import ConstraintManager
from sim.humanoid import HumanoidSim
from sim.motion_planner import MotionPlanner
from sim.pose_generator import PoseGenerator
from sim.reach_controller import ReachController
from sim.reachability import ReachabilitySolver
from state.world_state import PerfSnapshot

PHASE1_MAX_STEPS = 160
PHASE2_MAX_STEPS = 220
PHASE2_MAX_TURNS = 10

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
_CONTROLLER_DIR = Path(__file__).parent / "generated_controllers"
_DEBUG_LOG_PATH = Path("/Users/tanvi/Library/CloudStorage/OneDrive-UW/General/Hackathons/Sweccathon2026/twisterBenchmark/.cursor/debug-721478.log")


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    payload = {
        "sessionId": "721478",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return ""


class TwisterEnv(BaseEnv):
    def __init__(self) -> None:
        _CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random()
        self._sim = HumanoidSim()
        self._balance = BalanceController(self._sim, assist_strength=DEFAULT_BALANCE_ASSIST_STRENGTH)
        self._reach = ReachController(self._sim)
        self._reachability = ReachabilitySolver(self._sim, self._reach, self._balance)
        self._planner = MotionPlanner(self._sim)
        self._pose_gen = PoseGenerator()
        self._constraints = ConstraintManager(self._sim)
        self._mat = TwisterMat()
        self._spinner = Spinner(self._rng, reachability_checker=self._reachability)
        self._phase = 1
        self._phase1: Phase1State | None = None
        self._phase2: Phase2State | None = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._decision_times: list[float] = []
        self._move_history: list[dict[str, Any]] = []
        self._last_world_obs: dict[str, Any] = {}
        self._seed: int | None = None

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._seed = seed
        self._rng.seed(seed)
        self._sim.reset(seed=seed)
        self._constraints.unlock_all()
        self._phase = int(params.get("phase", 1))
        turns = int(params.get("max_turns", PHASE2_MAX_TURNS))
        assist_strength = float(params.get("balance_assist_strength", DEFAULT_BALANCE_ASSIST_STRENGTH))
        self._balance.set_assist_strength(assist_strength)
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._decision_times = []
        self._move_history = []
        self._mat = TwisterMat()
        self._spinner = Spinner(self._rng, reachability_checker=self._reachability)

        if self._phase == 1:
            command = self._spinner.spin(self._mat, limb_weights={"left_foot": 1.4, "right_foot": 1.4, "left_hand": 1.0, "right_hand": 1.0})
            self._phase1 = Phase1State(command=command)
            self._phase2 = None
            turn = 1
            locked = []
        else:
            command = self._spinner.spin(self._mat)
            self._phase1 = None
            self._phase2 = Phase2State(
                mat=self._mat,
                spinner=self._spinner,
                command=command,
                max_turns=turns,
            )
            self._phase2.constraints.manager = self._constraints
            turn = self._phase2.turn
            locked = self._phase2.constraints.to_dict()

        target = self._mat.circle_at(command.row, command.col).to_dict()
        ws = build_world_state(
            phase=self._phase,
            sim=self._sim,
            mat=self._mat,
            command=command,
            turn=turn,
            locked_limbs=locked,
            target_circle=target,
            balance=self._balance,
            perf=PerfSnapshot(),
        )
        obs = ws.to_observation()
        obs["system_prompt"] = _load_system_prompt()
        obs["placement_radius"] = PLACEMENT_RADIUS
        self._last_world_obs = obs
        # #region agent log
        _debug_log(
            "pre-fix",
            "H1,H2",
            "env.py:reset",
            "reset selected command and target",
            {
                "seed": seed,
                "phase": self._phase,
                "command": command.to_dict(),
                "target": target,
                "turn": turn,
                "locked_count": len(locked),
            },
        )
        # #endregion
        return obs

    def step(self, action: Any) -> StepResult:
        if self._phase == 1 and self._phase1 is None:
            raise RuntimeError("Call reset() before step()")
        if self._phase == 2 and self._phase2 is None:
            raise RuntimeError("Call reset() before step()")
        start = time.perf_counter()
        parsed = self._parse_action(action)
        # #region agent log
        _debug_log(
            "pre-fix",
            "H5",
            "env.py:step",
            "step parsed action",
            {
                "phase": self._phase,
                "step": self._step_count,
                "input_type": type(action).__name__,
                "parsed_keys": sorted(parsed.keys()),
                "uses_ik": bool(parsed.get("use_ik")),
                "has_controller": bool(parsed.get("controller")),
                "joint_target_count": len(parsed.get("joint_targets", {}) or {}),
            },
        )
        # #endregion
        perf = PerfSnapshot()
        perf.decision_ms = (time.perf_counter() - start) * 1000.0
        self._decision_times.append(perf.decision_ms / 1000.0)
        self._apply_action(parsed, perf)
        self._step_count += 1

        if self._phase == 1:
            assert self._phase1 is not None
            command = self._phase1.command
            target = self._mat.circle_at(command.row, command.col)
            success, terminated, reward, reason, reward_parts = evaluate_phase1_step(
                self._phase1,
                self._sim,
                self._mat,
                self._balance,
            )
            if self._step_count >= PHASE1_MAX_STEPS and not terminated:
                terminated = True
                reason = "timeout"
            self._cumulative_reward += reward
            ws = build_world_state(
                phase=1,
                sim=self._sim,
                mat=self._mat,
                command=command,
                turn=1,
                locked_limbs=[],
                target_circle=target.to_dict(),
                balance=self._balance,
                perf=perf,
                failure_status=reason,
            )
            obs = ws.to_observation()
            reachability = self._reachability.can_reach(command.limb, target.x, target.y)
            obs["reachability"] = {
                "feasible": reachability.feasible,
                "error_m": round(reachability.error_m, 4),
                "margin": round(reachability.margin, 4),
            }
            info = {
                "success": success,
                "termination_reason": reason,
                "steps_used": self._step_count,
                "placement_error": self._phase1.placement_errors[-1] if self._phase1.placement_errors else None,
                "avg_placement_error": statistics.fmean(self._phase1.placement_errors) if self._phase1.placement_errors else 0.0,
                "reward_breakdown": reward_parts,
                "cumulative_reward": self._cumulative_reward,
                "turns_survived": 1 if success else 0,
                "avg_decision_time_s": statistics.fmean(self._decision_times) if self._decision_times else 0.0,
                "reach_accuracy": max(0.0, 1.0 - (self._phase1.placement_errors[-1] / PLACEMENT_RADIUS)) if self._phase1.placement_errors else 0.0,
                "balance_success_rate": 1.0 if self._sim.is_upright() else 0.0,
                "constraint_violations": 0,
                "total_score": self._cumulative_reward,
            }
            # #region agent log
            _debug_log(
                "pre-fix",
                "H1,H3,H4",
                "env.py:step.phase1_eval",
                "phase1 evaluation result",
                {
                    "step": self._step_count,
                    "command": command.to_dict(),
                    "target": target.to_dict(),
                    "terminated": terminated,
                    "reason": reason,
                    "success": success,
                    "reward": reward,
                    "placement_error": info["placement_error"],
                    "upright": self._sim.is_upright(),
                    "fallen": self._sim.has_fallen(),
                    "torso": self._sim.get_torso_state(),
                },
            )
            # #endregion
            self._last_world_obs = obs
            return StepResult(observation=obs, reward=reward, terminated=terminated, truncated=False, info=info)

        assert self._phase2 is not None
        command = self._phase2.command
        target = self._mat.circle_at(command.row, command.col)
        turn_completed, terminated, reward, reason, reward_parts, violations = evaluate_phase2_step(
            self._phase2,
            self._sim,
            self._balance,
        )
        if self._step_count >= PHASE2_MAX_STEPS and not terminated:
            terminated = True
            reason = "timeout"
        self._cumulative_reward += reward
        ws = build_world_state(
            phase=2,
            sim=self._sim,
            mat=self._mat,
            command=command,
            turn=self._phase2.turn,
            locked_limbs=self._phase2.constraints.to_dict(),
            target_circle=target.to_dict(),
            balance=self._balance,
            perf=perf,
            failure_status=reason,
        )
        obs = ws.to_observation()
        obs["turn_completed"] = turn_completed
        info = {
            "success": turn_completed,
            "termination_reason": reason,
            "turn_completed": turn_completed,
            "turns_survived": self._phase2.turns_completed,
            "current_turn": self._phase2.turn,
            "steps_used": self._step_count,
            "placement_error": self._phase2.placement_errors[-1] if self._phase2.placement_errors else None,
            "avg_placement_error": statistics.fmean(self._phase2.placement_errors) if self._phase2.placement_errors else 0.0,
            "reward_breakdown": reward_parts,
            "constraint_violations": len(violations),
            "balance_success_rate": 1.0 if self._sim.is_upright() else 0.0,
            "avg_decision_time_s": statistics.fmean(self._decision_times) if self._decision_times else 0.0,
            "reach_accuracy": max(0.0, 1.0 - ((self._phase2.placement_errors[-1] if self._phase2.placement_errors else PLACEMENT_RADIUS) / PLACEMENT_RADIUS)),
            "total_score": self._cumulative_reward + float(self._phase2.turns_completed),
        }
        # #region agent log
        _debug_log(
            "pre-fix",
            "H1,H2,H3,H4",
            "env.py:step.phase2_eval",
            "phase2 evaluation result",
            {
                "step": self._step_count,
                "turn": self._phase2.turn,
                "turns_completed": self._phase2.turns_completed,
                "command": command.to_dict(),
                "target": target.to_dict(),
                "terminated": terminated,
                "reason": reason,
                "turn_completed": turn_completed,
                "reward": reward,
                "placement_error": info["placement_error"],
                "violations": violations,
                "locked": self._phase2.constraints.to_dict(),
                "upright": self._sim.is_upright(),
                "fallen": self._sim.has_fallen(),
                "torso": self._sim.get_torso_state(),
            },
        )
        # #endregion
        self._last_world_obs = obs
        return StepResult(observation=obs, reward=reward, terminated=terminated, truncated=False, info=info)

    def _apply_action(self, action: dict[str, Any], perf: PerfSnapshot) -> None:
        t0 = time.perf_counter()
        if action.get("controller"):
            action = self._run_controller(action["controller"])
        if action.get("use_ik"):
            cmd = self._phase1.command if self._phase == 1 and self._phase1 else self._phase2.command  # type: ignore[union-attr]
            circle = self._mat.circle_at(cmd.row, cmd.col)
            suggested = self._reach.suggest_targets(cmd.limb, circle.x, circle.y, ik_iterations=16)
            before_ee = self._sim.get_end_effector_positions()[cmd.limb]
            before_error = ((before_ee["x"] - circle.x) ** 2 + (before_ee["y"] - circle.y) ** 2) ** 0.5
            # #region agent log
            _debug_log(
                "pre-fix",
                "H1,H3",
                "env.py:_apply_action.ik_before",
                "before IK motion",
                {
                    "phase": self._phase,
                    "step": self._step_count,
                    "limb": cmd.limb,
                    "target": circle.to_dict(),
                    "end_effector": before_ee,
                    "xy_error": before_error,
                    "reach_vector": self._reach.reach_vector(cmd.limb, before_ee, circle.x, circle.y),
                    "primary_targets": {name: suggested[name] for name in self._reach.primary_joints(cmd.limb) if name in suggested},
                    "balance": {
                        "upright": self._sim.is_upright(),
                        "fallen": self._sim.has_fallen(),
                        "torso": self._sim.get_torso_state(),
                    },
                },
            )
            # #endregion
            if before_error <= PLACEMENT_RADIUS:
                perf.ik_ms = (time.perf_counter() - t0) * 1000.0
                return
            plan = self._planner.interpolate_to(suggested, steps=6)
            for step_targets in plan.steps:
                self._sim.set_targets(step_targets)
                self._sim.step_physics(substeps=max(2, PHYSICS_SUBSTEPS // 4))
            after_ee = self._sim.get_end_effector_positions()[cmd.limb]
            # #region agent log
            _debug_log(
                "pre-fix",
                "H1,H3,H4",
                "env.py:_apply_action.ik_after",
                "after IK motion",
                {
                    "phase": self._phase,
                    "step": self._step_count,
                    "limb": cmd.limb,
                    "target": circle.to_dict(),
                    "end_effector": after_ee,
                    "xy_error": ((after_ee["x"] - circle.x) ** 2 + (after_ee["y"] - circle.y) ** 2) ** 0.5,
                    "upright": self._sim.is_upright(),
                    "fallen": self._sim.has_fallen(),
                    "torso": self._sim.get_torso_state(),
                    "contacts": self._sim.get_contacts(),
                },
            )
            # #endregion
            perf.ik_ms = (time.perf_counter() - t0) * 1000.0
            return
        targets = action.get("joint_targets", {})
        if targets:
            self._sim.set_targets(
                targets,
                delta=bool(action.get("delta", False)),
                max_delta=float(action.get("max_delta", 8.0)),
            )
        perf.ik_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        self._sim.step_physics()
        perf.physics_ms = (time.perf_counter() - t1) * 1000.0
        perf.balance_ms = 0.0

    def _run_controller(self, controller: dict[str, Any]) -> dict[str, Any]:
        language = controller.get("language", "python")
        code = str(controller.get("code", ""))
        stem = f"controller_step{self._step_count:05d}"
        if language == "javascript":
            path = _CONTROLLER_DIR / f"{stem}.js"
            path.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                ["node", str(path)],
                input=json.dumps(self._last_world_obs),
                text=True,
                capture_output=True,
                check=False,
                timeout=8,
            )
        else:
            path = _CONTROLLER_DIR / f"{stem}.py"
            path.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                ["python3", str(path)],
                input=json.dumps(self._last_world_obs),
                text=True,
                capture_output=True,
                check=False,
                timeout=8,
            )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"use_ik": True}
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return payload.get("action", payload)
        except Exception:
            return {"use_ik": True}

    @staticmethod
    def _parse_action(action: Any) -> dict[str, Any]:
        if isinstance(action, dict):
            return action
        if isinstance(action, str):
            try:
                loaded = json.loads(action)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                return {"use_ik": True}
        return {"use_ik": True}


class MyEnv(TwisterEnv):
    """Backwards alias for local scaffold scripts."""
