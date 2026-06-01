"""IK-verified reachability checks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco

from sim.balance_controller import BalanceController
from sim.constants import PLACEMENT_RADIUS
from sim.humanoid import HumanoidSim
from sim.reach_controller import ReachController

_DEBUG_LOG_PATH = Path("/Users/tanvi/Library/CloudStorage/OneDrive-UW/General/Hackathons/Sweccathon2026/twisterBenchmark/.cursor/debug-721478.log")


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, object]) -> None:
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


@dataclass
class ReachabilityResult:
    feasible: bool
    error_m: float
    margin: float
    suggested_pose: dict[str, float]


class ReachabilitySolver:
    def __init__(self, sim: HumanoidSim, reach: ReachController, balance: BalanceController) -> None:
        self._sim = sim
        self._reach = reach
        self._balance = balance
        self._debug_mutation_logs = 0

    def can_reach(self, limb: str, x: float, y: float) -> ReachabilityResult:
        before_ee = self._sim.get_end_effector_positions()
        before_time = float(self._sim.data.time)
        saved_qpos = self._sim.data.qpos.copy()
        saved_qvel = self._sim.data.qvel.copy()
        saved_ctrl = self._sim.data.ctrl.copy()
        saved_xfrc = self._sim.data.xfrc_applied.copy()
        suggested = self._reach.suggest_targets(limb, x, y, ik_iterations=40)
        saved = self._sim.get_joint_targets_deg()
        self._sim.set_targets(suggested, delta=False)
        self._sim.step_physics(substeps=8)
        ee = self._sim.get_end_effector_positions()[limb]
        error = ((ee["x"] - x) ** 2 + (ee["y"] - y) ** 2) ** 0.5
        margin = self._balance.compute().margin
        self._sim.set_targets(saved, delta=False)
        self._sim.data.qpos[:] = saved_qpos
        self._sim.data.qvel[:] = saved_qvel
        self._sim.data.ctrl[:] = saved_ctrl
        self._sim.data.xfrc_applied[:] = saved_xfrc
        self._sim.data.time = before_time
        mujoco.mj_forward(self._sim.model, self._sim.data)
        after_ee = self._sim.get_end_effector_positions()
        max_shift = max(
            ((after_ee[name]["x"] - before_ee[name]["x"]) ** 2 + (after_ee[name]["y"] - before_ee[name]["y"]) ** 2) ** 0.5
            for name in before_ee
        )
        if max_shift > 0.001 and self._debug_mutation_logs < 8:
            self._debug_mutation_logs += 1
            # region agent log
            _debug_log(
                "pre-fix",
                "H6",
                "reachability.py:can_reach",
                "reachability check mutated live sim",
                {
                    "limb": limb,
                    "target": {"x": x, "y": y},
                    "sim_time_before": before_time,
                    "sim_time_after": float(self._sim.data.time),
                    "max_end_effector_xy_shift": max_shift,
                    "before": before_ee,
                    "after": after_ee,
                },
            )
            # endregion
        return ReachabilityResult(
            feasible=(error <= PLACEMENT_RADIUS and margin > -0.01),
            error_m=float(error),
            margin=float(margin),
            suggested_pose=suggested,
        )
