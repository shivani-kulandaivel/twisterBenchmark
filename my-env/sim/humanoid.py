"""MuJoCo humanoid wrapper with position-controlled joints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco

from sim.constants import (
    END_EFFECTOR_SITES,
    JOINT_SCHEMA,
    MAX_TORSO_TILT_DEG,
    MIN_TORSO_HEIGHT,
    PHYSICS_SUBSTEPS,
    PHYSICS_TIMESTEP,
    TORSO_BODY,
)

_MJCF_PATH = Path(__file__).parent / "mjcf" / "humanoid_twister.xml"


class HumanoidSim:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = PHYSICS_TIMESTEP

        self._joint_qpos_idx: dict[str, int] = {}
        self._joint_qvel_idx: dict[str, int] = {}
        self._actuator_idx: dict[str, int] = {}

        for name in JOINT_SCHEMA:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found in MJCF: {name}")
            self._joint_qpos_idx[name] = self.model.jnt_qposadr[joint_id]
            self._joint_qvel_idx[name] = self.model.jnt_dofadr[joint_id]
            act_name = f"act_{name}"
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id < 0:
                raise RuntimeError(f"Actuator not found in MJCF: {act_name}")
            self._actuator_idx[name] = act_id

        self._site_ids = {
            limb: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
            for limb, site in END_EFFECTOR_SITES.items()
        }
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)

        self._targets: dict[str, float] = {
            name: spec["neutral"] for name, spec in JOINT_SCHEMA.items()
        }

    def reset(self, seed: int | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)

        for name, spec in JOINT_SCHEMA.items():
            idx = self._joint_qpos_idx[name]
            self.data.qpos[idx] = spec["neutral"]
            self._targets[name] = spec["neutral"]

        self._apply_targets()
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def clamp_joint(self, name: str, value_deg: float) -> float:
        spec = JOINT_SCHEMA[name]
        return float(max(spec["low"], min(spec["high"], value_deg)))

    def set_targets(
        self,
        joint_targets: dict[str, float],
        *,
        delta: bool = False,
        max_delta: float = 15.0,
    ) -> None:
        for name, value in joint_targets.items():
            if name not in JOINT_SCHEMA:
                continue
            if delta:
                current = self._targets[name]
                delta_val = max(-max_delta, min(max_delta, float(value)))
                value = current + delta_val
            self._targets[name] = self.clamp_joint(name, float(value))

    def step_physics(self, substeps: int = PHYSICS_SUBSTEPS) -> None:
        self._apply_targets()
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

    def _apply_targets(self) -> None:
        for name, target in self._targets.items():
            self.data.ctrl[self._actuator_idx[name]] = target

    def get_joint_angles_deg(self) -> dict[str, float]:
        return {
            name: round(float(self.data.qpos[idx]), 2)
            for name, idx in self._joint_qpos_idx.items()
        }

    def get_joint_targets_deg(self) -> dict[str, float]:
        return dict(self._targets)

    def get_end_effector_positions(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for limb, site_id in self._site_ids.items():
            pos = self.data.site_xpos[site_id]
            result[limb] = {
                "x": round(float(pos[0]), 4),
                "y": round(float(pos[1]), 4),
                "z": round(float(pos[2]), 4),
            }
        return result

    def get_torso_state(self) -> dict[str, float]:
        pos = self.data.xpos[self._torso_id]
        mat = self.data.xmat[self._torso_id].reshape(3, 3)
        up = mat[:, 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
        return {
            "x": round(float(pos[0]), 4),
            "y": round(float(pos[1]), 4),
            "z": round(float(pos[2]), 4),
            "tilt_deg": round(tilt, 2),
        }

    def is_upright(self) -> bool:
        torso = self.get_torso_state()
        return torso["z"] >= MIN_TORSO_HEIGHT and torso["tilt_deg"] <= MAX_TORSO_TILT_DEG

    def has_fallen(self) -> bool:
        return not self.is_upright()

    def snapshot(self) -> dict[str, Any]:
        return {
            "joints": self.get_joint_angles_deg(),
            "joint_targets": self.get_joint_targets_deg(),
            "end_effectors": self.get_end_effector_positions(),
            "torso": self.get_torso_state(),
            "upright": self.is_upright(),
        }
