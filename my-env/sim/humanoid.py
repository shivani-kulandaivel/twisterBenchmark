"""MuJoCo humanoid wrapper with position-controlled joints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

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

        root_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_joint < 0:
            raise RuntimeError("Free joint 'root' not found in MJCF")
        self._root_qpos_idx = self.model.jnt_qposadr[root_joint]
        self._root_qvel_idx = self.model.jnt_dofadr[root_joint]

        self._foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("left_foot_geom", "right_foot_geom")
        ]

        self._targets: dict[str, float] = {
            name: spec["neutral"] for name, spec in JOINT_SCHEMA.items()
        }
        self._renderer: mujoco.Renderer | None = None

    def get_qpos(self) -> list[float]:
        return [float(v) for v in self.data.qpos]

    def set_qpos(self, qpos: list[float]) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def apply_joints_deg(self, joints: dict[str, float]) -> None:
        for name, value in joints.items():
            if name in self._joint_qpos_idx:
                self.data.qpos[self._joint_qpos_idx[name]] = float(value)
        mujoco.mj_forward(self.model, self.data)

    def render_png(self, width: int = 960, height: int = 720) -> bytes:
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera="showcase")
        flags = self._renderer.scene.flags
        for flag in ("mjRND_SHADOW", "mjRND_REFLECTION", "mjRND_SKYBOX", "mjRND_HAZE"):
            idx = getattr(mujoco.mjtRndFlag, flag, None)
            if idx is not None:
                flags[idx] = 1
        rgb = self._renderer.render()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install pillow for MuJoCo rendering: pip install pillow") from exc
        import io

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    def reset(self, seed: int | None = None) -> None:
        del seed
        mujoco.mj_resetData(self.model, self.data)

        root = self._root_qpos_idx
        self.data.qpos[root : root + 3] = 0.0
        self.data.qpos[root + 3 : root + 7] = (1.0, 0.0, 0.0, 0.0)

        for name, spec in JOINT_SCHEMA.items():
            idx = self._joint_qpos_idx[name]
            self.data.qpos[idx] = spec["neutral"]
            self._targets[name] = spec["neutral"]

        self._snap_feet_to_ground()
        self.data.qvel[:] = 0.0
        self._apply_targets()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(150):
            self._apply_targets()
            mujoco.mj_step(self.model, self.data)
            self._anchor_root()

    def _snap_feet_to_ground(self, ground_z: float = 0.0) -> None:
        """Translate the floating base so the lowest foot sole rests on the floor."""
        mujoco.mj_forward(self.model, self.data)
        lowest = min(
            float(self.data.geom_xpos[gid][2] - self.model.geom_size[gid][2])
            for gid in self._foot_geom_ids
        )
        self.data.qpos[self._root_qpos_idx + 2] += ground_z - lowest
        mujoco.mj_forward(self.model, self.data)

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

    # Pelvis anchor limits: the body may shift/lean/crouch within these bounds
    # but is kinematically prevented from toppling or walking away.
    _ANCHOR_XY = 0.22       # max pelvis horizontal drift (m)
    _ANCHOR_Z = (0.55, 1.0)  # pelvis height range (crouch .. stand)
    _ANCHOR_MAX_TILT = 45.0  # max pelvis lean before snapping upright (deg)

    def _anchor_root(self) -> None:
        """Keep the free pelvis within a stable envelope: clamp horizontal drift
        and height, and cap tilt (preserving heading) so the body cannot topple."""
        r = self._root_qpos_idx
        v = self._root_qvel_idx
        for i in (0, 1):
            p = float(self.data.qpos[r + i])
            if abs(p) > self._ANCHOR_XY:
                self.data.qpos[r + i] = math.copysign(self._ANCHOR_XY, p)
                self.data.qvel[v + i] = 0.0
        z = float(self.data.qpos[r + 2])
        if z < self._ANCHOR_Z[0] or z > self._ANCHOR_Z[1]:
            self.data.qpos[r + 2] = max(self._ANCHOR_Z[0], min(self._ANCHOR_Z[1], z))
            self.data.qvel[v + 2] = 0.0

        q = self.data.qpos[r + 3 : r + 7].copy()
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, q)
        R = mat.reshape(3, 3)
        up_z = max(-1.0, min(1.0, float(R[2, 2])))
        tilt = math.degrees(math.acos(up_z))
        if tilt > self._ANCHOR_MAX_TILT:
            yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
            half = yaw / 2.0
            self.data.qpos[r + 3 : r + 7] = (math.cos(half), 0.0, 0.0, math.sin(half))
            self.data.qvel[v + 3 : v + 6] = 0.0

    def step_physics(self, substeps: int = PHYSICS_SUBSTEPS) -> None:
        for _ in range(substeps):
            self._apply_targets()
            mujoco.mj_step(self.model, self.data)
            self._anchor_root()

    def _postural_tilt_deg(self) -> tuple[float, float]:
        up = self.data.xmat[self._torso_id].reshape(3, 3)[:, 2]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(up[0])))))
        roll = math.degrees(math.asin(max(-1.0, min(1.0, float(up[1])))))
        return pitch, roll

    def _stabilize(self, name: str, target: float, pitch: float, roll: float) -> float:
        # The pelvis is now kinematically anchored, so we no longer perturb joint
        # targets to chase balance — joints track the commanded (IK) targets
        # directly, which is far more accurate for placement.
        return target

    def _apply_targets(self) -> None:
        pitch, roll = self._postural_tilt_deg()
        for name, target in self._targets.items():
            self.data.ctrl[self._actuator_idx[name]] = self._stabilize(name, target, pitch, roll)

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

    # Joints that kinematically drive each end-effector (ancestors in the tree)
    # plus the support joints we allow the IK to recruit for leaning/crouching.
    _IK_CHAIN: dict[str, list[str]] = {
        "left_hand":  ["abdomen_pitch", "left_shoulder_pitch", "left_shoulder_roll", "left_elbow"],
        "right_hand": ["abdomen_pitch", "right_shoulder_pitch", "right_shoulder_roll", "right_elbow"],
        "left_foot":  ["left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle"],
        "right_foot": ["right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle"],
    }

    def _reach_posture(self, limb: str, tx: float, ty: float) -> dict[str, float]:
        """Gentle seed posture. With the pelvis softly anchored, the Jacobian IK
        chain (abdomen + arm for hands) reaches on its own, so we only add a light
        forward lean for hands to favour solutions that bend toward the mat."""
        posture: dict[str, float] = {}
        if limb in ("left_hand", "right_hand"):
            posture["abdomen_pitch"] = max(-20.0, min(20.0, ty * 18.0))
        return {k: self.clamp_joint(k, v) for k, v in posture.items()}

    def solve_ik(
        self,
        limb: str,
        target: tuple[float, float, float],
        *,
        max_iters: int = 60,
        tol: float = 0.01,
        damping: float = 0.12,
        max_step_deg: float = 12.0,
        with_posture: bool = True,
    ) -> dict[str, float]:
        """Damped-least-squares IK for one end-effector, solved on a scratch copy
        of the current state so the live sim is untouched. Returns target joint
        angles in degrees for the limb chain (and a lean/crouch posture)."""
        chain = self._IK_CHAIN.get(limb, [])
        if not chain:
            return {}

        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0.0

        # Seed with the reach posture so the optimizer starts from a leaned pose.
        posture = self._reach_posture(limb, target[0], target[1]) if with_posture else {}
        for name, val in posture.items():
            d.qpos[self._joint_qpos_idx[name]] = math.radians(val)

        site_id = self._site_ids[limb]
        dof_idx = [self._joint_qvel_idx[name] for name in chain]
        qpos_idx = [self._joint_qpos_idx[name] for name in chain]
        lows = np.array([math.radians(JOINT_SCHEMA[n]["low"]) for n in chain])
        highs = np.array([math.radians(JOINT_SCHEMA[n]["high"]) for n in chain])
        tgt = np.array(target, dtype=float)
        max_step = math.radians(max_step_deg)

        # Hands can't physically reach floor height (arm shorter than shoulder
        # height), and placement is scored on XY only — so down-weight the Z error
        # for hands to prioritise getting the XY right.
        z_weight = 0.2 if limb in ("left_hand", "right_hand") else 1.0

        jacp = np.zeros((3, self.model.nv))
        for _ in range(max_iters):
            mujoco.mj_forward(self.model, d)
            cur = d.site_xpos[site_id].copy()
            err = tgt - cur
            err[2] *= z_weight
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.model, d, jacp, None, site_id)
            J = jacp[:, dof_idx]                      # 3 x k
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)      # k
            dq = np.clip(dq, -max_step, max_step)
            q = d.qpos[qpos_idx] + dq
            q = np.clip(q, lows, highs)
            d.qpos[qpos_idx] = q

        result = {name: round(math.degrees(float(d.qpos[self._joint_qpos_idx[name]])), 1)
                  for name in chain}
        # Include the support posture (legs) for hands so the demo/LLM can crouch.
        for name, val in posture.items():
            result.setdefault(name, round(val, 1))
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "joints": self.get_joint_angles_deg(),
            "joint_targets": self.get_joint_targets_deg(),
            "end_effectors": self.get_end_effector_positions(),
            "torso": self.get_torso_state(),
            "upright": self.is_upright(),
        }
