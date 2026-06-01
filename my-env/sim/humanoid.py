"""MuJoCo humanoid wrapper with position-controlled joints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sim.constants import (
    BALANCE_COM_KD,
    BALANCE_COM_KP,
    BALANCE_F_MAX,
    BALANCE_ROT_KD,
    BALANCE_ROT_KP,
    BALANCE_T_MAX,
    END_EFFECTOR_SITES,
    JOINT_SCHEMA,
    MAX_TORSO_TILT_DEG,
    MIN_TORSO_HEIGHT,
    PHYSICS_SUBSTEPS,
    PHYSICS_TIMESTEP,
    SMOOTH_MAX_DELTA_DEG,
    TORSO_BODY,
)

_MJCF_PATH = Path(__file__).parent / "mjcf" / "humanoid_twister.xml"
_Z_WORLD = np.array([0.0, 0.0, 1.0])


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
        self._pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        root_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_joint < 0:
            raise RuntimeError("Free joint 'root' not found in MJCF")
        self._root_qpos_idx = self.model.jnt_qposadr[root_joint]
        self._root_qvel_idx = self.model.jnt_dofadr[root_joint]

        self._foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("left_foot_geom", "right_foot_geom")
        ]
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # Bodies whose contact with the floor counts as a collapse (NOT feet,
        # NOT forearms/hands — those are legal contact points in Twister).
        _collapse_bodies = (
            "pelvis", "lumbar", "torso", "head",
            "left_thigh", "right_thigh", "left_shin", "right_shin",
            "left_upper_arm", "right_upper_arm",
        )
        self._collapse_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in _collapse_bodies
        }

        self._targets: dict[str, float] = {
            name: spec["neutral"] for name, spec in JOINT_SCHEMA.items()
        }
        self._foot_site_to_geom = {
            "left_foot": self._foot_geom_ids[0],
            "right_foot": self._foot_geom_ids[1],
        }
        self._active_reach_limb: str | None = None
        self._lean_bias = (0.0, 0.0)
        self._support_override_xy: list[tuple[float, float]] | None = None

    def reset(self, seed: int | None = None) -> None:
        del seed
        mujoco.mj_resetData(self.model, self.data)

        root = self._root_qpos_idx
        self.data.qpos[root : root + 3] = 0.0
        self.data.qpos[root + 3 : root + 7] = (1.0, 0.0, 0.0, 0.0)

        for name, spec in JOINT_SCHEMA.items():
            idx = self._joint_qpos_idx[name]
            self.data.qpos[idx] = math.radians(spec["neutral"])
            self._targets[name] = spec["neutral"]

        self._snap_feet_to_ground()
        self.data.qvel[:] = 0.0
        self._apply_targets()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(150):
            self._apply_targets()
            mujoco.mj_step(self.model, self.data)

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

    def set_targets_smooth(self, joint_targets: dict[str, float]) -> None:
        """Set absolute targets with per-substep rate limiting during physics."""
        self._pending_targets = {
            name: self.clamp_joint(name, float(val)) for name, val in joint_targets.items()
        }

    def _advance_smooth_targets(self) -> None:
        pending = getattr(self, "_pending_targets", None)
        if not pending:
            return
        done = True
        for name, goal in pending.items():
            current = self._targets[name]
            step = max(-SMOOTH_MAX_DELTA_DEG, min(SMOOTH_MAX_DELTA_DEG, goal - current))
            self._targets[name] = current + step
            if abs(goal - self._targets[name]) > 0.05:
                done = False
        if done:
            self._pending_targets = None

    def step_physics(self, substeps: int = PHYSICS_SUBSTEPS) -> None:
        for _ in range(substeps):
            self._advance_smooth_targets()
            self._apply_targets()
            self._apply_balance_assist()
            mujoco.mj_step(self.model, self.data)

    def _apply_targets(self) -> None:
        # Targets are stored in degrees (human-facing); MuJoCo ctrl is radians.
        for name, target in self._targets.items():
            self.data.ctrl[self._actuator_idx[name]] = math.radians(target)

    def _apply_balance_assist(self) -> None:
        """Virtual-model balance: PD force/torque on the pelvis to keep the
        center of mass over the support polygon and the trunk upright.

        Models the stabilizing role a real player's core and stance muscles
        provide. Keeps full limb physics while preventing the whole body from
        toppling — a standard balance-assist / virtual model control technique.
        """
        pelvis = self._pelvis_id
        dof = self._root_qvel_idx

        # --- CoM over support (horizontal PD) ---
        cx, cy = self.com_xy()
        sx, sy = self.support_center_xy()
        com_vx = float(self.data.cvel[pelvis][3])  # linear vel proxy
        com_vy = float(self.data.cvel[pelvis][4])
        fx = BALANCE_COM_KP * (sx - cx) - BALANCE_COM_KD * com_vx
        fy = BALANCE_COM_KP * (sy - cy) - BALANCE_COM_KD * com_vy

        # --- Trunk upright (rotational PD about X and Y) ---
        up = self.data.xmat[self._pelvis_id].reshape(3, 3)[:, 2]
        # restoring torque ~ up x z_world
        rest = np.cross(up, _Z_WORLD)
        wx = float(self.data.qvel[dof + 3])
        wy = float(self.data.qvel[dof + 4])
        tx = BALANCE_ROT_KP * rest[0] - BALANCE_ROT_KD * wx
        ty = BALANCE_ROT_KP * rest[1] - BALANCE_ROT_KD * wy

        self.data.xfrc_applied[pelvis, 0] = float(np.clip(fx, -BALANCE_F_MAX, BALANCE_F_MAX))
        self.data.xfrc_applied[pelvis, 1] = float(np.clip(fy, -BALANCE_F_MAX, BALANCE_F_MAX))
        self.data.xfrc_applied[pelvis, 3] = float(np.clip(tx, -BALANCE_T_MAX, BALANCE_T_MAX))
        self.data.xfrc_applied[pelvis, 4] = float(np.clip(ty, -BALANCE_T_MAX, BALANCE_T_MAX))

    def com_xy(self) -> tuple[float, float]:
        """Whole-body center of mass (world XY)."""
        total = float(self.model.body_mass.sum())
        cx = float(sum(self.model.body_mass[i] * self.data.subtree_com[i][0] for i in range(self.model.nbody)) / total)
        cy = float(sum(self.model.body_mass[i] * self.data.subtree_com[i][1] for i in range(self.model.nbody)) / total)
        return cx, cy

    def set_active_reach(
        self,
        limb: str | None,
        lean_bias: tuple[float, float] = (0.0, 0.0),
        support_override_xy: list[tuple[float, float]] | None = None,
    ) -> None:
        """Tell the balance controller which limb is reaching (so a lifting foot
        is excluded from the support polygon) and how far to shift weight toward
        the reach."""
        self._active_reach_limb = limb
        self._lean_bias = lean_bias
        self._support_override_xy = support_override_xy

    def support_center_xy(self) -> tuple[float, float]:
        """Center of support polygon (world XY).

        Uses planted feet plus any additional end-effectors already grounded
        (e.g., a locked hand in phase 2). This gives the balance assist a
        realistic multi-contact support estimate.
        """
        if self._support_override_xy:
            cx = sum(p[0] for p in self._support_override_xy) / len(self._support_override_xy)
            cy = sum(p[1] for p in self._support_override_xy) / len(self._support_override_xy)
            return float(cx + self._lean_bias[0]), float(cy + self._lean_bias[1])

        planted = list(self._foot_geom_ids)
        if self._active_reach_limb in self._foot_site_to_geom:
            lifting = self._foot_site_to_geom[self._active_reach_limb]
            planted = [g for g in self._foot_geom_ids if g != lifting]

        pts = [self.data.geom_xpos[gid] for gid in planted]
        for limb, site_id in self._site_ids.items():
            if limb.endswith("foot"):
                continue
            if limb == self._active_reach_limb:
                continue
            pos = self.data.site_xpos[site_id]
            if float(pos[2]) <= 0.11:
                pts.append(pos)
        cx = sum(p[0] for p in pts) / len(pts) + self._lean_bias[0]
        cy = sum(p[1] for p in pts) / len(pts) + self._lean_bias[1]
        return float(cx), float(cy)

    def get_joint_angles_deg(self) -> dict[str, float]:
        return {
            name: round(math.degrees(float(self.data.qpos[idx])), 2)
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
        """Kept for observation/back-compat: true while standing tall and level."""
        torso = self.get_torso_state()
        return torso["z"] >= MIN_TORSO_HEIGHT and torso["tilt_deg"] <= MAX_TORSO_TILT_DEG

    def has_fallen(self) -> bool:
        """A Twister 'fall' is a collapse: a non-hand/foot body part touching the
        mat, the head/pelvis dropping to the floor, or the body tumbling out of
        control. Bending, crouching, or going on all fours is allowed."""
        reaching = self._active_reach_limb is not None
        # 1) Collapse contact: a 'core' body geom touches the floor.
        floor = self._floor_geom_id
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if floor not in (g1, g2):
                continue
            other = g2 if g1 == floor else g1
            body = int(self.model.geom_bodyid[other])
            if body in self._collapse_body_ids:
                return True
        # 2) Pelvis/head dropped near the floor (face-plant / sat down hard).
        pelvis_thresh = 0.28 if reaching else 0.32
        if float(self.data.xpos[self._pelvis_id][2]) < pelvis_thresh:
            return True
        # 3) Tumbling: large angular velocity of the root.
        wr = self.data.qvel[self._root_qvel_idx + 3 : self._root_qvel_idx + 6]
        tumble_thresh = 14.0 if reaching else 12.0
        if float(np.linalg.norm(wr)) > tumble_thresh:
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "joints": self.get_joint_angles_deg(),
            "joint_targets": self.get_joint_targets_deg(),
            "end_effectors": self.get_end_effector_positions(),
            "torso": self.get_torso_state(),
            "upright": self.is_upright(),
        }
