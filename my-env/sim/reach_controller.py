"""
Whole-body reach controller for the Twister humanoid.

Implements a hierarchical operational-space / whole-body controller
(Khatib 1987; Sentis & Khatib 2006; multi-contact WBC):

  Priority 1 — keep the center of mass over the foot support polygon (balance)
  Priority 2 — drive the commanded end-effector toward the target circle
  Priority 3 — regularize toward a neutral, athletic posture

Balance is the highest priority: an unbalanced reach pose is useless because
the body falls. Lower-priority tasks are projected into the null space of the
balance task so they never compromise stability.

The controller works purely in joint space (position-controlled hinges) and
never teleports the floating base, which keeps the physics stable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from sim.constants import JOINT_SCHEMA, TORSO_BODY

if TYPE_CHECKING:
    from sim.humanoid import HumanoidSim

# Max joint change per IK iteration (radians ~ 5 deg).
_MAX_DQ = math.radians(5.0)
_Z_UP = np.array([0.0, 0.0, 1.0])

CONTACT_Z: dict[str, float] = {
    "left_hand": 0.06,
    "right_hand": 0.06,
    "left_foot": 0.028,
    "right_foot": 0.028,
}


class ReachController:
    """Hierarchical IK: balance (CoM) > reach (end-effector) > posture."""

    def __init__(self, sim: HumanoidSim) -> None:
        self._sim = sim
        model = sim.model
        self._hinge_names = list(JOINT_SCHEMA.keys())
        self._hinge_qpos_idx = np.array(
            [sim._joint_qpos_idx[n] for n in self._hinge_names], dtype=np.int32
        )
        self._hinge_dof_idx = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self._hinge_names],
            dtype=np.int32,
        )
        self._n = len(self._hinge_names)
        self._torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
        self._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._name_to_hinge = {n: i for i, n in enumerate(self._hinge_names)}
        # Floating-base IK uses the 6 root DOFs (so it can plan a squat/lean)
        # plus the hinge DOFs. Only hinge angles are commanded; the base motion
        # is realized by physics + the balance assist.
        self._root_dof = int(sim._root_qvel_idx)
        self._root_qpos = int(sim._root_qpos_idx)
        self._root_dof_idx = np.arange(self._root_dof, self._root_dof + 6, dtype=np.int32)
        self._ik_dof_idx = np.concatenate([self._root_dof_idx, self._hinge_dof_idx])
        # Internals work in radians (MuJoCo-native); the public API is degrees.
        self._q_neutral = np.deg2rad(
            [JOINT_SCHEMA[n]["neutral"] for n in self._hinge_names]
        )
        self._lo = np.deg2rad([JOINT_SCHEMA[n]["low"] for n in self._hinge_names])
        self._hi = np.deg2rad([JOINT_SCHEMA[n]["high"] for n in self._hinge_names])

    # ---- public helpers -------------------------------------------------

    def contact_z(self, limb: str) -> float:
        return CONTACT_Z.get(limb, 0.06)

    def target_xyz(
        self,
        limb: str,
        x: float,
        y: float,
        z_override: float | None = None,
    ) -> tuple[float, float, float]:
        return (float(x), float(y), float(self.contact_z(limb) if z_override is None else z_override))

    def reach_vector(
        self, limb: str, end_effector: dict[str, float], target_x: float, target_y: float
    ) -> dict[str, float]:
        tz = self.contact_z(limb)
        return {
            "dx": round(target_x - end_effector["x"], 4),
            "dy": round(target_y - end_effector["y"], 4),
            "dz": round(tz - end_effector["z"], 4),
            "target_z": tz,
        }

    def primary_joints(self, limb: str) -> list[str]:
        side = "left" if limb.startswith("left") else "right"
        if limb.endswith("hand"):
            return [
                f"{side}_shoulder_pitch",
                f"{side}_shoulder_roll",
                f"{side}_elbow",
                "abdomen_pitch",
                f"{side}_hip_pitch",
                f"{side}_hip_roll",
            ]
        return [
            f"{side}_hip_pitch",
            f"{side}_hip_roll",
            f"{side}_knee",
            f"{side}_ankle",
            "abdomen_pitch",
        ]

    def suggest_targets(
        self,
        limb: str,
        target_x: float,
        target_y: float,
        *,
        ik_iterations: int = 80,
        anchor_limbs: list[str] | None = None,
        target_z: float | None = None,
    ) -> dict[str, float]:
        """Solve balanced IK from the current pose; return absolute joint targets."""
        sim = self._sim
        data = sim.data
        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()
        try:
            # Capture the planted-foot world positions to hold them fixed
            # (closed-chain constraint): both feet for a hand reach, the
            # stance foot only for a foot reach.
            planted_sites = self._anchor_sites(limb, anchor_limbs)
            mujoco.mj_forward(sim.model, data)
            planted_anchor = {
                sid: data.site_xpos[sid].copy() for sid in planted_sites
            }
            target = np.array(self.target_xyz(limb, target_x, target_y, target_z), dtype=np.float64)
            for _ in range(ik_iterations):
                self._ik_step(limb, target, planted_anchor)
            return {
                n: float(np.rad2deg(data.qpos[self._hinge_qpos_idx[i]]))
                for i, n in enumerate(self._hinge_names)
            }
        finally:
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            mujoco.mj_forward(sim.model, data)

    def _planted_foot_sites(self, limb: str) -> list[int]:
        sim = self._sim
        feet = ["left_foot", "right_foot"]
        if limb in feet:
            feet = [f for f in feet if f != limb]
        return [sim._site_ids[f] for f in feet]

    def _anchor_sites(self, limb: str, anchor_limbs: list[str] | None) -> list[int]:
        """Sites to keep fixed while solving IK."""
        sim = self._sim
        sites = set(self._planted_foot_sites(limb))
        if anchor_limbs:
            for l in anchor_limbs:
                if l == limb:
                    continue
                sid = sim._site_ids.get(l)
                if sid is not None:
                    sites.add(sid)
        return list(sites)

    def apply_step(
        self,
        limb: str,
        target_x: float,
        target_y: float,
        *,
        max_joint_rate: float = 3.0,
        anchor_limbs: list[str] | None = None,
    ) -> None:
        """Advance one control step: rate-limit current targets toward the balanced IK pose."""
        if limb.endswith("hand"):
            self._apply_hand_step(limb, target_x, target_y, max_joint_rate=max_joint_rate)
            return

        # Tell the balance controller which foot is planted and to weight-shift
        # toward the reach (human strategy), keeping the lean inside the support.
        self._sim.set_active_reach(limb, (0.0, 0.0))
        lean = self._lean_bias(limb, target_x, target_y)
        self._sim.set_active_reach(limb, lean)

        # Foot swing primitive: when far from the target, lift the foot so it can
        # move through free space instead of scraping the floor.
        target_z = None
        if limb.endswith("foot"):
            ee = self._sim.get_end_effector_positions()[limb]
            dist_xy = math.hypot(target_x - ee["x"], target_y - ee["y"])
            if dist_xy > 0.10:
                target_z = min(0.18, self.contact_z(limb) + 0.10 + 0.20 * dist_xy)
            else:
                target_z = self.contact_z(limb)

        goal = self.suggest_targets(
            limb,
            target_x,
            target_y,
            ik_iterations=80,
            anchor_limbs=anchor_limbs,
            target_z=target_z,
        )
        ee = self._sim.get_end_effector_positions()[limb]
        dist_xy = math.hypot(target_x - ee["x"], target_y - ee["y"])
        self._inject_hip_bend_fallback(
            goal,
            limb=limb,
            dist_xy=dist_xy,
            dx=target_x - ee["x"],
            dy=target_y - ee["y"],
        )
        current = self._sim.get_joint_targets_deg()
        stepped: dict[str, float] = {}
        joint_rate = max_joint_rate + (2.0 if limb.endswith("foot") else 0.0)
        for name, g in goal.items():
            c = current.get(name, JOINT_SCHEMA[name]["neutral"])
            stepped[name] = c + float(np.clip(g - c, -joint_rate, joint_rate))
        # Smooth updates over physics substeps for fluid, human-like motion.
        self._sim.set_targets_smooth(stepped)

    def _apply_hand_step(
        self,
        limb: str,
        target_x: float,
        target_y: float,
        *,
        max_joint_rate: float,
    ) -> None:
        """Dedicated hand controller.

        The whole-body closed-chain solver is useful for foot placement, but it
        can over-constrain arm motion. For hands we use a direct Jacobian step
        on arm + trunk joints so the hand visibly translates toward the target
        every turn.
        """
        sim = self._sim
        model, data = sim.model, sim.data
        side = "left" if limb.startswith("left") else "right"
        joint_names = [
            "abdomen_pitch",
            f"{side}_shoulder_pitch",
            f"{side}_shoulder_roll",
            f"{side}_elbow",
            "left_hip_pitch",
            "right_hip_pitch",
            "left_hip_roll",
            "right_hip_roll",
            "left_knee",
            "right_knee",
        ]
        cols = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names],
            dtype=np.int32,
        )
        site_id = sim._site_ids[limb]
        ee = sim.get_end_effector_positions()[limb]
        dist_xy = math.hypot(target_x - ee["x"], target_y - ee["y"])
        dx = target_x - ee["x"]
        dy = target_y - ee["y"]
        # Move in two phases: hover while far, then drop onto the circle.
        tz = 0.24 if dist_xy > 0.18 else self.contact_z(limb)
        target = np.array([target_x, target_y, tz], dtype=np.float64)

        # Hand reaches benefit from a bit more CoM shift.
        sim.set_active_reach(limb, self._lean_bias(limb, target_x, target_y))

        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        j = jp[:, cols]
        err = target - data.site_xpos[site_id]
        n = np.linalg.norm(err)
        if n > 0.08:
            err = err * (0.08 / n)

        dq = self._dls(j, damping=0.12) @ err
        max_rad = math.radians(max(2.5, max_joint_rate))
        dq = np.clip(dq, -max_rad, max_rad)

        current = sim.get_joint_targets_deg()
        stepped = {}
        for name, d in zip(joint_names, dq):
            stepped[name] = current.get(name, JOINT_SCHEMA[name]["neutral"]) + math.degrees(float(d))
        self._inject_hip_bend_fallback(
            stepped,
            limb=limb,
            dist_xy=dist_xy,
            dx=dx,
            dy=dy,
        )
        sim.set_targets_smooth(stepped)

    @staticmethod
    def _clip_joint(name: str, value: float) -> float:
        spec = JOINT_SCHEMA[name]
        return float(np.clip(value, spec["low"], spec["high"]))

    def _inject_hip_bend_fallback(
        self,
        targets: dict[str, float],
        *,
        limb: str,
        dist_xy: float,
        dx: float,
        dy: float,
    ) -> None:
        """Hinge at hips/torso when the target is hard to reach.

        Combines forward/back bend and lateral hip twist so the figure moves
        like a person leaning and rotating at the hips instead of staying rigid.
        """
        # Forward/back hip + trunk hinge (engages only on genuinely far reaches
        # so close, easy targets keep a natural near-upright posture).
        if dist_xy >= 0.22:
            bend = min(28.0, 8.0 + 48.0 * (dist_xy - 0.22))
            if dy >= 0.0:
                targets["left_hip_pitch"] = self._clip_joint(
                    "left_hip_pitch", min(targets.get("left_hip_pitch", 0.0), -bend)
                )
                targets["right_hip_pitch"] = self._clip_joint(
                    "right_hip_pitch", max(targets.get("right_hip_pitch", 0.0), bend)
                )
                targets["abdomen_pitch"] = self._clip_joint(
                    "abdomen_pitch", min(targets.get("abdomen_pitch", 0.0), -0.45 * bend)
                )
            else:
                targets["left_hip_pitch"] = self._clip_joint(
                    "left_hip_pitch", max(targets.get("left_hip_pitch", 0.0), bend)
                )
                targets["right_hip_pitch"] = self._clip_joint(
                    "right_hip_pitch", min(targets.get("right_hip_pitch", 0.0), -bend)
                )
                targets["abdomen_pitch"] = self._clip_joint(
                    "abdomen_pitch", max(targets.get("abdomen_pitch", 0.0), 0.45 * bend)
                )
            if limb.endswith("foot"):
                knee_bend = min(28.0, 8.0 + 0.45 * bend)
                targets["left_knee"] = self._clip_joint(
                    "left_knee", max(targets.get("left_knee", 0.0), knee_bend)
                )
                targets["right_knee"] = self._clip_joint(
                    "right_knee", max(targets.get("right_knee", 0.0), knee_bend)
                )

        # Gentle lateral hip lean toward sideways targets adds whole-body
        # variety (left/right weight shift) without over-rotating the trunk.
        if dist_xy >= 0.16 and abs(dx) > 0.04:
            twist = min(18.0, 30.0 * (abs(dx) - 0.04))
            if dx > 0.0:
                targets["left_hip_roll"] = self._clip_joint(
                    "left_hip_roll", max(targets.get("left_hip_roll", 0.0), twist)
                )
            else:
                targets["right_hip_roll"] = self._clip_joint(
                    "right_hip_roll", min(targets.get("right_hip_roll", 0.0), -twist)
                )

    def _lean_bias(self, limb: str, target_x: float, target_y: float) -> tuple[float, float]:
        """Shift the CoM target toward the reach, clamped to the support margin."""
        sx, sy = self._sim.support_center_xy()  # current (already includes prior bias=0 at call time)
        # Direction from current support center toward the target.
        dx, dy = target_x - sx, target_y - sy
        # Allow a modest weight shift; hands lean more than feet (feet need a planted base).
        max_x, max_y = (0.22, 0.16) if limb.endswith("hand") else (0.08, 0.06)
        return (float(np.clip(dx, -max_x, max_x)), float(np.clip(dy, -max_y, max_y)))

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _dls(j: np.ndarray, damping: float) -> np.ndarray:
        m = j.shape[0]
        return j.T @ np.linalg.solve(j @ j.T + (damping**2) * np.eye(m), np.eye(m))

    # Max Cartesian step per IK iteration (m). Keeps the linearization valid so
    # large reaches (hand from shoulder height down to the floor) converge
    # smoothly instead of overshooting.
    _MAX_CART_STEP = 0.04

    def _ik_step(self, limb: str, target_pos: np.ndarray, planted_anchor: dict[int, np.ndarray]) -> None:
        """One closed-chain damped-least-squares IK iteration (mutates
        ``data.qpos`` in place).

        Highest priority: keep the planted feet fixed in the world (a contact
        constraint). The base may translate/rotate only as a *consequence* of
        joint motion consistent with planted feet, so reaching down makes the
        body squat/lean — exactly what the physics will reproduce. The reach
        task is solved in the null space of that constraint.
        """
        sim = self._sim
        model, data = sim.model, sim.data
        ik = self._ik_dof_idx
        nik = len(ik)
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))

        # --- Constraint: planted feet stay at their anchor positions ---
        rows = []
        errs = []
        for sid, anchor in planted_anchor.items():
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            rows.append(jp[:, ik].copy())
            errs.append(anchor - data.site_xpos[sid])
        j_c = np.vstack(rows)
        err_c = np.concatenate(errs)
        j_c_pinv = self._dls(j_c, damping=0.02)
        dq_c = j_c_pinv @ err_c
        null_c = np.eye(nik) - j_c_pinv @ j_c

        # --- Reach task in the constraint null space ---
        mujoco.mj_jacSite(model, data, jp, jr, sim._site_ids[limb])
        j_t = jp[:, ik]
        err_t = target_pos - data.site_xpos[sim._site_ids[limb]]
        n = np.linalg.norm(err_t)
        if n > self._MAX_CART_STEP:
            err_t = err_t * (self._MAX_CART_STEP / n)
        j_t_ns = j_t @ null_c
        j_t_ns_pinv = self._dls(j_t_ns, damping=0.06)
        dq_t = null_c @ (j_t_ns_pinv @ (err_t - j_t @ dq_c))

        # --- Posture + base regularization in the remaining null space ---
        null_t = null_c @ (np.eye(nik) - j_t_ns_pinv @ j_t_ns)
        sec = np.zeros(nik)
        sec[6:] = 0.05 * (self._q_neutral - data.qpos[self._hinge_qpos_idx])
        up = data.xmat[self._pelvis_id].reshape(3, 3)[:, 2]
        rest = np.cross(up, _Z_UP)
        root_z = float(data.qpos[self._root_qpos + 2])
        if limb.endswith("hand"):
            # Pike strategy: keep the pelvis HIGH and let the trunk fold forward
            # (don't fight pelvis pitch), so only hands + feet touch the floor.
            sec[3] = 0.0          # allow forward/back pelvis pitch (the fold)
            sec[4] = 0.8 * rest[1]  # strongly resist sideways (roll) tipping
            sec[5] = 0.8 * rest[2]
            sec[2] = 1.6 * (0.74 - root_z)  # strongly prefer a tall pelvis
            # bias knees toward straight (pike, not squat)
            for kn in ("left_knee", "right_knee"):
                sec[6 + self._name_to_hinge[kn]] += 0.15 * (0.0 - data.qpos[self._hinge_qpos_idx[self._name_to_hinge[kn]]])
        else:
            sec[3:6] = 0.8 * rest
            sec[2] = 0.7 * (0.62 - root_z)
        dq_post = null_t @ sec

        # Always apply the full constraint correction; only rate-limit the reach
        # and posture so clipping can never let the planted feet drift.
        dq_rp = np.clip(dq_t + dq_post, -_MAX_DQ, _MAX_DQ)
        dq_full = np.zeros(model.nv)
        dq_full[ik] = dq_c + dq_rp
        mujoco.mj_integratePos(model, data.qpos, dq_full, 1.0)
        data.qpos[self._hinge_qpos_idx] = np.clip(
            data.qpos[self._hinge_qpos_idx], self._lo, self._hi
        )
        mujoco.mj_forward(model, data)
