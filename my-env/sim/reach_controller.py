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
        spine = ["lumbar_pitch", "lumbar_roll", "thoracic_pitch", "thoracic_yaw"]
        if limb.endswith("hand"):
            return [
                f"{side}_shoulder_pitch",
                f"{side}_shoulder_roll",
                f"{side}_elbow",
                *spine,
            ]
        return [
            f"{side}_hip_pitch",
            f"{side}_hip_roll",
            f"{side}_hip_yaw",
            f"{side}_knee",
            f"{side}_ankle",
            "lumbar_pitch",
            "lumbar_roll",
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

    def _limb_chain_joints(self, limb: str) -> list[str]:
        side = "left" if limb.startswith("left") else "right"
        if limb.endswith("hand"):
            return [f"{side}_shoulder_pitch", f"{side}_shoulder_roll", f"{side}_elbow"]
        return [
            f"{side}_hip_pitch",
            f"{side}_hip_roll",
            f"{side}_hip_yaw",
            f"{side}_knee",
            f"{side}_ankle",
        ]

    def correct_limb_xy(self, limb: str, target_x: float, target_y: float, *, iters: int = 14) -> None:
        """Post-physics kinematic correction to keep a locked limb on its circle."""
        sim = self._sim
        model, data = sim.model, sim.data
        tz = self.contact_z(limb)
        target = np.array([target_x, target_y, tz], dtype=np.float64)
        site_id = sim._site_ids[limb]
        chain = self._limb_chain_joints(limb)
        cols = np.array(
            [
                model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in chain
            ],
            dtype=np.int32,
        )
        for _ in range(iters):
            mujoco.mj_forward(model, data)
            err = target - data.site_xpos[site_id]
            if float(np.linalg.norm(err[:2])) < 0.006:
                break
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, site_id)
            dq = self._dls(jp[:, cols], damping=0.14) @ err
            dq = np.clip(dq, -math.radians(3.5), math.radians(3.5))
            for name, d in zip(chain, dq):
                idx = sim._joint_qpos_idx[name]
                lo = math.radians(JOINT_SCHEMA[name]["low"])
                hi = math.radians(JOINT_SCHEMA[name]["high"])
                data.qpos[idx] = float(np.clip(data.qpos[idx] + d, lo, hi))
                sim._targets[name] = math.degrees(data.qpos[idx])
        mujoco.mj_forward(model, data)

    def rehold_limb(self, limb: str, target_x: float, target_y: float) -> None:
        """Keep a locked end-effector on its circle using local joint corrections."""
        ee = self._sim.get_end_effector_positions()[limb]
        dist_xy = math.hypot(target_x - ee["x"], target_y - ee["y"])
        if dist_xy < 0.006:
            return
        self.correct_limb_xy(limb, target_x, target_y)

    def rehold_locked(
        self,
        locked: list[tuple[str, float, float]],
        *,
        skip_limb: str | None = None,
    ) -> None:
        for limb, x, y in locked:
            if limb == skip_limb:
                continue
            self.rehold_limb(limb, x, y)

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
            self._apply_hand_step(
                limb,
                target_x,
                target_y,
                max_joint_rate=max_joint_rate,
                anchor_limbs=anchor_limbs,
            )
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
            locked=bool(anchor_limbs),
            frozen_limbs=set(anchor_limbs or []),
        )
        current = self._sim.get_joint_targets_deg()
        stepped: dict[str, float] = {}
        joint_rate = max_joint_rate + (1.0 if limb.endswith("foot") else 0.0)
        if dist_xy > 0.35:
            joint_rate = max(joint_rate, 5.0)
        elif dist_xy > 0.20:
            joint_rate = max(joint_rate, 3.8)
        if anchor_limbs:
            joint_rate *= 0.78
        leg_spine = {
            "lumbar_pitch", "lumbar_roll", "thoracic_pitch",
            f"{limb.split('_')[0]}_hip_pitch",
            f"{limb.split('_')[0]}_hip_roll",
            f"{limb.split('_')[0]}_hip_yaw",
            f"{limb.split('_')[0]}_knee",
            f"{limb.split('_')[0]}_ankle",
        }
        for name, g in goal.items():
            c = current.get(name, JOINT_SCHEMA[name]["neutral"])
            rate = joint_rate * (1.1 if name in leg_spine else 0.65)
            stepped[name] = c + float(np.clip(g - c, -rate, rate))
        # Smooth updates over physics substeps for fluid, human-like motion.
        self._sim.set_targets_smooth(stepped)

    def _apply_hand_step(
        self,
        limb: str,
        target_x: float,
        target_y: float,
        *,
        max_joint_rate: float,
        anchor_limbs: list[str] | None = None,
    ) -> None:
        """Hand reach: closed-chain IK goal + rate-limited tracking.

        A single Jacobian step cannot cover large workspace gaps in one turn;
        the planted-foot IK solver plans spine/arm motion, then we rate-limit
        toward that pose. A small local Jacobian nudge keeps motion visible
        per step when locked limbs constrain the whole-body solve.
        """
        sim = self._sim
        ee = sim.get_end_effector_positions()[limb]
        dist_xy = math.hypot(target_x - ee["x"], target_y - ee["y"])
        locked = bool(anchor_limbs)
        if dist_xy > 0.18:
            tz = min(0.35, 0.25 + 0.35 * min(1.0, (dist_xy - 0.18) / 0.25))
        elif dist_xy > 0.12:
            tz = 0.14
        else:
            tz = self.contact_z(limb)

        lean = self._lean_bias(limb, target_x, target_y)
        support_pts: list[tuple[float, float]] | None = None
        if anchor_limbs:
            all_ee = sim.get_end_effector_positions()
            support_pts = [
                (all_ee[a]["x"], all_ee[a]["y"]) for a in anchor_limbs if a in all_ee
            ] or None
        sim.set_active_reach(limb, lean, support_override_xy=support_pts)

        ik_iters = 48 if dist_xy < 0.20 else (24 if locked else 36)
        goal = self.suggest_targets(
            limb,
            target_x,
            target_y,
            ik_iterations=ik_iters,
            anchor_limbs=anchor_limbs,
            target_z=tz,
        )
        self._inject_hip_bend_fallback(
            goal,
            limb=limb,
            dist_xy=dist_xy,
            dx=target_x - ee["x"],
            dy=target_y - ee["y"],
            locked=locked,
            frozen_limbs=set(anchor_limbs or []),
        )
        if dist_xy > 0.12:
            goal = self._hand_jacobian_nudge(
                limb, target_x, target_y, tz, goal, locked=locked, max_joint_rate=max_joint_rate
            )

        current = sim.get_joint_targets_deg()
        joint_rate = max_joint_rate
        if dist_xy > 0.40:
            joint_rate = max(joint_rate, 6.0)
        elif dist_xy > 0.25:
            joint_rate = max(joint_rate, 4.5)
        elif dist_xy < 0.12:
            joint_rate *= 0.85
        elif dist_xy < 0.20:
            joint_rate *= 0.92
        if locked:
            joint_rate *= 0.72
        arm_spine = {
            "lumbar_pitch", "lumbar_roll", "thoracic_pitch", "thoracic_yaw",
            f"{limb.split('_')[0]}_shoulder_pitch",
            f"{limb.split('_')[0]}_shoulder_roll",
            f"{limb.split('_')[0]}_elbow",
        }
        stepped: dict[str, float] = {}
        for name, g in goal.items():
            c = current.get(name, JOINT_SCHEMA[name]["neutral"])
            rate = joint_rate * (1.15 if name in arm_spine else 0.55)
            stepped[name] = c + float(np.clip(g - c, -rate, rate))
        sim.set_targets_smooth(stepped)

    def _hand_jacobian_nudge(
        self,
        limb: str,
        target_x: float,
        target_y: float,
        target_z: float,
        goal: dict[str, float],
        *,
        locked: bool,
        max_joint_rate: float,
    ) -> dict[str, float]:
        """Small arm+spine Jacobian correction blended into the IK goal."""
        sim = self._sim
        model, data = sim.model, sim.data
        side = "left" if limb.startswith("left") else "right"
        joint_names = [
            "lumbar_pitch",
            "lumbar_roll",
            "thoracic_pitch",
            "thoracic_yaw",
            f"{side}_shoulder_pitch",
            f"{side}_shoulder_roll",
            f"{side}_elbow",
        ]
        cols = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names],
            dtype=np.int32,
        )
        site_id = sim._site_ids[limb]
        target = np.array([target_x, target_y, target_z], dtype=np.float64)

        saved_qpos = data.qpos.copy()
        for name in JOINT_SCHEMA:
            data.qpos[sim._joint_qpos_idx[name]] = math.radians(goal.get(name, JOINT_SCHEMA[name]["neutral"]))
        mujoco.mj_forward(model, data)

        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        err = target - data.site_xpos[site_id]
        n = np.linalg.norm(err)
        max_cart = 0.035 if locked else 0.045
        if n > max_cart:
            err = err * (max_cart / n)
        damping = 0.18 if locked else 0.12
        dq = self._dls(jp[:, cols], damping=damping) @ err
        max_rad = math.radians(max(1.5 if locked else 2.0, max_joint_rate * 0.55))
        dq = np.clip(dq, -max_rad, max_rad)

        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)

        blend = 0.35 if locked else 0.50
        out = dict(goal)
        for name, d in zip(joint_names, dq):
            base = goal.get(name, JOINT_SCHEMA[name]["neutral"])
            out[name] = base + blend * math.degrees(float(d))
        return out

    def _inject_hip_bend_fallback(
        self,
        targets: dict[str, float],
        *,
        limb: str,
        dist_xy: float,
        dx: float,
        dy: float,
        locked: bool = False,
        frozen_limbs: set[str] | None = None,
    ) -> None:
        """If end-effector progress is hard, proactively bend at hips/torso.

        Hands prefer spine folding; feet prefer hip hinge. Keeps bends below
        tipping thresholds and scales down when other limbs are locked (phase 2).
        """
        frozen = frozen_limbs or set()

        def _frozen_side(side: str) -> bool:
            return any(l.startswith(side) for l in frozen)

        def _set_joint(name: str, value: float) -> None:
            side = "left" if name.startswith("left") else "right" if name.startswith("right") else ""
            if side and _frozen_side(side) and any(
                l.startswith(side) and l.endswith(("foot", "hand")) for l in frozen
            ):
                return
            targets[name] = value

        threshold = 0.30 if limb.endswith("hand") else 0.26
        if dist_xy < threshold:
            return

        is_hand = limb.endswith("hand")
        max_bend = 14.0 if is_hand else 20.0
        if locked:
            max_bend *= 0.75
        bend = min(max_bend, 4.0 + 34.0 * (dist_xy - threshold))
        sign = 1.0 if dy >= 0.0 else -1.0

        if is_hand:
            lp = sign * -0.60 * bend
            tp = sign * -0.40 * bend
            _set_joint(
                "lumbar_pitch",
                min(targets.get("lumbar_pitch", 0.0), lp)
                if sign > 0.0
                else max(targets.get("lumbar_pitch", 0.0), lp),
            )
            _set_joint(
                "thoracic_pitch",
                min(targets.get("thoracic_pitch", 0.0), tp)
                if sign > 0.0
                else max(targets.get("thoracic_pitch", 0.0), tp),
            )
            hip_frac = 0.22 * bend
            if sign > 0.0:
                _set_joint("left_hip_pitch", min(targets.get("left_hip_pitch", 0.0), -hip_frac))
                _set_joint("right_hip_pitch", max(targets.get("right_hip_pitch", 0.0), hip_frac))
            else:
                _set_joint("left_hip_pitch", max(targets.get("left_hip_pitch", 0.0), hip_frac))
                _set_joint("right_hip_pitch", min(targets.get("right_hip_pitch", 0.0), -hip_frac))
        elif sign > 0.0:
            _set_joint("left_hip_pitch", min(targets.get("left_hip_pitch", 0.0), -bend))
            _set_joint("right_hip_pitch", max(targets.get("right_hip_pitch", 0.0), bend))
            _set_joint("lumbar_pitch", min(targets.get("lumbar_pitch", 0.0), -0.40 * bend))
            _set_joint("thoracic_pitch", min(targets.get("thoracic_pitch", 0.0), -0.22 * bend))
        else:
            _set_joint("left_hip_pitch", max(targets.get("left_hip_pitch", 0.0), bend))
            _set_joint("right_hip_pitch", min(targets.get("right_hip_pitch", 0.0), -bend))
            _set_joint("lumbar_pitch", max(targets.get("lumbar_pitch", 0.0), 0.40 * bend))
            _set_joint("thoracic_pitch", max(targets.get("thoracic_pitch", 0.0), 0.22 * bend))

        if abs(dx) > 0.10:
            lateral = min(10.0, 7.0 * abs(dx))
            if locked:
                lateral *= 0.7
            roll_sign = 1.0 if dx >= 0.0 else -1.0
            cur = targets.get("lumbar_roll", 0.0)
            blended = 0.55 * cur + 0.45 * roll_sign * lateral
            _set_joint("lumbar_roll", float(np.clip(blended, -18.0, 18.0)))

        reach_xy = math.hypot(dx, dy)
        if reach_xy > 0.14:
            twist = min(14.0, 10.0 * reach_xy)
            if locked:
                twist *= 0.65
            yaw_sign = math.copysign(1.0, dx) if abs(dx) > 0.05 else math.copysign(1.0, dy)
            for hip_yaw in ("left_hip_yaw", "right_hip_yaw"):
                cur = targets.get(hip_yaw, 0.0)
                _set_joint(hip_yaw, cur + yaw_sign * 0.18 * twist)
            _set_joint(
                "thoracic_yaw",
                targets.get("thoracic_yaw", 0.0) + yaw_sign * 0.22 * twist,
            )

        if limb.endswith("foot"):
            knee_bend = min(22.0, 6.0 + 0.35 * bend)
            _set_joint("left_knee", max(targets.get("left_knee", 0.0), knee_bend))
            _set_joint("right_knee", max(targets.get("right_knee", 0.0), knee_bend))

    def _lean_bias(self, limb: str, target_x: float, target_y: float) -> tuple[float, float]:
        """Shift the CoM target toward the reach, clamped to the support margin."""
        sx, sy = self._sim.support_center_xy()
        dx, dy = target_x - sx, target_y - sy
        max_x, max_y = (0.20, 0.14) if limb.endswith("hand") else (0.08, 0.06)
        if limb.endswith("hand"):
            joints = self._sim.get_joint_angles_deg()
            spine_bend = abs(joints.get("lumbar_pitch", 0.0)) + abs(joints.get("thoracic_pitch", 0.0))
            if spine_bend > 8.0:
                scale = max(0.40, 1.0 - 0.035 * (spine_bend - 8.0))
                max_x *= scale
                max_y *= scale
        return (float(np.clip(dx, -max_x, max_x)), float(np.clip(dy, -max_y, max_y)))

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _dls(j: np.ndarray, damping: float) -> np.ndarray:
        m = j.shape[0]
        return j.T @ np.linalg.solve(j @ j.T + (damping**2) * np.eye(m), np.eye(m))

    # Max Cartesian step per IK iteration (m). Keeps the linearization valid so
    # large reaches (hand from shoulder height down to the floor) converge
    # smoothly instead of overshooting.
    _MAX_CART_STEP_HAND = 0.038
    _MAX_CART_STEP_FOOT = 0.032

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
        max_cart = self._MAX_CART_STEP_HAND if limb.endswith("hand") else self._MAX_CART_STEP_FOOT
        if n > max_cart:
            err_t = err_t * (max_cart / n)
        j_t_ns = j_t @ null_c
        reach_damp = 0.075 if limb.endswith("hand") else 0.085
        j_t_ns_pinv = self._dls(j_t_ns, damping=reach_damp)
        dq_t = null_c @ (j_t_ns_pinv @ (err_t - j_t @ dq_c))

        # --- Posture + base regularization in the remaining null space ---
        null_t = null_c @ (np.eye(nik) - j_t_ns_pinv @ j_t_ns)
        sec = np.zeros(nik)
        sec[6:] = 0.05 * (self._q_neutral - data.qpos[self._hinge_qpos_idx])
        up = data.xmat[self._pelvis_id].reshape(3, 3)[:, 2]
        rest = np.cross(up, _Z_UP)
        root_z = float(data.qpos[self._root_qpos + 2])
        spine_pitch = ("lumbar_pitch", "thoracic_pitch")
        spine_lateral = ("lumbar_roll", "thoracic_yaw")
        hip_yaw_names = ("left_hip_yaw", "right_hip_yaw")
        if limb.endswith("hand"):
            sec[3] = 0.0
            sec[4] = 0.55 * rest[1]
            sec[5] = 0.55 * rest[2]
            sec[2] = 1.35 * (0.74 - root_z)
            for sn in spine_pitch:
                idx = 6 + self._name_to_hinge[sn]
                sec[idx] = 0.018 * (
                    self._q_neutral[self._name_to_hinge[sn]]
                    - data.qpos[self._hinge_qpos_idx[self._name_to_hinge[sn]]]
                )
            for sn in spine_lateral:
                idx = 6 + self._name_to_hinge[sn]
                sec[idx] = 0.04 * (
                    self._q_neutral[self._name_to_hinge[sn]]
                    - data.qpos[self._hinge_qpos_idx[self._name_to_hinge[sn]]]
                )
            for kn in ("left_knee", "right_knee"):
                sec[6 + self._name_to_hinge[kn]] += 0.18 * (
                    0.0 - data.qpos[self._hinge_qpos_idx[self._name_to_hinge[kn]]]
                )
            for hn in hip_yaw_names:
                idx = 6 + self._name_to_hinge[hn]
                sec[idx] = 0.05 * (
                    self._q_neutral[self._name_to_hinge[hn]]
                    - data.qpos[self._hinge_qpos_idx[self._name_to_hinge[hn]]]
                )
        else:
            sec[3:6] = 0.65 * rest
            sec[2] = 0.55 * (0.62 - root_z)
            for sn in spine_pitch:
                idx = 6 + self._name_to_hinge[sn]
                sec[idx] = 0.025 * (
                    self._q_neutral[self._name_to_hinge[sn]]
                    - data.qpos[self._hinge_qpos_idx[self._name_to_hinge[sn]]]
                )
            for sn in (*spine_lateral, *hip_yaw_names):
                idx = 6 + self._name_to_hinge[sn]
                sec[idx] = 0.06 * (
                    self._q_neutral[self._name_to_hinge[sn]]
                    - data.qpos[self._hinge_qpos_idx[self._name_to_hinge[sn]]]
                )
        dq_post = null_t @ sec

        max_dq = _MAX_DQ if limb.endswith("hand") else math.radians(3.5)
        dq_rp = np.clip(dq_t + dq_post, -max_dq, max_dq)
        dq_full = np.zeros(model.nv)
        dq_full[ik] = dq_c + dq_rp
        mujoco.mj_integratePos(model, data.qpos, dq_full, 1.0)
        data.qpos[self._hinge_qpos_idx] = np.clip(
            data.qpos[self._hinge_qpos_idx], self._lo, self._hi
        )
        mujoco.mj_forward(model, data)
