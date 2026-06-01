"""Interactive Twister viewer — one spin at a time, built on TwisterEnv observations."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from env import TwisterEnv
from game.twister import MAT_ROWS
from sim.constants import PLACEMENT_RADIUS

LIMB_COLORS: dict[str, tuple[float, float, float, float]] = {
    "left_hand": (1.0, 0.2, 0.2, 0.95),
    "right_hand": (0.2, 0.4, 1.0, 0.95),
    "left_foot": (1.0, 0.6, 0.1, 0.95),
    "right_foot": (0.2, 0.8, 0.4, 0.95),
}


def _obs(result_or_dict: Any) -> dict[str, Any]:
    if isinstance(result_or_dict, dict):
        return result_or_dict
    raw = result_or_dict.observation
    if isinstance(raw, dict):
        return raw
    return raw.data if hasattr(raw, "data") else dict(raw)


class MatHighlighter:
    """Dim non-active mat circles; highlight target and locked placements."""

    _DIM = 0.42

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model
        self._base: dict[int, tuple[float, float, float, float]] = {}
        self._by_cell: dict[tuple[int, int], int] = {}
        self._all: list[int] = []
        for row in range(len(MAT_ROWS)):
            for col in range(6):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"mat_r{row}_c{col}")
                if gid < 0:
                    continue
                self._by_cell[(row, col)] = gid
                self._all.append(gid)
                self._base[gid] = tuple(model.geom_rgba[gid])
        self._target: int | None = None
        self._locked: set[int] = set()

    def _dim_inactive(self) -> None:
        active = set(self._locked)
        if self._target is not None:
            active.add(self._target)
        for gid in self._all:
            if gid in active:
                continue
            r, g, b, a = self._base[gid]
            self._model.geom_rgba[gid] = (r * self._DIM, g * self._DIM, b * self._DIM, a)

    def set_target(self, row: int | None, col: int | None) -> None:
        if self._target is not None:
            self._model.geom_rgba[self._target] = self._base[self._target]
            self._target = None
        if row is not None and col is not None:
            gid = self._by_cell.get((row, col))
            if gid is not None:
                self._target = gid
                self._model.geom_rgba[gid] = (1.0, 1.0, 1.0, 1.0)
        self._dim_inactive()

    def set_locked(self, locked: list[dict[str, Any]]) -> None:
        for gid in self._locked:
            self._model.geom_rgba[gid] = self._base[gid]
        self._locked.clear()
        for item in locked:
            circle = item.get("circle")
            if not circle:
                continue
            gid = self._by_cell.get((circle[0], circle[1]))
            if gid is None or gid == self._target:
                continue
            base = self._base[gid]
            self._model.geom_rgba[gid] = (min(1.0, base[0] * 1.15), base[1] * 0.55, base[2] * 0.2, 1.0)
            self._locked.add(gid)
        self._dim_inactive()


def _add_marker(
    scn: mujoco.MjvScene,
    idx: int,
    pos: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    *,
    size: float = 0.045,
    geom_type: int = mujoco.mjtGeom.mjGEOM_SPHERE,
) -> int:
    mujoco.mjv_initGeom(
        scn.geoms[idx],
        type=geom_type,
        size=[size, 0, 0],
        pos=list(pos),
        mat=np.eye(3).flatten(),
        rgba=list(rgba),
    )
    return idx + 1


def _draw_markers(
    viewer: mujoco.viewer.Handle,
    *,
    obs: dict[str, Any],
    limb: str | None,
    err_m: float | None,
) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0
    idx = 0
    target = obs.get("target_circle")
    locked = obs.get("locked_limbs") or []

    if target:
        tx, ty = target["x"], target["y"]
        idx = _add_marker(
            scn,
            idx,
            (tx, ty, 0.014),
            (1.0, 1.0, 1.0, 0.55),
            size=PLACEMENT_RADIUS,
            geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        )
        idx = _add_marker(scn, idx, (tx, ty, 0.022), (1.0, 1.0, 0.2, 0.95), size=0.04)

    for item in locked:
        circle = item.get("circle")
        if not circle:
            continue
        from game.twister import TwisterMat

        c = TwisterMat().circle_at(circle[0], circle[1])
        idx = _add_marker(
            scn,
            idx,
            (c.x, c.y, 0.016),
            (1.0, 0.45, 0.0, 0.75),
            size=0.085,
            geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        )

    if limb:
        ee = obs.get("end_effectors", {}).get(limb)
        if ee:
            if err_m is not None and err_m <= PLACEMENT_RADIUS:
                rgba = (0.2, 0.95, 0.35, 0.95)
            elif err_m is not None and err_m <= PLACEMENT_RADIUS * 2:
                rgba = (1.0, 0.85, 0.15, 0.95)
            else:
                rgba = LIMB_COLORS.get(limb, (0.9, 0.2, 0.9, 0.95))
            idx = _add_marker(scn, idx, (ee["x"], ee["y"], ee["z"]), rgba, size=0.05)

    scn.ngeom = idx


def _placement_err(
    obs: dict[str, Any],
    limb: str,
    target: dict[str, Any],
    info: dict[str, Any] | None = None,
) -> float:
    if info and info.get("placement_error") is not None:
        return float(info["placement_error"])
    ee = obs.get("end_effectors", {}).get(limb)
    if not ee:
        return float("inf")
    return math.hypot(ee["x"] - target["x"], ee["y"] - target["y"])


def _hold(
    viewer: mujoco.viewer.Handle,
    seconds: float,
    *,
    obs: dict[str, Any],
    limb: str | None,
    err_m: float | None,
) -> bool:
    deadline = time.time() + seconds
    while viewer.is_running() and time.time() < deadline:
        _draw_markers(viewer, obs=obs, limb=limb, err_m=err_m)
        viewer.sync()
        time.sleep(0.02)
    return viewer.is_running()


def _print_spin(turn: int, obs: dict[str, Any]) -> None:
    cmd = obs["command"]
    target = obs["target_circle"]
    print()
    print("═" * 54)
    print(f"  SPIN {turn}:  {cmd['instruction']}")
    print(f"  Target: {target['color']} [{target['row']},{target['col']}]  ({target['x']:+.2f}, {target['y']:+.2f})")
    locked = obs.get("locked_limbs") or []
    if locked:
        parts = ", ".join(f"{l['limb']}→{l['color']}" for l in locked)
        print(f"  Locked: {parts}")
    print("═" * 54)


def _print_result(
    turn: int,
    obs: dict[str, Any],
    *,
    limb: str,
    target: dict[str, Any],
    placed: bool,
    err_m: float,
) -> None:
    ee = obs["end_effectors"][limb]
    status = "PLACED" if placed else "missed"
    print(f"  Turn {turn}: {limb} at ({ee['x']:+.2f}, {ee['y']:+.2f}, {ee['z']:.2f})")
    print(f"  Target ({target['x']:+.2f}, {target['y']:+.2f}) — {err_m:.3f}m away [{status}]")


def _setup_camera(viewer: mujoco.viewer.Handle) -> None:
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18
    viewer.cam.distance = 3.8
    viewer.cam.lookat[:] = (0.0, 0.0, 0.85)


def run_spins(
    env: TwisterEnv,
    *,
    seed: int,
    num_spins: int,
    spin_hold: float,
    result_hold: float,
    step_pause: float,
    steps_per_turn: int,
) -> None:
    sim = env._sim
    highlighter = MatHighlighter(sim.model)
    obs = env.reset(seed=seed, phase=2, max_turns=num_spins)
    turns_done = 0

    print(f"Twister viewer — {num_spins} spins (close window to exit)\n")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        _setup_camera(viewer)

        while viewer.is_running() and turns_done < num_spins:
            cmd = obs["command"]
            limb = cmd["limb"]
            target = obs["target_circle"]
            turn = obs.get("turn", turns_done + 1)
            locked = obs.get("locked_limbs") or []

            turn_cmd = cmd
            turn_limb = limb
            turn_target = target

            highlighter.set_target(target["row"], target["col"])
            highlighter.set_locked(locked)
            _print_spin(turn, obs)
            if not _hold(viewer, spin_hold, obs=obs, limb=limb, err_m=None):
                break

            print(f"  Moving {limb.replace('_', ' ')}…")
            placed = False
            terminated = False
            last_info: dict[str, Any] = {}
            landing_obs = obs

            for _ in range(steps_per_turn):
                result = env.step({"use_ik": True})
                last_info = result.info
                landing_obs = _obs(result)

                err_m = _placement_err(landing_obs, turn_limb, turn_target, last_info)
                result_view = {**landing_obs, "command": turn_cmd, "target_circle": turn_target}
                _draw_markers(viewer, obs=result_view, limb=turn_limb, err_m=err_m)
                viewer.sync()
                time.sleep(step_pause)

                if last_info.get("turn_completed"):
                    placed = True
                    turns_done = int(last_info.get("turns_survived", turns_done + 1))
                    break
                if result.terminated or result.truncated:
                    terminated = True
                    break

            err_m = _placement_err(landing_obs, turn_limb, turn_target, last_info)
            placed = placed or err_m <= PLACEMENT_RADIUS
            result_view = {**landing_obs, "command": turn_cmd, "target_circle": turn_target}
            _print_result(turn, landing_obs, limb=turn_limb, target=turn_target, placed=placed, err_m=err_m)
            if not _hold(viewer, result_hold, obs=result_view, limb=turn_limb, err_m=err_m):
                break

            if terminated:
                print(f"\nEpisode ended: {last_info.get('termination_reason', 'unknown')}")
                break
            if not placed:
                print("\nTurn step budget exhausted without placement.")
                break

            obs = landing_obs

        print("\nDone.")


def _load_trace(path: Path) -> tuple[int | None, list[dict[str, Any]]]:
    seed: int | None = None
    actions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        evt = json.loads(line)
        if evt.get("event_type") == "episode_start":
            seed = evt.get("payload", {}).get("seed")
        elif evt.get("event_type") == "action":
            actions.append(evt.get("payload", {}).get("action", {"use_ik": True}))
    return seed, actions


def run_replay(
    env: TwisterEnv,
    actions: list[dict[str, Any]],
    *,
    seed: int | None,
    speed: float,
) -> None:
    sim = env._sim
    highlighter = MatHighlighter(sim.model)
    obs = env.reset(seed=seed, phase=2)
    target = obs.get("target_circle")
    if target:
        highlighter.set_target(target["row"], target["col"])

    interval = 0.15 / max(speed, 0.1)
    print(f"Replaying {len(actions)} steps.\n")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        _setup_camera(viewer)
        idx = 0
        last = time.time()

        while viewer.is_running():
            now = time.time()
            if idx < len(actions) and now - last >= interval:
                result = env.step(actions[idx])
                obs = _obs(result)
                target = obs.get("target_circle")
                if target:
                    highlighter.set_target(target["row"], target["col"])
                limb = obs["command"]["limb"]
                target = obs.get("target_circle") or {}
                err_m = _placement_err(obs, limb, target, result.info)
                _draw_markers(viewer, obs=obs, limb=limb, err_m=err_m)
                idx += 1
                last = now
            viewer.sync()
            time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description="Twister spin viewer (MuJoCo)")
    parser.add_argument("--spins", type=int, default=6, help="Successful placements to play for")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spin-hold",
        type=float,
        default=2.5,
        help="Seconds to show each new spin command before moving",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=2.5,
        help="Seconds to hold after each move so you can see the result",
    )
    parser.add_argument("--step-pause", type=float, default=0.04, help="Seconds between physics steps while moving")
    parser.add_argument("--steps-per-turn", type=int, default=160, help="Max physics steps per spin")
    parser.add_argument("--trace", type=Path, default=None, help="Replay a mesocosm .jsonl trace")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    args = parser.parse_args()

    env = TwisterEnv()

    if args.trace:
        if not args.trace.exists():
            raise SystemExit(f"Trace not found: {args.trace}")
        seed, actions = _load_trace(args.trace)
        if not actions:
            raise SystemExit(f"No actions in trace: {args.trace}")
        run_replay(env, actions, seed=seed, speed=args.speed)
        return

    run_spins(
        env,
        seed=args.seed,
        num_spins=args.spins,
        spin_hold=args.spin_hold,
        result_hold=args.hold,
        step_pause=args.step_pause,
        steps_per_turn=args.steps_per_turn,
    )


if __name__ == "__main__":
    main()
