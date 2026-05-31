"""
Interactive 3D viewer — humanoid on a Twister mat (MuJoCo).

Usage (macOS — requires mjpython for the 3D window):
    mjpython viewer.py                         # 5 Twister spins, 3s pauses
    mjpython viewer.py --spins 8 --pause 3
    mjpython viewer.py --trace data/traces/…   # replay a mesocosm run
    mjpython viewer.py --demo                  # single-turn heuristic demo

Controls (MuJoCo viewer):
    drag = orbit, scroll = zoom, right-drag = pan
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from env import TwisterEnv
from game.tasks import build_observation
from game.twister import MAT_ROWS, placement_error
from sim.constants import JOINT_SCHEMA, PLACEMENT_RADIUS

LIMB_COLORS: dict[str, tuple[float, float, float, float]] = {
    "left_hand": (1.0, 0.2, 0.2, 0.95),
    "right_hand": (0.2, 0.4, 1.0, 0.95),
    "left_foot": (1.0, 0.6, 0.1, 0.95),
    "right_foot": (0.2, 0.8, 0.4, 0.95),
}

VISUAL_GEOM_GROUP = 1
COLLISION_GEOM_GROUP = 3


class MatHighlighter:
    """Highlight target and locked circles on the Twister mat."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model
        self._base_rgba: dict[int, tuple[float, float, float, float]] = {}
        self._geom_ids: dict[tuple[int, int], int] = {}
        for row in range(len(MAT_ROWS)):
            for col in range(6):
                name = f"mat_r{row}_c{col}"
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    self._geom_ids[(row, col)] = gid
                    self._base_rgba[gid] = tuple(model.geom_rgba[gid])
        self._target_gid: int | None = None
        self._locked_gids: set[int] = set()

    def set_target(self, row: int | None, col: int | None) -> None:
        if self._target_gid is not None:
            self._model.geom_rgba[self._target_gid] = self._base_rgba[self._target_gid]
            self._target_gid = None
        if row is None or col is None:
            return
        gid = self._geom_ids.get((row, col))
        if gid is None:
            return
        self._target_gid = gid
        self._model.geom_rgba[gid] = (1.0, 1.0, 1.0, 1.0)

    def set_locked(self, locked: list[dict[str, Any]]) -> None:
        for gid in self._locked_gids:
            self._model.geom_rgba[gid] = self._base_rgba[gid]
        self._locked_gids.clear()
        for item in locked:
            row, col = item["circle"]
            gid = self._geom_ids.get((row, col))
            if gid is not None and gid != self._target_gid:
                self._model.geom_rgba[gid] = (1.0, 0.55, 0.0, 1.0)
                self._locked_gids.add(gid)


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
    target: dict[str, Any] | None,
    limb: str | None,
    limb_pos: dict[str, float] | None,
    locked: list[dict[str, Any]] | None = None,
) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0
    idx = 0

    if target:
        idx = _add_marker(
            scn,
            idx,
            (target["x"], target["y"], 0.05),
            (1.0, 1.0, 0.0, 0.85),
            size=0.055,
        )
        idx = _add_marker(
            scn,
            idx,
            (target["x"], target["y"], 0.002),
            (1.0, 1.0, 1.0, 0.35),
            size=PLACEMENT_RADIUS,
        )

    for item in locked or []:
        circle = item.get("circle")
        if not circle:
            continue
        row, col = circle
        from game.twister import TwisterMat

        c = TwisterMat().circle_at(row, col)
        idx = _add_marker(
            scn,
            idx,
            (c.x, c.y, 0.03),
            (1.0, 0.5, 0.0, 0.5),
            size=0.04,
        )

    if limb and limb_pos:
        rgba = LIMB_COLORS.get(limb, (0.9, 0.2, 0.9, 0.95))
        idx = _add_marker(
            scn,
            idx,
            (limb_pos["x"], limb_pos["y"], limb_pos["z"]),
            rgba,
            size=0.05,
        )

    scn.ngeom = idx


def _latest_trace(traces_dir: Path) -> Path | None:
    files = list(traces_dir.glob("*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _load_trace(path: Path) -> tuple[int | None, list[dict[str, Any]]]:
    seed: int | None = None
    actions: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["event_type"] == "episode_start":
            seed = event["payload"].get("seed")
        elif event["event_type"] == "action":
            actions.append(event["payload"]["action"])
    return seed, actions


def _obs_from_result(result: Any) -> dict[str, Any]:
    obs = result.observation
    if isinstance(obs, dict):
        return obs
    return obs.data if hasattr(obs, "data") else dict(obs)


def _naive_action(obs: dict[str, Any]) -> dict[str, Any]:
    """One IK physics step per viewer frame.

    Do not use controller mode here: each controller step can run up to
    max_steps inner physics iterations, which exhausts PHASE2_MAX_STEPS (~200)
    in a couple of viewer frames and ends the episode with reason \"timeout\".
    """
    del obs
    return {"use_ik": True}


def _apply_render_quality(model: mujoco.MjModel, quality: str) -> None:
    model.vis.global_.glow = 0.0
    if quality != "high":
        return
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1920)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 1080)
    model.vis.quality.shadowsize = max(model.vis.quality.shadowsize, 8192)


def _setup_camera(viewer: mujoco.viewer.Handle) -> None:
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18
    viewer.cam.distance = 3.8
    viewer.cam.lookat[:] = (0.0, 0.0, 0.85)


def _setup_visual(
    viewer: mujoco.viewer.Handle,
    *,
    quality: str = "standard",
    show_collision: bool = False,
) -> None:
    _setup_camera(viewer)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    for i in range(len(viewer.opt.geomgroup)):
        viewer.opt.geomgroup[i] = 1
    viewer.opt.geomgroup[VISUAL_GEOM_GROUP] = 1
    viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = 1 if show_collision else 0
    if quality == "high":
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_FOG] = False


def _sync_viewer(
    viewer: mujoco.viewer.Handle,
    seconds: float,
    *,
    highlighter: MatHighlighter | None = None,
    obs: dict[str, Any] | None = None,
    limb: str | None = None,
    locked: list[dict[str, Any]] | None = None,
) -> bool:
    deadline = time.time() + seconds
    while viewer.is_running() and time.time() < deadline:
        target = obs.get("target_circle") if obs else None
        limb_pos = obs["end_effectors"].get(limb) if obs and limb else None
        _draw_markers(
            viewer,
            target=target,
            limb=limb,
            limb_pos=limb_pos,
            locked=locked,
        )
        viewer.sync()
        time.sleep(0.01)
    return viewer.is_running()


def _print_spin_banner(turn: int, obs: dict[str, Any], pause_sec: float) -> None:
    cmd = obs["command"]
    target = obs["target_circle"]
    print()
    print("╔" + "═" * 52 + "╗", flush=True)
    print(f"║  SPIN {turn}:  {cmd['instruction']:<38}║", flush=True)
    print(
        f"║  Target: {target['color']} circle [{target['row']},{target['col']}]"
        f"  ({target['x']:+.2f}, {target['y']:+.2f})     ║",
        flush=True,
    )
    locked = obs.get("locked_limbs") or []
    if locked:
        parts = [f"{l['limb']}→{l['color']}" for l in locked]
        line = "  Locked: " + ", ".join(parts)
        print(f"║{line:<52}║")
    print("╚" + "═" * 52 + "╝")
    print(f"  (waiting {pause_sec:.0f}s — white circle = target, yellow dot = aim point)")


def _print_landing(turn: int, obs: dict[str, Any], *, placed: bool, pause_sec: float) -> None:
    cmd = obs["command"]
    limb = cmd["limb"]
    target = obs["target_circle"]
    ee = obs["end_effectors"][limb]
    err = obs.get("placement_error", "?")
    status = "PLACED ✓" if placed else "missed"
    print()
    print(f"  → Turn {turn} result: {limb} landed at ({ee['x']:+.2f}, {ee['y']:+.2f}, {ee['z']:.2f})")
    print(f"    Target was ({target['x']:+.2f}, {target['y']:+.2f})  —  {err}m away  [{status}]")
    print(f"  (holding {pause_sec:.0f}s — colored dot shows where the limb ended up)")


def _advance_demo_spin(env: TwisterEnv, obs: dict[str, Any]) -> dict[str, Any]:
    """Advance to the next spinner command (demo pacing when a turn is missed)."""
    state = env._phase2
    assert state is not None
    state.command = state.spinner.spin(state.mat)
    state.turn += 1
    target = state.mat.circle_at(state.command.row, state.command.col)
    limb_pos = env._sim.get_end_effector_positions()[state.command.limb]
    err = placement_error(limb_pos, target)
    return build_observation(
        phase=2,
        sim=env._sim,
        mat=state.mat,
        command=state.command,
        turn=state.turn,
        locked_limbs=state.constraints.to_dict(),
        target_circle=target.to_dict(),
        placement_error_m=err,
        reach=env._reach,
    )


def run_spins(
    env: TwisterEnv,
    *,
    seed: int,
    num_spins: int,
    pause_sec: float,
    steps_per_turn: int,
    render_quality: str = "standard",
    show_collision: bool = False,
) -> None:
    sim = env._sim
    _apply_render_quality(sim.model, render_quality)
    highlighter = MatHighlighter(sim.model)
    obs = env.reset(seed=seed, phase=2, max_turns=num_spins)
    turn = obs["turn"]
    terminated = False

    print(f"Twister spin simulation — {num_spins} turns, {pause_sec:.0f}s between spin/move", flush=True)
    print("Close the MuJoCo window to exit.\n", flush=True)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        _setup_visual(viewer, quality=render_quality, show_collision=show_collision)

        while viewer.is_running() and not terminated and turn <= num_spins:
            cmd = obs["command"]
            limb = cmd["limb"]
            locked = obs.get("locked_limbs") or []
            target = obs["target_circle"]
            turn_cmd = cmd
            turn_target = target
            turn_limb = limb

            highlighter.set_target(target["row"], target["col"])
            highlighter.set_locked(locked)
            _print_spin_banner(turn, obs, pause_sec)

            if not _sync_viewer(
                viewer,
                pause_sec,
                highlighter=highlighter,
                obs=obs,
                limb=limb,
                locked=locked,
            ):
                break

            print(f"  Moving {limb.replace('_', ' ')}…")
            placed = False
            result = None
            for step_i in range(steps_per_turn):
                if not viewer.is_running():
                    break
                result = env.step(_naive_action(obs))
                obs = _obs_from_result(result)
                err = obs.get("placement_error", 999)
                placed = err != "?" and float(err) <= PLACEMENT_RADIUS

                _draw_markers(
                    viewer,
                    target=obs.get("target_circle"),
                    limb=limb,
                    limb_pos=obs["end_effectors"].get(limb),
                    locked=obs.get("locked_limbs"),
                )
                viewer.sync()
                time.sleep(0.04)

                if result.terminated or result.truncated:
                    reason = result.info.get("termination_reason", "")
                    if reason == "fall":
                        print("  (figure lost balance — keeping viewer open)")
                    else:
                        terminated = True
                    break
                if placed:
                    break

            if not viewer.is_running():
                break

            landing_obs = {
                **obs,
                "command": turn_cmd,
                "target_circle": turn_target,
                "end_effectors": obs["end_effectors"],
                "placement_error": round(
                    placement_error(
                        obs["end_effectors"][turn_limb],
                        env._mat.circle_at(turn_target["row"], turn_target["col"]),
                    ),
                    4,
                ),
            }
            _print_landing(turn, landing_obs, placed=placed, pause_sec=pause_sec)

            if not _sync_viewer(
                viewer,
                pause_sec,
                highlighter=highlighter,
                obs=landing_obs,
                limb=turn_limb,
                locked=obs.get("locked_limbs"),
            ):
                break

            if terminated:
                reason = result.info.get("termination_reason", "ended") if result else "ended"
                if reason != "fall":
                    print(f"\nEpisode ended: {reason}")
                    break

            if placed:
                if obs["turn"] <= turn and turn < num_spins:
                    obs = _advance_demo_spin(env, obs)
                turn = obs["turn"]
            elif turn < num_spins:
                obs = _advance_demo_spin(env, obs)
                turn = obs["turn"]
            else:
                break

        print("\nDone — close the MuJoCo window to exit.")


def run_replay(
    env: TwisterEnv,
    actions: list[dict[str, Any]],
    *,
    seed: int | None,
    speed: float,
    loop: bool,
    render_quality: str = "standard",
    show_collision: bool = False,
) -> None:
    sim = env._sim
    _apply_render_quality(sim.model, render_quality)
    highlighter = MatHighlighter(sim.model)
    obs = env.reset(seed=seed)
    target = obs.get("target_circle")
    if target:
        highlighter.set_target(target["row"], target["col"])

    print(f"Replaying {len(actions)} steps. Close the MuJoCo window to exit.\n")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        _setup_visual(viewer, quality=render_quality, show_collision=show_collision)
        action_idx = 0
        last_advance = time.time()

        while viewer.is_running():
            if action_idx < len(actions):
                now = time.time()
                if now - last_advance >= 0.15 / max(speed, 0.1):
                    result = env.step(actions[action_idx])
                    obs = _obs_from_result(result)
                    target = obs.get("target_circle")
                    if target:
                        highlighter.set_target(target["row"], target["col"])
                    limb = obs["command"]["limb"]
                    _draw_markers(
                        viewer,
                        target=target,
                        limb=limb,
                        limb_pos=obs["end_effectors"].get(limb),
                    )
                    action_idx += 1
                    last_advance = now
                    if action_idx >= len(actions) and loop:
                        obs = env.reset(seed=seed)
                        action_idx = 0

            viewer.sync()
            time.sleep(0.01)


def run_demo(
    env: TwisterEnv,
    *,
    seed: int,
    speed: float,
    max_steps: int,
    render_quality: str = "standard",
    show_collision: bool = False,
) -> None:
    sim = env._sim
    _apply_render_quality(sim.model, render_quality)
    highlighter = MatHighlighter(sim.model)
    obs = env.reset(seed=seed)
    target = obs.get("target_circle")
    if target:
        highlighter.set_target(target["row"], target["col"])

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        _setup_visual(viewer, quality=render_quality, show_collision=show_collision)
        last_step = time.time()
        step_num = 0

        while viewer.is_running() and step_num < max_steps:
            now = time.time()
            if now - last_step >= 0.12 / max(speed, 0.1):
                result = env.step(_naive_action(obs))
                obs = _obs_from_result(result)
                step_num += 1
                target = obs.get("target_circle")
                if target:
                    highlighter.set_target(target["row"], target["col"])
                limb = obs["command"]["limb"]
                _draw_markers(
                    viewer,
                    target=target,
                    limb=limb,
                    limb_pos=obs["end_effectors"].get(limb),
                )
                last_step = now
                if result.terminated or result.truncated:
                    reason = result.info.get("termination_reason", "")
                    if reason != "fall":
                        break
            viewer.sync()
            time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Twister viewer (MuJoCo)")
    parser.add_argument("--trace", type=Path, help="Replay a mesocosm trace .jsonl")
    parser.add_argument("--demo", action="store_true", help="Single-turn heuristic demo")
    parser.add_argument("--spins", type=int, default=5, help="Number of Twister spins (default mode)")
    parser.add_argument("--pause", type=float, default=3.0, help="Seconds between spin and move")
    parser.add_argument("--steps-per-turn", type=int, default=120, help="Max physics steps per move")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop trace replay")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--quality",
        choices=("standard", "high"),
        default="high",
        help="Visual quality (high enables shadows, skybox, 1080p offscreen buffer)",
    )
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="Render collision geoms (group 3) in addition to visual geoms",
    )
    args = parser.parse_args()

    env = TwisterEnv()

    if args.trace:
        if not args.trace.exists():
            print(f"Trace not found: {args.trace}")
            raise SystemExit(1)
        seed, actions = _load_trace(args.trace)
        if not actions:
            print(f"Trace has no actions: {args.trace}")
            raise SystemExit(1)
        run_replay(
            env,
            actions,
            seed=seed,
            speed=args.speed,
            loop=args.loop,
            render_quality=args.quality,
            show_collision=args.show_collision,
        )
        return

    if args.demo:
        run_demo(
            env,
            seed=args.seed,
            speed=args.speed,
            max_steps=args.max_steps,
            render_quality=args.quality,
            show_collision=args.show_collision,
        )
        return

    run_spins(
        env,
        seed=args.seed,
        num_spins=args.spins,
        pause_sec=args.pause,
        steps_per_turn=args.steps_per_turn,
        render_quality=args.quality,
        show_collision=args.show_collision,
    )


if __name__ == "__main__":
    main()
