#!/usr/bin/env python3
"""
Fast demo: runs TwisterEnv locally with a scripted heuristic agent.
No Ollama needed — completes in a few seconds.
Writes data/traces/<uuid>.jsonl  then exports showcase/replay.json with 3D frames.

Usage:
    python run_demo.py
    python run_demo.py --episodes 2 --steps 12
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from env import TwisterEnv
from sim.humanoid import HumanoidSim


def _heuristic_action(sim: HumanoidSim, obs: dict, step: int, rng: random.Random) -> dict:
    """
    IK-based agent: solves whole-body Jacobian IK from the body's CURRENT pose,
    then blends toward it so the motion looks natural rather than snapping. Because
    the IK is re-solved each step from the leaned pose, the body progressively
    bends/crouches toward targets that a stationary arm could never reach.
    """
    command = obs.get("command", {})
    limb = command.get("limb", "left_hand")
    target = obs.get("target_circle", {})
    current_joints = obs.get("joints", {})

    tx = target.get("x", 0.0)
    ty = target.get("y", 0.0)
    # Aim hands just above the mat; feet onto it.
    tz = 0.12 if limb in ("left_hand", "right_hand") else 0.0

    suggested = sim.solve_ik(limb, (tx, ty, tz))

    # Blend toward the IK solution each step + tiny noise for liveliness.
    joint_targets = dict(current_joints)
    for jname, goal in suggested.items():
        cur = current_joints.get(jname, goal)
        joint_targets[jname] = round(cur + 0.75 * (goal - cur) + rng.uniform(-1.0, 1.0), 1)

    return {"joint_targets": joint_targets, "delta": False}


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

def _evt(episode_id: str, step: int, event_type: str, payload: dict) -> str:
    return json.dumps({
        "episode_id": episode_id,
        "step": step,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    })


def run_episode(env: TwisterEnv, seed: int, max_steps: int, rng: random.Random) -> list[str]:
    episode_id = str(uuid.uuid4())
    lines: list[str] = []

    obs = env.reset(seed=seed)
    lines.append(_evt(episode_id, 0, "reset", {"seed": seed}))
    lines.append(_evt(episode_id, 0, "observation", {"phase": "after_env", "data": obs}))

    for step in range(1, max_steps + 1):
        action = _heuristic_action(env._sim, obs, step, rng)

        reasoning = (
            f"Step {step}: limb={obs.get('command',{}).get('limb','?')} "
            f"target=({obs.get('target_circle',{}).get('x',0):.2f}, "
            f"{obs.get('target_circle',{}).get('y',0):.2f}) "
            f"error={obs.get('placement_error','?')}"
        )
        lines.append(_evt(episode_id, step, "model_call", {"text": reasoning}))
        lines.append(_evt(episode_id, step, "action", {"action": action}))

        result = env.step(action)
        obs = result.observation
        reason = result.info.get("termination_reason", "")

        # In demo mode, ignore falls so the body keeps animating all steps
        terminated = result.terminated and reason != "fall"

        lines.append(_evt(episode_id, step, "step_result", {
            "reward": result.reward,
            "terminated": terminated,
            "truncated": result.truncated,
            "info": result.info,
        }))
        lines.append(_evt(episode_id, step, "observation", {"phase": "after_env", "data": obs}))

        if terminated or result.truncated:
            print(f"  episode seed={seed} ended at step {step}, "
                  f"reward={result.reward:.2f}, reason={reason}")
            break

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = TwisterEnv()

    traces_dir = ROOT / "data" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    print(f"Running {args.episodes} episodes x {args.steps} steps (phase {args.phase})...")
    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"  [{ep+1}/{args.episodes}] seed={seed}")
        lines = run_episode(env, seed=seed, max_steps=args.steps, rng=rng)
        all_lines.extend(lines)

    trace_path = traces_dir / f"{uuid.uuid4()}.jsonl"
    trace_path.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\nTrace saved -> {trace_path}")

    # Export replay with 3D frames
    print("Rendering 3D frames...")
    showcase_dir = ROOT / "showcase"
    from showcase.export_replay import export
    out = export(trace_path, showcase_dir)
    frames = list((showcase_dir / "frames").glob("*.png"))
    print(f"replay.json saved -> {out}  ({len(frames)} frames rendered)")

    # Open the viewer
    import webbrowser
    viewer_url = f"http://localhost:8080"
    print(f"\nOpening showcase: {viewer_url}")
    webbrowser.open(viewer_url)


if __name__ == "__main__":
    main()
