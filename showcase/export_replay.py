#!/usr/bin/env python3
"""Convert a Mesocosm trace JSONL into replay.json + MuJoCo PNG frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.humanoid import HumanoidSim  # noqa: E402


def _load_trace(trace_path: Path) -> list[tuple[str, list[dict]]]:
    """Parse a trace JSONL into one (episode_id, steps) pair per episode it
    contains. A single trace file may hold several episodes (run_llm --episodes N)."""
    episodes: dict[str, dict] = {}
    order: list[str] = []

    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        eid = ev.get("episode_id") or trace_path.stem
        if eid not in episodes:
            episodes[eid] = {"reasoning": {}, "actions": {}, "rewards": {}, "obs": {}}
            order.append(eid)
        d = episodes[eid]
        step = int(ev.get("step", 0))
        et = ev["event_type"]
        payload = ev.get("payload", {})
        if et == "model_call":
            d["reasoning"][step] = payload.get("text", "")
        elif et == "action":
            d["actions"][step] = payload.get("action")
        elif et == "step_result":
            d["rewards"][step] = float(payload.get("reward", 0))
        elif et == "observation" and payload.get("phase") == "after_env":
            d["obs"][step] = payload.get("data", {})

    result: list[tuple[str, list[dict]]] = []
    for eid in order:
        d = episodes[eid]
        obs_by_step = d["obs"]
        if not obs_by_step:
            continue
        steps = []
        for step in sorted(obs_by_step):
            obs = obs_by_step[step]
            steps.append(
                {
                    "step": step,
                    "observation": obs,
                    "view": _normalize_view(obs),
                    "reasoning": d["reasoning"].get(step, ""),
                    "action": d["actions"].get(step, {}),
                    "reward": d["rewards"].get(step, 0),
                    "terminated": step == max(obs_by_step) and d["rewards"].get(step, 0) >= 1.0,
                }
            )
        result.append((eid, steps))
    return result


def _normalize_view(obs: dict) -> dict:
    """Flatten single- and multi-command observations into one shape the viewer
    can render directly: a list of goals (limb, target, live error, placed flag)."""
    radius = float(obs.get("placement_radius", 0.08))
    ee = obs.get("end_effectors", {})
    goals = []

    if obs.get("multi") or obs.get("num_targets", 1) > 1:
        commands = obs.get("commands", []) or []
        targets = obs.get("targets", []) or []
        errors = obs.get("placement_errors", []) or []
        for i, cmd in enumerate(commands):
            tgt = targets[i] if i < len(targets) else {}
            err = errors[i] if i < len(errors) else None
            goals.append(_goal(cmd, tgt, err, radius, ee))
    else:
        cmd = obs.get("command", {}) or {}
        tgt = obs.get("target") or obs.get("target_circle") or {}
        err = obs.get("placement_error")
        if cmd:
            goals.append(_goal(cmd, tgt, err, radius, ee))

    command_text = " + ".join(g["instruction"] for g in goals if g["instruction"]) or "—"
    return {
        "command_text": command_text,
        "goals": goals,
        "placement_radius": radius,
        "upright": obs.get("upright", True),
    }


def _goal(cmd: dict, tgt: dict, err, radius: float, ee: dict) -> dict:
    limb = cmd.get("limb", "")
    instruction = cmd.get("instruction") or cmd.get("goal") or ""
    if not instruction and limb:
        instruction = f"{limb.replace('_', ' ')} → {cmd.get('color', '?')}"
    lpos = ee.get(limb, {})
    return {
        "limb": limb,
        "instruction": instruction,
        "color": cmd.get("color") or tgt.get("color"),
        "target": {"x": tgt.get("x"), "y": tgt.get("y")},
        "limb_pos": {"x": lpos.get("x"), "y": lpos.get("y"), "z": lpos.get("z")},
        "error": round(err, 4) if isinstance(err, (int, float)) else None,
        "placed": isinstance(err, (int, float)) and err <= radius,
    }


def _restore_sim(sim: HumanoidSim, observation: dict) -> None:
    physics = observation.get("physics_state") or {}
    qpos = physics.get("qpos")
    if qpos:
        sim.set_qpos(qpos)
        return
    sim.reset()
    joints = observation.get("joints") or {}
    if joints:
        sim.apply_joints_deg(joints)


def export(trace_paths, showcase_dir: Path) -> Path:
    """Render frames and write replay.json for one or more traces. Each episode's
    frames live in frames/<ep8>/ and are cached (skipped if already rendered)."""
    if isinstance(trace_paths, (str, Path)):
        trace_paths = [trace_paths]
    frames_dir = showcase_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sim: HumanoidSim | None = None
    replay: dict[str, list[dict]] = {}

    for trace_path in trace_paths:
        for episode_id, steps in _load_trace(Path(trace_path)):
            if episode_id in replay:
                continue  # first occurrence wins (dedupe across traces)
            ep8 = str(episode_id)[:8]
            sub = frames_dir / ep8
            for row in steps:
                frame_name = f"step_{row['step']:02d}.png"
                frame_path = sub / frame_name
                if not frame_path.exists():
                    sub.mkdir(parents=True, exist_ok=True)
                    if sim is None:
                        sim = HumanoidSim()
                    _restore_sim(sim, row["observation"])
                    frame_path.write_bytes(sim.render_png())
                row["frame"] = f"frames/{ep8}/{frame_name}"
            replay[episode_id] = steps

    out = showcase_dir / "replay.json"
    out.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "domain_id": "twister-body",
                "binding_vow_version": "1.0.0",
                "visibility": "gallery_public",
                "replay": replay,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def recent_traces(n: int) -> list[Path]:
    """The n most recently modified trace files, newest first."""
    traces = sorted((ROOT / "data" / "traces").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return list(reversed(traces[-n:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="*",
        default=None,
        help="Path(s) to trace .jsonl (default: newest in data/traces/)",
    )
    parser.add_argument(
        "--all",
        type=int,
        metavar="N",
        help="Combine the N most recent traces into one replay (for the episode dropdown)",
    )
    args = parser.parse_args()

    showcase_dir = Path(__file__).parent
    if args.traces:
        paths = [Path(t) for t in args.traces]
    elif args.all:
        paths = recent_traces(args.all)
        if not paths:
            raise SystemExit("No trace files found in data/traces/")
    else:
        latest = recent_traces(1)
        if not latest:
            raise SystemExit("No trace files found in data/traces/")
        paths = latest

    out = export(paths, showcase_dir)
    print(f"Wrote {out} with MuJoCo frames in {showcase_dir / 'frames'}")


if __name__ == "__main__":
    main()
