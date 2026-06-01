#!/usr/bin/env python3
"""
Run the Twister benchmark against any LiteLLM-compatible model.
Works with Gemini, OpenAI, Anthropic, or Ollama — no cloud platform needed.

Usage:
    python run_llm.py --model gemini/gemini-2.0-flash --episodes 3
    python run_llm.py --model openai/gpt-4o-mini --episodes 3
    python run_llm.py --model ollama/llama3.2:1b --episodes 3

Required env vars (set the one that matches your model):
    GEMINI_API_KEY      for gemini/...
    OPENAI_API_KEY      for openai/...
    ANTHROPIC_API_KEY   for anthropic/...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import litellm
except ImportError:
    sys.exit("litellm not found — run: pip install litellm")

try:
    import requests
except ImportError:
    sys.exit("requests not found — run: pip install requests")

ADAPTER_URL = "http://localhost:8765"
SYSTEM_PROMPT_FILE = ROOT / "system_prompt.txt"


def _load_env_file() -> None:
    """Load KEY=VALUE pairs from a local .env file into the environment.
    Keeps API keys out of shell commands (which can trip secret detection)."""
    envf = ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _system_prompt() -> str:
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    return "You control a humanoid body playing Twister. Respond with only JSON: {\"joint_targets\": {...}}"


def _adapter(method: str, path: str, body: dict | None = None) -> dict:
    url = ADAPTER_URL + path
    try:
        if method == "POST":
            r = requests.post(url, json=body, timeout=30)
        else:
            r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        sys.exit(f"Cannot reach adapter at {ADAPTER_URL}. Start it with: python adapter.py")


def _call_llm(model: str, messages: list[dict], retries: int = 3) -> str:
    kwargs: dict = {"temperature": 0.2, "timeout": 120}
    # Force JSON-only output — prevents prose/code responses
    # Works for Ollama, OpenAI, Groq, Anthropic via litellm
    kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(retries):
        try:
            resp = litellm.completion(model=model, messages=messages, **kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "quota" in msg.lower() or "rate" in msg.lower()
            if is_rate_limit and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    rate-limited (quota), waiting {wait}s before retry {attempt+2}/{retries}...")
                time.sleep(wait)
                continue
            raise
    return ""


def _evt(episode_id: str, step: int, event_type: str, payload: dict) -> str:
    return json.dumps({
        "episode_id": episode_id,
        "step": step,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    })


# Joints that matter per limb — the LLM only sees these
_LIMB_JOINTS = {
    "left_hand":  ["left_shoulder_pitch", "left_shoulder_roll", "left_elbow"],
    "right_hand": ["right_shoulder_pitch", "right_shoulder_roll", "right_elbow"],
    "left_foot":  ["left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle"],
    "right_foot": ["right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle"],
}

# Physics-correct IK for position-controlled joints.
#
# ARM GEOMETRY (both arms, in torso/parent frame):
#   Left arm starts pointing in +X; right arm starts pointing in -X.
#   Combined arm direction after pitch θ and roll φ (both in parent frame):
#     left:  (cos θ,  sin θ·sin φ, −sin θ·cos φ)
#     right: (−cos θ, −sin θ·sin φ,  sin θ·cos φ)
#   For left  arm: POSITIVE pitch → arm swings DOWN (toward −Z at +90°)
#   For right arm: NEGATIVE pitch → arm swings DOWN (toward −Z at −90°)
#
# LEG GEOMETRY:
#   Both thighs start pointing in −Z (straight down).
#   After hip_pitch θ: thigh direction = (−sin θ,  0,        −cos θ)
#   After hip_roll  φ: combined        = (−sin θ,  cos θ·sin φ, −cos θ·cos φ)
#   Foot XY from hip pivot: L·(−sin θ,  cos θ·sin φ)
#   foot_x_world ≈ hip_x_offset + L·(−sin θ) + foot_site_x_offset
#   foot_y_world ≈ L · cos θ · sin φ
#
# Placement scoring is XY-only (height ignored), so we only need the XY projection.
def _suggest_joints(limb: str, tx: float, ty: float, cur_joints: dict) -> dict:
    """Return physics-correct target angles to position the limb XY above (tx, ty)."""
    s: dict[str, float] = {}
    L_arm = 0.40   # effective arm length (upper 0.22 + forearm 0.20, slight elbow bend)
    L_leg = 0.62   # effective leg reach (thigh 0.34 + shin 0.28 effective)

    if limb in ("left_hand", "right_hand"):
        is_left = (limb == "left_hand")
        side = "left" if is_left else "right"
        sx = 0.12 if is_left else -0.12  # shoulder X in world

        dx = tx - sx  # required X displacement from shoulder to target
        dy = ty       # required Y displacement (shoulder at y≈0)

        if is_left:
            # Left: hand_x = sx + L·cos θ  →  cos θ = dx/L
            cos_t = max(-1.0, min(1.0, dx / L_arm))
            theta = math.degrees(math.acos(cos_t))   # [0°, 180°]; +90° = arm straight down
            sin_t = math.sin(math.radians(theta))
            # Left: hand_y = L·sin θ·sin φ  →  sin φ = dy / (L·sin θ)
            if sin_t > 0.05:
                sin_p = max(-1.0, min(1.0, dy / (L_arm * sin_t)))
                phi = math.degrees(math.asin(sin_p))
            else:
                phi = 0.0
            theta = max(-140.0, min(140.0, theta))
            phi   = max(-110.0, min( 30.0, phi))
        else:
            # Right: hand_x = sx + L·(−cos θ)  →  cos θ = −dx/L
            cos_t = max(-1.0, min(1.0, -dx / L_arm))
            theta = -math.degrees(math.acos(cos_t))  # [−180°, 0°]; −90° = arm straight down
            sin_t = math.sin(math.radians(abs(theta)))
            # Right: hand_y = L·(−sin θ·sin φ)  →  sin φ = −dy / (L·sin|θ|)
            if sin_t > 0.05:
                sin_p = max(-1.0, min(1.0, -dy / (L_arm * sin_t)))
                phi = math.degrees(math.asin(sin_p))
            else:
                phi = 0.0
            theta = max(-140.0, min(140.0, theta))
            phi   = max( -30.0, min(110.0, phi))

        s[f"{side}_shoulder_pitch"] = round(theta, 1)
        s[f"{side}_shoulder_roll"]  = round(phi,   1)
        s[f"{side}_elbow"]          = 30.0  # gentle bend, natural reach

        # Slight torso lean toward ty target (helps shoulder reach further)
        s["abdomen_pitch"] = round(max(-20.0, min(20.0, ty * 15.0)), 1)

    elif limb in ("left_foot", "right_foot"):
        side = "left" if limb == "left_foot" else "right"
        # Foot neutral XY: (±0.18, 0) — hip offset ±0.09 + foot-site X offset ±0.09
        fx0 = 0.18 if limb == "left_foot" else -0.18

        dx = tx - fx0   # required X shift from neutral foot position
        dy = ty         # required Y shift

        # foot_x = fx0 + L·(−sin θ)  →  sin θ = −dx / L
        sin_t = max(-0.99, min(0.99, -dx / L_leg))
        theta = math.degrees(math.asin(sin_t))
        cos_t = math.cos(math.radians(theta))

        # foot_y = L·cos θ·sin φ  →  sin φ = dy / (L·cos θ)
        if abs(cos_t) > 0.05:
            sin_p = max(-0.99, min(0.99, dy / (L_leg * cos_t)))
            phi = math.degrees(math.asin(sin_p))
        else:
            phi = 0.0

        knee  = max(0.0, 12.0 + abs(dx) * 8.0)
        ankle = max(-35.0, min(35.0, -dy * 6.0))

        s[f"{side}_hip_pitch"] = round(max(-110.0, min(35.0, theta)), 1)
        s[f"{side}_hip_roll"]  = round(max( -35.0, min(35.0, phi)),   1)
        s[f"{side}_knee"]      = round(min(135.0, knee), 1)
        s[f"{side}_ankle"]     = round(ankle, 1)

    return s


_IK_SIM = None  # lazily-created HumanoidSim used only for IK suggestions


def _ik_suggest(obs: dict, limb: str, tx: float, ty: float) -> dict:
    """Reconstruct the body's current pose from the observation and run the real
    Jacobian IK to suggest joint targets. Returns {} if pose data is unavailable."""
    qpos = obs.get("physics_state", {}).get("qpos")
    if not qpos:
        return {}
    global _IK_SIM
    try:
        if _IK_SIM is None:
            from sim.humanoid import HumanoidSim
            _IK_SIM = HumanoidSim()
        _IK_SIM.set_qpos(qpos)
        tz = 0.12 if limb in ("left_hand", "right_hand") else 0.0
        return _IK_SIM.solve_ik(limb, (tx, ty, tz))
    except Exception:
        return {}


def _move_hint(dx: float, dy: float, dz: float) -> str:
    def _dir(v: float, pos_label: str, neg_label: str) -> str:
        if abs(v) < 0.02:
            return ""
        return f"{abs(v):.2f}m {pos_label if v > 0 else neg_label}"
    hints = [h for h in [_dir(dx, "LEFT(+X)", "RIGHT(-X)"), _dir(dy, "FWD(+Y)", "BACK(-Y)"),
                         _dir(dz, "UP", "DOWN")] if h]
    return ", ".join(hints) if hints else "on target"


def _build_multi_llm_obs(obs: dict) -> dict:
    """Observation for the multi-command task: one entry per limb + merged IK."""
    ee = obs.get("end_effectors", {})
    commands = obs.get("commands", [])
    targets = obs.get("targets", [])
    errors = obs.get("placement_errors", [None] * len(commands))

    tasks = []
    merged: dict[str, float] = {}
    abdomen_vals: list[float] = []
    for i, cmd in enumerate(commands):
        limb = cmd.get("limb", "")
        tgt = targets[i] if i < len(targets) else {}
        lpos = ee.get(limb, {})
        lx, ly, lz = lpos.get("x", 0.0), lpos.get("y", 0.0), lpos.get("z", 0.0)
        tx, ty = tgt.get("x", 0.0), tgt.get("y", 0.0)
        dx, dy, dz = round(tx - lx, 3), round(ty - ly, 3), round(0.0 - lz, 3)
        suggested = _ik_suggest(obs, limb, tx, ty)
        # Merge: abdomen_pitch is shared between arms — average it.
        for j, v in suggested.items():
            if j == "abdomen_pitch":
                abdomen_vals.append(v)
            else:
                merged[j] = v
        tasks.append({
            "goal": cmd.get("instruction", ""),
            "limb": limb,
            "limb_pos": {"x": round(lx, 3), "y": round(ly, 3), "z": round(lz, 3)},
            "target_pos": {"x": round(tx, 3), "y": round(ty, 3), "z": 0.0},
            "move_hint": _move_hint(dx, dy, dz),
            "placement_error_m": errors[i] if i < len(errors) else None,
            "suggested_joints": suggested,
        })
    if abdomen_vals:
        merged["abdomen_pitch"] = round(sum(abdomen_vals) / len(abdomen_vals), 1)

    return {
        "task_type": "place ALL listed limbs on their targets at the same time",
        "num_targets": len(commands),
        "goal_radius_m": 0.08,
        "tasks": tasks,
        "suggested_joints_all": merged,
        "upright": obs.get("upright", True),
    }


def _build_llm_obs(obs: dict) -> dict:
    """Compact, spatially-enriched observation for the LLM. Strips noise, adds hints."""
    if obs.get("multi"):
        return _build_multi_llm_obs(obs)
    command = obs.get("command", {})
    limb = command.get("limb", "")
    ee = obs.get("end_effectors", {})
    target = obs.get("target_circle", {})
    joints = obs.get("joints", {})

    lpos = ee.get(limb, {})
    lx, ly, lz = lpos.get("x", 0.0), lpos.get("y", 0.0), lpos.get("z", 0.0)
    tx, ty = target.get("x", 0.0), target.get("y", 0.0)
    dx, dy, dz = round(tx - lx, 3), round(ty - ly, 3), round(0.0 - lz, 3)
    hdist = round((dx**2 + dy**2) ** 0.5, 3)

    def _dir(v: float, pos_label: str, neg_label: str) -> str:
        if abs(v) < 0.02:
            return ""
        return f"{abs(v):.2f}m {pos_label if v > 0 else neg_label}"

    hints = [h for h in [_dir(dx, "LEFT(+X)", "RIGHT(-X)"), _dir(dy, "FWD(+Y)", "BACK(-Y)"),
                          _dir(dz, "UP", "DOWN")] if h]
    hint_str = ", ".join(hints) if hints else "on target"

    relevant = _LIMB_JOINTS.get(limb, [])
    suggested = _ik_suggest(obs, limb, tx, ty) or _suggest_joints(limb, tx, ty, joints)

    return {
        "goal": command.get("instruction", ""),
        "limb": limb,
        "limb_pos":    {"x": round(lx, 3), "y": round(ly, 3), "z": round(lz, 3)},
        "target_pos":  {"x": round(tx, 3), "y": round(ty, 3), "z": 0.0},
        "move_needed": {"dx": dx, "dy": dy, "dz": dz, "horizontal_dist": hdist},
        "move_hint":   hint_str,
        "placement_error_m": obs.get("placement_error"),
        "goal_radius_m": 0.08,
        "current_joints": {j: round(joints.get(j, 0.0), 1) for j in relevant},
        "abdomen_pitch": round(joints.get("abdomen_pitch", 0.0), 1),
        "suggested_joints": suggested,
        "upright": obs.get("upright", True),
    }


def _parse_raw(raw: str) -> dict:
    text = raw.strip()
    # Strip markdown fences
    if "```" in text:
        parts = text.splitlines()
        inside = []
        in_fence = False
        for line in parts:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                inside.append(line)
        if inside:
            text = "\n".join(inside).strip()

    # 1) Common case: the whole response is the JSON object.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "joint_targets" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) Outer-object case: take from the FIRST '{' to the LAST '}' (handles
    #    nested joint_targets correctly, and any prose before/after the JSON).
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        try:
            parsed = json.loads(text[first : last + 1])
            if isinstance(parsed, dict) and "joint_targets" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # 3) Fallback: scan every '{...}' span and keep one that has joint_targets.
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        for j in range(len(text), i, -1):
            if text[j - 1] != "}":
                continue
            try:
                parsed = json.loads(text[i:j])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "joint_targets" in parsed:
                return parsed

    return {}


def run_episode(model: str, seed: int, max_steps: int, verbose: bool, delay: float = 1.0,
                num_targets: int = 1) -> list[str]:
    episode_id = str(uuid.uuid4())
    lines: list[str] = []
    sys_prompt = _system_prompt()

    # Reset — adapter needs episode_id; returns {"data": obs, "system_prompt": ...}
    reset_resp = _adapter("POST", "/reset", {"episode_id": episode_id, "seed": seed,
                                             "scenario_params": {"phase": 1, "num_targets": num_targets}})
    obs = reset_resp.get("data", reset_resp)
    # Use system prompt from adapter if available, else our file
    sp = reset_resp.get("system_prompt") or sys_prompt

    lines.append(_evt(episode_id, 0, "reset", {"seed": seed}))
    lines.append(_evt(episode_id, 0, "observation", {"phase": "after_env", "data": obs}))

    messages: list[dict] = [{"role": "system", "content": sp}]

    for step in range(1, max_steps + 1):
        obs_text = json.dumps(_build_llm_obs(obs), separators=(",", ":"))
        messages.append({"role": "user", "content": obs_text})

        t0 = time.time()
        raw = _call_llm(model, messages)
        elapsed = time.time() - t0

        if verbose:
            if obs.get("multi"):
                cmds = " + ".join(c.get("instruction", "?") for c in obs.get("commands", []))
                errs = obs.get("placement_errors")
                err = ",".join(f"{e:.3f}" for e in errs) if errs else "?"
                print(f"    step {step:2d}  cmd='{cmds}'  errors=[{err}]  ({elapsed:.1f}s)")
            else:
                cmd = obs.get("command", {}).get("instruction", "?")
                err = obs.get("placement_error", "?")
                print(f"    step {step:2d}  cmd={cmd!r}  error={err}  ({elapsed:.1f}s)")
            print(f"           llm -> {raw[:120]}")

        lines.append(_evt(episode_id, step, "model_call", {"text": raw}))
        action = _parse_raw(raw)
        lines.append(_evt(episode_id, step, "action", {"action": action}))
        # Stateless: reset to just system prompt each step — keeps context tiny for small models
        messages = [messages[0]]

        # Step — adapter needs episode_id
        result = _adapter("POST", "/step", {"episode_id": episode_id, "action": action})
        obs = result.get("observation", {})
        if isinstance(obs, dict) and "data" in obs:
            obs = obs["data"]
        reward = float(result.get("reward", 0))
        terminated = bool(result.get("terminated", False))
        truncated = bool(result.get("truncated", False))
        info = result.get("info", {})

        lines.append(_evt(episode_id, step, "step_result", {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }))
        lines.append(_evt(episode_id, step, "observation", {"phase": "after_env", "data": obs}))

        if terminated or truncated:
            reason = info.get("termination_reason", "?")
            print(f"  episode ended at step {step}  reward={reward:.2f}  reason={reason}")
            break

        if delay > 0:
            time.sleep(delay)

    _adapter("POST", "/close", {"episode_id": episode_id})
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemini/gemini-2.0-flash", help="LiteLLM model string")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between steps (throttle API)")
    parser.add_argument("--targets", type=int, default=1, help="Number of simultaneous limb targets")
    args = parser.parse_args()

    _load_env_file()

    print(f"Model: {args.model}  episodes={args.episodes}  max_steps={args.steps}")
    print(f"Adapter: {ADAPTER_URL}")

    # Verify adapter is up
    try:
        requests.get(ADAPTER_URL + "/health", timeout=3)
    except Exception:
        # /health may not exist; that's fine — reset will catch a real outage
        pass

    traces_dir = ROOT / "data" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    all_lines: list[str] = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"\n[{ep+1}/{args.episodes}] seed={seed}")
        lines = run_episode(args.model, seed=seed, max_steps=args.steps, verbose=args.verbose,
                            delay=args.delay, num_targets=args.targets)
        all_lines.extend(lines)

    trace_path = traces_dir / f"{uuid.uuid4()}.jsonl"
    trace_path.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\nTrace saved -> {trace_path}")

    # Auto-export replay with 3D frames. Bundle the most recent traces so the
    # viewer's episode dropdown lets you switch between recent runs (cached
    # frames make re-exporting old episodes essentially free).
    print("Rendering 3D frames...")
    from showcase.export_replay import export, recent_traces
    showcase_dir = ROOT / "showcase"
    out = export(recent_traces(6), showcase_dir)
    frames = list((showcase_dir / "frames").rglob("*.png"))
    print(f"replay.json -> {out}  ({len(frames)} frames)")
    print("\nRefresh http://localhost:8080 to see the results!")


if __name__ == "__main__":
    main()
