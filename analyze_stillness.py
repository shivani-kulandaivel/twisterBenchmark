"""Reconstruct, per step, what the LLM saw vs. what it output — to explain why
the model freezes on one action. Uses the latest trace; no API calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_llm import _build_llm_obs

trace = sorted((ROOT / "data" / "traces").glob("*.jsonl"),
               key=lambda p: p.stat().st_mtime)[-1]
print(f"Trace: {trace.name}\n")

# Group events by episode, in order.
events = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
by_ep: dict[str, list] = {}
for e in events:
    by_ep.setdefault(e["episode_id"], []).append(e)

# Analyse only the first episode for clarity.
eid = next(iter(by_ep))
evs = by_ep[eid]
print(f"Episode {eid[:8]}\n")

# Walk steps: the obs the model saw at step N is the 'observation/after_env' logged
# at step N-1 (or the reset obs for step 1).
last_obs = None
prev_llm_obs_json = None
for e in evs:
    et = e["event_type"]
    step = e["step"]
    if et == "observation":
        obs = e["payload"]["data"]
        # This obs becomes the INPUT for the next model call.
        last_obs = obs
    elif et == "action":
        action = e["payload"]["action"]
        # Reconstruct what the model was shown for THIS step (from last_obs).
        if last_obs is not None:
            llm_obs = _build_llm_obs(last_obs)
            sug = llm_obs.get("suggested_joints", {})
            hint = llm_obs.get("move_hint", "")
            curj = llm_obs.get("current_joints", {})
            llm_obs_json = json.dumps(llm_obs, sort_keys=True)
            changed = "CHANGED" if llm_obs_json != prev_llm_obs_json else "SAME-AS-PREV"
            prev_llm_obs_json = llm_obs_json
            print(f"step {step}: obs-to-model {changed}")
            print(f"   move_hint     : {hint}")
            print(f"   current_joints: {curj}")
            print(f"   suggested(IK) : {sug}")
            print(f"   model OUTPUT  : {action.get('joint_targets', {})}")
            print()
