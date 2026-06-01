"""Fast headless IK tuning harness — no rendering. Reports reach quality per seed."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from env import TwisterEnv

import os

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
POSTURE = os.environ.get("POSTURE", "1") == "1"
TZ = float(os.environ.get("TZ", "0.12"))


def run(seed: int) -> dict:
    env = TwisterEnv()
    obs = env.reset(seed=seed)
    limb = obs["command"]["limb"]
    tx = obs["target_circle"]["x"]
    ty = obs["target_circle"]["y"]
    tz = TZ if limb in ("left_hand", "right_hand") else 0.0

    best = float("inf")
    fell = False
    last = None
    for _ in range(STEPS):
        suggested = env._sim.solve_ik(limb, (tx, ty, tz), with_posture=POSTURE)
        cur = obs["joints"]
        jt = dict(cur)
        for j, g in suggested.items():
            jt[j] = cur.get(j, g) + 0.80 * (g - cur.get(j, g))
        res = env.step({"joint_targets": jt, "delta": False})
        obs = res.observation
        err = float(res.info.get("placement_error", "nan"))
        best = min(best, err)
        last = err
        if res.info.get("termination_reason") == "fall":
            fell = True
            break
    return {"seed": seed, "limb": limb, "tx": tx, "ty": ty,
            "best": best, "last": last, "fell": fell}


solved = 0
reachable_solved = 0
for s in range(N):
    r = run(42 + s)
    dist = (r["tx"] ** 2 + r["ty"] ** 2) ** 0.5
    hit = r["best"] <= 0.08
    solved += hit
    tag = "HIT " if hit else "    "
    fall = "FELL" if r["fell"] else ""
    print(f"  seed={r['seed']} {r['limb']:11s} tgt=({r['tx']:+.2f},{r['ty']:+.2f}) "
          f"d={dist:.2f}  best={r['best']:.3f} {tag}{fall}")

print(f"\n  solved {solved}/{N} within 0.08m")
