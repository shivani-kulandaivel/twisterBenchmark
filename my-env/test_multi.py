"""Headless test of the 2-command task using merged IK. No API calls."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from env import TwisterEnv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def merged_ik(env, obs):
    jt = dict(obs["joints"])
    abdomen = []
    for i, cmd in enumerate(obs["commands"]):
        limb = cmd["limb"]
        tgt = obs["targets"][i]
        tz = 0.12 if limb in ("left_hand", "right_hand") else 0.0
        sug = env._sim.solve_ik(limb, (tgt["x"], tgt["y"], tz))
        for j, v in sug.items():
            if j == "abdomen_pitch":
                abdomen.append(v)
            else:
                jt[j] = jt.get(j, v) + 0.8 * (v - jt.get(j, v))
    if abdomen:
        jt["abdomen_pitch"] = sum(abdomen) / len(abdomen)
    return jt


solved = 0
for s in range(N):
    env = TwisterEnv()
    obs = env.reset(seed=42 + s, num_targets=2)
    limbs = [c["limb"] for c in obs["commands"]]
    best = None
    fell = False
    for _ in range(STEPS):
        res = env.step({"joint_targets": merged_ik(env, obs), "delta": False})
        obs = res.observation
        errs = obs.get("placement_errors", [])
        if best is None or max(errs) < max(best):
            best = errs
        if res.info.get("termination_reason") == "fall":
            fell = True
            break
        if res.info.get("success") == "True":
            break
    hit = best is not None and all(e <= 0.08 for e in best)
    solved += hit
    bests = ",".join(f"{e:.3f}" for e in best) if best else "?"
    print(f"  seed={42+s} {str(limbs):42s} best=[{bests}] {'BOTH-HIT' if hit else ''}{'  FELL' if fell else ''}")

print(f"\n  both-placed {solved}/{N}")
