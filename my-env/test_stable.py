"""Does the body stay upright doing NOTHING? Isolates the anchor from the IK."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from env import TwisterEnv

env = TwisterEnv()
obs = env.reset(seed=42)
neutral = dict(obs["joints"])
print(f"after reset: torso={env._sim.get_torso_state()}  upright={env._sim.is_upright()}")
for i in range(1, 31):
    res = env.step({"joint_targets": neutral, "delta": False})
    ts = env._sim.get_torso_state()
    if i % 5 == 0 or res.info.get("termination_reason") == "fall":
        print(f"  step {i:2d}: torso_z={ts['z']:.3f} tilt={ts['tilt_deg']:.1f}  "
              f"upright={env._sim.is_upright()} reason={res.info.get('termination_reason','')}")
    if res.info.get("termination_reason") == "fall":
        print("  -> FELL while doing nothing")
        break
