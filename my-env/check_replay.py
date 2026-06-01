import json
d = json.load(open("showcase/replay.json", encoding="utf-8"))
ep = list(d["replay"].keys())[0]
steps = d["replay"][ep]
print(f"Episode: {ep[:8]}...")
print(f"Steps: {len(steps)}")
for s in steps:
    print(f"  step={s['step']}  frame={s.get('frame','NONE')}  reward={s['reward']}")
