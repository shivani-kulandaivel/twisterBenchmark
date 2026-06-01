import json
import sys
from pathlib import Path

trace = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
    (Path(__file__).parent / "data" / "traces").glob("*.jsonl"),
    key=lambda p: p.stat().st_mtime)[-1]

print(f"Trace: {trace.name}\n")
episodes = {}
for line in trace.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    ev = json.loads(line)
    if ev["event_type"] != "step_result":
        continue
    eid = ev["episode_id"][:8]
    info = ev["payload"]["info"]
    episodes.setdefault(eid, []).append({
        "step": ev["step"],
        "err": float(info.get("placement_error", "nan")),
        "success": info.get("success"),
        "reason": info.get("termination_reason", ""),
    })

for eid, steps in episodes.items():
    first = steps[0]["err"]
    best = min(s["err"] for s in steps)
    last = steps[-1]["err"]
    reason = steps[-1]["reason"] or "ran-out"
    hit = "SUCCESS" if any(s["success"] == "True" for s in steps) else ""
    print(f"  ep {eid}: start={first:.3f}  best={best:.3f}  end={last:.3f}  "
          f"({len(steps)} steps, {reason}) {hit}")
