"""Confirm whether _parse_raw drops the model's JSON."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_llm import _parse_raw

trace = sorted((ROOT / "data" / "traces").glob("*.jsonl"),
               key=lambda p: p.stat().st_mtime)[-1]
for l in trace.read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    e = json.loads(l)
    if e["event_type"] == "model_call":
        raw = e["payload"]["text"]
        parsed = _parse_raw(raw)
        print("=== RAW MODEL OUTPUT ===")
        print(raw)
        print("=== _parse_raw RESULT ===")
        print(parsed)
        print("=== json.loads(raw) directly ===")
        try:
            print(json.loads(raw))
        except Exception as ex:
            print("FAILED:", ex)
        break
