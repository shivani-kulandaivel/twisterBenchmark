"""Session debug logging (NDJSON). Remove after debug session."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_DEBUG_LOG = Path(__file__).parent.parent / ".cursor" / "debug-f3bce4.log"


def dbg(location: str, message: str, data: dict[str, Any], hypothesis_id: str) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "f3bce4",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
            "runId": data.get("runId", "pre-fix"),
        }
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
    # #endregion
