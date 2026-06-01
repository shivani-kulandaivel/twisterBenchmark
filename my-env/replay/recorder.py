"""Replay recorder utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReplayStep:
    step: int
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    terminated: bool
    info: dict[str, Any]


class ReplayRecorder:
    def __init__(self) -> None:
        self.steps: list[ReplayStep] = []

    def append(
        self,
        step: int,
        observation: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        terminated: bool,
        info: dict[str, Any],
    ) -> None:
        self.steps.append(
            ReplayStep(
                step=step,
                observation=observation,
                action=action,
                reward=float(reward),
                terminated=bool(terminated),
                info=info,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "step": s.step,
                    "observation": s.observation,
                    "action": s.action,
                    "reward": s.reward,
                    "terminated": s.terminated,
                    "info": s.info,
                }
                for s in self.steps
            ]
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
