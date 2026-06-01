"""Reward shaping for Twister episodes."""

from __future__ import annotations


def reward_breakdown(
    placed: bool,
    error_m: float,
    stability_score: float,
    fell: bool,
    slipped: bool,
) -> dict[str, float]:
    if fell:
        return {"fall_penalty": -0.2, "total": -0.2}
    if slipped:
        return {"constraint_penalty": -0.2, "total": -0.2}
    placement = 1.0 if placed else max(0.0, 0.4 - error_m)
    stability = 0.1 * max(0.0, min(1.0, stability_score))
    efficiency = -0.01 * min(1.0, error_m)
    total = placement + stability + efficiency
    return {
        "placement": float(placement),
        "stability": float(stability),
        "efficiency": float(efficiency),
        "total": float(total),
    }
