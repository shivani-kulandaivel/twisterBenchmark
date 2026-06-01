"""Scripted AI benchmark."""

from __future__ import annotations

from env import TwisterEnv


def run_ai_bench(episodes: int = 10, phase: int = 2) -> dict[str, float]:
    successes = 0
    turns = []
    fails = 0
    illegal = 0
    for ep in range(episodes):
        env = TwisterEnv()
        env.reset(seed=ep, phase=phase)
        done = False
        info = {}
        while not done:
            res = env.step({"use_ik": True})
            done = bool(res.terminated or res.truncated)
            info = res.info
        if info.get("termination_reason") in {"success", "max_turns"}:
            successes += 1
        if info.get("termination_reason") == "fall":
            fails += 1
        illegal += int(info.get("constraint_violations", 0))
        turns.append(float(info.get("turns_survived", 0)))
    return {
        "successful_moves": float(successes),
        "game_length": sum(turns) / max(1, len(turns)),
        "balance_failures": float(fails),
        "illegal_moves": float(illegal),
        "avg_turns_survived": sum(turns) / max(1, len(turns)),
    }
