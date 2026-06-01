"""Agent benchmark metrics from environment episodes."""

from __future__ import annotations

from env import TwisterEnv


def run_agent_bench(episodes: int = 8) -> dict[str, float]:
    turns: list[float] = []
    decision_times: list[float] = []
    reach_acc: list[float] = []
    balance: list[float] = []
    violations = 0.0
    score = 0.0
    for ep in range(episodes):
        env = TwisterEnv()
        env.reset(seed=ep, phase=2)
        done = False
        info = {}
        while not done:
            step = env.step({"use_ik": True})
            done = bool(step.terminated or step.truncated)
            info = step.info
        turns.append(float(info.get("turns_survived", 0.0)))
        decision_times.append(float(info.get("avg_decision_time_s", 0.0)))
        reach_acc.append(float(info.get("reach_accuracy", 0.0)))
        balance.append(float(info.get("balance_success_rate", 0.0)))
        violations += float(info.get("constraint_violations", 0.0))
        score += float(info.get("total_score", 0.0))
    n = max(1, len(turns))
    return {
        "survival_turns": sum(turns) / n,
        "avg_decision_time_s": sum(decision_times) / n,
        "reach_accuracy": sum(reach_acc) / n,
        "balance_success_rate": sum(balance) / n,
        "constraint_violations": violations / n,
        "tokens_generated": 0.0,
        "planning_depth": 0.0,
        "action_quality": score / n,
        "total_score": score / n,
    }
