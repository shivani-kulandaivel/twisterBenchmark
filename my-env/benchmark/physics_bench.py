"""Physics benchmark measurements."""

from __future__ import annotations

import random
import time

from sim.balance_controller import BalanceController
from sim.humanoid import HumanoidSim
from sim.reach_controller import ReachController


def run_physics_bench(samples: int = 100) -> dict[str, float]:
    sim = HumanoidSim()
    sim.reset(seed=0)
    reach = ReachController(sim)
    balance = BalanceController(sim)
    rng = random.Random(0)

    t0 = time.perf_counter()
    for _ in range(samples):
        sim.step_physics(substeps=2)
    physics_ms = ((time.perf_counter() - t0) * 1000.0) / samples

    t1 = time.perf_counter()
    for _ in range(samples):
        x = rng.uniform(-0.3, 0.3)
        y = rng.uniform(-0.2, 0.3)
        reach.suggest_targets("left_hand", x, y, ik_iterations=8)
    ik_ms = ((time.perf_counter() - t1) * 1000.0) / samples

    t2 = time.perf_counter()
    for _ in range(samples):
        balance.compute()
    balance_ms = ((time.perf_counter() - t2) * 1000.0) / samples

    return {
        "collision_detection_ms": physics_ms,
        "ik_solve_ms": ik_ms,
        "balance_solve_ms": balance_ms,
        "ms_per_sim_step": physics_ms,
        "physics_fps": 1000.0 / max(1e-6, physics_ms),
    }
