from sim.balance_controller import BalanceController
from sim.humanoid import HumanoidSim
from sim.reach_controller import ReachController
from sim.reachability import ReachabilitySolver


def test_reachability_returns_struct():
    sim = HumanoidSim()
    sim.reset(seed=0)
    solver = ReachabilitySolver(sim, ReachController(sim), BalanceController(sim))
    result = solver.can_reach("left_hand", 0.1, 0.1)
    assert isinstance(result.feasible, bool)
    assert isinstance(result.error_m, float)
