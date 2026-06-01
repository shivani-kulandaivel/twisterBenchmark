from sim.balance_controller import BalanceController
from sim.humanoid import HumanoidSim


def test_balance_metrics_shape():
    sim = HumanoidSim()
    sim.reset(seed=0)
    bal = BalanceController(sim)
    m = bal.compute()
    assert len(m.com) == 3
    assert len(m.support_polygon) >= 4
