from sim.constraints import ConstraintManager
from sim.humanoid import HumanoidSim


def test_constraints_lock_unlock():
    sim = HumanoidSim()
    sim.reset(seed=0)
    mgr = ConstraintManager(sim)
    mgr.lock("left_hand", 0.1, 0.1, "red")
    assert len(mgr.active_constraints()) == 1
    mgr.unlock("left_hand")
    assert len(mgr.active_constraints()) == 0
