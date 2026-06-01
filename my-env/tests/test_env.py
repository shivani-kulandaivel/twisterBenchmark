from env import TwisterEnv


def test_env_reset_step():
    env = TwisterEnv()
    obs = env.reset(seed=1, phase=1)
    assert "command" in obs
    step = env.step({"use_ik": True})
    assert isinstance(step.reward, float)
