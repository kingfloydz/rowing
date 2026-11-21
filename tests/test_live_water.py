from types import SimpleNamespace
import torch
from rowing_mjlab.mdp import recovery_immersion_cost
from rowing_mjlab.curriculum import PenaltyCurriculum


def test_only_late_recovery_interior_is_penalized():
    phase=torch.tensor([.2,.435,.48,.55,.75,.98,.015,.06])
    areas=torch.tensor([[1.,.50]]).expand(len(phase),-1)
    c=PenaltyCurriculum(len(phase),"cpu")
    s=SimpleNamespace(penalty_curriculum=c,water_level=torch.zeros_like(phase),
                      blade_area=SimpleNamespace(wetted_fraction=lambda *args:areas),
                      reference=lambda:(phase,None,None,None))
    env=SimpleNamespace(rowing=s,sim=SimpleNamespace(data=None))
    c.set_iteration(c.threshold-1)
    torch.testing.assert_close(recovery_immersion_cost(env),torch.zeros_like(phase))
    c.advance()
    torch.testing.assert_close(recovery_immersion_cost(env),torch.tensor([0.,0.,0.,.75,.75,.75,0.,0.]))


def test_training_substeps_apply_episode_loads_and_selective_reset():
    from mjlab.envs import ManagerBasedRlEnv
    from rowing_mjlab.config import env_cfg
    env=ManagerBasedRlEnv(env_cfg(2),device="cpu")
    try:
        env.reset()
        water=env.rowing.live_water
        theta=water.theta.clone()
        for _ in range(2):
            env.step(torch.zeros(2,env.action_manager.total_action_dim))
        assert torch.isfinite(water.out).all() and water.out.abs().max()>0
        torch.testing.assert_close(water.theta,theta)
        remaining=water.out[1].clone()
        env.action_manager.reset(torch.tensor([0]))
        torch.testing.assert_close(water.theta[1],theta[1])
        torch.testing.assert_close(water.out[1],remaining)
        assert water.out[0].count_nonzero()==0 and water.fresh[0]
    finally:
        env.close()
