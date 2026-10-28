import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from rowing_mjlab import mdp


class ForwardSpeedTests(unittest.TestCase):
    def env(self,vel,rotation=None):
        n=len(vel)
        r=torch.eye(3).repeat(n,1,1) if rotation is None else torch.tensor(rotation,dtype=torch.float32)
        d=SimpleNamespace(cvel=torch.zeros(n,1,6),xmat=r[:,None],
                          xipos=torch.zeros(n,1,3),subtree_com=torch.zeros(n,1,3))
        d.cvel[:,0,3:]=torch.tensor(vel,dtype=torch.float32)
        return SimpleNamespace(rowing=SimpleNamespace(boat=0),sim=SimpleNamespace(data=d))

    def test_forward_reverse_sideways_and_heave(self):
        env=self.env([[2,0,0],[-2,0,0],[0,2,0],[0,0,3],[0,0,0]])
        torch.testing.assert_close(mdp.boat_forward_speed(env),torch.tensor([2.,-2.,0.,0.,0.]))

    def test_heading_and_pitch_do_not_reward_vertical_motion(self):
        rot=Rotation.from_euler('zy',[[90,0],[0,30]],degrees=True).as_matrix()
        env=self.env([[0,2,0],[2,0,5]],rot)
        torch.testing.assert_close(mdp.boat_forward_speed(env),torch.tensor([2.,2.]))

    def test_velocity_reference_point(self):
        env=self.env([[0,0,0]])
        env.sim.data.cvel[0,0,2]=2.
        env.sim.data.xipos[0,0,1]=1.
        torch.testing.assert_close(mdp.boat_forward_speed(env),torch.tensor([-2.]))

    def test_lateral_cost_follows_heading_and_excludes_heave(self):
        rot=Rotation.from_euler('z',np.array([[90],[0],[0],[0]]),degrees=True).as_matrix()
        env=self.env([[2,0,0],[0,-2,0],[3,0,0],[0,0,5]],rot)
        torch.testing.assert_close(mdp.boat_lateral_velocity_cost(env),torch.tensor([4.,4.,0.,0.]))

    def test_angular_cost_includes_all_axes_without_linear_speed(self):
        env=self.env([[8,0,0],[0,9,0]])
        env.sim.data.cvel[:,0,:3]=torch.tensor([[1.,2.,3.],[-1.,0.,0.]])
        torch.testing.assert_close(mdp.boat_angular_velocity_cost(env),torch.tensor([14.,1.]))


if __name__=='__main__':
    unittest.main()
