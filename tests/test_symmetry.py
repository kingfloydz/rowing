import unittest
from types import SimpleNamespace
import torch
import mujoco
from hydrodynamics.models import ROOT
from rowing_mjlab.symmetry import BilateralSymmetry, MIRROR_PAIRS


class SymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
        cls.names = tuple(cls.m.joint(i).name for i in range(cls.m.njnt) if cls.m.joint(i).name.endswith('_joint'))
        cls.sym = BilateralSymmetry(cls.m,cls.names,'cpu')

    def state(self, batch=3):
        q=torch.tensor(self.m.qpos0,dtype=torch.float32).expand(batch,-1).clone()
        v=torch.zeros(batch,self.m.nv)
        c=q[:,[self.m.joint(n).qposadr[0] for n in self.names]].clone()
        return q,v,c

    def test_assembled_pose_has_zero_cost(self):
        self.assertTrue(torch.equal(self.sym.cost(*self.state()),torch.zeros(3)))

    def test_all_pairs_respect_mirror_sign(self):
        s=self.sym
        for i,(_,sign) in enumerate(MIRROR_PAIRS):
            q,v,c=self.state()
            q[:,s.q[0][i]]+=.2; q[:,s.q[1][i]]+=sign*.2
            v[:,s.v[0][i]]=.5; v[:,s.v[1][i]]=sign*.5
            c[:,s.commands[0][i]]+=.2; c[:,s.commands[1][i]]+=sign*.2
            self.assertTrue(torch.all(s.cost(q,v,c)==0))
            q[:,s.q[1][i]]-=sign*.4
            self.assertTrue(torch.all(s.cost(q,v,c)>0))

    def test_velocity_and_command_independently_penalized(self):
        s=self.sym
        for kind in ('velocity','command'):
            q,v,c=self.state()
            if kind=='velocity':v[:,s.v[0][0]]=1
            else:c[:,s.commands[0][0]]+=.3
            self.assertTrue(torch.all(s.cost(q,v,c)>0))

    def test_small_offsets_tolerated_and_batch_gradient_finite(self):
        s=self.sym;q,v,c=self.state()
        q[:,s.q[0]]+=.02;c[:,s.commands[0]]+=.02
        self.assertTrue(torch.all(s.cost(q,v,c)==0))
        c[1,s.commands[0]]+=.3;c.requires_grad_(True)
        cost=s.cost(q,v,c);self.assertEqual(float(cost[0].detach()),0)
        self.assertGreater(float(cost[1].detach()),0);self.assertEqual(float(cost[2].detach()),0)
        cost.sum().backward();self.assertTrue(torch.isfinite(c.grad).all())

    def test_mdp_wiring(self):
        from rowing_mjlab import mdp
        q,v,c=self.state();c[:,self.sym.commands[0][0]]+=.3
        env=SimpleNamespace(rowing=SimpleNamespace(symmetry=self.sym,commanded_targets=c),
                            sim=SimpleNamespace(data=SimpleNamespace(qpos=q,qvel=v)))
        torch.testing.assert_close(mdp.bilateral_asymmetry_cost(env),self.sym.cost(q,v,c))
        self.assertTrue(torch.all(mdp.bilateral_command_rmse_deg(env)>0))
        self.assertTrue(torch.all(mdp.bilateral_position_rmse_deg(env)==0))

    def test_midline_yaw_roll_penalized_independently(self):
        s=self.sym
        for i in range(2):
            for kind in ('position','velocity','command'):
                q,v,c=self.state()
                if kind=='position': q[:,s.center_q[i]]+=.3
                elif kind=='velocity': v[:,s.center_v[i]]=1.
                else: c[:,s.center_commands[i]]+=.3
                self.assertTrue(torch.all(s.cost(q,v,c)>0), (i,kind))

    def test_midline_pitch_is_allowed(self):
        q,v,c=self.state()
        joint=self.m.joint('waist_pitch_joint')
        q[:,int(joint.qposadr[0])]+=.3
        v[:,int(joint.dofadr[0])]=1.
        c[:,self.names.index('waist_pitch_joint')]+=.3
        self.assertTrue(torch.all(self.sym.cost(q,v,c)==0))


if __name__=='__main__':
    unittest.main()
