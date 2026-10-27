import unittest
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from rowing_mjlab import mdp
from rowing_mjlab.asset import build_spec, reset_float_pose
from hydrodynamics.models import ROOT, load_closed_surface, hydrostatic_wrench
import json


class RollTests(unittest.TestCase):
    def test_reward_and_termination_use_root_roll(self):
        angles=np.array([0,10,-10,30,61,-61,180])
        quat=Rotation.from_euler('xyz',np.column_stack((angles,np.full(7,4),np.full(7,37))),degrees=True).as_quat()
        q=torch.zeros(7,7,dtype=torch.float64)
        q[:,3:7]=torch.tensor(quat[:,[3,0,1,2]])
        env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q)))
        torch.testing.assert_close(mdp.roll_upright(env),torch.exp(-torch.tensor(angles/10).square()))
        torch.testing.assert_close(mdp.boat_roll_abs_deg(env),torch.tensor(abs(angles),dtype=torch.float64))
        self.assertEqual(mdp.capsized(env).tolist(),[False,False,False,False,True,True,True])

    def test_training_reset_balances_weight_and_pitch(self):
        model=build_spec().compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        pos,quat=reset_float_pose()
        np.testing.assert_allclose(model.qpos0[:7],np.r_[pos,quat],atol=1e-12)
        mass=model.body_mass.sum();cg=data.subtree_com[model.body('boat').id]
        level=json.loads((ROOT/'data/rowing_reference_provenance.json').read_text())['water_level_m_at_repository_reset']
        forces=[];moments=[]
        for mesh,body in [('shell_white','boat'),('fin','boat'),('port_blade_3mm','port_oar'),('starboard_blade_3mm','starboard_oar')]:
            tri,_=load_closed_surface(ROOT/'boat_meshes'/f'{mesh}.stl')
            i=model.body(body).id
            force,torque,_=hydrostatic_wrench(tri@data.xmat[i].reshape(3,3).T+data.xpos[i],cg,level)
            forces.append(force);moments.append(torque)
        self.assertAlmostEqual(np.sum(forces,axis=0)[2],mass*9.81,places=5)
        self.assertAlmostEqual(np.sum(moments,axis=0)[1],0.,places=5)
        q=torch.tensor(model.qpos0)[None]
        env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q)))
        self.assertAlmostEqual(float(mdp.roll_upright(env)[0]),1.)


if __name__=='__main__':
    unittest.main()
