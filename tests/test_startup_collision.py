import unittest
import numpy as np
import torch
import mujoco
from types import SimpleNamespace
from rowing_mjlab import mdp
from rowing_mjlab.reference import OarReference, PERIOD
from rowing_mjlab.asset import build_spec


class StartupCollisionTests(unittest.TestCase):
    def test_oar_rewards_are_independent_and_bilateral(self):
        sigma=torch.deg2rad(torch.tensor([10.,4.,15.,10.,4.,15.]))
        q=torch.zeros(3,6)
        for axis in range(3):
            q[axis,axis]=sigma[axis]
        state=SimpleNamespace(oar_q=torch.arange(6),ref=SimpleNamespace(sigma=sigma),
                              reference=lambda:(None,None,None,torch.zeros(3,9)))
        env=SimpleNamespace(rowing=state,sim=SimpleNamespace(data=SimpleNamespace(qpos=q)))
        scores=torch.stack([mdp.track_oar_sweep(env),mdp.track_oar_elevation(env),mdp.track_oar_feather(env)],dim=-1)
        expected=torch.ones(3,3);expected.fill_diagonal_(float(np.exp(-.5)))
        torch.testing.assert_close(scores,expected)

    def test_first_cycle_rate_and_blend(self):
        ref=OarReference('cpu');p0=torch.tensor(.2);initial=torch.zeros(6)
        t=torch.tensor([0.,.5,1.,PERIOD,2*PERIOD-1e-6,2*PERIOD,2.5*PERIOD,3*PERIOD,4*PERIOD],dtype=torch.float64)
        phase,rate,blend,target=ref.target(t,p0,initial)
        expected=(p0+torch.tensor([0.,.5/(2*PERIOD),1/(2*PERIOD),.5,1-1e-6/(2*PERIOD),1.,1.5,2.,3.],dtype=torch.float64))%1
        torch.testing.assert_close(phase,expected)
        torch.testing.assert_close(rate,torch.tensor([.5,.5,.5,.5,.5,1.,1.,1.,1.]))
        torch.testing.assert_close(blend,torch.tensor([0.,.5,1.,1.,1.,1.,1.,1.,1.],dtype=torch.float64))
        torch.testing.assert_close(target[0],torch.zeros(6,dtype=torch.float64))
        torch.testing.assert_close(target[2],ref(phase[2]))

    def test_cross_oar_contacts_generated(self):
        spec=build_spec();m=spec.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        a=m.geom('oar_collision_port_shaft').id;b=m.geom('oar_collision_starboard_shaft').id
        self.assertTrue(m.geom_contype[a]&m.geom_conaffinity[b])
        def cross_contacts():
            return [c for c in d.contact if {int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])}
                    == {m.body('port_oar').id,m.body('starboard_oar').id}]
        self.assertFalse(cross_contacts(),'Unexpected oar contact in calibrated reset pose')
        # Put just the starboard shaft collider onto the port shaft in world space.
        body=int(m.geom_bodyid[b]);R=d.xmat[body].reshape(3,3)
        geom=next(g for g in spec.geoms if g.name=='oar_collision_starboard_shaft')
        center=d.geom_xpos[a]+np.array([.005,0,0])
        axis=d.geom_xmat[a].reshape(3,3)[:,2]*m.geom_size[a,1]
        geom.fromto=np.r_[R.T@(center-axis-d.xpos[body]),R.T@(center+axis-d.xpos[body])]
        # Recompile to rebuild collision BVHs after moving a collider.
        m=spec.compile();d=mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        self.assertTrue(cross_contacts(),'Oar-oar intersections must produce physical contacts')


if __name__=='__main__':
    unittest.main()
