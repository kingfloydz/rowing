"""End-to-end check of the configured contact filter, including oar contacts."""
import unittest
import numpy as np
import mujoco
import torch
from types import SimpleNamespace
from rowing_mjlab.config import env_cfg
from rowing_mjlab.asset import build_spec
from rowing_mjlab import mdp


class ContactSensorTests(unittest.TestCase):
    def test_sensor_reports_cross_oar_contact(self):
        from mjlab.envs import ManagerBasedRlEnv
        cfg=env_cfg(1,play=True)
        self.assertEqual(cfg.scene.sensors[0].secondary_policy,'any')
        env=ManagerBasedRlEnv(cfg,device='cpu')
        try:
            m=env.sim.mj_model;d=mujoco.MjData(m)
            sensor_ids=[i for i in range(m.nsensor) if m.sensor(i).name.startswith('forbidden_contact')]
            self.assertTrue(sensor_ids)
            self.assertTrue(np.all(m.sensor_refid[sensor_ids]==-1))
            # Put both shafts on the same horizontal line across the hull.
            for side,sign in [('port',1),('starboard',-1)]:
                d.qpos[m.joint('rowing/'+side+'_sweep').qposadr[0]]=sign*.7853981634
                d.qpos[m.joint('rowing/'+side+'_elevation').qposadr[0]]=sign*.3141592654
                d.qpos[m.joint('rowing/'+side+'_feather').qposadr[0]]=0
            mujoco.mj_forward(m,d)
            oars={m.body('rowing/port_oar').id,m.body('rowing/starboard_oar').id}
            self.assertTrue(any({int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])}==oars for c in d.contact))
            found=torch.tensor(d.sensordata[[m.sensor_adr[i] for i in sensor_ids]])[None,:]
            proxy=SimpleNamespace(num_envs=1,scene={'forbidden_contact':SimpleNamespace(data=SimpleNamespace(found=found))})
            self.assertEqual(float(mdp.collision_cost(proxy)[0]),1.)
        finally:env.close()


if __name__=='__main__':unittest.main()
