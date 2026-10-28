import json
import unittest
import mujoco
import numpy as np
import torch
import trimesh
from hydrodynamics.models import ROOT
from rowing_mjlab.asset import build_spec
from rowing_mjlab.reference import OarReference


class CrossingTests(unittest.TestCase):
    def test_full_cycle_clearance_depth_and_joint_limits(self):
        model=build_spec().compile();data=mujoco.MjData(model)
        ref=OarReference('cpu')
        water=json.loads((ROOT/'data/reset_float_pose.json').read_text())['water_level_m']
        sides=('port','starboard')
        joint_ids=[model.joint(s+'_'+j).id for s in sides for j in ('sweep','elevation','feather')]
        qa=model.jnt_qposadr[joint_ids]
        bodies=[model.body(s+'_oar').id for s in sides]
        sites=[model.site(s+'_grasp').id for s in sides]
        vertices=[trimesh.load_mesh(ROOT/f'boat_meshes/{s}_blade_3mm.stl').vertices for s in sides]
        source_phase=np.linspace(0,1,1201)
        q=ref(torch.tensor(source_phase+ref.phase_offset,dtype=torch.float32)).numpy()
        self.assertTrue(np.all(q>=model.jnt_range[joint_ids,0]-1e-6))
        self.assertTrue(np.all(q<=model.jnt_range[joint_ids,1]+1e-6))
        np.testing.assert_allclose(q[0],q[-1],atol=1e-6)
        crossings=0
        for phase,angles in zip(source_phase,q):
            data.qpos[:]=model.qpos0;data.qpos[qa]=angles
            mujoco.mj_forward(model,data)
            heights=[(v@data.xmat[b].reshape(3,3).T+data.xpos[b])[:,2]-water for v,b in zip(vertices,bodies)]
            if .05<=phase<=.35:
                self.assertLess(max(h.max() for h in heights),-.005)
            if .50<=phase<=.90:
                self.assertGreater(min(h.min() for h in heights),.020)
            right,left=data.site_xpos[sites]
            if np.linalg.norm(left[:2]-right[:2])<.15:
                self.assertGreater(left[2]-right[2],.060)
                crossings+=1
            for contact in data.contact:
                self.assertNotEqual({int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])},set(bodies))
        self.assertGreater(crossings,20)


if __name__=='__main__':unittest.main()
