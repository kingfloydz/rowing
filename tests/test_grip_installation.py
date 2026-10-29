"""Check physical bilateral grip installation, not raw quaternion signs."""
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from rowing_mjlab.asset import build_spec

class GripInstallationTests(unittest.TestCase):
    def test_mirrored_effective_installation_and_reset_closure(self):
        m=build_spec().compile();d=mujoco.MjData(m);mirror=np.diag([1,-1,1]);effective=[]
        for side in ('left','right'):
            e=m.equality(side+'_hand_fixed').id;h=m.eq_obj1id[e]
            mount=Rotation.from_quat(m.body_quat[h][[1,2,3,0]])
            weld=Rotation.from_quat(m.eq_data[e,6:10][[1,2,3,0]])
            effective.append((mount*weld).as_matrix())
        np.testing.assert_allclose(effective[0],mirror@effective[1]@mirror,atol=1e-10)
        # Equal physical feather rotation must stay mirrored over its full range.
        for angle in np.linspace(0,np.deg2rad(65),9):
            twist=Rotation.from_rotvec([0,angle,0]).as_matrix()
            np.testing.assert_allclose(effective[0]@twist,mirror@(effective[1]@twist)@mirror,atol=1e-10)
        mujoco.mj_forward(m,d)
        error=d.efc_pos[d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        self.assertLess(np.max(np.abs(error)),1e-7)
        self.assertFalse(any(c.dist<-.0001 for c in d.contact))
