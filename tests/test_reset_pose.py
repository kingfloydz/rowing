import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from hydrodynamics.models import ROOT
from rowing_mjlab.reset_pose import apply_reset_joints, reset_joint_positions


class ResetPoseTests(unittest.TestCase):
    def test_rebasing_preserves_kinematics_and_grip_closure(self):
        original = mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
        root = ET.parse(ROOT/'scene.xml').getroot()
        apply_reset_joints(root)
        root.find('compiler').set('meshdir',str(ROOT))
        updated = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        old_data, new_data = mujoco.MjData(original), mujoco.MjData(updated)
        mujoco.mj_forward(original,old_data)
        mujoco.mj_forward(updated,new_data)
        for side in ('left','right'):
            hand = original.body(side+'_dex1_base_link').id
            np.testing.assert_allclose(new_data.xpos[hand],old_data.xpos[hand],atol=1e-10)
            np.testing.assert_allclose(new_data.xmat[hand],old_data.xmat[hand],atol=1e-10)
        # Rebasing must preserve every body's physical transform at equal q,
        # including configurations away from the new reset.
        for offset in (0., .03):
            old_data.qpos[:] = updated.qpos0
            old_data.qpos[7:] += offset
            new_data.qpos[:] = old_data.qpos
            mujoco.mj_forward(original,old_data)
            mujoco.mj_forward(updated,new_data)
            np.testing.assert_allclose(new_data.xpos,old_data.xpos,atol=1e-10)
            np.testing.assert_allclose(new_data.xmat,old_data.xmat,atol=1e-10)

    def test_symmetric_axial_angles_and_outward_rotation(self):
        original = mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
        values = reset_joint_positions()
        self.assertAlmostEqual(values['left_shoulder_yaw_joint'],-values['right_shoulder_yaw_joint'])
        for side, sign in (('left',1),('right',-1)):
            name = side+'_shoulder_yaw_joint'
            old = original.qpos0[original.joint(name).qposadr[0]]
            self.assertGreater(sign*np.rad2deg(values[name]-old),50.)
