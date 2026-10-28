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
            eq=updated.equality(side+'_hand_fixed').id
            mask=(new_data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(new_data.efc_id==eq)
            np.testing.assert_allclose(new_data.efc_pos[mask],0.,atol=1e-7)
        left=updated.site('left_gripper_center').id
        right=updated.site('right_gripper_center').id
        self.assertGreater(new_data.site_xpos[left,2]-new_data.site_xpos[right,2],.003)
        for side in ('left','right'):
            eq=updated.equality(side+'_foot_fixed').id
            mask=(new_data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(new_data.efc_id==eq)
            np.testing.assert_allclose(new_data.efc_pos[mask],0.,atol=1e-7)
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

    def test_outward_rotation_and_joint_limits(self):
        original = mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
        values = reset_joint_positions()
        # Unequal gate heights and refitted grip frames do not imply mirrored yaw.
        for name,value in values.items():
            bounds=original.jnt_range[original.joint(name).id]
            self.assertGreaterEqual(value,bounds[0])
            self.assertLessEqual(value,bounds[1])
        # Check the elbow's actual direction, rather than a historical shoulder
        # yaw offset that need not hold after refitting the shoulder and wrist.
        data=mujoco.MjData(original)
        for name,value in values.items():data.qpos[original.joint(name).qposadr[0]]=value
        mujoco.mj_kinematics(original,data)
        torso=original.body('torso_link').id
        ids=[original.joint('right_'+part+'_joint').id for part in ('shoulder_yaw','elbow','wrist_roll')]
        s,e,w=data.xanchor[ids].copy()
        chord=w-s;projection=s+chord*((e-s)@chord)/(chord@chord)
        outward=-((e-projection)@data.xmat[torso].reshape(3,3))[1]
        self.assertGreater(outward,0.)
        axis=data.xaxis[ids[1]]
        upper,lower=e-s,w-e
        upper-=axis*(upper@axis);lower-=axis*(lower@axis)
        bend=np.arctan2(axis@np.cross(upper,lower),upper@lower)
        self.assertLess(abs(np.rad2deg(bend)),6.)
