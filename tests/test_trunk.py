"""CPU checks for sign, reference frame, phase and reward integration."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from rowing_mjlab.trunk import TrunkReference, trunk_vertical_angle, angular_error


def rotation(axis, deg):
    angles = np.asarray(deg)
    if angles.ndim == 1 and len(axis) == 1:
        angles = angles[:, None]
    return torch.tensor(Rotation.from_euler(axis, angles, degrees=True).as_matrix(), dtype=torch.float32)


class TrunkTests(unittest.TestCase):
    def test_world_vertical_and_sign(self):
        torso = rotation('y', [-18, 0, 34])
        boat = torch.eye(3).expand(3,3,3)
        torch.testing.assert_close(torch.rad2deg(trunk_vertical_angle(torso, boat)),
                                   torch.tensor([-18.,0.,34.]), atol=1e-5, rtol=1e-5)
        # Boat pitch must not be subtracted from world-vertical torso tilt.
        torch.testing.assert_close(trunk_vertical_angle(torso, rotation('y', [15,15,15])),
                                   trunk_vertical_angle(torso, boat))

    def test_yaw_invariance_and_roll(self):
        yaw = rotation('z', 73)
        torso = rotation('y', 20)
        torch.testing.assert_close(trunk_vertical_angle(yaw@torso, yaw),
                                   trunk_vertical_angle(torso, torch.eye(3)))
        self.assertAlmostEqual(float(trunk_vertical_angle(rotation('x',25),torch.eye(3))),0.)

    def test_measured_reference_and_phase(self):
        ref = TrunkReference('cpu')
        phase = torch.linspace(0,1,801)
        values = ref(phase)
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(float(values[0]),float(values[-1]))
        deg = torch.rad2deg(values)
        self.assertTrue(-20 < float(deg[0]) < -15)
        self.assertTrue(32 < float(deg.max()) < 37)
        self.assertTrue(.43 < float(phase[deg.argmax()]) < .56)
        self.assertGreater(float(ref(torch.tensor(.58))),0)
        self.assertTrue(torch.allclose(ref(phase+1),values,atol=2e-6))
        initial = torch.tensor(.12)
        torch.testing.assert_close(ref.target(phase,torch.zeros_like(phase),initial),
                                   initial.expand_as(phase))
        torch.testing.assert_close(ref.target(phase,torch.ones_like(phase),initial), values)

    def test_runtime_mdp_observation_reward_and_metric(self):
        from rowing_mjlab import mdp
        ref = TrunkReference('cpu')
        phase = torch.tensor([0.,.25,.4])
        target = ref(phase)
        offset = torch.tensor([0.,10.,-20.])*torch.pi/180
        torso = rotation('y',torch.rad2deg(target+offset).numpy())
        xmat = torch.stack([torch.eye(3).expand(3,3,3),torso],dim=1).reshape(3,2,9)
        state = SimpleNamespace(boat=0,torso=1,trunk_ref=ref,initial_trunk=torch.tensor(.1),
            reference=lambda:(phase,torch.ones(3),torch.ones(3),torch.zeros(3,9)))
        env = SimpleNamespace(rowing=state,sim=SimpleNamespace(data=SimpleNamespace(xmat=xmat)))
        obs = mdp.privileged_trunk(env)
        self.assertEqual(obs.shape,(3,2))
        torch.testing.assert_close(obs[:,1],target)
        torch.testing.assert_close(mdp.trunk_error_deg(env),torch.tensor([0.,10.,20.]),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(mdp.track_trunk(env),torch.exp(-torch.tensor([0.,1.,4.])),atol=1e-6,rtol=1e-5)

    def test_actual_model_axis_convention(self):
        import mujoco
        from hydrodynamics.models import ROOT
        model=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
        data=mujoco.MjData(model)
        mujoco.mj_kinematics(model,data)
        torso=torch.tensor(data.xmat[model.body('torso_link').id].reshape(3,3))
        boat=torch.tensor(data.xmat[model.body('boat').id].reshape(3,3))
        self.assertLess(float((torso[:,0]*boat[:,0]).sum()),-.9)
        self.assertGreater(float(torso[2,2]),.9)
        angle=trunk_vertical_angle(torso,boat)
        self.assertTrue(-10 < float(torch.rad2deg(angle)) < 0)


if __name__ == '__main__':
    unittest.main()
