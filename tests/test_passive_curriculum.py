import unittest
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from rowing_mjlab.asset import build_spec,h2_names
from rowing_mjlab.curriculum import PenaltyCurriculum
from rowing_mjlab import mdp


class PassiveCurriculumTests(unittest.TestCase):
    def test_hinges_are_passive_connector_axes_and_reset_closed(self):
        m=build_spec().compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        self.assertEqual(len(h2_names()),31)
        self.assertEqual(m.nq,48);self.assertEqual(m.nv,47)
        for side in ('left','right'):
            j=m.joint(side+'_hand_passive_roll').id
            self.assertEqual(m.jnt_type[j],mujoco.mjtJoint.mjJNT_HINGE)
            np.testing.assert_allclose(m.jnt_axis[j],[1,0,0])
            self.assertFalse(m.jnt_limited[j]);self.assertEqual(m.qpos0[m.jnt_qposadr[j]],0)
            self.assertAlmostEqual(m.dof_damping[m.jnt_dofadr[j]],.02)
            self.assertNotIn(side+'_hand_passive_roll',h2_names())
            e=m.equality(side+'_hand_fixed').id
            mask=(d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id==e)
            np.testing.assert_allclose(d.efc_pos[mask],0,atol=1e-6)

    def test_completed_iteration_boundary_is_independent_of_environment_count(self):
        for n in (1,3,2048):
            c=PenaltyCurriculum(n,'cpu');c.set_iteration(999)
            self.assertTrue(torch.all(c.scale('action')==1))
            c.advance()
            self.assertEqual(c.completed_iterations,1000)
            self.assertTrue(torch.all(c.scale('action')==40))
            self.assertTrue(torch.all(c.scale('angular')==5))

    def test_reward_scaling_and_checkpoint_roundtrip(self):
        c=PenaltyCurriculum(2,'cpu');c.set_iteration(1000)
        env=SimpleNamespace(rowing=SimpleNamespace(boat=0,penalty_curriculum=c),
            sim=SimpleNamespace(data=SimpleNamespace(cvel=torch.ones(2,1,6))),
            action_manager=SimpleNamespace(action=torch.ones(2,29),prev_action=torch.zeros(2,29)))
        torch.testing.assert_close(-.05*mdp.curriculum_action_rate_cost(env),torch.tensor([-2.,-2.]))
        torch.testing.assert_close(-mdp.curriculum_angular_velocity_cost(env),torch.tensor([-15.,-15.]))
        restored=PenaltyCurriculum(2,'cpu');restored.load_state_dict(c.state_dict())
        self.assertEqual(restored.completed_iterations,1000)
        resized=PenaltyCurriculum(3,'cpu');resized.load_state_dict(c.state_dict())
        self.assertEqual(resized.completed_iterations,1000)

    def test_resume_uses_current_threshold(self):
        c=PenaltyCurriculum(1,'cpu')
        c.load_state_dict({'completed_iterations':1500,'threshold':2000})
        self.assertEqual(c.threshold,1000)
        self.assertTrue(torch.all(c.action_scale==40))

    def test_runner_checkpoint_preserves_curriculum(self):
        import tempfile
        from pathlib import Path
        from rowing_mjlab.runner import RowingRunner
        c=PenaltyCurriculum(2,'cpu');c.set_iteration(1000)
        env=SimpleNamespace(common_step_counter=123,rowing=SimpleNamespace(penalty_curriculum=c))
        runner=object.__new__(RowingRunner)
        runner.env=SimpleNamespace(unwrapped=env)
        runner.alg=SimpleNamespace(save=lambda:{},load=lambda *args:True)
        runner.cfg={'upload_model':False};runner.current_learning_iteration=10
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'checkpoint.pt');runner.save(path)
            c.set_iteration(0);runner.load(path,map_location='cpu')
            self.assertEqual(c.completed_iterations,1000)
            self.assertEqual(runner.current_learning_iteration,1000)
            runner.load(path,load_cfg={'iteration':False},map_location='cpu')
            self.assertEqual(c.completed_iterations,0)

    def test_runner_advances_once_after_successful_ppo_update(self):
        from unittest.mock import patch
        from rowing_mjlab.runner import RowingRunner,MjlabOnPolicyRunner
        c=PenaltyCurriculum(2,'cpu');c.set_iteration(999)
        def initialize(runner):
            runner.env=SimpleNamespace(unwrapped=SimpleNamespace(rowing=SimpleNamespace(penalty_curriculum=c)))
            runner.alg=SimpleNamespace(update=lambda:{'loss':1.})
        with patch.object(MjlabOnPolicyRunner,'__init__',initialize):runner=RowingRunner()
        self.assertEqual(runner.alg.update(),{'loss':1.})
        self.assertEqual(c.completed_iterations,1000)
        self.assertTrue(torch.all(c.action_scale==40))


if __name__=='__main__':unittest.main()
