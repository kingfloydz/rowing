"""Behavioral checks for the GPU training task; Warp CPU runs the same hydro kernels."""
from types import SimpleNamespace
import numpy as np
import pytest
import mujoco
torch = pytest.importorskip('torch')
wp = pytest.importorskip('warp')
pytest.importorskip('mjlab')
from scipy.spatial.transform import Rotation
from hydrodynamics.models import ROOT
from hydrodynamics.training_adapter import TrainingRowingHydrodynamics
from rowing_mjlab.hydro import WarpHydrodynamics
from rowing_mjlab.reference import OarReference, PERIOD


def test_reference_start_and_complete_half_speed_cycle():
    ref = OarReference('cpu')
    m = mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    from rowing_mjlab.asset import OAR_NAMES
    q = torch.tensor([m.qpos0[m.joint(n).qposadr[0]] for n in OAR_NAMES],dtype=torch.float32)
    phase0 = ref.nearest(q)
    t = torch.tensor([0.,1e-5,PERIOD,2*PERIOD-1e-5,2*PERIOD,2*PERIOD+PERIOD/4])
    phase,rate,blend,target = ref.target(t,phase0,q)
    torch.testing.assert_close(target[0],q)
    assert torch.max(abs(target[1]-q)) < 1e-6
    torch.testing.assert_close((phase[2]-phase0).remainder(1),torch.tensor(.5))
    torch.testing.assert_close(phase[4],phase0,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close((phase[5]-phase0).remainder(1),torch.tensor(.25),atol=1e-6,rtol=1e-6)
    assert rate.tolist() == [.5,.5,.5,.5,1.,1.]
    torch.testing.assert_close(ref(torch.tensor([0.])),ref(torch.tensor([1.])))


@pytest.fixture(scope='module')
def hydro_comparison():
    m = mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    cpu = TrainingRowingHydrodynamics(m)
    gpu = WarpHydrodynamics(m,5,'cpu')
    states, expected = [],[]
    cases = [(0.12,0.,0.),(0.12,15.,4.),(0.03,-35.,-2.),(.8,5.,1.),(.10,58.,2.5)]
    rng = np.random.default_rng(5)
    for z,roll,u in cases:
        d = mujoco.MjData(m)
        d.qpos[2] = z
        q = Rotation.from_euler('xyz',[roll,7,-13],degrees=True).as_quat()
        d.qpos[3:7] = q[[3,0,1,2]]
        d.qvel[:] = rng.normal(0,.15,m.nv)
        d.qvel[0] = u
        for side in ('port','starboard'):
            for joint in ('sweep','elevation','feather'):
                j=m.joint(side+'_'+joint)
                d.qpos[j.qposadr[0]] += rng.uniform(-.12,.12)
        mujoco.mj_forward(m,d)
        _,loads,_ = cpu.force_terms(d)
        out = []
        for name in ('boat','port_oar','starboard_oar'):
            body=m.body(name).id
            f=sum((v['force_N'] for v in loads if v['body']==body),np.zeros(3))
            t=sum((v['torque_Nm']+np.cross(v['point_m']-d.xipos[body],v['force_N'])
                   for v in loads if v['body']==body),np.zeros(3))
            r=d.xmat[body].reshape(3,3)
            out.append(np.r_[r.T@f,r.T@t])
        states.append(d)
        expected.append(out)
    def array(name,dtype,shape=None):
        a=np.stack([getattr(d,name) for d in states]).astype(np.float32)
        if shape:a=a.reshape(shape)
        return wp.array(a,dtype=dtype,device='cpu')
    data=SimpleNamespace(xpos=array('xpos',wp.vec3),xipos=array('xipos',wp.vec3),
        xmat=array('xmat',wp.mat33,(5,m.nbody,3,3)),cvel=array('cvel',wp.spatial_vector),
        subtree_com=array('subtree_com',wp.vec3),xfrc_applied=wp.zeros((5,m.nbody),dtype=wp.spatial_vector,device='cpu'))
    gpu(None,data)
    return np.array(expected),gpu.loads.numpy(),data,gpu


def test_warp_hydro_matches_cpu_including_rotations_and_partial_wetting(hydro_comparison):
    expected,actual,_,_=hydro_comparison
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=.003)


def test_hydro_replaces_loads_without_accumulation(hydro_comparison):
    _,first,data,gpu=hydro_comparison
    gpu(None,data)
    np.testing.assert_array_equal(first,gpu.loads.numpy())
    # Unforced bodies receive no accidental external load.
    force=data.xfrc_applied.numpy()
    other=np.ones(force.shape[1],bool);other[gpu.body_ids]=False
    assert np.count_nonzero(force[:,other])==0


def test_compact_history_exact_across_resets_and_rollout_boundary():
    from tensordict import TensorDict
    from rsl_rl.storage import RolloutStorage
    from rowing_mjlab.storage import HistoryStorage
    n,steps,d=3,7,5
    history=torch.randn(n,126,d)
    obs=TensorDict({'actor':history.clone(),'privileged':torch.randn(n,27)},batch_size=[n])
    storage=HistoryStorage(n,steps,obs,[2],'cpu')
    expected=[]
    for rollout in range(2):
        expected.clear()
        for t in range(steps):
            obs=TensorDict({'actor':history.clone(),'privileged':torch.randn(n,27)},batch_size=[n])
            expected.append(history.clone())
            tr=RolloutStorage.Transition()
            tr.observations=obs;tr.actions=torch.zeros(n,2);tr.values=torch.zeros(n,1)
            tr.rewards=torch.zeros(n);tr.dones=torch.tensor([t==2,t==0,t==5])
            tr.actions_log_prob=torch.zeros(n);tr.distribution_params=(torch.zeros(n,2),torch.ones(n,2))
            storage.add_transition(tr)
            new=torch.randn(n,d)
            history=torch.cat((history[:,1:],new[:,None]),dim=1)
            history[tr.dones]=new[tr.dones,None].expand(-1,126,-1)
        actual=storage.reconstruct(torch.arange(n*steps))['actor']
        torch.testing.assert_close(actual,torch.stack(expected).flatten(0,1),rtol=0,atol=0)
        assert storage.timeline.numel()<steps*n*126*d/4
        storage.clear()


def test_actor_privilege_isolation_and_independent_critic():
    from tensordict import TensorDict
    from rowing_mjlab.networks import TemporalModel
    torch.set_num_threads(2)
    obs=TensorDict({'actor':torch.randn(2,126,107),'privileged':torch.randn(2,27)},batch_size=[2])
    groups={'actor':['actor'],'critic':['actor','privileged']}
    actor=TemporalModel(obs,groups,'actor',29,hidden_dims=[32],activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.3,'std_type':'scalar'})
    critic=TemporalModel(obs,groups,'critic',1,hidden_dims=[32],activation='elu',obs_normalization=True)
    changed=obs.clone();changed['privileged']+=100
    torch.testing.assert_close(actor(obs),actor(changed),rtol=0,atol=0)
    assert not torch.allclose(critic(obs),critic(changed))
    assert actor.encoder.context[-1].out_features==16
    assert not ({id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()})
    actor(obs).sum().backward()
    assert actor.encoder.input.weight.grad is not None
    exported=actor.as_jit().eval()
    torch.testing.assert_close(exported(obs['actor']),actor(obs))


def test_mjlab_reset_head_constraints_termination_and_privileged_layout():
    from mjlab.envs import ManagerBasedRlEnv
    from rowing_mjlab.config import env_cfg
    from rowing_mjlab import mdp
    env=ManagerBasedRlEnv(env_cfg(2,play=True),device='cpu')
    try:
        obs,_=env.reset()
        assert obs['actor'].shape==(2,126,107)
        assert obs['privileged'].shape==(2,27)
        m=env.sim.mj_model;s=env.rowing
        assert m.nu==31 and m.neq==4
        assert 'joint_velocity' not in env.cfg.rewards
        # Compact hands must not start inside oversized oar collision proxies.
        initial=mujoco.MjData(m)
        mujoco.mj_forward(m,initial)
        assert initial.ncon==0
        assert m.joint('rowing/boat_free').type==mujoco.mjtJoint.mjJNT_FREE
        assert m.joint('rowing/seat_pelvis_hinge').type==mujoco.mjtJoint.mjJNT_HINGE
        torch.testing.assert_close(env.sim.data.qpos[:],s.q0.expand(2,-1))
        torch.testing.assert_close(env.sim.data.qvel[:],torch.zeros_like(env.sim.data.qvel[:]))
        for _ in range(3):
            obs,reward,done,timeout,_=env.step(torch.zeros(2,29))
            assert torch.isfinite(reward).all() and not done.any()
        a=env.action_manager.get_term('h2')
        a.process_actions(torch.ones(2,29))
        head=[s.h2_names.index(n) for n in ('head_pitch_joint','head_yaw_joint')]
        torch.testing.assert_close(a.targets[:,head],s.q0[s.h2_q[head]].expand(2,-1))
        # Current world position is not an input to either observation function.
        before=mdp.privileged(env).clone()
        env.sim.data.qpos[:,0]+=100
        torch.testing.assert_close(mdp.privileged(env),before)
        env.sim.data.qpos[0,3:7]=torch.tensor([np.cos(np.deg2rad(61)/2),np.sin(np.deg2rad(61)/2),0,0])
        assert mdp.capsized(env).tolist()==[True,False]
        env.episode_length_buf[:]=1499
        env.step(torch.zeros(2,29))
        assert env.reset_time_outs.all()
        obs,_=env.reset(env_ids=torch.tensor([0]))
        torch.testing.assert_close(obs['actor'][0],obs['actor'][0,-1:].expand(126,-1))
    finally:
        env.close()


def test_compact_hand_pose_closure_axes_and_grip_placement():
    import json
    baseline=json.loads((ROOT/'designs/compact_hand/pose_baseline.json').read_text())
    for filename in ('scene.xml','preview.xml'):
        m=mujoco.MjModel.from_xml_path(str(ROOT/filename));d=mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        assert np.max(abs(d.efc_pos[d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY]))<1e-8
        locations=[]
        for h,o,sign in [('left','starboard',1),('right','port',-1)]:
            bid=m.body(h+'_dex1_base_link').id;oid=m.body(o+'_oar').id
            rh=d.xmat[bid].reshape(3,3);ro=d.xmat[oid].reshape(3,3)
            center=d.xpos[bid]+rh@np.array([.029,0,0])
            local=ro.T@(center-d.xpos[oid])
            np.testing.assert_allclose(local[[0,2]],0,atol=1e-9)
            assert abs(rh[:,2]@ro[:,1])>1-1e-10
            t=sign*local[1];locations.append(t)
            assert t-.035>.74425 and t+.035<.86216
            axis=d.xaxis[m.joint(h+'_elbow_joint').id]
            assert axis@baseline['elbow_axes_world'][h]>np.cos(np.deg2rad(5))
            assert .23<m.body_mass[bid]<.25
            assert mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,h+'_dex1_finger_link_1')==-1
        np.testing.assert_allclose(locations[0],locations[1],atol=1e-9)
