"""Added inertia solve, passive frequency response, state/reset and batch checks."""
import copy
from types import SimpleNamespace

import mujoco
import mujoco_warp as mw
import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from hydrodynamics.hull_radiation import load_parameters, frequency_response, validate_parameters
from rowing_mjlab.radiation import HullRadiation


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def simple_model():
    return mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody><body name="boat"><freejoint/>
        <geom type="sphere" size=".1" mass="17" contype="0" conaffinity="0"/>
        <body pos=".5 0 .2"><joint axis="0 1 0"/>
          <geom type="box" size=".1 .2 .3" mass="2" contype="0" conaffinity="0"/>
        </body></body></worldbody></mujoco>''')


def test_parameters_are_psd_causal_and_have_no_extra_steady_drag():
    p = load_parameters()
    a, b = frequency_response(p, np.r_[0., np.logspace(-5, 5, 300)])
    assert np.linalg.eigvalsh(a).min() > 0
    assert np.linalg.eigvalsh(b).min() >= -1e-10
    np.testing.assert_array_equal(b[0], np.zeros((6, 6)))
    np.testing.assert_allclose(a[0], np.array(p['added_mass_inf'])+p['radiation_delta_mass'])
    np.testing.assert_allclose(a[-1], p['added_mass_inf'], atol=1e-6)
    _, peak = frequency_response(p, p['pole_rad_s'])
    np.testing.assert_allclose(peak, .5*p['pole_rad_s']*np.array(p['radiation_delta_mass']))
    bad = copy.deepcopy(p)
    bad['radiation_delta_mass'][0][0] = -1
    with pytest.raises(ValueError, match='positive semidefinite'):
        validate_parameters(bad)


def test_native_acceleration_uses_augmented_mass_and_kirchhoff_bias():
    """Independent mj_jac reference, arbitrary attitude, offset and moving child."""
    m = simple_model()
    n = 4
    rng = np.random.default_rng(193)
    with wp.ScopedDevice('cpu'):
        wm = mw.put_model(m)
        wd = mw.make_data(m, nworld=n, nconmax=1, njmax=1)
        fluid = HullRadiation(m, n, 'cpu', 1)
        water = wp.array(np.tile([0, 1000, .2, -.1, .05], (n, 1)), dtype=float, device='cpu')
        loads = wp.zeros((n, 3, 6), dtype=float, device='cpu')
        wm.callback.control = lambda wm, wd: fluid(wm, wd, water, loads)
        expected_m, expected_bias, expected_acc = [], [], []
        qpos, qvel, applied = [], [], []
        ref = np.asarray(fluid.parameters['reference_point_body_m'])
        a = np.asarray(fluid.parameters['added_mass_inf'])
        # Constant transformation to body origin for independent Newton-Euler reference.
        t = np.eye(6)
        t[:3, 3:] = -skew(ref)
        a_origin = t.T@a@t
        for _ in range(n):
            d = mujoco.MjData(m)
            rot = Rotation.random(random_state=rng)
            xyzw = rot.as_quat()
            d.qpos[3:7] = xyzw[[3, 0, 1, 2]]
            d.qpos[7] = rng.normal()
            d.qvel[:] = rng.normal(size=m.nv)
            d.qfrc_applied[:] = rng.normal(size=m.nv)*10
            mujoco.mj_forward(m, d)
            rigid = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, d, rigid)
            jv, jw = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            r = d.xmat[1].reshape(3, 3)
            mujoco.mj_jac(m, d, jv, jw, d.xpos[1]+r@ref, 1)
            jac = np.vstack((r.T@jv, r.T@jw))
            matrix = rigid+jac.T@a@jac
            v = r.T@(d.qvel[:3]-[.2, -.1, .05])
            w = d.qvel[3:6]
            momentum = a_origin@np.r_[v, w]
            conv = np.r_[np.cross(w, momentum[:3]),
                         np.cross(v, momentum[:3])+np.cross(w, momentum[3:])]
            conv -= a_origin@np.r_[np.cross(w, v), np.zeros(3)]
            bias = d.qfrc_bias.copy()
            bias[:6] += np.r_[r@conv[:3], conv[3:]]
            expected_m.append(matrix)
            expected_bias.append(bias)
            expected_acc.append(np.linalg.solve(matrix, d.qfrc_applied-bias))
            qpos.append(d.qpos.copy()); qvel.append(d.qvel.copy()); applied.append(d.qfrc_applied.copy())
        wp.to_torch(wd.qpos)[:] = torch.tensor(np.array(qpos), dtype=torch.float32)
        wp.to_torch(wd.qvel)[:] = torch.tensor(np.array(qvel), dtype=torch.float32)
        wp.to_torch(wd.qfrc_applied)[:] = torch.tensor(np.array(applied), dtype=torch.float32)
        mw.forward(wm, wd)
        adr, nnz = wm.M_rowadr.numpy(), wm.M_rownnz.numpy()
        actual_m = np.zeros((n, m.nv, m.nv))
        for i in range(m.nv):
            cols = []
            c = i
            while c >= 0:
                cols.append(c); c = m.dof_parentid[c]
            for j, c in enumerate(reversed(cols)):
                actual_m[:, i, c] = actual_m[:, c, i] = wd.M.numpy()[:, adr[i]+j]
        np.testing.assert_allclose(actual_m, expected_m, atol=3e-5, rtol=3e-6)
        np.testing.assert_allclose(wd.qfrc_bias.numpy(), expected_bias, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(wd.qacc.numpy(), expected_acc, atol=3e-4, rtol=3e-5)
        original_m, original_state = wd.M.numpy().copy(), fluid.state.numpy().copy()
        mw.forward(wm, wd)
        np.testing.assert_array_equal(wd.M.numpy(), original_m)
        np.testing.assert_array_equal(fluid.state.numpy(), original_state)
        np.testing.assert_array_equal(m.body_mass, [0, 17, 2])


def memory_fixture(n):
    """Isolate memory kernel with stationary coordinates, no mechanics integration."""
    m = simple_model()
    p = load_parameters()
    p['reference_point_body_m'] = [0., 0., 0.]
    fluid = HullRadiation(m, n, 'cpu', 1, p)
    model = SimpleNamespace(M_rowadr=wp.array([0, 1, 3, 6, 10, 15], dtype=int, device='cpu'),
                            M_rownnz=wp.array([1, 2, 3, 4, 5, 6], dtype=int, device='cpu'))
    data = SimpleNamespace(
        xmat=wp.array(np.tile(np.eye(3), (n, 2, 1, 1)), dtype=wp.mat33, device='cpu'),
        xpos=wp.zeros((n, 2), dtype=wp.vec3, device='cpu'),
        xipos=wp.zeros((n, 2), dtype=wp.vec3, device='cpu'),
        qvel=wp.zeros((n, 6), dtype=float, device='cpu'),
        time=wp.zeros(n, dtype=float, device='cpu'),
        M=wp.zeros((n, 21), dtype=float, device='cpu'),
        qfrc_bias=wp.zeros((n, 6), dtype=float, device='cpu'),
        xfrc_applied=wp.zeros((n, 2), dtype=wp.spatial_vector, device='cpu'))
    water=wp.array(np.tile([0, 1000, 0, 0, 0], (n, 1)), dtype=float, device='cpu')
    loads=wp.zeros((n, 3, 6), dtype=float, device='cpu')
    def update(t, velocity):
        wp.to_torch(data.time)[:] = t
        wp.to_torch(data.qvel)[:] = torch.as_tensor(velocity, dtype=torch.float32)
        data.M.zero_(); data.qfrc_bias.zero_(); data.xfrc_applied.zero_(); loads.zero_()
        fluid(model, data, water, loads)
        return fluid.wrench.numpy().copy()
    return fluid, update


def test_time_domain_matches_broadband_frequency_response_and_selective_reset():
    freq = np.array([.1, .5, 1., 3.])
    fluid, update = memory_fixture(4)
    dt = .01
    time = np.arange(2001)*dt
    response = []
    for t in time:
        v = np.zeros((4, 6)); v[:, 2] = np.sin(2*np.pi*freq*t)
        response.append(update(float(t), v)[:, 2])
    response = np.asarray(response)
    # Bilinear discretization warps frequency; compare its exact transfer function.
    for i, f in enumerate(freq):
        t = time[1000:]
        basis = np.column_stack((np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t)))
        fit = np.linalg.lstsq(basis, -response[1000:, i], rcond=None)[0]
        w = 2/dt*np.tan(np.pi*f*dt)
        a, b = frequency_response(fluid.parameters, w)
        target = [b[2, 2], w*(a[2, 2]-fluid.parameters['added_mass_inf'][2][2])]
        np.testing.assert_allclose(fit, target, rtol=2e-4, atol=.003)
    before = fluid.state.numpy().copy()
    update(float(time[-1]), v)
    np.testing.assert_array_equal(before, fluid.state.numpy())
    fluid.reset(torch.tensor([1, 3]))
    np.testing.assert_array_equal(fluid.state.numpy()[[0, 2]], before[[0, 2]])
    np.testing.assert_array_equal(fluid.state.numpy()[[1, 3], :18], 0)
    assert (fluid.state.numpy()[[1, 3], 18] == -1).all()
    update(float(time[-1]), v)
    np.testing.assert_array_equal(fluid.wrench.numpy()[[1, 3]], 0)


def test_random_velocity_memory_is_passive_and_constant_speed_settles_to_zero():
    fluid, update = memory_fixture(1)
    rng = np.random.default_rng(71)
    dt, w0 = .01, fluid.parameters['pole_rad_s']
    dm = np.asarray(fluid.parameters['radiation_delta_mass'])
    previous_v, previous_r = np.zeros(6), np.zeros(6)
    previous_state = np.zeros(12)
    work, loss = 0., 0.
    update(0., previous_v[None])
    for k in range(1, 501):
        v = rng.normal(size=6)
        r = -update(k*dt, v[None])[0]
        state = fluid.state.numpy()[0, :12]
        average_rate = .5*(state[6:]+previous_state[6:])
        work += dt*np.dot(.5*(v+previous_v), .5*(r+previous_r))
        loss += dt*2*w0**3*(average_rate@dm@average_rate)
        previous_v, previous_r, previous_state = v, r, state.copy()
    energy = .5*w0*w0*(state[6:]@dm@state[6:]+w0*w0*(state[:6]@dm@state[:6]))
    assert work > 0 and loss > 0 and energy > 0
    np.testing.assert_allclose(work, energy+loss, rtol=2e-5)
    for k in range(501, 2001):
        update(k*dt, np.ones((1, 6)))
    assert np.max(abs(fluid.wrench.numpy())) < .002


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required for graph validation')
def test_cuda_graph_native_step_with_radiation():
    m = simple_model()
    with wp.ScopedDevice('cuda:0'):
        wm = mw.put_model(m)
        wd = mw.make_data(m, nworld=32, nconmax=1, njmax=1)
        fluid = HullRadiation(m, 32, 'cuda:0', 1)
        water = wp.array(np.tile([0, 1000, 0, 0, 0], (32, 1)), dtype=float)
        loads = wp.zeros((32, 3, 6), dtype=float)
        def callback(model, data):
            data.xfrc_applied.zero_(); loads.zero_()
            fluid(model, data, water, loads)
        wm.callback.control = callback
        wp.to_torch(wd.qvel)[:, 2] = .3
        # Warm up all kernels before capture, then begin a fresh memory interval.
        mw.step(wm, wd)
        fluid.reset()
        with wp.ScopedCapture() as capture:
            mw.step(wm, wd)
        for _ in range(20):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        assert np.isfinite(wd.qacc.numpy()).all()
        assert np.max(abs(fluid.wrench.numpy())) > 0
        before = fluid.state.numpy().copy()
        fluid.reset(torch.tensor([0], device='cuda:0'))
        np.testing.assert_array_equal(fluid.state.numpy()[1:], before[1:])


@pytest.mark.parametrize('enabled', [True, False])
def test_mjlab_radiation_step_and_partial_episode_reset(monkeypatch, enabled):
    # TensorBoard's official no-TensorFlow fallback avoids an unrelated old
    # TensorFlow/NumPy ABI conflict in the local test environment. No files or
    # installed packages are changed; this model does not depend on TensorFlow.
    import sys
    from types import ModuleType
    monkeypatch.setitem(sys.modules, 'tensorboard.compat.notf', ModuleType('tensorboard.compat.notf'))
    from mjlab.envs import ManagerBasedRlEnv
    from rowing_mjlab.config import env_cfg
    cfg = env_cfg(2, play=True)
    cfg.actions['h2'].hull_radiation = enabled
    env = ManagerBasedRlEnv(cfg, device='cpu')
    try:
        env.reset()
        fluid = env.rowing.hydro.radiation
        assert (fluid is not None) == enabled
        if enabled:
            np.testing.assert_array_equal(fluid.state.numpy()[:, :12], 0)
        env.sim.data.qvel[:, 2] = torch.tensor([.1, -.12])
        for _ in range(3):
            obs, reward, done, timeout, _ = env.step(torch.zeros(2, 29))
            assert torch.isfinite(reward).all()
            assert torch.isfinite(env.sim.data.qpos).all()
            assert not done.any()
        if not enabled:
            return
        assert np.max(abs(fluid.wrench.numpy())) > .001
        env.sim.forward()
        before = fluid.state.numpy().copy()
        force_before = env.rowing.hydro.loads.numpy().copy()
        env.sim.forward()
        np.testing.assert_array_equal(fluid.state.numpy(), before)
        np.testing.assert_array_equal(env.rowing.hydro.loads.numpy(), force_before)
        env.reset(env_ids=torch.tensor([1]))
        np.testing.assert_array_equal(fluid.state.numpy()[0], before[0])
        np.testing.assert_array_equal(fluid.state.numpy()[1, :12], 0)
        np.testing.assert_array_equal(fluid.wrench.numpy()[1], 0)
    finally:
        env.close()
