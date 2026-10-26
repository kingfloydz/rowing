"""Independent mechanics, wetting, passivity and entry-impulse checks."""
from types import SimpleNamespace

import mujoco
import mujoco_warp as mw
import numpy as np
import pytest
import torch
import warp as wp
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

from hydrodynamics.blade_inertia import load_parameters, make_strips
from rowing_mjlab.blade_inertia import BladeInertia


def model_and_geometry():
    m = mujoco.MjModel.from_xml_string('''<mujoco>
      <option gravity="1 -2 -9.81" integrator="implicitfast" timestep=".002"/>
      <worldbody><body name="boat"><freejoint/>
        <geom type="sphere" size=".15" mass="17" contype="0" conaffinity="0"/>
        <body name="port" pos="0 .6 0"><joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/>
          <geom type="box" pos=".02 .1 0" size=".02 .3 .1" mass="1.5" contype="0" conaffinity="0"/>
        </body>
        <body name="starboard" pos="0 -.6 0"><joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/>
          <geom type="box" pos=".02 -.1 0" size=".02 .3 .1" mass="1.5" contype="0" conaffinity="0"/>
        </body>
      </body></worldbody></mujoco>''')
    p = load_parameters()
    corners = np.array([[0, -.24, -.1], [0, .24, -.1], [0, .24, .1], [0, -.24, .1]])
    triangles = corners[[[0, 1, 2], [0, 2, 3]]]
    return m, p, make_strips([triangles, triangles], p)


class Evaluator:
    def __init__(self, n=1, device='cpu'):
        self.m, self.p, geometry = model_and_geometry()
        self.n, self.device = n, device
        with wp.ScopedDevice(device):
            self.wm = mw.put_model(self.m)
            self.wd = mw.make_data(self.m, nworld=n, nconmax=1, njmax=1)
        self.fluid = BladeInertia(self.m, n, device, [2, 3], parameters=self.p, geometry=geometry)
        self.water = wp.array(np.tile([0, 1000, 0, 0, 0], (n, 1)), dtype=float, device=device)
        self.loads = wp.zeros((n, 3, 6), dtype=float, device=device)
        def callback(m, d):
            d.xfrc_applied.zero_(); self.loads.zero_()
            self.fluid(m, d, self.water, self.loads)
        self.wm.callback.control = callback

    def evaluate(self, qpos, qvel, level=0., current=(0., 0., 0.)):
        wp.to_torch(self.wd.qpos)[:] = torch.as_tensor(np.asarray(qpos), dtype=torch.float32, device=self.device)
        wp.to_torch(self.wd.qvel)[:] = torch.as_tensor(np.asarray(qvel), dtype=torch.float32, device=self.device)
        wp.to_torch(self.water)[:, 0] = float(level)
        wp.to_torch(self.water)[:, 2:] = torch.tensor(current, device=self.device)
        mw.forward(self.wm, self.wd)
        return self.fluid.diagnostics.numpy().copy()

    def matrix(self):
        result = np.zeros((self.n, self.m.nv, self.m.nv))
        raw = self.wd.M.numpy()
        adr, nnz = self.wm.M_rowadr.numpy(), self.wm.M_rownnz.numpy()
        for row in range(self.m.nv):
            col, slot = row, adr[row]+nnz[row]-1
            while col >= 0:
                result[:, row, col] = result[:, col, row] = raw[:, slot]
                col = self.m.dof_parentid[col]; slot -= 1
        return result


def jacobian(m, d, body):
    jv, jw = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jv, jw, d.xpos[body], body)
    return jv, jw


def test_full_wet_mass_bias_and_native_acceleration_against_spatial_reference():
    e = Evaluator(3)
    m, fluid = e.m, e.fluid
    rng = np.random.default_rng(20)
    qs, vs, expected_m, expected_bias, expected_acc = [], [], [], [], []
    current = np.array([.3, -.2, .1])
    for _ in range(e.n):
        d = mujoco.MjData(m)
        quat = Rotation.random(random_state=rng).as_quat()
        d.qpos[3:7] = quat[[3, 0, 1, 2]]
        d.qpos[7:] = rng.normal(0, .2, 6)
        d.qvel[:] = rng.normal(0, .7, m.nv)
        mujoco.mj_forward(m, d)
        mass = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, mass)
        bias = d.qfrc_bias.copy()
        plus, minus = mujoco.MjData(m), mujoco.MjData(m)
        for target, dt in ((plus, 1e-6), (minus, -1e-6)):
            target.qpos[:] = d.qpos
            mujoco.mj_integratePos(m, target.qpos, d.qvel, dt)
            mujoco.mj_forward(m, target)
        for side, body in enumerate([2, 3]):
            spatial_a = np.zeros((6, 6))
            for strip in range(side*fluid.ns, (side+1)*fluid.ns):
                for k in range(4):
                    point = .5*(fluid.geometry.endpoints[strip, k]+fluid.geometry.endpoints[strip, k+1])
                    b = np.r_[[1., 0., 0.], np.cross(point, [1., 0., 0.])]
                    spatial_a += fluid.geometry.full_mass[strip]/4*np.outer(b, b)
            rot = d.xmat[body].reshape(3, 3)
            jv, jw = jacobian(m, d, body)
            jac = np.vstack((rot.T@jv, rot.T@jw))
            mass += jac.T@spatial_a@jac
            vp, wp_ = jacobian(m, plus, body)
            vm, wm_ = jacobian(m, minus, body)
            a0 = rot.T@((vp-vm)@d.qvel/2e-6)
            alpha0 = rot.T@((wp_-wm_)@d.qvel/2e-6)
            v = rot.T@(jv@d.qvel-current)
            om = rot.T@jw@d.qvel
            momentum = spatial_a@np.r_[v, om]
            wrench = spatial_a@np.r_[a0-np.cross(om, v), alpha0]
            wrench += np.r_[np.cross(om, momentum[:3]),
                            np.cross(v, momentum[:3])+np.cross(om, momentum[3:])]
            bias += jac.T@wrench
        qs.append(d.qpos.copy()); vs.append(d.qvel.copy())
        expected_m.append(mass); expected_bias.append(bias)
        expected_acc.append(np.linalg.solve(mass, -bias))
    diag = e.evaluate(qs, vs, level=10., current=current)
    np.testing.assert_allclose(diag[:, :, 18], 8., atol=1e-5)
    np.testing.assert_array_equal(diag[:, :, 19], 0)
    np.testing.assert_array_equal(diag[:, :, 12:18], 0)
    np.testing.assert_allclose(e.matrix(), expected_m, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(e.wd.qfrc_bias.numpy(), expected_bias, atol=8e-5, rtol=3e-5)
    np.testing.assert_allclose(e.wd.qacc.numpy(), expected_acc, atol=1e-3, rtol=2e-4)
    assert min(np.linalg.eigvalsh(e.matrix()).flatten()) > 0


def test_partial_wetting_mass_rate_and_energy_balance_under_rotation():
    e = Evaluator(4)
    m = e.m
    rng = np.random.default_rng(51)
    qs = np.tile(m.qpos0, (4, 1))
    for i in range(4):
        quat = Rotation.from_euler('xyz', rng.normal(0, .08, 3)).as_quat()
        qs[i, 3:7] = quat[[3, 0, 1, 2]]
    vs = rng.normal(0, .3, (4, m.nv)); vs[:, 2] = [.3, -.3, .5, -.5]
    diag = e.evaluate(qs, vs)
    hydro_generalized = e.wd.qfrc_smooth.numpy().copy()
    for i in range(4):
        d = mujoco.MjData(m); d.qpos[:] = qs[i]; d.qvel[:] = vs[i]
        mujoco.mj_forward(m, d)
        hydro_generalized[i] += d.qfrc_bias
    h = 2e-4
    plus, minus = qs.copy(), qs.copy()
    for i in range(4):
        mujoco.mj_integratePos(m, plus[i], vs[i], h)
        mujoco.mj_integratePos(m, minus[i], vs[i], -h)
    dp = e.evaluate(plus, vs)
    dn = e.evaluate(minus, vs)
    np.testing.assert_allclose((dp[:, :, 18]-dn[:, :, 18])/(2*h), diag[:, :, 19], atol=.003, rtol=.002)
    energy_dot = (dp[:, :, 20].sum(1)-dn[:, :, 20].sum(1))/(2*h)
    power = (hydro_generalized*vs).sum(1)
    loss = diag[:, :, 22].sum(1)
    np.testing.assert_allclose(power+energy_dot, -loss, atol=.006, rtol=.003)
    assert np.all(power+energy_dot < .006)
    assert np.all(diag[:, :, 21] <= 0)
    # Independent high-accuracy quadrature for a stationary vertical half-wet plate.
    q = np.tile(m.qpos0, (4, 1)); v = np.zeros_like(vs)
    d0 = e.evaluate(q, v)
    eps = e.p['wetting_depth_m']
    def smooth(z):
        u = np.clip(-z/eps, 0., 1.)
        return u*u*(3-2*u)
    fraction = quad(smooth, -.1, .1, points=[-eps, 0], epsabs=1e-12)[0]/.2
    np.testing.assert_allclose(d0[:, :, 18], 8*fraction**2, atol=3e-6)


def test_dry_zero_selective_reset_and_same_time_idempotency():
    e = Evaluator(2)
    q = np.tile(e.m.qpos0, (2, 1))
    v = np.ones((2, e.m.nv))*.2
    dry = e.evaluate(q, v, level=-10)
    np.testing.assert_array_equal(dry[:, :, :23], 0)
    wet = e.evaluate(q, v)
    mass = e.wd.M.numpy().copy(); force = e.wd.xfrc_applied.numpy().copy()
    np.testing.assert_array_equal(e.evaluate(q, v), wet)
    np.testing.assert_array_equal(e.wd.M.numpy(), mass)
    np.testing.assert_array_equal(e.wd.xfrc_applied.numpy(), force)
    e.fluid.reset(torch.tensor([1]))
    np.testing.assert_array_equal(e.fluid.diagnostics.numpy()[0], wet[0])
    np.testing.assert_array_equal(e.fluid.diagnostics.numpy()[1], 0)
    np.testing.assert_array_equal(e.evaluate(q, v), wet)
    e.fluid.set_mass_scale(0., env_ids=torch.tensor([1]))
    after = e.evaluate(q, v)
    np.testing.assert_array_equal(after[0], wet[0])
    np.testing.assert_array_equal(after[1, :, :23], 0)


def test_flat_entry_impulse_and_exit_no_recruitment_force():
    # Horizontal blade, prescribed normal entry speed: impulse = added mass * V.
    # Evaluate the two kernels on CPU kinematics to avoid integrating the prescribed motion.
    e = Evaluator()
    q = e.m.qpos0.copy()
    q[3:7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
    v = np.zeros(e.m.nv); v[2] = -1.
    d = mujoco.MjData(e.m)
    arrays = {'xpos':wp.vec3, 'xipos':wp.vec3, 'xmat':wp.mat33, 'subtree_com':wp.vec3,
              'cvel':wp.spatial_vector, 'cdof':wp.spatial_vector, 'cdof_dot':wp.spatial_vector}
    data = SimpleNamespace(M=wp.zeros_like(e.wd.M), qfrc_bias=wp.zeros_like(e.wd.qfrc_bias),
                           xfrc_applied=wp.zeros_like(e.wd.xfrc_applied), qvel=wp.zeros_like(e.wd.qvel))
    for name, dtype in arrays.items():
        setattr(data, name, wp.zeros_like(getattr(e.wd, name)))
    def sample(z, vz):
        d.qpos[:] = q; d.qpos[2] = z
        d.qvel[:] = v; d.qvel[2] = vz
        mujoco.mj_forward(e.m, d)
        for name in arrays:
            src = getattr(d, name)
            if name == 'xmat': src = src.reshape(-1, 3, 3)
            wp.to_torch(getattr(data, name))[:] = torch.tensor(src, dtype=torch.float32)
        wp.to_torch(data.qvel)[:] = torch.tensor(d.qvel, dtype=torch.float32)
        data.M.zero_(); data.qfrc_bias.zero_(); data.xfrc_applied.zero_(); e.loads.zero_()
        e.fluid(e.wm, data, e.water, e.loads)
        return e.fluid.diagnostics.numpy()[0, 0].copy()
    depths = np.linspace(-.001, .011, 241)
    history = np.array([sample(-depth, -1.) for depth in depths])
    impulse = np.trapezoid(history[:, 14], depths)  # Fz, dt=ddepth / 1 m/s
    np.testing.assert_allclose(impulse, 8., rtol=3e-4)
    assert history[:, 21].max() <= 0
    for depth in [.001, .004, .008]:
        diag = sample(-depth, 1.)
        assert diag[19] < 0
        np.testing.assert_allclose(diag[12:18], 0., atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
def test_cuda_capture_blade_inertia():
    e = Evaluator(32, 'cuda:0')
    q = np.tile(e.m.qpos0, (32, 1)); v = np.zeros((32, e.m.nv)); v[:, 0] = 1.; v[:, 2] = -.2
    e.evaluate(q, v)
    expected = e.fluid.diagnostics.numpy().copy()
    with wp.ScopedDevice('cuda:0'), wp.ScopedCapture() as capture:
        mw.forward(e.wm, e.wd)
    wp.capture_launch(capture.graph)
    wp.synchronize()
    np.testing.assert_allclose(e.fluid.diagnostics.numpy(), expected, atol=1e-6)
    assert np.isfinite(e.wd.qacc.numpy()).all()


@pytest.mark.parametrize('enabled', [True, False])
def test_mjlab_random_actions_and_blade_toggle(monkeypatch, enabled):
    import sys
    from types import ModuleType
    # Test-process-only official TensorBoard fallback; avoid unrelated old TF ABI.
    monkeypatch.setitem(sys.modules, 'tensorboard.compat.notf', ModuleType('tensorboard.compat.notf'))
    from mjlab.envs import ManagerBasedRlEnv
    from rowing_mjlab.config import env_cfg
    cfg = env_cfg(2, play=True)
    cfg.actions['h2'].blade_inertia = enabled
    env = ManagerBasedRlEnv(cfg, device='cpu')
    try:
        env.reset()
        fluid = env.rowing.hydro.blade_inertia
        assert (fluid is not None) == enabled
        generator = torch.Generator().manual_seed(314)
        for _ in range(20):
            obs, reward, _, _, _ = env.step(2*torch.rand((2, 29), generator=generator)-1)
            assert torch.isfinite(reward).all()
            assert torch.isfinite(env.sim.data.qpos).all()
            assert torch.isfinite(env.sim.data.qvel).all()
            if enabled:
                diag = fluid.diagnostics.numpy()
                assert np.isfinite(diag).all()
                assert np.all(diag[:, :, 18] >= 0) and np.all(diag[:, :, 18] <= 8.0001)
                assert np.all(diag[:, :, 21] <= 1e-6)
        env.sim.forward()
        if enabled:
            before = fluid.diagnostics.numpy().copy()
        env.reset(env_ids=torch.tensor([0]))
        if enabled:
            np.testing.assert_array_equal(fluid.diagnostics.numpy()[1], before[1])
    finally:
        env.close()
