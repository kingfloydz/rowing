"""GPU hull added inertia + passive radiation memory for mjwarp 3.11 forward/step.

One thread per world; no history convolution, CPU transfers or dynamic allocation
in the callback. The root 6x6 inertia is inserted BEFORE native factorization.
Do not use with mjwarp.step1/step2 (factorization occurs at a different point).
"""
import numpy as np
import mujoco
import warp as wp

from hydrodynamics.hull_radiation import load_parameters, validate_parameters


@wp.kernel(enable_backward=False)
def hull_inertia_radiation(
    body: int, reference: wp.vec3, rho_ref: float, pole: float,
    ainf: wp.array(dtype=wp.spatial_matrix), delta: wp.array(dtype=wp.spatial_matrix),
    scales: wp.array2d(dtype=float), water: wp.array2d(dtype=float),
    rowadr: wp.array(dtype=int), rownnz: wp.array(dtype=int),
    xmat: wp.array2d(dtype=wp.mat33), xpos: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3), qvel: wp.array2d(dtype=float),
    time: wp.array(dtype=float), mass: wp.array2d(dtype=float),
    bias: wp.array2d(dtype=float), state: wp.array2d(dtype=float),
    radiation: wp.array2d(dtype=float), loads: wp.array3d(dtype=float),
    xfrc: wp.array2d(dtype=wp.spatial_vector),
):
    world = wp.tid()
    rot = xmat[world, body]
    rt = wp.transpose(rot)
    density = water[world, 1]/rho_ref
    am = ainf[0]*(density*scales[world, 0])
    dm = delta[0]*(density*scales[world, 1])
    w0 = pole*scales[world, 2]
    # Free-joint qvel: world linear velocity of body origin, BODY angular velocity.
    j = wp.spatial_matrix(0.0)
    skew_ref = wp.skew(reference)
    for r in range(3):
        for c in range(3):
            j[r, c] = rot[c, r]
            j[r, c+3] = -skew_ref[r, c]
        j[r+3, r+3] = 1.0
    current = wp.vec3(water[world, 2], water[world, 3], water[world, 4])
    v_origin = rt*(wp.vec3(qvel[world, 0], qvel[world, 1], qvel[world, 2])-current)
    omega = wp.vec3(qvel[world, 3], qvel[world, 4], qvel[world, 5])
    v = v_origin+wp.cross(omega, reference)
    nu = wp.spatial_vector(v[0], v[1], v[2], omega[0], omega[1], omega[2])

    added = wp.transpose(j)*am*j
    for r in range(6):
        for c in range(r+1):
            # MuJoCo Warp 3.11 CSR M, lower triangle, root ancestors consecutive.
            adr = rowadr[r]+rownnz[r]-1-r+c
            mass[world, adr] = mass[world, adr]+added[r, c]

    # Body-frame Kirchhoff terms plus dJ/dt*qvel. No gravity or rigid mass edit.
    momentum = am*nu
    p = wp.vec3(momentum[0], momentum[1], momentum[2])
    h = wp.vec3(momentum[3], momentum[4], momentum[5])
    f_bias = wp.cross(omega, p)
    t_bias = wp.cross(v, p)+wp.cross(omega, h)
    dv = -wp.cross(omega, v_origin)
    frame_bias = am*wp.spatial_vector(dv[0], dv[1], dv[2], 0.0, 0.0, 0.0)
    b = wp.transpose(j)*(frame_bias+wp.spatial_vector(
        f_bias[0], f_bias[1], f_bias[2], t_bias[0], t_bias[1], t_bias[2]))
    for r in range(6):
        bias[world, r] = bias[world, r]+b[r]

    # state = [z(6), zdot(6), previous velocity(6), previous time].
    # Trapezoidal discretization is A-stable and preserves positive realness.
    # Advance only when physical time changes: repeated forward calls and graph
    # capture must not invent elapsed time or repeatedly advance the memory.
    now = time[world]
    previous_time = state[world, 18]
    dt = now-previous_time
    rate = wp.spatial_vector()
    for k in range(6):
        z = state[world, k]
        zd = state[world, k+6]
        if previous_time < 0.0 or dt < 0.0:
            z = 0.0
            zd = 0.0
        elif dt > 0.0:
            half = 0.5*dt
            wh = w0*half
            den = (1.0+wh)*(1.0+wh)
            vel_sum = state[world, k+12]+nu[k]
            zn = ((1.0+2.0*wh-wh*wh)*z+2.0*half*zd+half*half*vel_sum)/den
            zd = (-2.0*w0*w0*half*z+(1.0-2.0*wh-wh*wh)*zd+half*vel_sum)/den
            z = zn
        state[world, k] = z
        state[world, k+6] = zd
        state[world, k+12] = nu[k]
        rate[k] = zd
    state[world, 18] = now
    wrench = -(dm*rate)*(w0*w0)
    for k in range(6):
        radiation[world, k] = wrench[k]
    fw = rot*wp.vec3(wrench[0], wrench[1], wrench[2])
    tw = rot*wp.vec3(wrench[3], wrench[4], wrench[5])
    tw += wp.cross(xpos[world, body]+rot*reference-xipos[world, body], fw)
    xfrc[world, body] = xfrc[world, body]+wp.spatial_vector(fw[0], fw[1], fw[2], tw[0], tw[1], tw[2])
    local_t = rt*tw
    for k in range(3):
        loads[world, 0, k] = loads[world, 0, k]+wrench[k]
        loads[world, 0, k+3] = loads[world, 0, k+3]+local_t[k]


class HullRadiation:
    """Fixed nominal-immersion prior; coefficients can be replaced after BEM.

    Use only as the LAST stage of WarpHydrodynamics' control callback. ``M`` and
    qfrc_bias must have been rebuilt by a full forward pass. The source model's
    first joint must be the boat's free joint. Requires mjwarp 3.11 CSR M layout.
    """
    def __init__(self, model, nworld, device, body, parameters=None):
        p = load_parameters() if parameters is None else validate_parameters(parameters)
        if p.get('damping_ratio') != 1.0:
            raise ValueError('This realization requires critical damping (zeta=1)')
        if (model.njnt < 1 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
                or model.jnt_bodyid[0] != body or model.jnt_dofadr[0] != 0
                or not np.array_equal(model.dof_parentid[:6], np.arange(-1, 5))):
            raise ValueError('Hull must own the first free joint and its first six DOFs')
        if model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER,
                                       mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
            raise ValueError('Hull radiation supports Euler or implicitfast, not RK4')
        if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION:
            raise ValueError('Hydrodynamics requires the control callback to be enabled')
        self.parameters = p
        self.nworld, self.device, self.body = nworld, device, body
        self.reference = wp.vec3(*p['reference_point_body_m'])
        self.ainf = wp.array(np.asarray(p['added_mass_inf'], np.float32)[None],
                             dtype=wp.spatial_matrix, device=device)
        self.delta = wp.array(np.asarray(p['radiation_delta_mass'], np.float32)[None],
                              dtype=wp.spatial_matrix, device=device)
        self.state = wp.zeros((nworld, 19), dtype=float, device=device)
        self.wrench = wp.zeros((nworld, 6), dtype=float, device=device)
        self.scales = wp.ones((nworld, 3), dtype=float, device=device)
        # Torch aliases only for reset/domain randomization, never in callback.
        self._state = wp.to_torch(self.state)
        self._wrench = wp.to_torch(self.wrench)
        self._scales = wp.to_torch(self.scales)
        self.reset()

    def reset(self, env_ids=None):
        """Selective asynchronous environment reset, including same-time resets."""
        ids = slice(None) if env_ids is None else env_ids
        self._state[ids] = 0
        self._state[ids, 18] = -1
        self._wrench[ids] = 0

    def set_scales(self, added_mass=1.0, radiation=1.0, pole=1.0, env_ids=None):
        """Set positive scalar multipliers BETWEEN episodes; resets their memory.

        Separate mass/damping/pole sensitivity tests retain continuous passivity.
        Do not independently perturb individual matrix entries (can destroy PSD).
        """
        values = np.asarray([added_mass, radiation, pole], dtype=float)
        if not np.isfinite(values).all() or np.any(values[:2] < 0) or values[2] <= 0:
            raise ValueError('Mass/radiation scales must be >=0, pole scale >0')
        ids = slice(None) if env_ids is None else env_ids
        for k, value in enumerate(values):
            self._scales[ids, k] = float(value)
        self.reset(env_ids)

    def __call__(self, model, data, water, loads):
        if not hasattr(model, 'M_rowadr') or not hasattr(data, 'M'):
            raise RuntimeError('Hull added inertia requires MuJoCo Warp 3.11 CSR mass matrix')
        wp.launch(hull_inertia_radiation, self.nworld, inputs=[
            self.body, self.reference, self.parameters['rho_reference_kg_m3'],
            self.parameters['pole_rad_s'], self.ainf, self.delta, self.scales, water,
            model.M_rowadr, model.M_rownnz, data.xmat, data.xpos, data.xipos,
            data.qvel, data.time, data.M, data.qfrc_bias, self.state, self.wrench,
            loads, data.xfrc_applied], device=self.device)
