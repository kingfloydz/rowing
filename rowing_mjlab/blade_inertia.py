"""Batched variable blade inertia and passive entry/exit closure, mjwarp 3.11.

Two kernels: independent strips, then one writer per world for both blades.
No temporal state, acceleration differencing, atomics, or online CPU geometry.
"""
import numpy as np
import mujoco
import warp as wp
from pathlib import Path

from hydrodynamics.blade_inertia import load_parameters, repository_strips, validate_parameters
from hydrodynamics.models import ROOT

Vec4 = wp.types.vector(length=4, dtype=wp.float32)
Mat39 = wp.types.matrix(shape=(3, 9), dtype=wp.float32)


@wp.func
def wet_step(u: float):
    x = wp.clamp(u, 0.0, 1.0)
    return x*x*(3.0-2.0*x)


@wp.func
def wet_integral(u: float):
    result = float(0.0)
    if u >= 1.0:
        result = u-0.5
    elif u > 0.0:
        result = u*u*u*(1.0-0.5*u)
    return result


@wp.func
def cell_wet(a: float, b: float, adot: float, bdot: float):
    # Depths normalized by smoothing depth. Integrate wet_step along the cell.
    fraction = float(0.0)
    rate = float(0.0)
    if a >= 1.0 and b >= 1.0:
        fraction = 1.0
    elif a > 0.0 or b > 0.0:
        diff = b-a
        if wp.abs(diff) < 1.0e-3:
            mid = wp.clamp(0.5*(a+b), 0.0, 1.0)
            fraction = wet_step(mid)
            rate = 6.0*mid*(1.0-mid)*0.5*(adot+bdot)
        else:
            fraction = (wet_integral(b)-wet_integral(a))/diff
            rate = (wet_step(b)*bdot-wet_step(a)*adot-fraction*(bdot-adot))/diff
        fraction = wp.clamp(fraction, 0.0, 1.0)
    return wp.vec2(fraction, rate)


@wp.kernel(enable_backward=False)
def integrate_blade_strips(
    ns: int, depth: float, rho_ref: float,
    endpoints: wp.array2d(dtype=wp.vec3), full_mass: wp.array(dtype=float),
    bodies: wp.array(dtype=int), roots: wp.array(dtype=int),
    dofs: wp.array2d(dtype=int), counts: wp.array(dtype=int),
    scales: wp.array2d(dtype=float), water: wp.array2d(dtype=float),
    xpos: wp.array2d(dtype=wp.vec3), xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33), subtree_com: wp.array2d(dtype=wp.vec3),
    cvel: wp.array2d(dtype=wp.spatial_vector), cdof_dot: wp.array2d(dtype=wp.spatial_vector),
    qvel: wp.array2d(dtype=float), chunks: wp.array3d(dtype=float),
):
    world, strip = wp.tid()
    side = strip//ns
    body = bodies[side]
    rot = xmat[world, body]
    origin = xpos[world, body]
    com = xipos[world, body]
    root_com = subtree_com[world, roots[side]]
    cv = cvel[world, body]
    omega = wp.vec3(cv[0], cv[1], cv[2])
    vroot = wp.vec3(cv[3], cv[4], cv[5])
    current = wp.vec3(water[world, 2], water[world, 3], water[world, 4])
    normal = rot*wp.vec3(1.0, 0.0, 0.0)
    ndot = wp.cross(omega, normal)
    # RNE spatial bias acceleration, WITHOUT fictitious gravity acceleration.
    ca = wp.spatial_vector()
    for k in range(counts[side]):
        d = dofs[side, k]
        ca += cdof_dot[world, d]*qvel[world, d]
    alpha = wp.vec3(ca[0], ca[1], ca[2])
    alin = wp.vec3(ca[3], ca[4], ca[5])
    wet = Vec4(0.0)
    wet_rate = Vec4(0.0)
    fraction = float(0.0)
    fraction_rate = float(0.0)
    max_vz = float(0.0)
    for k in range(4):
        pa = origin+rot*endpoints[strip, k]
        pb = origin+rot*endpoints[strip, k+1]
        va = vroot+wp.cross(omega, pa-root_com)
        vb = vroot+wp.cross(omega, pb-root_com)
        max_vz = wp.max(max_vz, wp.max(wp.abs(va[2]), wp.abs(vb[2])))
        cell = cell_wet((water[world, 0]-pa[2])/depth, (water[world, 0]-pb[2])/depth,
                        -va[2]/depth, -vb[2]/depth)
        wet[k] = cell[0]
        wet_rate[k] = cell[1]
        fraction += 0.25*cell[0]
        fraction_rate += 0.25*cell[1]
    nominal = full_mass[strip]*water[world, 1]/rho_ref*scales[world, side]
    inertia = wp.mat33(0.0)  # active local twist components: vx, omega_y, omega_z
    bias_f = wp.vec3(0.0)
    bias_t = wp.vec3(0.0)
    entry_f = wp.vec3(0.0)
    entry_t = wp.vec3(0.0)
    energy = float(0.0)
    loss = float(0.0)
    entry_power = float(0.0)
    for k in range(4):
        r = 0.5*(endpoints[strip, k]+endpoints[strip, k+1])
        point = origin+rot*r
        absolute_v = vroot+wp.cross(omega, point-root_com)
        v = absolute_v-current
        vn = wp.dot(v, normal)
        m = 0.25*nominal*wet[k]*fraction
        mdot = 0.25*nominal*(wet_rate[k]*fraction+wet[k]*fraction_rate)
        b = wp.vec3(1.0, r[2], -r[1])
        inertia += m*wp.outer(b, b)
        # Physical point acceleration at qacc=0 from spatial bias acceleration.
        a0 = alin+wp.cross(alpha, point-root_com)+wp.cross(omega, absolute_v)
        # Positive bias wrench; its negative is the fixed-mass fluid reaction.
        bf = m*(normal*(wp.dot(normal, a0)+wp.dot(ndot, v))+vn*ndot)
        bt = -m*vn*wp.cross(normal, v)+wp.cross(point-com, bf)
        ef = -wp.max(mdot, 0.0)*vn*normal
        bias_f += bf
        bias_t += bt
        entry_f += ef
        entry_t += wp.cross(point-com, ef)
        energy += 0.5*m*vn*vn
        loss += 0.5*wp.abs(mdot)*vn*vn
        entry_power -= wp.max(mdot, 0.0)*vn*vn
    chunks[world, strip, 0] = inertia[0, 0]
    chunks[world, strip, 1] = inertia[0, 1]
    chunks[world, strip, 2] = inertia[0, 2]
    chunks[world, strip, 3] = inertia[1, 1]
    chunks[world, strip, 4] = inertia[1, 2]
    chunks[world, strip, 5] = inertia[2, 2]
    for k in range(3):
        chunks[world, strip, 6+k] = bias_f[k]
        chunks[world, strip, 9+k] = bias_t[k]
        chunks[world, strip, 12+k] = entry_f[k]
        chunks[world, strip, 15+k] = entry_t[k]
    chunks[world, strip, 18] = nominal*fraction*fraction
    chunks[world, strip, 19] = 2.0*nominal*fraction*fraction_rate
    chunks[world, strip, 20] = energy
    chunks[world, strip, 21] = entry_power
    chunks[world, strip, 22] = loss
    chunks[world, strip, 23] = max_vz


@wp.kernel(enable_backward=False)
def assemble_blade_inertia(
    ns: int, bodies: wp.array(dtype=int), roots: wp.array(dtype=int),
    dofs: wp.array2d(dtype=int), counts: wp.array(dtype=int),
    chunks: wp.array3d(dtype=float), diagnostics: wp.array3d(dtype=float),
    rowadr: wp.array(dtype=int), rownnz: wp.array(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3), xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33), subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector), mass: wp.array2d(dtype=float),
    bias: wp.array2d(dtype=float), xfrc: wp.array2d(dtype=wp.spatial_vector),
    loads: wp.array3d(dtype=float),
):
    world = wp.tid()  # both sides share root mass entries; one writer avoids races
    for side in range(2):
        for k in range(24):
            value = float(0.0)
            for s in range(ns):
                if k == 23:
                    value = wp.max(value, chunks[world, side*ns+s, k])
                else:
                    value += chunks[world, side*ns+s, k]
            diagnostics[world, side, k] = value
        a = wp.mat33(diagnostics[world, side, 0], diagnostics[world, side, 1], diagnostics[world, side, 2],
                     diagnostics[world, side, 1], diagnostics[world, side, 3], diagnostics[world, side, 4],
                     diagnostics[world, side, 2], diagnostics[world, side, 4], diagnostics[world, side, 5])
        body = bodies[side]
        rot = xmat[world, body]
        rt = wp.transpose(rot)
        origin = xpos[world, body]
        com = xipos[world, body]
        root_com = subtree_com[world, roots[side]]
        bf = wp.vec3(diagnostics[world, side, 6], diagnostics[world, side, 7], diagnostics[world, side, 8])
        bt = wp.vec3(diagnostics[world, side, 9], diagnostics[world, side, 10], diagnostics[world, side, 11])
        ef = wp.vec3(diagnostics[world, side, 12], diagnostics[world, side, 13], diagnostics[world, side, 14])
        et = wp.vec3(diagnostics[world, side, 15], diagnostics[world, side, 16], diagnostics[world, side, 17])
        j = Mat39(0.0)
        for k in range(counts[side]):
            d = dofs[side, k]
            screw = cdof[world, d]
            w = wp.vec3(screw[0], screw[1], screw[2])
            linear = wp.vec3(screw[3], screw[4], screw[5])
            local_v = rt*(linear+wp.cross(w, origin-root_com))
            local_w = rt*w
            j[0, k] = local_v[0]
            j[1, k] = local_w[1]
            j[2, k] = local_w[2]
            bq = wp.dot(linear+wp.cross(w, com-root_com), bf)+wp.dot(w, bt)
            bias[world, d] = bias[world, d]+bq
        for r in range(counts[side]):
            d = dofs[side, r]
            ar = a*wp.vec3(j[0, r], j[1, r], j[2, r])
            for c in range(r+1):
                adr = rowadr[d]+rownnz[d]-1-r+c
                value = wp.dot(ar, wp.vec3(j[0, c], j[1, c], j[2, c]))
                mass[world, adr] = mass[world, adr]+value
        xfrc[world, body] = xfrc[world, body]+wp.spatial_vector(ef[0], ef[1], ef[2], et[0], et[1], et[2])
        fl = rt*ef
        tl = rt*et
        for k in range(3):
            loads[world, side+1, k] = loads[world, side+1, k]+fl[k]
            loads[world, side+1, k+3] = loads[world, side+1, k+3]+tl[k]


class BladeInertia:
    """Full forward/step callback stage; uniform steady current, fixed water level.

    Added mass and entry are one closure; no separate empirical slam multiplier.
    Diagnostics columns are documented in BLADE_INERTIA.md. Reset clears outputs
    only: analytic wetting rates need no previous-frame memory or time guard.
    """
    def __init__(self, model, nworld, device, body_ids, repository=ROOT, parameters=None, geometry=None):
        repository = Path(repository)
        p = load_parameters(repository/'data/blade_inertia.json') if parameters is None else validate_parameters(dict(parameters))
        if model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
            raise ValueError('Blade inertia supports full Euler/implicitfast step, not RK4')
        if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION:
            raise ValueError('Blade inertia requires an enabled control callback')
        self.nworld, self.device, self.ns = nworld, device, p['strips_per_blade']
        geometry = repository_strips(repository, p) if geometry is None else geometry
        if geometry.endpoints.shape != (2*self.ns, 5, 3) or geometry.full_mass.shape != (2*self.ns,):
            raise ValueError('Strip geometry shape disagrees with parameters')
        if not np.isfinite(geometry.endpoints).all() or not np.isfinite(geometry.full_mass).all() or np.any(geometry.full_mass < 0):
            raise ValueError('Strip coordinates/masses must be finite and masses nonnegative')
        chains = []
        for body in body_ids:
            ancestor = body
            while ancestor and model.body_dofnum[ancestor] == 0:
                ancestor = model.body_parentid[ancestor]
            d = model.body_dofadr[ancestor]+model.body_dofnum[ancestor]-1 if ancestor else -1
            chain = []
            while d >= 0:
                chain.append(d)
                d = model.dof_parentid[d]
            if not 1 <= len(chain) <= 9:
                raise ValueError('Blade must have 1..9 ancestor DOFs (root plus oar joints)')
            chains.append(list(reversed(chain)))
        if len(chains) != 2 or body_ids[0] == body_ids[1]:
            raise ValueError('Two distinct blade bodies required')
        ids = np.full((2, 9), -1, np.int32)
        for side, chain in enumerate(chains):
            ids[side, :len(chain)] = chain
        self.parameters = dict(p, geometry=geometry.metadata)
        self.geometry = geometry
        self.body_ids = list(body_ids)
        self.dofs = wp.array(ids, dtype=int, device=device)
        self.counts = wp.array([len(c) for c in chains], dtype=int, device=device)
        self.bodies = wp.array(body_ids, dtype=int, device=device)
        self.roots = wp.array(model.body_rootid[body_ids], dtype=int, device=device)
        self.endpoints = wp.array(geometry.endpoints, dtype=wp.vec3, device=device)
        self.full_mass = wp.array(geometry.full_mass, dtype=float, device=device)
        self.scales = wp.ones((nworld, 2), dtype=float, device=device)
        self.chunks = wp.zeros((nworld, 2*self.ns, 24), dtype=float, device=device)
        self.diagnostics = wp.zeros((nworld, 2, 24), dtype=float, device=device)
        self._chunks = wp.to_torch(self.chunks)
        self._diagnostics = wp.to_torch(self.diagnostics)
        self._scales = wp.to_torch(self.scales)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._chunks[ids] = 0
        self._diagnostics[ids] = 0

    def set_mass_scale(self, value=1.0, env_ids=None):
        if not np.isfinite(value) or value < 0:
            raise ValueError('Mass scale must be finite and nonnegative')
        ids = slice(None) if env_ids is None else env_ids
        self._scales[ids] = float(value)
        self.reset(env_ids)

    def __call__(self, model, data, water, loads):
        p = self.parameters
        wp.launch(integrate_blade_strips, (self.nworld, 2*self.ns), inputs=[
            self.ns, p['wetting_depth_m'], p['rho_reference_kg_m3'], self.endpoints,
            self.full_mass, self.bodies, self.roots, self.dofs, self.counts, self.scales,
            water, data.xpos, data.xipos, data.xmat, data.subtree_com, data.cvel,
            data.cdof_dot, data.qvel, self.chunks], device=self.device)
        wp.launch(assemble_blade_inertia, self.nworld, inputs=[
            self.ns, self.bodies, self.roots, self.dofs, self.counts, self.chunks,
            self.diagnostics, model.M_rowadr, model.M_rownnz, data.xpos, data.xipos,
            data.xmat, data.subtree_com, data.cdof, data.M, data.qfrc_bias,
            data.xfrc_applied, loads], device=self.device)
