"""Batched Warp port of hydrodynamics.training_adapter (nominal parameters).

Exact water-plane clipping and hydrostatic triangle integration. Each thread
integrates a small fixed chunk; a second pass reduces chunks, avoiding global
atomic contention and an environments x faces x vertices temporary tensor.
All geometry preparation is offline at construction. No CPU work in callbacks.
The physical limitations of HYDRODYNAMICS_TRAINING.md still apply.
"""
import numpy as np
import warp as wp

from hydrodynamics.models import ROOT, load_closed_surface, projected_blade
from hydrodynamics.blade_inertia import load_parameters as load_blade_parameters
from hydrodynamics.training import TrainingHullResistance

CHUNK = 32
Mat43 = wp.types.matrix(shape=(4,3),dtype=wp.float32)


@wp.func
def sstep(x: float):
    a = wp.clamp(x, 0.0, 1.0)
    return a*a*(3.0-2.0*a)


@wp.func
def blade_force(v: wp.vec3, n: wp.vec3, c: wp.vec3, area: float, rho: float):
    vn = wp.dot(v, n)
    vt = wp.dot(v, c)
    u = wp.sqrt(vn*vn+vt*vt)
    f = wp.vec3(0.0)
    if u > 1.0e-12:
        # Algebraic equivalent of harmonic CD=2*sin(alpha)^2+.01,
        # CL=.9*sin(2*alpha); retains the sign at negative chordwise flow.
        cd = 2.0*vn*vn/(u*u)+0.01
        cl = 1.8*wp.abs(vn)*vt/(u*u)
        e = (vt*c+vn*n)/u
        el = (wp.abs(vn)*c-wp.sign(vn)*vt*n)/u
        f = 0.5*rho*area*u*u*(-cd*e+cl*el)
    tangent = v-vt*c-vn*n
    return f-0.005*rho*area*wp.length(tangent)*tangent


@wp.func
def fin_force(v: wp.vec3, c: wp.vec3, n: wp.vec3, area: float, ar: float, rho: float):
    u = wp.dot(v, c)
    vn = wp.dot(v, n)
    speed = wp.sqrt(u*u+vn*vn)
    result = wp.vec3(0.0)
    if speed > 1.0e-12 and ar > 1.0e-12:
        cs = u/speed
        sn = vn/speed
        slope = 2.0*wp.pi/(1.0+2.0/(0.8*ar))
        cl = slope*sn*cs
        cdi = cl*cl/(wp.pi*0.8*ar)
        q = 0.5*rho*area*speed*speed
        attached = q*(cl*(sn*c-cs*n)-cdi*(cs*c+sn*n))
        separated = -q*1.2*wp.abs(sn)*sn*n
        blend = sstep((wp.atan2(wp.abs(vn),wp.abs(u))-wp.pi/12.0)/(wp.pi/9.0))
        result = (1.0-blend)*attached+blend*separated
    return result


@wp.kernel
def integrate_chunks(
    triangles: wp.array2d(dtype=wp.vec3), starts: wp.array(dtype=int),
    counts: wp.array(dtype=int), kinds: wp.array(dtype=int),
    bodies: wp.array(dtype=int), roots: wp.array(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
    xipos: wp.array2d(dtype=wp.vec3), cvel: wp.array2d(dtype=wp.spatial_vector),
    subtree_com: wp.array2d(dtype=wp.vec3), water: wp.array2d(dtype=float),
    blade_wetting_depth: float, result: wp.array3d(dtype=float),
):
    world, chunk = wp.tid()
    kind = kinds[chunk]  # hull=0, fin=1, blade volumes=2/4, surfaces=3/5
    body = bodies[kind]
    origin = xpos[world,body]
    com = xipos[world,body]
    rot = xmat[world,body]
    ex = rot*wp.vec3(1.0,0.0,0.0)
    ey = rot*wp.vec3(0.0,1.0,0.0)
    ez = rot*wp.vec3(0.0,0.0,1.0)
    cv = cvel[world,body]
    omega = wp.vec3(cv[0],cv[1],cv[2])
    v0 = wp.vec3(cv[3],cv[4],cv[5])
    root_com = subtree_com[world,roots[body]]
    level = water[world,0]
    rho = water[world,1]
    current = wp.vec3(water[world,2],water[world,3],water[world,4])
    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    weighted_point = wp.vec3(0.0)
    total_area = float(0.0)
    low = float(1.0e20)
    high = float(-1.0e20)
    for offset in range(counts[chunk]):
        i = starts[chunk]+offset
        a = rot*triangles[i,0]+origin
        b = rot*triangles[i,1]+origin
        c = rot*triangles[i,2]+origin
        triangle_depth = level-(a[2]+b[2]+c[2])/3.0
        # At most four vertices after clipping a triangle by one plane.
        poly = Mat43(0.0)
        size = int(0)
        for edge in range(3):
            p = a
            q = b
            if edge == 1:
                p = b
                q = c
            elif edge == 2:
                p = c
                q = a
            dp = p[2]-level
            dq = q[2]-level
            if dp <= 0.0:
                poly[size] = p
                size += 1
            if (dp < 0.0 and dq > 0.0) or (dp > 0.0 and dq < 0.0):
                poly[size] = p+(dp/(dp-dq))*(q-p)
                size += 1
        for j in range(1,size-1):
            p = poly[0]
            q = poly[j]
            r = poly[j+1]
            cross = wp.cross(q-p,r-p)
            area = 0.5*wp.length(cross)
            if area > 1.0e-15:
                n = cross/(2.0*area)
                center = (p+q+r)/3.0
                if kind == 0 or kind == 1 or kind == 2 or kind == 4:
                    pa = rho*9.81*wp.max(level-p[2],0.0)
                    pb = rho*9.81*wp.max(level-q[2],0.0)
                    pc = rho*9.81*wp.max(level-r[2],0.0)
                    f = -n*area*(pa+pb+pc)/3.0
                    integral = area/12.0*((p+q+r-3.0*com)*(pa+pb+pc)
                                         +(p-com)*pa+(q-com)*pb+(r-com)*pc)
                    force += f
                    torque -= wp.cross(integral,n)
                vel = v0+wp.cross(omega,center-root_com)-current
                f_drag = wp.vec3(0.0)
                if kind == 0:
                    nt = n-wp.dot(n,ex)*ex
                    vn = wp.dot(vel,nt)
                    f_drag = -0.25*rho*area*wp.abs(vn)*vn*nt
                elif kind == 3 or kind == 5:
                    chord = ey
                    if kind == 5:
                        chord = -ey
                    f_drag = blade_force(vel,ex,chord,area,rho)
                    # Match the cubic waterline transition used by blade
                    # added inertia during entry and extraction.
                    f_drag *= sstep((triangle_depth+blade_wetting_depth)/(2.0*blade_wetting_depth))
                force += f_drag
                torque += wp.cross(center-com,f_drag)
                weight = area
                axis = ex
                if kind == 1:
                    weight = 0.5*area*wp.abs(wp.dot(n,ey))
                    axis = ez
                total_area += weight
                weighted_point += weight*center
                for k in range(3):
                    point = p
                    if k == 1:
                        point = q
                    elif k == 2:
                        point = r
                    coord = wp.dot(point-origin,axis)
                    low = wp.min(low,coord)
                    high = wp.max(high,coord)
    for k in range(3):
        result[world,chunk,k] = force[k]
        result[world,chunk,k+3] = torque[k]
        result[world,chunk,k+7] = weighted_point[k]
    result[world,chunk,6] = total_area
    result[world,chunk,10] = low
    result[world,chunk,11] = high


@wp.func
def hull_drag(u: float, length: float, area: float, rho: float,
              nodes: wp.array(dtype=float), coefs: wp.array2d(dtype=float),
              cr_lo: float, cr_hi: float, slope_lo: float, slope_hi: float,
              form: float):
    re = wp.max(wp.abs(u)*length/1.0e-6,1.0)
    log_re = wp.log(re)/wp.log(10.0)
    lam = 1.328/wp.sqrt(re)
    turb = 0.075/wp.pow(wp.max(log_re,5.0)-2.0,2.0)
    blend = sstep(log_re-5.0)
    cf = (1.0+form)*((1.0-blend)*lam+blend*turb)
    fn = wp.abs(u)/wp.sqrt(9.81*length)
    lo = nodes[0]
    hi = nodes[nodes.shape[0]-1]
    cr = float(0.0)
    if fn < lo:
        ratio = fn/lo
        cr = cr_lo*wp.pow(ratio,4.0)*wp.exp((4.0-slope_lo)*(1.0-ratio))
    elif fn > hi:
        x = wp.log(fn/hi)
        cr = cr_hi*wp.exp(slope_hi*x+(2.0-slope_hi)*(x-0.1*(1.0-wp.exp(-x/0.1))))
    else:
        index = int(0)
        for j in range(nodes.shape[0]-2):
            if fn >= nodes[j+1]:
                index = j+1
        dx = fn-nodes[index]
        cr = ((coefs[0,index]*dx+coefs[1,index])*dx+coefs[2,index])*dx+coefs[3,index]
    return 0.5*rho*area*u*u*(cf+cr)


@wp.kernel
def reduce_chunks(
    chunks: wp.array3d(dtype=float), ranges: wp.array2d(dtype=int),
    bodies: wp.array(dtype=int), roots: wp.array(dtype=int),
    xmat: wp.array2d(dtype=wp.mat33), xipos: wp.array2d(dtype=wp.vec3),
    cvel: wp.array2d(dtype=wp.spatial_vector), subtree_com: wp.array2d(dtype=wp.vec3),
    water: wp.array2d(dtype=float), nodes: wp.array(dtype=float),
    coefs: wp.array2d(dtype=float), constants: wp.array(dtype=float),
    loads: wp.array3d(dtype=float), xfrc: wp.array2d(dtype=wp.spatial_vector),
):
    world, slot = wp.tid()  # boat, port, starboard; unique writer per body
    first = 2*slot
    body = bodies[first]
    com = xipos[world,body]
    rot = xmat[world,body]
    ex = rot*wp.vec3(1.0,0.0,0.0)
    ey = rot*wp.vec3(0.0,1.0,0.0)
    cv = cvel[world,body]
    omega = wp.vec3(cv[0],cv[1],cv[2])
    v0 = wp.vec3(cv[3],cv[4],cv[5])
    current = wp.vec3(water[world,2],water[world,3],water[world,4])
    total_f = wp.vec3(0.0)
    total_t = wp.vec3(0.0)
    for kind in range(first,first+2):
        area = float(0.0)
        center = wp.vec3(0.0)
        low = float(1.0e20)
        high = float(-1.0e20)
        for chunk in range(ranges[kind,0],ranges[kind,1]):
            total_f += wp.vec3(chunks[world,chunk,0],chunks[world,chunk,1],chunks[world,chunk,2])
            total_t += wp.vec3(chunks[world,chunk,3],chunks[world,chunk,4],chunks[world,chunk,5])
            area += chunks[world,chunk,6]
            center += wp.vec3(chunks[world,chunk,7],chunks[world,chunk,8],chunks[world,chunk,9])
            low = wp.min(low,chunks[world,chunk,10])
            high = wp.max(high,chunks[world,chunk,11])
        if (kind == 0 or kind == 1) and area > 1.0e-10 and high-low > 1.0e-8:
            center /= area
            vel = v0+wp.cross(omega,center-subtree_com[world,roots[body]])-current
            f = wp.vec3(0.0)
            if kind == 0:
                u = wp.dot(vel,ex)
                drag = hull_drag(u,high-low,area,water[world,1],nodes,coefs,
                                 constants[0],constants[1],constants[2],constants[3],constants[4])
                f = -wp.sign(u)*drag*ex
            else:
                f = fin_force(vel,ex,ey,area,(high-low)*(high-low)/area,water[world,1])
            total_f += f
            total_t += wp.cross(center-com,f)
    # MuJoCo xfrc_applied: world force then world torque, about body COM.
    xfrc[world,body] = wp.spatial_vector(total_f[0],total_f[1],total_f[2],total_t[0],total_t[1],total_t[2])
    local_f = wp.transpose(rot)*total_f
    local_t = wp.transpose(rot)*total_t
    for k in range(3):
        loads[world,slot,k] = local_f[k]
        loads[world,slot,k+3] = local_t[k]


class WarpHydrodynamics:
    """Owns persistent buffers; callback is compatible with mjlab CUDA capture."""
    def __init__(self, model, nworld, device, prefix="", repository=ROOT, hull_radiation=False, blade_inertia=False):
        from pathlib import Path
        repository = Path(repository)
        surfaces = [load_closed_surface(repository/'boat_meshes'/f'{name}.stl')[0]
                    for name in ('shell_white','fin')]
        for side in ('port','starboard'):
            path = repository/'boat_meshes'/f'{side}_blade_3mm.stl'
            surfaces += [load_closed_surface(path)[0],projected_blade(path)[0]]
        starts, counts, kinds, ranges = [], [], [], []
        face_start = 0
        for kind, triangles in enumerate(surfaces):
            start = len(starts)
            for offset in range(0,len(triangles),CHUNK):
                starts.append(face_start+offset)
                counts.append(min(CHUNK,len(triangles)-offset))
                kinds.append(kind)
            ranges.append((start,len(starts)))
            face_start += len(triangles)
        law = TrainingHullResistance(path=repository/'data/parameters.json')
        self.nworld = nworld
        self.device = device
        blade_params = load_blade_parameters(repository/'data/blade_inertia.json')
        self.blade_wetting_depth = float(blade_params['wetting_depth_m'])
        self.nchunks = len(starts)
        self.face_count = face_start
        self.triangles = wp.array(np.concatenate(surfaces).astype(np.float32),dtype=wp.vec3,device=device)
        self.starts = wp.array(starts,dtype=int,device=device)
        self.counts = wp.array(counts,dtype=int,device=device)
        self.kinds = wp.array(kinds,dtype=int,device=device)
        self.ranges = wp.array(ranges,dtype=int,device=device)
        self.body_ids = [int(model.body(prefix+name).id) for name in ('boat','boat','port_oar','port_oar','starboard_oar','starboard_oar')]
        self.bodies = wp.array(self.body_ids,dtype=int,device=device)
        self.roots = wp.array(model.body_rootid,dtype=int,device=device)
        self.nodes = wp.array(law.fn,dtype=float,device=device)
        self.coefs = wp.array(law.interpolator.c,dtype=float,device=device)
        self.constants = wp.array([law.cr[0],law.cr[-1],*law.end_slopes,law.form_factor],dtype=float,device=device)
        water = np.zeros((nworld,5),dtype=np.float32)
        water[:,1] = 1000.0
        self.water = wp.array(water,dtype=float,device=device)
        self.chunks = wp.zeros((nworld,self.nchunks,12),dtype=float,device=device)
        self.loads = wp.zeros((nworld,3,6),dtype=float,device=device)
        self.radiation = None
        if hull_radiation:
            from .radiation import HullRadiation
            from hydrodynamics.hull_radiation import load_parameters
            self.radiation = HullRadiation(model,nworld,device,self.body_ids[0],
                parameters=load_parameters(repository/'data/hull_radiation.json'))
        self.blade_inertia = None
        if blade_inertia:
            from .blade_inertia import BladeInertia
            self.blade_inertia = BladeInertia(model,nworld,device,
                [self.body_ids[2],self.body_ids[4]],repository=repository)

    def reset(self, env_ids=None):
        if self.radiation is not None:
            self.radiation.reset(env_ids)
        if self.blade_inertia is not None:
            self.blade_inertia.reset(env_ids)

    def __call__(self, model, data):
        wp.launch(integrate_chunks,(self.nworld,self.nchunks),inputs=[
            self.triangles,self.starts,self.counts,self.kinds,self.bodies,self.roots,
            data.xpos,data.xmat,data.xipos,data.cvel,data.subtree_com,self.water,
            self.blade_wetting_depth,self.chunks],device=self.device)
        wp.launch(reduce_chunks,(self.nworld,3),inputs=[
            self.chunks,self.ranges,self.bodies,self.roots,data.xmat,data.xipos,
            data.cvel,data.subtree_com,self.water,self.nodes,self.coefs,self.constants,
            self.loads,data.xfrc_applied],device=self.device)
        if self.radiation is not None:
            self.radiation(model,data,self.water,self.loads)
        if self.blade_inertia is not None:
            self.blade_inertia(model,data,self.water,self.loads)
