"""SI-unit reduced hydrodynamics. Calm water, rigid geometry, no online CFD.

Literature parameters and fitted parameters are separate from unvalidated closures.
See HYDRODYNAMICS_REPORT.md for scope, equations and experimental provenance.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
import trimesh

RHO = 1000.0
G = 9.81
ROOT = Path(__file__).resolve().parents[1]


def clip_underwater(poly, level=0.0):
    """Sutherland-Hodgman clipping; preserve winding; flat free surface."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = a[2]-level, b[2]-level
        if da <= 0: out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            out.append(a + da/(da-db)*(b-a))
    return np.asarray(out).reshape(-1, 3)


def wet_triangles(triangles, level=0.0):
    """Retain fully wet triangles in a batch; clip only waterline intersections."""
    triangles = np.asarray(triangles)
    z = triangles[:, :, 2]-level
    full = triangles[np.all(z <= 0, axis=1)]
    partial = triangles[(np.min(z, axis=1) < 0) & (np.max(z, axis=1) > 0)]
    extra = []
    for tri in partial:
        p = clip_underwater(tri, level)
        extra.extend([p[[0, j, j+1]] for j in range(1, len(p)-1)])
    return np.concatenate((full, np.asarray(extra).reshape(-1, 3, 3)))


def triangle_geometry(tris):
    cross = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    area = np.linalg.norm(cross, axis=1)*.5
    normal = cross/np.maximum(2*area[:, None], 1e-30)
    return area, normal, tris.mean(axis=1)


def hydrostatic_wrench(triangles_world, origin, level=0.0, rho=RHO):
    """Exact integration of linear pressure and quadratic moment on triangles.

    Mesh must enclose displaced volume and have outward winding. Free-surface
    caps are unnecessary because their gauge pressure is zero.
    Returns force, torque about origin, wetted surface area.
    """
    tris = wet_triangles(triangles_world, level)
    if not len(tris): return np.zeros(3), np.zeros(3), 0.0
    area, normal, _ = triangle_geometry(tris)
    p = rho*G*np.maximum(level-tris[:, :, 2], 0)
    force = -normal*(area*p.mean(axis=1))[:, None]
    # integral(r*p)dA = A/12 * [(sum r)(sum p) + sum(r_i*p_i)]
    r = tris-np.asarray(origin)
    integral_rp = area[:, None]/12*(r.sum(axis=1)*p.sum(axis=1)[:, None]
                                  + (r*p[:, :, None]).sum(axis=1))
    torque = -np.cross(integral_rp, normal)
    return force.sum(axis=0), torque.sum(axis=0), float(area.sum())


def load_closed_surface(path):
    mesh = trimesh.load_mesh(path, process=True)
    pieces = list(mesh.split(only_watertight=False))
    good = [p for p in pieces if p.is_watertight and abs(p.volume) > 1e-9]
    for p in good:
        if p.volume < 0: p.invert()
    if not good: raise ValueError(f'No closed displaced-volume surface: {path}')
    return np.concatenate([p.triangles for p in good]), {
        'closed_components': len(good), 'omitted_faces': sum(len(p.faces) for p in pieces if p not in good),
        'enclosed_volume_m3': float(sum(p.volume for p in good)),
        'bounds_m': mesh.bounds.tolist(),
    }


def projected_blade(path):
    """Union of actual mesh triangle projections, preserving concave outline."""
    from shapely import Polygon, unary_union, constrained_delaunay_triangles
    mesh = trimesh.load_mesh(path, process=True)
    projected = [Polygon(t[:,1:]) for t in mesh.triangles]
    outline=unary_union([p for p in projected if p.area>1e-14])
    pieces=constrained_delaunay_triangles(outline)
    tris=np.array([np.column_stack((np.full(3,mesh.bounds[:,0].mean()),np.asarray(p.exterior.coords)[:3])) for p in pieces.geoms])
    # Subdivide to improve rotational-velocity integration; each triangle gets 4.
    for _ in range(2):
        sub=[]
        for a,b,c in tris:
            ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
            sub.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
        tris=np.asarray(sub)
    area = float(triangle_geometry(tris)[0].sum())
    span = float(np.ptp(mesh.vertices[:, 1]))
    chord = area/span  # area-equivalent rectangular chord, explicitly approximate
    return tris, {'area_m2': area, 'span_m': span, 'equivalent_chord_m': chord,
                  'max_vertical_chord_m': float(np.ptp(mesh.vertices[:, 2])),
                  'thickness_m': float(np.ptp(mesh.vertices[:, 0]))}


def ittc57(speed, length, nu=1e-6):
    """ITTC turbulent correlation with low-Re regularisation (not validated there)."""
    re = np.maximum(np.abs(speed)*length/nu, 1e5)
    return .075/(np.log10(re)-2)**2


@dataclass
class HullResistance:
    fn_nodes: np.ndarray
    residual_nodes: np.ndarray
    form_factor: float = .012
    nu: float = 1e-6

    def coefficient(self, speed, length):
        fn = np.abs(speed)/np.sqrt(G*length)
        # Bounds are intentionally clamped, not an unbounded high-order extrapolation.
        f = np.clip(fn, self.fn_nodes[0], self.fn_nodes[-1])
        cr = PchipInterpolator(self.fn_nodes, self.residual_nodes)(f)
        # Below lowest measured Fn, force residual coefficient to vanish as Fn^4.
        cr = cr*np.minimum(1, (fn/self.fn_nodes[0])**4)
        return (1+self.form_factor)*ittc57(speed, length, self.nu)+np.maximum(cr, 0)

    def magnitude(self, speed, length, wetted_area, rho=RHO):
        return .5*rho*wetted_area*np.asarray(speed)**2*self.coefficient(speed, length)


@dataclass
class BladeDrag:
    relative_depth: np.ndarray
    cd_nodes: np.ndarray

    def cd(self, top_depth, chord):
        h = np.clip(np.asarray(top_depth)/chord, self.relative_depth[0], self.relative_depth[-1])
        return PchipInterpolator(self.relative_depth, self.cd_nodes)(h)


def normal_plate_force(velocity, normal, area, cd, rho=RHO):
    """Flat-plate normal-force closure. Dissipative; arbitrary-angle use unvalidated."""
    v = np.asarray(velocity)
    vn = np.sum(v*normal, axis=-1)
    return -.5*rho*np.asarray(area)[..., None]*np.asarray(cd)[..., None]*np.linalg.norm(v, axis=-1)[..., None]*vn[..., None]*normal


def plate_added_mass(span, chord, rho=RHO):
    """Grift2019 Eq3.3 (Patton), AR=2 coefficient .84, extrapolated for CAD blade."""
    return .84*np.pi/4*rho*span*chord**2


def start_plate_force(t, acceleration, target_speed, cd, entrainment=0.0,
                      tail_amplitude=0.0, tail_scale=1.0, added_mass=True):
    """Published start-from-rest apparatus benchmark only, NOT a general stroke law.

    Returns transducer force, including plate inertia mp*a. Entrainment follows
    Grift Eq4.1, with Patton mass rather than Yu. Transition tail is an explicitly
    empirical smooth envelope, not a fit to individual vortex peaks.
    """
    t=np.asarray(t); a=acceleration; vmax=target_speed; tc=vmax/a
    accelerating=(t>=0)&(t<tc)
    v=np.clip(a*t, 0, vmax)
    ma=plate_added_mass(.2,.1) if added_mass else 0
    f=.5*RHO*.02*cd*v*v + (.4+ma)*a*accelerating
    f += entrainment*a*np.maximum(t-.07,0)*accelerating
    theta=np.maximum(t-tc,0)*vmax/.1
    f += tail_amplitude*.5*RHO*.02*vmax*vmax/(1+theta/tail_scale)**2*(t>=tc)
    return f


def load_parameters(path=None):
    p=json.loads(Path(path or ROOT/'data'/'parameters.json').read_text(encoding='utf-8'))
    return (HullResistance(np.array(p['hull']['fn_nodes']), np.array(p['hull']['residual_nodes']),
                           p['hull']['form_factor'], p['hull']['nu']),
            BladeDrag(np.array(p['blade']['relative_depth']), np.array(p['blade']['cd_nodes'])), p)
