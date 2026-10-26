"""Offline engineering prior for hull inertia/radiation; no CFD or stroke frequency.

These are geometry-based surrogate coefficients, NOT identified free-surface
BEM coefficients. See HULL_RADIATION.md for assumptions and sensitivity ranges.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
import trimesh

from .models import ROOT, load_closed_surface, wet_triangles, hydrostatic_wrench

PARAMETER_PATH = ROOT/'data/hull_radiation.json'


def validate_parameters(p):
    if p['schema_version'] != 1 or p['dof_order'] != ['u', 'v', 'w', 'p', 'q', 'r']:
        raise ValueError('Unsupported hull radiation schema / DOF convention')
    for key in ('added_mass_inf', 'radiation_delta_mass'):
        a = np.asarray(p[key], dtype=float)
        if a.shape != (6, 6) or not np.isfinite(a).all():
            raise ValueError(f'{key} must be a finite 6 x 6 matrix')
        if not np.allclose(a, a.T, atol=1e-10) or np.linalg.eigvalsh(a).min() < -1e-10:
            raise ValueError(f'{key} must be symmetric positive semidefinite')
    ref = np.asarray(p['reference_point_body_m'])
    if ref.shape != (3,) or not np.isfinite(ref).all():
        raise ValueError('Invalid hull reference point')
    for key in ('rho_reference_kg_m3', 'pole_rad_s'):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f'{key} must be positive')
    return p


def load_parameters(path=PARAMETER_PATH):
    return validate_parameters(json.loads(Path(path).read_text(encoding='utf-8')))


def frequency_response(p, omega):
    """Return A(omega), B(omega), positive damping convention, omega in rad/s.

    Z(s) = DeltaA * omega0**2 * s / (s + omega0)**2;
    A = Ainf + Im(Z)/omega; B = Re(Z). Includes the omega=0 limit.
    """
    w = np.asarray(omega, dtype=float)
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('Frequencies must be finite and non-negative')
    w0 = p['pole_rad_s']
    den = (w0*w0+w*w)**2
    da = np.asarray(p['radiation_delta_mass'])
    a = np.asarray(p['added_mass_inf']) + (w0*w0*(w0*w0-w*w)/den)[..., None, None]*da
    b = (2*w0**3*w*w/den)[..., None, None]*da
    return a, b


def estimate_parameters(mesh_path, water_level, sections=241, rho=1000.0,
                        instantaneous_fraction=0.5):
    """Integrate half-ellipse strip estimates over the ACTUAL nominal wet hull.

    The one-half exterior-domain factor and the additional instantaneous
    fraction are distinct assumptions. Neither is an exact free-surface BEM
    boundary condition. Surge uses a triaxial ellipsoid shape coefficient.
    """
    if sections < 21 or rho <= 0 or not 0 < instantaneous_fraction <= 1:
        raise ValueError('Invalid strip estimation settings')
    mesh_path = Path(mesh_path)
    triangles, _ = load_closed_surface(mesh_path)
    wet = wet_triangles(triangles, water_level)
    if not len(wet):
        raise ValueError('The hull does not intersect the selected water level')
    lo, hi = wet.min(axis=(0, 1)), wet.max(axis=(0, 1))
    length, beam, draft = hi-lo
    f, torque, _ = hydrostatic_wrench(triangles, np.zeros(3), water_level, rho)
    volume = f[2]/(rho*9.81)
    ref = np.array([-torque[1]/f[2], 0., water_level])
    mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
                           faces=np.arange(triangles.size//3).reshape(-1, 3), process=True)
    # Midpoint quadrature: avoids degenerate sections at the two waterline ends.
    dx = length/sections
    strip = np.zeros((6, 6))
    for x in lo[0]+(np.arange(sections)+.5)*dx:
        section = mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
        if section is None:
            continue
        pts = []
        for edge in section.discrete:
            for a, b in zip(edge[:-1], edge[1:]):
                if a[2] <= water_level:
                    pts.append(a)
                if (a[2]-water_level)*(b[2]-water_level) < 0:
                    pts.append(a+(water_level-a[2])/(b[2]-a[2])*(b-a))
        if not pts:
            continue
        pts = np.asarray(pts)
        half_beam = np.ptp(pts[:, 1])/2
        depth = water_level-pts[:, 2].min()
        # Per-length inertia of half of a reflected elliptic cross-section.
        ay = .5*rho*np.pi*depth**2
        az = .5*rho*np.pi*half_beam**2
        ar = rho*np.pi/16*(half_beam**2-depth**2)**2
        jy = np.array([0, 1, 0, 0, 0, x-ref[0]])
        jz = np.array([0, 0, 1, 0, -(x-ref[0]), 0])
        strip += dx*(ay*np.outer(jy, jy)+az*np.outer(jz, jz))
        strip[3, 3] += dx*ar
    axes = np.array([length/2, beam/2, draft])
    abc = axes.prod()
    alpha = abc*quad(lambda s: 1/((axes[0]**2+s)*np.sqrt(np.prod(axes**2+s))),
                     0, np.inf, epsabs=1e-10)[0]
    strip[0, 0] = rho*volume*alpha/(2-alpha)
    ainf = instantaneous_fraction*strip
    delta = (1-instantaneous_fraction)*strip
    result = dict(
        schema_version=1, dof_order=['u', 'v', 'w', 'p', 'q', 'r'],
        status='unvalidated geometry-based passive engineering prior, not BEM data',
        reference_point_body_m=ref.tolist(), rho_reference_kg_m3=rho,
        pole_rad_s=float(np.sqrt(9.81/beam)), damping_ratio=1.0,
        added_mass_inf=ainf.tolist(), radiation_delta_mass=delta.tolist(),
        geometry=dict(mesh=mesh_path.name, sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                      water_level_body_m=water_level, waterline_length_m=float(length),
                      maximum_wet_beam_m=float(beam), draft_m=float(draft),
                      displaced_volume_m3=float(volume), strip_sections=sections),
        assumptions=dict(half_domain_factor=.5, instantaneous_fraction=instantaneous_fraction,
                         radiation='one critically damped pole pair shared by six DOFs',
                         pole_rule='sqrt(g / maximum_wet_beam); not stroke frequency',
                         speed='zero-forward-speed coefficients used as a training approximation'),
        sources=[
            'https://home.hvl.no/ansatte/gste/ftp/MarinLab_files/Litteratur/Marine_Hydrodynamics_Newman_2018.pdf',
            'https://doi.org/10.1017/jfm.2020.1104',
            'https://wec-sim.github.io/WEC-Sim/main/theory/theory.html',
        ])
    return validate_parameters(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write JSON; otherwise print only')
    parser.add_argument('--sections', type=int, default=241)
    args = parser.parse_args()
    provenance = json.loads((ROOT/'data/rowing_reference_provenance.json').read_text())
    p = estimate_parameters(ROOT/'boat_meshes/shell_white.stl',
                            provenance['water_level_m_at_repository_reset'], args.sections)
    text = json.dumps(p, indent=2, ensure_ascii=False)+'\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
