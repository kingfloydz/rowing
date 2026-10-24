"""Mechanics kernels with explicit fluid-domain assumptions (SI units).

No empirical rowing coefficients are supplied here. A mathematically valid
inertia matrix is not evidence that it describes a surface-piercing oar.
See HYDRODYNAMICS_CORE.md for the distinction between force decomposition,
potential-flow mechanics, and the unresolved free-surface/vorticity closure.
"""
from dataclasses import dataclass
import numpy as np


def _array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return a


def skew(vector):
    x, y, z = _array(vector, (3,), 'vector')
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def point_twist_map(offset):
    """nu_P = X nu_O; offset = OP, same axes, nu=[v, omega]."""
    x = np.eye(6)
    x[:3, 3:] = -skew(offset)
    return x


def shift_wrench(wrench_at_point, offset):
    """[F, M_O] = X.T [F, M_P]; M_P is an intrinsic moment, not r x F."""
    return point_twist_map(offset).T @ _array(wrench_at_point, (6,), 'wrench')


def check_inertia(matrix, characteristic_length=1.):
    """Validate reciprocal, positive-semidefinite fluid kinetic energy.

    Scale angular coordinates by a declared length before numerical checks;
    force and torque blocks have different SI units. Never silently symmetrize
    or clip negative eigenvalues from an invalid hydrodynamic calculation.
    """
    if not np.isfinite(characteristic_length) or characteristic_length <= 0:
        raise ValueError('Positive finite characteristic length required')
    m = _array(matrix, (6, 6), 'added inertia')
    d = np.diag([1., 1., 1.] + [1./characteristic_length]*3)
    scaled = d @ m @ d
    scale = max(float(np.linalg.norm(scaled, 2)), 1e-30)
    if not np.allclose(scaled, scaled.T, rtol=0, atol=1e-10*scale):
        raise ValueError('Added inertia violates reciprocity/symmetry')
    if np.linalg.eigvalsh(scaled).min() < -1e-10*scale:
        raise ValueError('Added inertia has negative kinetic energy')
    return m.copy()


@dataclass(frozen=True)
class UnboundedPotentialInertia:
    """Constant body-frame Kirchhoff tensor in otherwise still unbounded fluid.

    Assumes an incompressible, inviscid, irrotational fluid, fixed fully wetted
    rigid geometry, no circulation and no free surface/wall. No wave radiation,
    separation, entry/exit or varying wet mass is represented. A uniform steady
    current is allowed through the relative translational velocity.

    Source and geometry identity are mandatory provenance, not an accuracy
    certificate. Matrix blocks have units kg, kg m, kg m^2 respectively.
    """
    matrix: np.ndarray
    geometry_id: str
    source: str
    characteristic_length: float = 1.

    def __post_init__(self):
        if not self.geometry_id.strip() or not self.source.strip():
            raise ValueError('Geometry identity and coefficient source required')
        m = check_inertia(self.matrix, self.characteristic_length)
        m.setflags(write=False)
        object.__setattr__(self, 'matrix', m)

    @classmethod
    def sphere(cls, radius, rho=1000.):
        """Exact sphere: translational added mass = displaced mass / 2.

        Rotation about its centre produces no normal boundary velocity, so its
        inviscid rotational added inertia is zero, not a fitted small value.
        """
        if not np.isfinite(radius) or radius <= 0 or not np.isfinite(rho) or rho <= 0:
            raise ValueError('Positive finite radius and density required')
        mass = .5*rho*4*np.pi*radius**3/3
        return cls(np.diag([mass]*3+[0.]*3), f'sphere_radius_{radius:g}',
                   'Kirchhoff/Lamb unbounded potential-flow sphere', radius)

    def energy(self, body_twist):
        nu = _array(body_twist, (6,), 'body twist')
        return float(.5*nu @ self.matrix @ nu)

    def coriolis_wrench(self, body_twist):
        """C_A(nu) nu; includes translation-rotation and Munk moment terms."""
        nu = _array(body_twist, (6,), 'body twist')
        p, h = np.split(self.matrix @ nu, 2)
        v, omega = np.split(nu, 2)
        return np.r_[np.cross(omega, p), np.cross(v, p)+np.cross(omega, h)]

    def wrench(self, body_twist, body_twist_derivative):
        """Force ON body, about the tensor origin, expressed in body axes.

        The derivative differentiates BODY COMPONENTS; it is not simply the
        inertial acceleration rotated into the body. For uniform current,
        v_dot_body = R.T a_relative_world - omega_body x v_relative_body.
        """
        derivative = _array(body_twist_derivative, (6,), 'body twist derivative')
        return -self.matrix @ derivative-self.coriolis_wrench(body_twist)

    def about_new_origin(self, new_to_old):
        x = point_twist_map(new_to_old)
        return UnboundedPotentialInertia(x.T @ self.matrix @ x,
            self.geometry_id, self.source, self.characteristic_length)


def cummins_kernel(frequency, radiation_damping, time):
    """K(t)=2/pi integral_0^infinity B(w) cos(w t) dw, truncated reference.

    B is [frequency, dof, dof], from ONE consistent linear radiation problem
    for a finite 3-D body at zero forward speed and fixed mean wetted geometry.
    Viscous steady damping is not radiation: B(0) must vanish. Finite bandwidth is
    explicit: the caller must demonstrate cutoff/grid convergence and recover
    A(w) from the same kernel/A_infinity. This routine does NOT identify a
    kernel from mean drag, provide coefficients, or describe forward-speed
    rowing. Oscillatory/sign-changing kernels are permitted.
    """
    w = np.asarray(frequency, float)
    b = np.asarray(radiation_damping, float)
    t = np.asarray(time, float)
    if (w.ndim != 1 or len(w) < 2 or w[0] != 0 or not np.all(np.isfinite(w))
            or np.any(np.diff(w) <= 0) or t.ndim != 1
            or not np.all(np.isfinite(t)) or np.any(t < 0)):
        raise ValueError('Finite increasing frequencies starting at zero and nonnegative times required')
    if (b.ndim != 3 or b.shape[0] != len(w) or b.shape[1] != b.shape[2]
            or not np.all(np.isfinite(b))):
        raise ValueError('Radiation damping must have shape [frequency, dof, dof]')
    scale = max(float(np.max(np.abs(b))), 1e-30)
    if not np.allclose(b, b.transpose(0, 2, 1), atol=1e-10*scale, rtol=0):
        raise ValueError('Zero-speed radiation damping must be reciprocal')
    if np.linalg.eigvalsh(b).min() < -1e-10*scale:
        raise ValueError('Negative radiation damping violates the zero-speed passive model')
    if np.max(np.abs(b[0])) > 1e-8*scale:
        raise ValueError('Zero-speed finite-body radiation requires B(0)=0; steady drag is a different contribution')
    return np.array([2/np.pi*np.trapezoid(b*np.cos(w*ti)[:, None, None], w, axis=0)
                     for ti in t])
