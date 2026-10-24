"""Literature-based closures; domains of applicability are deliberately explicit.

Reijtenbagh et al., JFM 1028 A30 (2026), DOI 10.1017/jfm.2025.11107.
The scalar history model is for one-direction translating, submerged plates.
It replaces the excess-force closure, rather than being added to inviscid added
mass. Rotating/reversing/entering blades require additional validation.
"""
import numpy as np


def history_integral(time_edges, velocity):
    """Abel integral int_0^t v'(s)/sqrt(t-s) ds, exact for linear v per interval.

    Inputs are samples at increasing time edges starting at zero and v(0)=0.
    O(N^2) reference implementation, not the batched-training implementation.
    No instantaneous-acceleration feedback and no numerical derivative of force.
    """
    t = np.asarray(time_edges, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if t.ndim != 1 or v.shape != t.shape or len(t) < 2:
        raise ValueError('Need matching one-dimensional time and velocity arrays')
    if t[0] != 0 or v[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError('History must start from rest at t=0; times must increase')
    a = np.diff(v)/np.diff(t)
    out = np.zeros_like(t)
    for i in range(1, len(t)):
        weights = 2*(np.sqrt(t[i]-t[:i])-np.sqrt(t[i]-t[1:i+1]))
        out[i] = weights@a[:i]
    return out


def acceleration_reynolds(kind, parameter, length, nu=1e-6):
    """Paper Eqs. 3.7, 3.15; parameter is a, J, or Q^2 in SI units."""
    if parameter <= 0 or length <= 0 or nu <= 0:
        raise ValueError('Positive motion parameter, length and viscosity required')
    if kind == 'constant_acceleration':
        return np.sqrt(parameter*length**3)/nu
    if kind == 'constant_jerk':
        return (2*length**5*parameter)**(1/3)/nu
    if kind == 'sqrt_velocity':
        return (.5*length**4*parameter)**(1/3)/nu
    raise ValueError(kind)


def prescribed_motion(t, kind, parameter, final_speed):
    t = np.maximum(np.asarray(t, dtype=float), 0)
    if parameter <= 0 or final_speed <= 0:
        raise ValueError('Positive motion parameter and final speed required')
    if kind == 'constant_acceleration':
        stop = final_speed/parameter
        v = np.minimum(parameter*t, final_speed)
        a = np.where(t < stop, parameter, 0.)
    elif kind == 'constant_jerk':
        stop = np.sqrt(2*final_speed/parameter)
        v = np.minimum(.5*parameter*t*t, final_speed)
        a = np.where(t < stop, parameter*t, 0.)
    elif kind == 'sqrt_velocity':
        stop = final_speed**2/parameter
        v = np.minimum(np.sqrt(parameter*t), final_speed)
        # Singular idealized start: evaluate only at t>0; numerical history uses v.
        a = np.where((t > 0) & (t < stop), .5*np.sqrt(parameter)/np.sqrt(np.maximum(t, 1e-30)), 0.)
    else:
        raise ValueError(kind)
    return v, a, stop


def history_excess_force(t, kind, parameter, final_speed, area=.02,
                         length=.1, rho=998., nu=.978e-6, constant=1.5):
    """Analytic Abel integral, including relaxation, with common C_1=1.5.

    Motion-specific constants follow from integration: C_J=C_1*2/3=1,
    C_Q=C_1*pi/4=1.1781. This is not three separately fitted coefficients.
    Uses nominal motion; measured motion may differ (paper Appendix A).
    """
    t = np.maximum(np.asarray(t, dtype=float), 0)
    _, _, stop = prescribed_motion(t, kind, parameter, final_speed)
    h = np.minimum(t, stop)
    if kind == 'constant_acceleration':
        integral = 2*parameter*(np.sqrt(t)-np.sqrt(np.maximum(t-stop, 0)))
    elif kind == 'constant_jerk':
        tail = np.maximum(t-h, 0)
        integral = parameter*(2*t*(np.sqrt(t)-np.sqrt(tail))
                              - 2/3*(t**1.5-tail**1.5))
    else:
        integral = np.sqrt(parameter)*np.arcsin(np.sqrt(np.divide(h, t, out=np.zeros_like(t), where=t>0)))
    re = acceleration_reynolds(kind, parameter, length, nu)
    return .5*constant*rho*area*np.sqrt(nu*re)*integral


def rotating_added_mass_wrench(velocity, acceleration, normal, omega, mass):
    """Energy-consistent anisotropic mass at a rigidly carried point, fixed m.

    Returns point force and intrinsic orientation torque. Add r x F separately.
    n_dot=omega x n. Identity: F.v + M.omega = -d[m(n.v)^2/2]/dt.
    Varying wetted mass and wave-current gradients are outside this closure.
    """
    v, a, n, om = map(np.asarray, (velocity, acceleration, normal, omega))
    ndot = np.cross(om, n)
    vn = np.dot(n, v)
    force = -mass*(n*(np.dot(n, a)+np.dot(ndot, v))+vn*ndot)
    torque = mass*vn*np.cross(n, v)
    return force, torque
