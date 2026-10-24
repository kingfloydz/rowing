"""Restricted empirical references, not unrestricted CAD rowing models.

The caller supplies physical conditions. Reject unsupported geometry/motion
instead of silently clamping coefficients or inventing a blending factor.
Historical exploratory functions remain in models.py for reproducibility.
"""
from dataclasses import dataclass
import numpy as np
from .models import G, load_parameters
from .polars import BladePolar
from .unsteady import history_excess_force


class OutsideReferenceDomain(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise OutsideReferenceDomain(message)


def validate_components(components):
    """Reject known duplicate closures. A valid selection is not calibration."""
    names = tuple(components)
    if len(set(names)) != len(names):
        raise ValueError('A force component cannot be included twice')
    groups = [
        {'grift_depth', 'caplan_static', 'barre_dynamic'},
        {'potential_inertia', 'history_excess', 'barre_dynamic'},
        {'day_total_resistance', 'full_wave_resistance'},
    ]
    for group in groups:
        conflict = group.intersection(names)
        if len(conflict) > 1:
            raise ValueError('Overlapping physical contributions: '+', '.join(sorted(conflict)))
    rejected = {'fitted_tail', 'fitted_wake', 'uncalibrated_crossflow',
                'uncalibrated_fin', 'dode_response_table'}
    if rejected.intersection(names):
        raise ValueError('Research-only closures are excluded from the reference core')
    known = {'hydrostatics', 'grift_depth', 'caplan_static', 'barre_dynamic',
             'potential_inertia', 'history_excess', 'day_total_resistance',
             'full_wave_resistance', 'linear_radiation', 'viscous_resistance'}
    if set(names)-known:
        raise ValueError('Unknown component: '+', '.join(sorted(set(names)-known)))
    if 'linear_radiation' in names and 'potential_inertia' in names:
        raise ValueError('Radiation already requires its own consistent A_infinity; do not add unbounded inertia')
    return names


@dataclass
class DayReferenceResistance:
    """Original Day2011 hull, natural-transition steady towing reference only.

    Fn support is measured; zero velocity is an exact physical limit. No
    Froude-only transfer to an unrelated hull and no silent low/high extrapolation.
    """
    def force(self, speed, *, geometry_id, length=7.937, wetted_area,
              rho=1000., steady=True):
        law, _, _ = load_parameters()
        u = np.asarray(speed, float)
        _require(geometry_id == 'day2011_reference_hull', 'Day coefficients do not identify the CAD hull')
        _require(np.isfinite(length) and abs(length-7.937) < 1e-8,
                 'Only the reference length is admitted; scale transfer needs separate evidence')
        _require(steady, 'Steady drag reference is not a validated unsteady hull force')
        _require(np.all(np.isfinite(u)) and np.all(u >= 0),
                 'Original towing data support forward speed; reverse towing is unvalidated')
        _require(np.isfinite(wetted_area) and wetted_area > 0 and np.isfinite(rho) and rho > 0,
                 'Positive finite wetted area and density required')
        _require(np.isclose(wetted_area,2.34,rtol=0,atol=1e-8),
                 'Reference displacement/trim has wetted area 2.34 m^2; other loading is unvalidated')
        fn = u/np.sqrt(G*length)
        supported = (u == 0) | ((fn >= law.fn_nodes[0]-1e-12) & (fn <= law.fn_nodes[-1]+1e-12))
        _require(np.all(supported), f'Fn outside [{law.fn_nodes[0]:.6f}, {law.fn_nodes[-1]:.6f}]')
        return -law.magnitude(u, length, wetted_area, rho)


def grift_steady_normal_force(speed, top_depth, *, width=.2, height=.1,
                              fully_wet=True, steady=True, normal_translation=True):
    """Grift2019 depth sweep at U=.30 m/s; rectangular AR=2 plate, water.

    Width is horizontal .20 m, height vertical .10 m. h is upper-edge depth.
    The broad experimental Re range is not a measured joint Cd(Re,h) table.
    """
    _require(fully_wet and steady and normal_translation,
             'Depth coefficients require steady, fully wetted, broadside translation')
    _require(np.isclose(width, .2, rtol=0, atol=1e-10) and np.isclose(height, .1, rtol=0, atol=1e-10),
             'Only the measured rectangular plate geometry is admitted')
    u, h = np.broadcast_arrays(np.asarray(speed, float), np.asarray(top_depth, float))
    _require(np.all(np.isfinite(u)) and np.all(np.isfinite(h)) and np.all(np.abs(np.abs(u)-.3) < 1e-10),
             'This depth sweep was measured at 0.30 m/s')
    _require(np.all((h >= 0) & (h <= .2)), 'Measured full-wet upper-edge depth is 0..0.20 m')
    _, law, _ = load_parameters()
    return -.5*1000.*width*height*law.cd(h, height)*np.abs(u)*u


def caplan_static_force(alpha, *, geometry_id='caplan2007_flat_big_blade',
                        speed=.75, area=.007742, upper_edge_depth=0.,
                        steady=True, spanwise_speed=0.):
    """Return measured-reference [drag, lift], not a global CAD force vector.

    Positive coefficients follow Caplan's conventions; dimensional area and
    velocity are fixed to this reference. The polar alone predicts no moment.
    """
    _require(geometry_id == 'caplan2007_flat_big_blade', 'Static polar geometry mismatch')
    _require(steady and spanwise_speed == 0, 'Only static planar incidence is measured')
    _require(np.isclose(speed, .75, rtol=0, atol=1e-10)
             and np.isclose(area, .007742, rtol=0, atol=1e-10)
             and upper_edge_depth == 0, 'Speed, area and immersion must match Caplan reference')
    alpha = np.asarray(alpha, float)
    _require(np.all(np.isfinite(alpha)), 'Finite incidence required')
    cd, cl = BladePolar.caplan_flat().coefficients(alpha)
    return .5*999.*area*speed**2*np.stack((cd, cl), axis=-1)


def reijtenbagh_start_force(time, kind, parameter, final_speed, *,
                            full_wet_fixed_depth, straight_normal_motion,
                            quiescent_start, free_surface_negligible,
                            area=.02, length=.1, rho=998., nu=.978e-6,
                            upper_edge_depth=.15):
    """Restricted normal-translation excess force, NOT a cyclic rowing kernel.

    Free-surface-negligible is an explicit modelling assumption to substantiate,
    not a universal numerical h/c threshold inferred from this paper. Caller
    must add steady force of the SAME geometry/conditions, not another inertia.
    Geometry here is deliberately restricted to the existing reference plate.
    """
    _require(full_wet_fixed_depth and straight_normal_motion and quiescent_start
             and free_surface_negligible, 'Starting-plate theory does not cover turning, ventilation or cyclic wake re-entry')
    _require(np.isclose(area, .02,rtol=0,atol=1e-10)
             and np.isclose(length, .1,rtol=0,atol=1e-10)
             and np.isclose(upper_edge_depth,.15,rtol=0,atol=1e-10),
             'Reference geometry and 0.15 m upper-edge depth only in this implementation')
    _require(np.isfinite(parameter) and np.isfinite(final_speed)
             and np.isclose(final_speed,.6,rtol=0,atol=1e-10),
             'This checked reference covers the 0.60 m/s nonuniform-start experiments')
    bounds={'constant_jerk':(.1,.8),'sqrt_velocity':(.16,.64)}
    _require(kind in bounds,'The checked reference covers constant-jerk and sqrt-velocity starts only')
    _require(bounds[kind][0]<=parameter<=bounds[kind][1],'Motion parameter outside compared experiment range')
    t = np.asarray(time, float)
    _require(t.ndim == 1 and len(t) >= 2 and np.all(np.isfinite(t))
             and t[0] == 0 and np.all(np.diff(t) > 0), 'Increasing time from quiescent t=0 required')
    _require(np.isfinite(rho) and rho > 0 and np.isfinite(nu) and nu > 0,
             'Positive finite density and viscosity required')
    return history_excess_force(t, kind, parameter, final_speed, area, length, rho, nu)
