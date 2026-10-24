"""Independent mechanics, analytical radiation, and domain rejection checks."""
import numpy as np
import pytest
import mujoco
from scipy.spatial.transform import Rotation
from hydrodynamics.core import (UnboundedPotentialInertia, point_twist_map,
    shift_wrench, cummins_kernel)
from hydrodynamics.reference_models import (validate_components, OutsideReferenceDomain,
    DayReferenceResistance, grift_steady_normal_force, caplan_static_force,
    reijtenbagh_start_force)
from hydrodynamics.mujoco_adapter import couple_unbounded_inertia


def test_full_inertia_energy_and_origin_covariance():
    rng=np.random.default_rng(441)
    for _ in range(20):
        a=rng.normal(size=(6,6));m=a.T@a
        inertia=UnboundedPotentialInertia(m,'synthetic_tensor','mechanics test')
        nu,derivative=rng.normal(size=(2,6));r=rng.normal(size=3)
        force=inertia.wrench(nu,derivative)
        dt=1e-6
        edot=(inertia.energy(nu+dt*derivative)-inertia.energy(nu-dt*derivative))/(2*dt)
        np.testing.assert_allclose(force@nu,-edot,rtol=2e-8,atol=2e-8)
        x=point_twist_map(r)
        shifted=inertia.about_new_origin(r)
        new_nu=np.linalg.solve(x,nu);new_derivative=np.linalg.solve(x,derivative)
        np.testing.assert_allclose(shifted.energy(new_nu),inertia.energy(nu),rtol=1e-12)
        np.testing.assert_allclose(shifted.wrench(new_nu,new_derivative),shift_wrench(force,r),atol=1e-11)


def test_inertia_rejects_negative_energy_and_nonreciprocal_coefficients():
    bad=np.eye(6);bad[0,0]=-1
    with pytest.raises(ValueError,match='negative'):
        UnboundedPotentialInertia(bad,'bad','test')
    bad=np.eye(6);bad[0,3]=.2
    with pytest.raises(ValueError,match='reciprocity'):
        UnboundedPotentialInertia(bad,'bad','test')


def test_rotating_frame_derivative_recovers_sphere_world_force():
    inertia=UnboundedPotentialInertia.sphere(.2)
    r=Rotation.from_rotvec([.3,-.5,.2]).as_matrix()
    v=np.array([1.,-.4,.8]);a=np.array([.7,.1,-.6]);om=np.array([.6,.8,-.2])
    nu=np.r_[r.T@v,om]
    derivative=np.r_[r.T@a-np.cross(om,nu[:3]),[.3,-.2,.1]]
    wrench=inertia.wrench(nu,derivative)
    np.testing.assert_allclose(r@wrench[:3],-inertia.matrix[0,0]*a,atol=1e-12)
    np.testing.assert_allclose(wrench[3:],0,atol=1e-12)


def test_coupled_tensor_matches_independent_rigid_plus_fluid_mass_solve():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
    <worldbody><body><freejoint/><geom type="sphere" size=".2" mass="2"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    a=np.eye(6);a[0,4]=.7;a[2,3]=-.4
    inertia=UnboundedPotentialInertia(a.T@a,'synthetic_coupled','independent matrix test')
    rigid=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,rigid)
    applied=np.array([10.,2.,-3.,.7,-.8,.1])
    expected=np.linalg.solve(rigid+inertia.matrix,applied)
    result=couple_unbounded_inertia(m,d,applied,1,d.xpos[1],inertia)
    np.testing.assert_allclose(d.qacc,expected,rtol=1e-10,atol=1e-10)
    assert result['residual_SI']<1e-10


def test_cummins_transform_matches_analytic_sign_changing_passive_kernel():
    # Positive-real Z(s)=b*s/(s^2+2*lambda*s+omega0^2), with B(0)=0.
    # Its kernel changes sign and has zero integral. This analytical reference
    # checks the transform, not a particular hull's hydrodynamic coefficients.
    w=np.linspace(0,400,40001);t=np.linspace(.03,4,70)
    lam=.7;om=3.;b=2.
    damping=2*b*lam*w*w/((om*om-w*w)**2+4*lam*lam*w*w)
    calculated=cummins_kernel(w,damping[:,None,None],t)[:,0,0]
    wd=np.sqrt(om*om-lam*lam)
    expected=b*np.exp(-lam*t)*(np.cos(wd*t)-lam/wd*np.sin(wd*t))
    np.testing.assert_allclose(calculated,expected,atol=3e-4)
    assert calculated.min()<-.5


def test_radiation_does_not_accept_steady_drag_as_radiation_damping():
    with pytest.raises(ValueError,match='B\\(0\\)=0'):
        cummins_kernel(np.linspace(0,5,10),np.ones((10,1,1)),np.array([0.,1.]))


@pytest.mark.parametrize('components',[
    ['caplan_static','grift_depth'],['barre_dynamic','potential_inertia'],
    ['barre_dynamic','history_excess'],['history_excess','potential_inertia'],
    ['day_total_resistance','full_wave_resistance'],['linear_radiation','potential_inertia'],
    ['hydrostatics','fitted_wake'],['hydrostatics','hydrostatics'],
])
def test_force_double_counting_and_research_only_closures_are_rejected(components):
    with pytest.raises(ValueError):validate_components(components)


def test_reference_domains_reject_cad_transfer_partial_wetting_and_rotation():
    hull=DayReferenceResistance()
    assert hull.force(0,geometry_id='day2011_reference_hull',wetted_area=2.34)==0
    assert hull.force(3,geometry_id='day2011_reference_hull',wetted_area=2.34)<0
    for speed,geometry in [(5.,'day2011_reference_hull'),(3.,'rowing_CAD')]:
        with pytest.raises(OutsideReferenceDomain):
            hull.force(speed,geometry_id=geometry,wetted_area=2.34)
    assert grift_steady_normal_force(.3,.1)<0
    with pytest.raises(OutsideReferenceDomain):grift_steady_normal_force(.3,-.01)
    with pytest.raises(OutsideReferenceDomain):caplan_static_force(.5,steady=False)
    with pytest.raises(OutsideReferenceDomain):
        reijtenbagh_start_force(np.linspace(0,1,10),'constant_acceleration',.8,.4,
            full_wet_fixed_depth=True,straight_normal_motion=False,
            quiescent_start=True,free_surface_negligible=True)
