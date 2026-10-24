from pathlib import Path
import sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hydrodynamics.polars import BladePolar
from hydrodynamics.unsteady import (history_integral, history_excess_force,
    prescribed_motion, rotating_added_mass_wrench)
from hydrodynamics.mujoco_adapter import RowingHydrodynamics


def test_abel_integral_against_three_analytic_motions():
    t=np.linspace(0,2,4001)
    cases=[(.7*t,1.4*np.sqrt(t)),(.2*t*t,(.4*4/3)*t**1.5),(.6*np.sqrt(t),np.full_like(t,.3*np.pi))]
    for v,exact in cases:
        numeric=history_integral(t,v)
        np.testing.assert_allclose(numeric[t>.1],exact[t>.1],rtol=.0005,atol=1e-6)


def test_history_has_relaxation_and_expected_viscosity_cancellation():
    t=np.array([.1,.5,1.,2.,100.,10000.])
    f=history_excess_force(t,'constant_acceleration',.8,.4)
    assert f[-1]>0 and f[-1]<f[-2]<f[-3]
    np.testing.assert_allclose(f,history_excess_force(t,'constant_acceleration',.8,.4,nu=4e-6),rtol=1e-12)
    assert abs(f[-2]/f[-1]-10)<.02


def test_rotating_mass_power_equals_fluid_energy_derivative():
    rng=np.random.default_rng(123)
    for _ in range(50):
        v,a,n,omega=rng.normal(size=(4,3));n/=np.linalg.norm(n);mass=3.1
        f,m=rotating_added_mass_wrench(v,a,n,omega,mass)
        edot=mass*(n@v)*(n@a+np.cross(omega,n)@v)
        np.testing.assert_allclose(f@v+m@omega,-edot,atol=1e-12)
        # A finite-difference energy check makes this independent of force algebra.
        dt=1e-6;ndot=np.cross(omega,n)
        ep=.5*mass*((n+dt*ndot)@(v+dt*a))**2
        em=.5*mass*((n-dt*ndot)@(v-dt*a))**2
        np.testing.assert_allclose((ep-em)/(2*dt),edot,rtol=1e-6,atol=1e-8)


def test_polar_passivity_and_normal_orientation_invariance():
    polar=BladePolar.caplan_flat();rng=np.random.default_rng(77)
    v=rng.normal(size=(200,3));n=np.array([1.,0,0]);c=np.array([0.,1,0])
    force=polar.force(v,n,c,.02)
    assert np.all(np.einsum('ij,ij->i',force,v)<=1e-10)
    np.testing.assert_allclose(force,polar.force(v,-n,c,.02),atol=1e-12)
    np.testing.assert_array_equal(polar.force(np.zeros((2,3)),n,c,.02),0)


def test_extended_scene_options_and_no_added_mass_double_count():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));d=mujoco.MjData(m)
    h=RowingHydrodynamics(m,blade_model='caplan_flat',added_mass_rotation=True)
    j=m.joint('boat_free').id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j]
    d.qpos[qa+2]=.1193943;d.qvel[va]=1;d.qvel[va+5]=.15
    result=h.forward(d,added_mass=False)
    assert not any(x['label']=='blade_added_mass_rotation' for x in result['loads'])
    for _ in range(5):
        result=h.step(d)
        assert result['added_mass_residual_N']<1e-4
    assert any(x['label']=='blade_caplan_static_polar' for x in result['loads'])
    assert np.all(np.isfinite(d.qpos))
