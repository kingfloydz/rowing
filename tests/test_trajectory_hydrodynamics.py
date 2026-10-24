"""Physical identities and independent quadrature checks for new closures."""
import numpy as np
from scipy.integrate import quad
from hydrodynamics.polars import BladePolar
from hydrodynamics.trajectory import (BladeTrajectoryParameters,blade_trajectory_force,
    WakeMomentumParameters,wake_momentum_force,passive_surge_memory)
from hydrodynamics.waves import (steady_coefficient,mean_wave_correction,
                                quasi_static_mean_correction)


def test_no_lag_no_inertia_equals_existing_static_polar():
    t=np.linspace(0,2,101);angle=.3*np.sin(t)
    v=np.column_stack((np.cos(t),np.sin(2*t)))
    n=np.column_stack((np.cos(angle),np.sin(angle)))
    c=np.column_stack((-np.sin(angle),np.cos(angle)))
    polar=BladePolar.caplan_flat()
    a=blade_trajectory_force(t,v,np.zeros_like(v),angle,.3*np.cos(t),polar,.02,.275,.02/.275,
                             BladeTrajectoryParameters(1,1,0,0))['force']
    expected=np.array([polar.force(vi,ni,ci,.02) for vi,ni,ci in zip(v,n,c)])
    np.testing.assert_allclose(a,expected,atol=1e-12)


def test_wake_energy_balance_and_timestep_convergence():
    polar=BladePolar.caplan_flat();t=np.linspace(0,2,801)
    v=np.column_stack((np.sin(np.pi*t)**2,np.zeros_like(t)))
    p=WakeMomentumParameters(2,1,1,.7,0)
    args=(t,v,np.zeros_like(v),np.zeros_like(t),np.zeros_like(t),polar,.02,.275,.02/.275,p)
    a=wake_momentum_force(*args,substeps=2);b=wake_momentum_force(*args,substeps=4)
    np.testing.assert_allclose(a['force'],b['force'],atol=2e-5)
    f=b['quasi_steady_relative_force'];w=b['wake_velocity'];rel=v-w
    relative_dissipation=-np.sum(f*rel,axis=1)
    assert relative_dissipation.min()>-1e-10
    external_work=np.trapezoid(-np.sum(f*v,axis=1),t)
    accounted=b['wake_energy'][-1]+np.trapezoid(relative_dissipation+b['wake_dissipation'],t)
    assert abs(external_work-accounted)<2e-4
    # The wake can return stored energy locally while the complete model is passive.
    assert np.any(np.sum(f*v,axis=1)>0)


def test_radiation_memory_energy_identity():
    t=np.linspace(0,10,10001);v=np.sin(1.7*t);acc=1.7*np.cos(1.7*t)
    a=passive_surge_memory(t,v,acc,2.,[.8,1.2],[.6,2.])
    work=np.trapezoid(-a['force']*v,t)
    expected=a['stored_energy'][-1]-a['stored_energy'][0]+np.trapezoid(a['dissipation'],t)
    assert abs(work-expected)<2e-5


def test_wave_steady_bessel_against_independent_integral():
    for f in [.3,.57,.9]:
        integral=quad(lambda z:2*(1+z*z)**1.5*np.exp(-(1+z*z)/f**2),0,12,epsabs=1e-11)[0]/f**8
        np.testing.assert_allclose(steady_coefficient(f),integral,rtol=1e-8)


def test_wave_low_frequency_limit_and_phase_integral():
    for f in [.3,.57]:
        phase=np.linspace(0,2*np.pi,10001);eps=.001
        factor=1+eps*np.sin(phase)
        correction=(np.trapezoid(steady_coefficient(f*factor)*factor**2,phase)/(2*np.pi)-steady_coefficient(f))/eps**2
        np.testing.assert_allclose(quasi_static_mean_correction(f),correction,rtol=3e-5)
        np.testing.assert_allclose(mean_wave_correction(f,.005),correction,rtol=2e-4)
