import numpy as np
import pytest
from hydrodynamics.prescribed_stroke import PrescribedTankStroke


def test_motion_derivatives_and_published_local_velocities():
    stroke=PrescribedTankStroke()
    t=np.linspace(-1,1,10001);th=2*np.arctan(np.exp(3*t));kin=stroke.kinematics(th)
    np.testing.assert_allclose(np.gradient(kin['position'],t,axis=0)[1:-1],kin['velocity'][1:-1],atol=1e-6)
    np.testing.assert_allclose(np.gradient(kin['velocity'],t,axis=0)[1:-1],kin['acceleration'][1:-1],atol=1e-5)
    middle=len(t)//2
    np.testing.assert_allclose(kin['velocity'][middle],[1.014,0],atol=1e-12)
    # Rigid-body acceleration must satisfy a dot outward radius = -Le*omega^2.
    outward=-np.column_stack((np.cos(th),np.sin(th)))
    np.testing.assert_allclose(np.sum(kin['acceleration']*outward,axis=1),-stroke.lever*kin['omega']**2,atol=1e-12)


def test_froude_similarity_force_and_moment_scaling():
    theta=np.deg2rad(np.arange(20.,161.,5.));s=2.3
    a=PrescribedTankStroke();b=PrescribedTankStroke(rotation_rate=3/np.sqrt(s),length_scale=s)
    fa=a.force(theta)['force'];fb=b.force(theta)['force']
    np.testing.assert_allclose(fb,fa*s**3,rtol=1e-12)
    np.testing.assert_allclose(b.moment_from_force(theta,fb),a.moment_from_force(theta,fa)*s**4,rtol=1e-12)


def test_empirical_support_and_force_sign():
    theta=np.deg2rad(np.arange(20.,161.,5.));stroke=PrescribedTankStroke()
    force=stroke.force(theta)['force'];kin=stroke.kinematics(theta)
    assert np.all(np.sum(force*kin['velocity'],axis=1)<=1e-12)
    assert np.all(force[:,0]<0)
    with pytest.raises(ValueError):PrescribedTankStroke(rotation_rate=2.).force(theta)
    with pytest.raises(ValueError):PrescribedTankStroke(advance_number=.99).force(theta)
    with pytest.raises(ValueError):stroke.kinematics([0.,np.pi/2])
