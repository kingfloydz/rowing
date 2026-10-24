"""Physics and extrapolation tests for the opt-in training model."""
import json
from pathlib import Path
import numpy as np
import pytest
import mujoco
from hydrodynamics.models import ROOT
from hydrodynamics.training import (TrainingParameters,TrainingHullResistance,
    sample_episode,skin_friction,blade_traction,fin_force)
from hydrodynamics.training_adapter import TrainingRowingHydrodynamics


def test_hull_zero_reversal_and_no_endpoint_clipping():
    law=TrainingHullResistance()
    u=np.r_[0.,np.geomspace(1e-9,5,300)]
    r=law.magnitude(u,5.87765685,2.3670595)
    assert r[0]==0 and np.all(np.isfinite(r)) and np.all(r[1:]>0)
    np.testing.assert_array_equal(r,law.magnitude(-u,5.87765685,2.3670595))
    assert law.magnitude(5,5.87765685,2.3670595)>law.magnitude(4,5.87765685,2.3670595)
    assert law.components(4,5.87765685,2.3670595)['outside_reference']
    assert not law.components(3,5.87765685,2.3670595)['outside_reference']
    assert law.components(4.1,5.87765685,2.3670595)['outside_target_speed']


def test_residual_joins_are_c1_and_reference_nodes_preserved():
    law=TrainingHullResistance()
    np.testing.assert_allclose(law.residual(law.fn),law.cr,rtol=1e-12)
    for edge in law.fn[[0,-1]]:
        h=1e-7
        left=(law.residual(edge)-law.residual(edge-h))/h
        right=(law.residual(edge+h)-law.residual(edge))/h
        np.testing.assert_allclose(left,right,rtol=.002,atol=2e-8)
    assert law.residual(0)==0
    re=np.array([1e6,1e7])
    np.testing.assert_allclose(skin_friction(re),.075/(np.log10(re)-2)**2)


def test_episode_sampling_is_repeatable_and_immutable():
    from dataclasses import FrozenInstanceError
    assert sample_episode(1)==sample_episode(1)
    assert sample_episode(1)!=sample_episode(2)
    with pytest.raises(FrozenInstanceError):sample_episode(1).residual_scale=2
    for i in range(30):
        law=TrainingHullResistance(sample_episode(i))
        f=law.magnitude(np.linspace(0,4,201),5.87765685,2.3670595)
        assert np.all(np.isfinite(f)) and np.all(f>=0)


@pytest.mark.parametrize('mode',['normal','harmonic'])
def test_blade_passivity_symmetry_and_finite_zero(mode):
    rng=np.random.default_rng(17);v=rng.normal(size=(400,3))*4
    n=np.array([1.,0,0]);c=np.array([0.,1.,0])
    for p in [TrainingParameters(),sample_episode(23)]:
        f=blade_traction(v,n,c,.02,p,mode=mode)
        assert np.max(np.sum(f*v,axis=1))<=1e-9
        np.testing.assert_allclose(blade_traction(-v,n,c,.02,p,mode=mode),-f,atol=1e-10)
        np.testing.assert_allclose(blade_traction(v,-n,c,.02,p,mode=mode),f,atol=1e-10)
        np.testing.assert_array_equal(blade_traction(np.zeros((1,3)),n,c,.02,p,mode=mode),0)


def test_fin_lift_passivity_small_angle_and_reverse_flow():
    p=TrainingParameters();c=np.array([1.,0,0]);n=np.array([0.,1,0])
    for alpha in np.linspace(-np.pi,np.pi,361):
        v=4*np.array([np.cos(alpha),np.sin(alpha),0])
        f=fin_force(v,c,n,.03,2,p)
        assert f@v<=1e-9
        np.testing.assert_allclose(fin_force(-v,c,n,.03,2,p),-f,atol=1e-10)
    assert fin_force(np.array([3.,.05,0]),c,n,.03,2,p)[1]<0
    np.testing.assert_array_equal(fin_force(np.zeros(3),c,n,.03,2,p),0)


@pytest.fixture(scope='module')
def scene():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    return m,TrainingRowingHydrodynamics(m)


def test_scene_generalized_power_and_dry_external_force(scene):
    m,h=scene;d=mujoco.MjData(m);j=m.joint('boat_free').id
    qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j]
    d.qpos[qa+2]=.1193943
    d.qvel[va:va+6]=[3.,.2,.1,.3,-.2,.15]
    mujoco.mj_forward(m,d);q,loads,_=h.force_terms(d)
    dissipative=np.zeros(m.nv)
    for a in loads:
        if 'buoyancy' not in a['label']:
            mujoco.mj_applyFT(m,d,a['force_N'],a['torque_Nm'],a['point_m'],a['body'],dissipative)
    assert dissipative@d.qvel<=1e-8
    np.testing.assert_allclose(dissipative@d.qvel,h.last_diagnostics['relative_dissipative_power_W'],rtol=1e-10)
    parameters=h.training_parameters
    h.force_terms(d);assert parameters is h.training_parameters
    d.qpos[qa+2]=3;mujoco.mj_forward(m,d)
    ext=np.zeros(m.nv);ext[va]=7
    h.forward(d,ext)
    np.testing.assert_allclose(d.qfrc_applied,ext,atol=1e-9)
    with pytest.raises(ValueError):h.forward(d,added_mass=True)


def test_short_scene_integration_at_rest_and_four_m_s(scene):
    m,h=scene;j=m.joint('boat_free').id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j]
    for speed in [0.,4.]:
        d=mujoco.MjData(m);d.qpos[qa+2]=.1193943;d.qvel[va]=speed
        for _ in range(8):h.step(d)
        assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))


def test_invalid_training_inputs_rejected():
    with pytest.raises(ValueError):TrainingParameters(residual_scale=-1)
    with pytest.raises(ValueError):TrainingParameters(high_fn_power=9)
    with pytest.raises(ValueError):TrainingHullResistance().magnitude(2,0,2)
