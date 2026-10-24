from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
import mujoco
from hydrodynamics.models import (hydrostatic_wrench, normal_plate_force,
    projected_blade,plate_added_mass,load_parameters,wet_triangles)
from hydrodynamics.mujoco_adapter import RowingHydrodynamics,solve_normal_added_mass


def test_box_archimedes_and_moment():
    box=trimesh.creation.box(extents=[2,1,1])
    f,t,area=hydrostatic_wrench(box.triangles,[1,0,0])
    np.testing.assert_allclose(f,[0,0,9810],atol=1e-9)
    np.testing.assert_allclose(t,[0,9810,0],atol=1e-9)
    assert abs(area-5)<1e-12
    f,t,_=hydrostatic_wrench(box.triangles,[0,0,0],level=-1)
    np.testing.assert_array_equal(f,0);np.testing.assert_array_equal(t,0)


def test_rotated_submerged_box_no_horizontal_buoyancy():
    box=trimesh.creation.box(extents=[2,1,.5])
    box.apply_transform(trimesh.transformations.euler_matrix(.2,.3,.5))
    box.apply_translation([.2,-.4,-2])
    f,t,_=hydrostatic_wrench(box.triangles,[0,0,0])
    np.testing.assert_allclose(f,[0,0,9810],atol=1e-8)
    np.testing.assert_allclose(t,np.cross([.2,-.4,-2],f),atol=1e-8)


def test_plate_force_is_passive_and_obeys_reversal():
    rng=np.random.default_rng(20261024);v=rng.normal(size=(100,3));n=np.array([1.,0,0])
    f=normal_plate_force(v,n,.02,1.3)
    assert np.all(np.sum(f*v,axis=1)<=1e-12)
    np.testing.assert_allclose(normal_plate_force(-v,n,.02,1.3),-f)
    np.testing.assert_allclose(normal_plate_force(v,-n,.02,1.3),f)


def test_blades_mirror_area_and_dry_cutoff():
    a,ma=projected_blade(ROOT/'boat_meshes/port_blade_3mm.stl')
    b,mb=projected_blade(ROOT/'boat_meshes/starboard_blade_3mm.stl')
    assert abs(ma['area_m2']-mb['area_m2'])<1e-12
    assert abs(ma['area_m2']-.0755812923)<1e-8
    assert len(wet_triangles(a+np.array([0,0,1])))==0


def test_implicit_added_mass_matches_analytic_mass_sum():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body><freejoint/><geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    B=np.zeros((1,model.nv));B[0,0]=1
    force=np.zeros(model.nv);force[0]=10
    f,error=solve_normal_added_mass(model,data,force,B,np.array([3.]),np.array([0.]))
    assert abs(data.qacc[0]-10/(2+3))<1e-10
    assert abs(f[0]+6)<1e-10 and error<1e-10


def test_reference_coefficients_and_added_mass():
    hull,blade,_=load_parameters()
    assert 1.0<float(blade.cd(0,.1))<1.2
    assert 1.5<float(blade.cd(.02,.1))<1.7
    assert abs(plate_added_mass(.2,.1)-1.3194689145)<1e-10
    assert hull.magnitude(0,7.937,2.34)==0
    np.testing.assert_allclose(hull.magnitude(-2,7.937,2.34),hull.magnitude(2,7.937,2.34))


def test_scene_external_force_preserved_and_no_nan():
    model=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));data=mujoco.MjData(model)
    hydro=RowingHydrodynamics(model)
    j=model.joint('boat_free').id;qa=model.jnt_qposadr[j];va=model.jnt_dofadr[j]
    # Entire boat and blades out of the water: hydro loads must vanish.
    data.qpos[qa+2]=2;mujoco.mj_forward(model,data)
    external=np.zeros(model.nv);external[va]=5
    r=hydro.forward(data,external_qforce=external)
    np.testing.assert_allclose(data.qfrc_applied,external,atol=1e-10)
    assert r['added_mass_forces_N']==[]
    data.qpos[qa+2]=.1193943;data.qvel[va]=1;mujoco.mj_forward(model,data)
    for _ in range(5):
        r=hydro.step(data)
        assert r['added_mass_residual_N']<1e-4
    assert np.all(np.isfinite(data.qpos))
