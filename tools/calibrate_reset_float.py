"""Offline heave/trim equilibrium with zero roll and unchanged internal pose."""
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from scipy.optimize import root
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hydrodynamics.models import ROOT, load_closed_surface, hydrostatic_wrench
from rowing_mjlab.reset_pose import apply_reset_joints


def calibrate(model=None, qpos=None):
    if model is None:
        scene=ET.parse(ROOT/'scene.xml').getroot()
        apply_reset_joints(scene)
        scene.find('compiler').set('meshdir',str(ROOT))
        model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
    data=mujoco.MjData(model)
    if qpos is not None:data.qpos[:]=qpos
    # Express the geometry in the hull frame before solving world heave/trim.
    data.qpos[:7]=[0,0,0,1,0,0,0]
    mujoco.mj_kinematics(model,data)
    mass=model.body_mass.sum()
    cg=(model.body_mass[:,None]*data.xipos).sum(axis=0)/mass
    level=json.loads((ROOT/'data/rowing_reference_provenance.json').read_text())['water_level_m_at_repository_reset']
    surfaces=[]
    for mesh,body in [('shell_white','boat'),('fin','boat'),
                      ('port_blade_3mm','port_oar'),('starboard_blade_3mm','starboard_oar')]:
        triangles,_=load_closed_surface(ROOT/'boat_meshes'/f'{mesh}.stl')
        b=model.body(body).id
        surfaces.append(triangles@data.xmat[b].reshape(3,3).T+data.xpos[b])
    def wrench(state):
        z,pitch=state;rot=Rotation.from_rotvec([0,pitch,0]).as_matrix()
        offset=np.array([0,0,z]);center=rot@cg+offset
        loads=[hydrostatic_wrench(t@rot.T+offset,center,level) for t in surfaces]
        return np.sum([v[0] for v in loads],axis=0),np.sum([v[1] for v in loads],axis=0)
    def residual(state):
        force,torque=wrench(state)
        return np.array([force[2]-mass*9.81,torque[1]])
    sol=root(residual,[-.004,0.])
    if not sol.success or np.max(np.abs(residual(sol.x)))>1e-5:
        raise RuntimeError(f'Float calibration failed: {sol.message}')
    z,pitch=sol.x
    if abs(z)>.02 or abs(pitch)>np.deg2rad(2):
        raise ValueError('Float correction exceeds small-reset adjustment bounds')
    force,torque=wrench(sol.x)
    return dict(position_m=[0.,0.,float(z)],quaternion_wxyz=[float(np.cos(pitch/2)),0.,float(np.sin(pitch/2)),0.],
                pitch_deg=float(np.rad2deg(pitch)),water_level_m=level,mass_kg=float(mass),
                buoyancy_N=float(force[2]),weight_N=float(mass*9.81),
                moment_about_system_cg_Nm=torque.tolist(),
                note='Heave and pitch balanced; roll/yaw zero; internal joints follow reset_joint_pose.json. Residual roll requires active control. Recalibrate after mass/geometry changes.')


if __name__=='__main__':
    result=calibrate()
    (ROOT/'data/reset_float_pose.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
