"""Calibrate a geometric left-over-right reference; no dynamics guarantee."""
import sys
import json
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import trimesh
from scipy.optimize import brentq, least_squares
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from hydrodynamics.models import ROOT
from rowing_mjlab.reset_pose import apply_reset_joints


def main():
    root = ET.parse(ROOT/'scene.xml').getroot()
    apply_reset_joints(root)
    root.find('compiler').set('meshdir', str(ROOT))
    floating = json.loads((ROOT/'data/reset_float_pose.json').read_text())
    boat = root.find("./worldbody/body[@name='boat']")
    boat.set('pos', ' '.join(map(str, floating['position_m'])))
    boat.set('quat', ' '.join(map(str, floating['quaternion_wxyz'])))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    source = np.loadtxt(ROOT/'data/rowing_reference.csv', delimiter=',', skiprows=1)
    spline = CubicSpline(source[:,0], source[:,1:], bc_type='periodic')
    sides = ('port', 'starboard')
    qa = [model.joint(s+'_'+j).qposadr[0] for s in sides for j in ('sweep','elevation','feather')]
    meshes = [trimesh.load_mesh(ROOT/f'boat_meshes/{s}_blade_3mm.stl').vertices for s in sides]
    bodies = [model.body(s+'_oar').id for s in sides]
    sites = [model.site(s+'_grasp').id for s in sides]
    water = floating['water_level_m']

    def pose(q):
        data.qpos[:] = model.qpos0
        data.qpos[qa] = q
        mujoco.mj_kinematics(model,data)

    def top(i):
        return (meshes[i]@data.xmat[bodies[i]].reshape(3,3).T+data.xpos[bodies[i]])[:,2].max()-water

    def bottom(i):
        return (meshes[i]@data.xmat[bodies[i]].reshape(3,3).T+data.xpos[bodies[i]])[:,2].min()-water

    phases = np.linspace(0,1,801)
    records = []
    for phase in phases:
        q = spline(phase).copy()
        pose(q)
        distance = np.linalg.norm(data.site_xpos[sites[0],:2]-data.site_xpos[sites[1],:2])
        # Smoothly widen the height gap before hands approach one another.
        u = np.clip((.30-distance)/.15,0,1)
        window = u*u*u*(10-15*u+6*u*u)
        a = np.deg2rad(.85)*window
        # Left hand up a, right hand down 4a (joint axes have opposite signs).
        q[[1,4]] += [4*a,a]
        # Common hand-height increase buries both blades in the drive interior.
        # Full strength on [.05,.35], quintic ramps outside it.
        u = min(np.clip((phase-.01)/.04,0,1),np.clip((.40-phase)/.05,0,1))
        drive = u*u*u*(10-15*u+6*u*u)
        def depth(c):
            qq=q.copy();qq[[1,4]] += [-c,c];pose(qq)
            return max(top(0),top(1))+.008
        common = brentq(depth,0,.16) if depth(0)>0 and drive>0 else 0.
        q[[1,4]] += [-common*drive,common*drive]
        pose(q)
        # Recovery is an interior interval, not entry/extraction transitions.
        u=min(np.clip((phase-.44)/.06,0,1),np.clip((.97-phase)/.07,0,1))
        recovery=u*u*u*(10-15*u+6*u*u)
        def clearance(c):
            qq=q.copy();qq[[1,4]] += [c,-c];pose(qq)
            return min(bottom(0),bottom(1))-.020
        lift=brentq(clearance,0,.16) if clearance(0)<0 and recovery>0 else 0.
        q[[1,4]] += [lift*recovery,-lift*recovery]
        pose(q)
        records.append([phase,*(q-spline(phase)),float(window),float(common*drive)])
    records=np.array(records);records[-1,1:]=records[0,1:]
    np.savetxt(ROOT/'data/oar_crossing_correction.csv',records[:,:7],delimiter=',',
               header='source_phase,port_sweep,port_elevation,port_feather,starboard_sweep,starboard_elevation,starboard_feather',comments='',fmt='%.12g')

    # Small arm-only IK restores both hand welds after the +/-5 mm gate move.
    data.qpos[:]=model.qpos0;mujoco.mj_forward(model,data)
    old=json.loads((ROOT/'data/reset_joint_pose.json').read_text())
    solution={}
    for side in ('left','right'):
        names=[side+'_'+j+'_joint' for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
        idx=[model.joint(n).qposadr[0] for n in names]
        joint_ids=[model.joint(n).id for n in names]
        initial=model.qpos0[idx].copy()
        eq=model.equality(side+'_hand_fixed').id
        mujoco.mj_kinematics(model,data)
        body1,body2 = model.eq_obj1id[eq],model.eq_obj2id[eq]
        rel=model.eq_data[eq]
        relrot=Rotation.from_quat(rel[6:10][[1,2,3,0]]).as_matrix()
        target_rot=data.xmat[body2].reshape(3,3)@relrot.T
        target_pos=data.xpos[body2]-target_rot@rel[3:6]
        def residual(x):
            data.qpos[idx]=x;mujoco.mj_kinematics(model,data)
            rotation=Rotation.from_matrix(target_rot.T@data.xmat[body1].reshape(3,3)).as_rotvec()
            return np.r_[(data.xpos[body1]-target_pos)*1000,rotation*1000,(x-initial)*.01]
        fit=least_squares(residual,initial,bounds=(model.jnt_range[joint_ids,0],model.jnt_range[joint_ids,1]),max_nfev=4000,xtol=1e-12,gtol=1e-12,ftol=1e-12)
        if np.max(abs(residual(fit.x)[:6]))>1e-4:raise RuntimeError(f'{side} reset IK failed: {residual(fit.x)}')
        solution.update(zip(names,map(float,fit.x)))
    old['joint_positions_rad'].update(solution)
    old['note']='Arm-only IK preserves hand welds after starboard/left-hand gate +5 mm and port/right-hand gate -5 mm. Other reset joints unchanged.'
    (ROOT/'data/reset_joint_pose.json').write_text(json.dumps(old,indent=2)+'\n')
    meta=dict(left_hand_gate='starboard',right_hand_gate='port',gate_height_change_m={'starboard':.005,'port':-.005},
              differential_left_right_ratio=[1,4],differential_peak_deg=[.85,3.4],
              common_depth_correction='Common hand-raising angle plus differential left-up/right-down correction; common term is separate from 1:4 ratio',
              drive_source_phase=[.05,.35],blade_top_margin_m=.008,
              recovery_source_phase=[.50,.90],recovery_bottom_clearance_m=.020,
              transitions_source_phase=[[.90,1.0],[0.0,.05],[.35,.50]],
              elevation_limits_deg={'port':[-15,35],'starboard':[-35,15]},
              reset_port_elevation_change_rad=-.01,
              source='Original literature CSV unchanged; additive correction sampled in unshifted source phase',
              limitation='Fixed reset hull geometric calibration; actual tracking and roll require dynamics validation')
    (ROOT/'data/oar_crossing.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
