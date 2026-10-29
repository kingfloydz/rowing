"""Fit a closed physical reset and fixed coaxial gripper mounts for the priors."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from rowing_mjlab.asset import build_spec
from rowing_mjlab.retarget import elbow_calibration
from hydrodynamics.models import ROOT


def main():
    geometry=json.loads((ROOT/'data/rig_geometry.json').read_text())
    if geometry.get('mirrored_effective_grip_installation'):
        raise RuntimeError('Use tools/symmetrize_grip_installation.py: independent mount fitting would undo mirrored feather installation.')
    out=ROOT/'renders/reference10_literature';backup=out/'previous_installation';backup.mkdir(exist_ok=True)
    for name in ('scene.xml','preview.xml','data/reset_joint_pose.json','data/reset_float_pose.json','data/rig_geometry.json'):
        dst=backup/Path(name).name
        if not dst.exists():shutil.copy2(ROOT/name,dst)
    a=np.load(ROOT/'data/reference20/trajectory.npz');names=list(a['names']);m=build_spec().compile();d=mujoco.MjData(m)
    qi=np.array([m.joint(n).qposadr[0] for n in names]);jids=[m.joint(n).id for n in names]
    cal=elbow_calibration(m);ei=[names.index(s+'_elbow_joint') for s in ('left','right')]
    bend=abs(a['q'][:,ei]-np.array([c['zero_rad'] for c in cal]))
    ids=np.where(np.max(bend,axis=1)<np.deg2rad(5.8))[0]
    if not len(ids):raise RuntimeError('No nearly extended reset phase in literature reference')
    seat=names.index('seat_slide');ids=sorted(ids,key=lambda i:abs(a['q'][i,seat]-m.qpos0[qi[seat]]))
    eq=[m.equality(s+'_'+p+'_fixed').id for p in ('hand','foot') for s in ('left','right')]
    pivot=np.array([.029,0,0]);bounds=m.jnt_range[jids].copy();bounds[:,0]+=.015;bounds[:,1]-=.015
    free=np.array([i for i,n in enumerate(names) if not n.startswith(('port_','starboard_')) and i not in ei])
    wrist=np.array([i for i,n in enumerate(names) if 'wrist_' in n])
    mount_before=[Rotation.from_quat(m.body_quat[m.eq_obj1id[e]][[1,2,3,0]]) for e in eq[:2]]
    pairs=[]
    for side,oar in (('left','starboard'),('right','port')):
        shaft=m.geom('oar_collision_'+oar+'_shaft').id
        knee=m.geom('h2_collision_'+side+'_knee_link_0').id
        for part in ('wrist_roll','wrist_pitch','wrist_yaw','elbow'):
            body=m.body(side+'_'+part+'_link').id
            for g in range(m.ngeom):
                if m.geom_bodyid[g]==body and m.geom_contype[g]:pairs.extend([(g,shaft),(g,knee)])
    solutions=[]
    for k in ids[::max(1,len(ids)//6)][:6]:
        initial=a['q'][k].copy()
        def pose(x):
            q=initial.copy();q[free]=x;d.qpos[:]=m.qpos0;d.qpos[qi]=q;mujoco.mj_kinematics(m,d)
            return d.xmat.reshape(-1,3,3)
        def closure(x):
            r=pose(x);res=[]
            for j,e in enumerate(eq):
                aa,b=m.eq_obj1id[e],m.eq_obj2id[e];rel=m.eq_data[e];rot=Rotation.from_quat(rel[6:10][[1,2,3,0]]).as_matrix()
                if j<2:
                    target=d.xpos[b]+r[b]@(rot.T@(pivot-rel[3:6]));res.extend(10*(d.xpos[aa]+r[aa]@pivot-target))
                else:
                    res.extend(10*(d.xpos[aa]+r[aa]@rel[3:6]-d.xpos[b]));res.extend(Rotation.from_matrix((r[aa]@rot).T@r[b]).as_rotvec())
            return np.array(res)
        def objective(x):
            q=initial.copy();q[free]=x
            angles=mount_changes(x)
            return np.sum((x-initial[free])**2)+.15*np.sum(q[wrist]**2)+4*np.sum(angles**2)
        def mount_changes(x):
            r=pose(x);angles=[]
            for old,e in zip(mount_before,eq[:2]):
                hand,oar=m.eq_obj1id[e],m.eq_obj2id[e];parent=m.body_parentid[hand]
                rel=Rotation.from_quat(m.eq_data[e,6:10][[1,2,3,0]])
                desired=Rotation.from_matrix(r[parent]).inv()*Rotation.from_matrix(r[oar])*rel.inv()
                angles.append((old.inv()*desired).magnitude())
            return np.array(angles)
        def clearance(x):
            pose(x);distances=[mujoco.mj_geomDistance(m,d,aa,b,.05,None)-.002 for aa,b in pairs]
            return np.r_[distances,np.deg2rad(60)-mount_changes(x)]
        fit=minimize(objective,np.clip(initial[free],bounds[free,0]+1e-6,bounds[free,1]-1e-6),method='SLSQP',bounds=list(zip(bounds[free,0],bounds[free,1])),constraints=[dict(type='eq',fun=closure),dict(type='ineq',fun=clearance)],options=dict(maxiter=350,ftol=1e-11,eps=1e-7))
        err=np.max(abs(closure(fit.x)));print('Reset phase',a['phase'][k],'closure',err,flush=True)
        if err<1e-7 and min(clearance(fit.x))>=-1e-7:
            q=initial.copy();q[free]=fit.x;solutions.append((fit.fun,k,q))
    if not solutions:raise RuntimeError('No physical reset closed')
    _,k,q=min(solutions,key=lambda x:x[0]);d.qpos[:]=m.qpos0;d.qpos[qi]=q;mujoco.mj_kinematics(m,d)
    records={}
    for side,e in zip(('left','right'),eq[:2]):
        hand,oar=m.eq_obj1id[e],m.eq_obj2id[e];parent=m.body_parentid[hand]
        oldpos=m.body_pos[hand].copy();oldrot=Rotation.from_quat(m.body_quat[hand][[1,2,3,0]])
        relrot=Rotation.from_quat(m.eq_data[e,6:10][[1,2,3,0]])
        desired=Rotation.from_matrix(d.xmat[parent].reshape(3,3)).inv()*Rotation.from_matrix(d.xmat[oar].reshape(3,3))*relrot.inv()
        m.body_pos[hand]=oldpos+oldrot.apply(pivot)-desired.apply(pivot)
        m.body_quat[hand]=desired.as_quat()[[3,0,1,2]]
        records[side]=dict(previous_position_m=oldpos.tolist(),previous_quaternion_wxyz=oldrot.as_quat()[[3,0,1,2]].tolist(),
            position_m=m.body_pos[hand].tolist(),quaternion_wxyz=m.body_quat[hand].tolist(),
            mount_rotation_change_deg=float(np.rad2deg((oldrot.inv()*desired).magnitude())),relpose=m.eq_data[e,3:10].tolist())
    mujoco.mj_forward(m,d)
    contacts=[dict(geoms=[m.geom(c.geom1).name,m.geom(c.geom2).name],depth_m=float(-c.dist)) for c in d.contact if c.dist<-.0001]
    error=max(abs(d.efc_pos[d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY]),default=0)
    report=dict(reset_phase=float(a['phase'][k]),reset_q=dict(zip(names,map(float,q))),grip=records,
        max_equality_error=float(error),contacts=contacts,note='Rigid wrist-to-gripper mount adjusted about grip center; sleeve remains coaxial and weld is constant. Reset elbows retain literature-derived near-extension angles.')
    (out/'reset_fit.json').write_text(json.dumps(report,indent=2)+'\n');np.savez(out/'reset_fit.npz',q=q,names=names)
    mujoco.mj_saveModel(m,str(out/'reset_fit.mjb'))
    print(json.dumps(report,indent=2))
    if error>1e-7 or contacts:raise RuntimeError('Reset candidate requires closure/contact repair before installation')


if __name__=='__main__':main()
