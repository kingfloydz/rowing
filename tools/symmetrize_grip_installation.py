"""Mirror wrist-to-oar installation, then solve reset with mounts held fixed."""
import sys, types, json
from pathlib import Path
sys.modules['tensorboard.compat.notf']=types.ModuleType('tensorboard.compat.notf')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco, numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, minimize
from rowing_mjlab.asset import build_spec
from hydrodynamics.models import ROOT


def main():
    m=build_spec().compile();d=mujoco.MjData(m);pivot=np.array([.029,0,0]);S=np.diag([1,-1,1])
    eq=[m.equality(s+'_'+p+'_fixed').id for p in ('hand','foot') for s in ('left','right')]
    rotations=[];before=[]
    for i,e in enumerate(eq[:2]):
        h=m.eq_obj1id[e];mount=R.from_quat(m.body_quat[h][[1,2,3,0]]);rel=R.from_quat(m.eq_data[e,6:10][[1,2,3,0]])
        effective=mount*rel
        rotations.append(effective if i==0 else R.from_matrix(S@effective.as_matrix()@S))
        before.append((m.body_pos[h].copy(),mount,rel))
    delta=(rotations[0].inv()*rotations[1]).as_rotvec()
    common=rotations[0]*R.from_rotvec(delta/2);records={}
    for i,(s,e) in enumerate(zip(('left','right'),eq[:2])):
        h=m.eq_obj1id[e];pos,old,rel=before[i]
        effective=common if i==0 else R.from_matrix(S@common.as_matrix()@S)
        desired=effective*rel.inv()
        m.body_pos[h]=pos+old.apply(pivot)-desired.apply(pivot);m.body_quat[h]=desired.as_quat()[[3,0,1,2]]
        records[s]=dict(previous_position_m=pos.tolist(),previous_quaternion_wxyz=old.as_quat()[[3,0,1,2]].tolist(),position_m=m.body_pos[h].tolist(),quaternion_wxyz=m.body_quat[h].tolist(),mount_rotation_change_deg=float(np.rad2deg((old.inv()*desired).magnitude())),relpose=m.eq_data[e,3:10].tolist())
    names=list(np.load(ROOT/'data/reference20/trajectory.npz')['names']);qi=np.array([m.joint(n).qposadr[0] for n in names]);q0=m.qpos0[qi].copy()
    free=np.array([i for i,n in enumerate(names) if not n.startswith(('port_','starboard_')) or n.endswith('_feather') ])
    bounds=np.array([m.joint(names[i]).range for i in free]);bounds[:,0]+=.005;bounds[:,1]-=.005
    for j,i in enumerate(free):
        if 'elbow' in names[i]:bounds[j]=[1.23,1.34]
        if names[i] in ('waist_yaw_joint','waist_roll_joint'):bounds[j]=[-.04,.04]
    initial_common=common
    def set_mounts(v):
        common=initial_common*R.from_rotvec(v)
        for i,e in enumerate(eq[:2]):
            h=m.eq_obj1id[e];pos,old,rel=before[i]
            effective=common if i==0 else R.from_matrix(S@common.as_matrix()@S)
            desired=effective*rel.inv()
            m.body_pos[h]=pos+old.apply(pivot)-desired.apply(pivot);m.body_quat[h]=desired.as_quat()[[3,0,1,2]]
    bounds=np.vstack([bounds,[[-3.14,3.14]]*3]);initial=np.r_[q0[free],np.zeros(3)]
    feather=[list(free).index(names.index(s+'_feather')) for s in ('port','starboard')]
    initial[feather]=0
    def closure(x):
        set_mounts(x[-3:])
        d.qpos[:]=m.qpos0;d.qpos[qi[free]]=x[:-3];mujoco.mj_forward(m,d)
        return np.r_[d.efc_pos[d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY],x[feather[0]]-x[feather[1]]]
    def clearance(x):
        closure(x)
        return min([float(c.dist) for c in d.contact]+[.002])-.00001
    def residual(x):return np.r_[100*closure(x),.01*(x-initial),100*min(0,clearance(x))]
    fit=least_squares(residual,np.clip(initial,bounds[:,0],bounds[:,1]),bounds=(bounds[:,0],bounds[:,1]),max_nfev=800,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    fit=minimize(lambda x:np.sum((x-initial)**2),fit.x,method='SLSQP',bounds=list(zip(*bounds.T)),constraints=[dict(type='eq',fun=closure),dict(type='ineq',fun=clearance)],options=dict(maxiter=500,ftol=1e-12))
    fit=least_squares(closure,fit.x,bounds=(bounds[:,0],bounds[:,1]),max_nfev=200,ftol=1e-14,xtol=1e-14,gtol=1e-14)
    error=float(max(abs(closure(fit.x))));q=d.qpos[qi].copy()
    contacts=[dict(geoms=[m.geom(c.geom1).name,m.geom(c.geom2).name],depth_m=float(-c.dist)) for c in d.contact if c.dist<-.0001]
    print('closure',error,'contacts',contacts,flush=True)
    if error>1e-7 or contacts:raise RuntimeError('Fixed symmetric mount reset did not pass closure/contact check')
    for s,e in zip(('left','right'),eq[:2]):
        h=m.eq_obj1id[e];records[s]['position_m']=m.body_pos[h].tolist();records[s]['quaternion_wxyz']=m.body_quat[h].tolist()
        old=R.from_quat(np.array(records[s]['previous_quaternion_wxyz'])[[1,2,3,0]])
        records[s]['mount_rotation_change_deg']=float(np.rad2deg((old.inv()*R.from_quat(m.body_quat[h][[1,2,3,0]])).magnitude()))
    out=ROOT/'renders/reference10_literature'
    report=dict(reset_phase=json.loads((ROOT/'data/reference20/metadata.json').read_text())['reset_phase'],reset_q=dict(zip(names,map(float,q))),grip=records,max_equality_error=error,contacts=contacts,note='Mirrored effective wrist-to-oar installation held fixed during reset IK; grip center and ten reference channels retained.',mirrored_installation_before_deg=float(np.rad2deg(np.linalg.norm(delta))),mirrored_installation_after_deg=0.)
    (out/'reset_fit.json').write_text(json.dumps(report,indent=2)+'\n');np.savez(out/'reset_fit.npz',q=q,names=names);mujoco.mj_saveModel(m,str(out/'reset_fit.mjb'))
    from tools.install_literature_reset import main as install
    install()
    path=ROOT/'data/rig_geometry.json';geometry=json.loads(path.read_text());geometry['mirrored_effective_grip_installation']=True;path.write_text(json.dumps(geometry,indent=2)+'\n')

if __name__=='__main__':main()
