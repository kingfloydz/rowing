"""Choose a nearly extended, outward right-elbow reset with exact grip closure."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from rowing_mjlab.asset import build_spec
from rowing_mjlab.retarget import elbow_calibration
from hydrodynamics.models import ROOT


def main():
    m=build_spec().compile();d=mujoco.MjData(m)
    names=['right_'+j+'_joint' for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
    joints=[m.joint(n).id for n in names];idx=m.jnt_qposadr[joints]
    initial=m.qpos0[idx].copy()
    zero=elbow_calibration(m)[1]['zero_rad']
    eq=m.equality('right_hand_fixed').id;a,b=m.eq_obj1id[eq],m.eq_obj2id[eq]
    mujoco.mj_kinematics(m,d)
    rel=m.eq_data[eq]
    target_r=d.xmat[b].reshape(3,3)@Rotation.from_quat(rel[6:10][[1,2,3,0]]).as_matrix().T
    target_p=d.xpos[b]-target_r@rel[3:6]
    torso=m.body('torso_link').id

    def geometry(x):
        d.qpos[idx]=x;mujoco.mj_kinematics(m,d)
        s,e,w=d.xanchor[np.array(joints)[[2,3,4]]].copy()
        chord=w-s;projection=s+chord*((e-s)@chord)/(chord@chord)
        outward=-((e-projection)@d.xmat[torso].reshape(3,3))[1]
        return outward

    def closure(x):
        geometry(x)
        return np.r_[d.xpos[a]-target_p,Rotation.from_matrix(target_r.T@d.xmat[a].reshape(3,3)).as_rotvec()]

    def objective(x):
        return (x[3]-zero)**2+1e-4*np.sum((x-initial)**2)

    before=dict(projected_bend_deg=float(np.rad2deg(initial[3]-zero)),outward_m=float(geometry(initial)))
    bounds=m.jnt_range[joints]
    constraints=[dict(type='eq',fun=closure),dict(type='ineq',fun=lambda x:geometry(x))]
    candidates=[];rng=np.random.default_rng(42)
    for k in range(14):
        start=initial if k==0 else np.clip(initial+rng.normal(0,.45,7),bounds[:,0]+1e-6,bounds[:,1]-1e-6)
        fit=minimize(objective,start,method='SLSQP',bounds=bounds,constraints=constraints,
                     options=dict(maxiter=600,ftol=1e-13))
        if np.max(abs(closure(fit.x)))<1e-8 and geometry(fit.x)>=-1e-9:candidates.append(fit)
    if not candidates:raise RuntimeError('No outward right-arm solution preserving the grip was found')
    fit=min(candidates,key=lambda f:f.fun)
    old=json.loads((ROOT/'data/reset_joint_pose.json').read_text())
    old['joint_positions_rad'].update(zip(names,map(float,fit.x)))
    old['right_arm_posture']='Minimum projected elbow bend on outward branch, with exact existing hand weld and unchanged oar pose.'
    (ROOT/'data/reset_joint_pose.json').write_text(json.dumps(old,indent=2)+'\n')
    report=dict(before=before,after=dict(projected_bend_deg=float(np.rad2deg(fit.x[3]-zero)),outward_m=float(geometry(fit.x))),
                max_grip_closure_error=float(np.max(abs(closure(fit.x)))),joint_positions_rad=dict(zip(names,map(float,fit.x))))
    out=ROOT/'renders/installation';out.mkdir(parents=True,exist_ok=True)
    (out/'right_arm_adjustment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
