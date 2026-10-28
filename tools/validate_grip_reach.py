"""Offline bilateral arm IK at prescribed seat/trunk/oar references.

Tests geometric reach only, not collisions, leg closure or dynamic tracking.
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from rowing_mjlab.asset import build_spec
from rowing_mjlab.reference import RowingReference, IMITATION_NAMES
from rowing_mjlab.trunk import TrunkReference, trunk_vertical_angle
from hydrodynamics.models import ROOT


def validate(samples=101, reset_feather=False):
    m=build_spec().compile();d=mujoco.MjData(m)
    ref=RowingReference('cpu',m);trunk=TrunkReference('cpu')
    ids=[m.joint(n).qposadr[0] for n in IMITATION_NAMES]
    tj=[m.joint(n).id for n in ('seat_pelvis_hinge','waist_pitch_joint')]
    tq=m.jnt_qposadr[tj];base=m.qpos0[tq].copy()
    torso=m.body('torso_link').id;boat=m.body('boat').id
    previous={};records=[];poses=[]
    for phase in np.linspace(0,1,samples):
        d.qpos[:]=m.qpos0
        d.qpos[ids]=ref(torch.tensor([phase],dtype=torch.float64)).numpy()[0]
        target=float(trunk(torch.tensor(phase,dtype=torch.float64)))
        def tr(x):
            d.qpos[tq]=x;mujoco.mj_kinematics(m,d)
            actual=float(trunk_vertical_angle(torch.tensor(d.xmat[torso].reshape(3,3)),torch.tensor(d.xmat[boat].reshape(3,3))))
            return np.r_[100*(actual-target),.0001*(x-base)]
        fit=least_squares(tr,base,bounds=m.jnt_range[tj].T)
        tr(fit.x)
        if reset_feather:
            d.qpos[:]=m.qpos0
            for gate in ('port','starboard'):
                d.qpos[m.joint(gate+'_feather').qposadr[0]]=np.pi/2*phase
            mujoco.mj_kinematics(m,d)
        row={'phase':float(phase)}
        for side in ('left','right'):
            names=[side+'_'+n+'_joint' for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
            ji=[m.joint(n).id for n in names];qi=m.jnt_qposadr[ji]
            eq=m.equality(side+'_hand_fixed').id;a,b=m.eq_obj1id[eq],m.eq_obj2id[eq]
            rel=m.eq_data[eq];rr=Rotation.from_quat(rel[6:10][[1,2,3,0]]).as_matrix()
            target_r=d.xmat[b].reshape(3,3)@rr.T;target_p=d.xpos[b]-target_r@rel[3:6]
            start=previous.get(side,m.qpos0[qi]).copy()
            def residual(x):
                d.qpos[qi]=x;mujoco.mj_kinematics(m,d)
                return np.r_[(d.xpos[a]-target_p)*10,Rotation.from_matrix(target_r.T@d.xmat[a].reshape(3,3)).as_rotvec(),.00001*(x-start)]
            bounds=m.jnt_range[ji].T
            fit=least_squares(residual,start,bounds=bounds,max_nfev=300,ftol=1e-10,gtol=1e-10,xtol=1e-10)
            rng=np.random.default_rng(42)
            for attempt in range(6):
                if np.linalg.norm(residual(fit.x)[:6])<1e-5:break
                seed=np.clip(m.qpos0[qi]+rng.normal(0,1.,7),bounds[0]+1e-6,bounds[1]-1e-6)
                candidate=least_squares(residual,seed,bounds=bounds,max_nfev=200)
                if np.linalg.norm(residual(candidate.x))<np.linalg.norm(residual(fit.x)):fit=candidate
            err=residual(fit.x)
            previous[side]=fit.x.copy()
            row[side+'_position_error_m']=float(np.linalg.norm(err[:3])/10)
            row[side+'_orientation_error_deg']=float(np.rad2deg(np.linalg.norm(err[3:6])))
            row[side+'_joint_margin_deg']=float(np.rad2deg(np.min(np.minimum(fit.x-bounds[0],bounds[1]-fit.x))))
        records.append(row);poses.append(d.qpos.copy())
    report={key:max(r[key] for r in records) for key in records[0] if 'error' in key}
    report.update({key:min(r[key] for r in records) for key in records[0] if 'margin' in key})
    report['samples']=samples
    report['passed']=all(report[s+'_position_error_m']<1e-4 and report[s+'_orientation_error_deg']<.1 for s in ('left','right'))
    report['scope']=('Reset body with feather 0..90 deg' if reset_feather else 'Arms only; prescribed seat/trunk/oars; elbow priors allowed to adapt; no collision, leg or dynamics certification')
    out=ROOT/'renders'/('reset_feather_reach' if reset_feather else 'grip_reach');out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(dict(summary=report,phases=records),indent=2))
    np.savez(out/'poses.npz',phase=np.linspace(0,1,samples),qpos=poses)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=101)
    parser.add_argument('--reset-feather',action='store_true')
    args=parser.parse_args();result=validate(args.samples,args.reset_feather)
    sys.exit(0 if result['passed'] else 1)
