"""Retarget motion priors to the current H2 with exact saved attachment geometry.

Writes a separate candidate and a validity report; does not activate training.
All six oar coordinates are fixed. Human elbow zero is calibrated using H2
maximum extension; seat motion is a soft shape prior, not human metre transfer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import mujoco
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'data/reference9'
QA = ROOT/'results/reference9'
OARS = [s+'_'+j for s in ('port','starboard') for j in ('sweep','elevation','feather')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=80)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--restarts', type=int, default=5)
    args = parser.parse_args()
    m = mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    joints = [j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
              and m.joint(j).name not in OARS and not m.joint(j).name.startswith('head_')]
    names = [m.joint(j).name for j in joints]
    qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    oqa = np.array([m.joint(n).qposadr[0] for n in OARS])
    lo, hi = m.jnt_range[joints].T.copy()
    lo += 1e-5; hi -= 1e-5
    seatidx = names.index('seat_slide')
    eidx = [names.index(s+'_elbow_joint') for s in ('left','right')]
    extension = []
    minimum_bend = []
    for side, idx in zip(('left','right'),eidx):
        ids = [m.joint(side+'_'+part+'_joint').id for part in ('shoulder_yaw','elbow','wrist_roll')]
        def bend(q):
            d.qpos[:] = m.qpos0; d.qpos[qa[idx]] = q
            mujoco.mj_kinematics(m,d)
            a = d.xanchor[ids[1]]-d.xanchor[ids[0]]
            b = d.xanchor[ids[2]]-d.xanchor[ids[1]]
            return np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))
        opt = minimize_scalar(bend,bounds=(lo[idx],hi[idx]),method='bounded')
        extension.append(float(opt.x)); minimum_bend.append(float(opt.fun))
        lo[idx] = opt.x  # anatomically flexed branch; no elbow hyperextension

    raw = np.loadtxt(ROOT/'data/rowing_reference.csv',delimiter=',',skiprows=1)
    oars = CubicSpline(raw[:,0],raw[:,1:],bc_type='periodic')
    prior = np.loadtxt(OUT/'aligned_priors.csv',delimiter=',',skiprows=1)
    priors = CubicSpline(prior[:,0],prior[:,1:],bc_type='periodic')
    phase0 = json.loads((ROOT/'designs/compact_hand/reference_start.json').read_text())['shared_phase']
    seat_mapping=json.loads((OUT/'seat_mapping.json').read_text())
    # Geometric seed only; offset and span are fitted by project_motion_reference.
    # Feasible output is determined by the full attachments, not this assumption.
    def targets(phi):
        p = priors(phi % 1)
        seat = seat_mapping['offset_m']+seat_mapping['span_m']*p[0]
        elbow = np.clip(np.array(extension)+p[1],lo[eidx],hi[eidx])
        return seat, elbow

    rows = []
    n = args.nodes
    phases = (phase0+np.arange(n)/n) % 1
    xs = np.tile(m.qpos0[qa],(n,1))
    x = np.clip(m.qpos0[qa],lo+1e-8,hi-1e-8)
    base = x.copy()
    weights = np.full(len(joints),.025)
    weights[seatidx] = 3.
    weights[eidx] = .12
    for name in ('waist_yaw_joint','waist_roll_joint'):
        weights[names.index(name)] = .3
    rng = np.random.default_rng(42)
    # Pose residual and its Jacobian, with cached forward-position evaluation.
    cache_x = None
    current_phi = 0.
    def evaluate(x):
        nonlocal cache_x
        if cache_x is None or not np.array_equal(x,cache_x):
            d.qpos[:] = m.qpos0
            d.qpos[qa] = x; d.qpos[oqa] = oars(current_phi)
            mujoco.mj_fwdPosition(m,d)
            cache_x = x.copy()
        mask = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        return d.efc_pos[mask],d.efc_J.reshape(d.nefc,m.nv)[mask][:,va]

    for sweep in range(args.passes):
        worst = 0.
        for k,phi in enumerate(phases):
            current_phi=float(phi);cache_x=None
            target=base.copy();target[seatidx],target[eidx]=targets(phi)
            seed=x if sweep==0 else .5*(x+xs[k])
            prev=x.copy()
            def residual(v):
                eq,_=evaluate(v)
                return np.r_[eq*2000,weights*(v-target),.10*(v-prev)]
            def jac(v):
                _,j=evaluate(v)
                return np.vstack([j*2000,np.diag(weights),np.eye(len(v))*.10])
            fit=least_squares(residual,np.clip(seed,lo+1e-8,hi-1e-8),jac=jac,
                              bounds=(lo,hi),max_nfev=220,ftol=1e-8,xtol=1e-9,gtol=1e-8)
            if abs(evaluate(fit.x)[0]).max() > 1e-4:
                for attempt in range(args.restarts):
                    initial=base.copy() if attempt==0 else base+rng.normal(0,.5,len(base))
                    initial[seatidx]=target[seatidx]
                    if attempt>0: initial[eidx]=target[eidx]
                    trial=least_squares(residual,np.clip(initial,lo+1e-8,hi-1e-8),jac=jac,
                                        bounds=(lo,hi),max_nfev=300,ftol=1e-8,xtol=1e-9,gtol=1e-8)
                    if np.linalg.norm(trial.fun)<np.linalg.norm(fit.fun): fit=trial
                    if abs(evaluate(fit.x)[0]).max()<1e-5: break
            x=fit.x;xs[k]=x
            eq,_=evaluate(x);err=float(abs(eq).max());worst=max(worst,err)
            rows.append([sweep,float(phi),err,fit.nfev,float(np.linalg.norm(fit.fun))])
            if k%10==0:
                print(f'pass={sweep} node={k}/{n} phase={phi:.4f} closure={err:.6g}',flush=True)
        print(f'pass={sweep} worst_closure={worst:.6g}',flush=True)

    order=np.argsort(phases);p=phases[order];v=xs[order]
    # Periodic cubic in a rotated coordinate, then evaluate a common 401 grid.
    # Shape-preserving interpolation avoids joint-limit overshoot. Validate
    # dense closures explicitly; periodic value equality alone is insufficient.
    spl=PchipInterpolator(np.r_[p[-2:]-1,p,p[:2]+1],np.vstack([v[-2:],v,v[:2]]))
    grid=np.linspace(0,1,401)
    full=np.tile(m.qpos0,(len(grid),1))
    full[:,qa]=spl(grid)
    full[:,oqa]=oars(grid)
    full[-1]=full[0]

    # Quantify physical weld errors, not mixed position/quaternion residual units.
    errors=[]
    for q in full:
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        row=[]
        for eid in range(m.neq):
            a,b=int(m.eq_obj1id[eid]),int(m.eq_obj2id[eid])
            ra=d.xmat[a].reshape(3,3);rb=d.xmat[b].reshape(3,3)
            rel=m.eq_data[eid,3:6]
            rrel=Rotation.from_quat(m.eq_data[eid,6:10][[1,2,3,0]]).as_matrix()
            row.extend([np.linalg.norm(d.xpos[a]+ra@rel-d.xpos[b]),
                        np.linalg.norm(Rotation.from_matrix((ra@rrel).T@rb).as_rotvec())])
        errors.append(row)
    errors=np.array(errors)
    qlim=np.maximum(np.maximum(lo-full[:,qa],full[:,qa]-hi),0)
    period=60/26
    fs=CubicSpline(grid,full,bc_type='periodic')
    dq=fs(grid,1)/period
    outnames=OARS+['seat_slide','left_elbow_joint','right_elbow_joint']
    outqa=[m.joint(n).qposadr[0] for n in outnames]
    candidate=full[:,outqa]
    np.savetxt(OUT/'rowing_reference9_candidate.csv',np.c_[grid,candidate],delimiter=',',
               header=','.join(['phase']+outnames),comments='',fmt='%.12g')
    np.savez_compressed(OUT/'h2_trajectory_candidate.npz',phase=grid,qpos=full,
                        qpos_derivative=dq,joint_names=np.array(names),joint_qpos_indices=qa)
    np.savetxt(QA/'closure_errors.csv',np.c_[grid,errors],delimiter=',',comments='',
               header='phase,'+','.join(m.equality(i).name+'_'+s for i in range(m.neq) for s in ('position_m','rotation_rad')))
    np.savetxt(QA/'optimization_log.csv',rows,delimiter=',',comments='',
               header='pass,phase,max_equality_residual,nfev,residual_norm')
    report={
        'status':'candidate_not_activated',
        'scene_sha256':hashlib.sha256((ROOT/'scene.xml').read_bytes()).hexdigest(),
        'oar_source_sha256':hashlib.sha256((ROOT/'data/rowing_reference.csv').read_bytes()).hexdigest(),
        'oars_unchanged_max_abs_rad':float(abs(candidate[:,:6]-raw[:,1:]).max()),
        'period_s':period,'nodes':n,'passes':args.passes,'restarts':args.restarts,
        'max_attachment_position_error_m':float(errors[:,::2].max()),
        'max_attachment_rotation_error_deg':float(np.rad2deg(errors[:,1::2]).max()),
        'max_joint_limit_violation':float(qlim.max()),
        'seat_seed_range_m':[float(min(targets(p)[0] for p in grid)),float(max(targets(p)[0] for p in grid))],
        'seat_origin':'geometry-informed prior, not anchored to reset',
        'seat_actual_range_m':[float(candidate[:,6].min()),float(candidate[:,6].max())],
        'elbow_extension_coordinates_deg':np.rad2deg(extension).tolist(),
        'minimum_geometric_arm_bend_deg':np.rad2deg(minimum_bend).tolist(),
        'elbow_prior_conversion':'q_extension + human flexion, clipped to limits; weak heuristic, NOT exact anatomy equivalence',
        'elbow_actual_range_deg':[[float(np.rad2deg(candidate[:,i]).min()),float(np.rad2deg(candidate[:,i]).max())] for i in (7,8)],
        'periodic_value_error':float(abs(full[-1]-full[0]).max()),
        'max_robot_joint_speed_rad_s':float(abs(dq[:,qa[1:]]).max()),
        'collision_check':'pending','dynamics_check':'not performed',
        'limitations':['Independent studies; event alignment does not establish experimental simultaneity',
                       'Sequential constrained least squares with periodic interpolation, not global optimality proof',
                       'Candidate must pass attachment, limits, continuity and collision gates before training'],
    }
    report['attachment_gate_passed']=bool(errors[:,::2].max()<.001 and np.rad2deg(errors[:,1::2]).max()<.5 and qlim.max()<1e-6)
    (OUT/'conversion_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
    ax[0].plot(grid,np.rad2deg(candidate[:,:6]));ax[0].set_ylabel('Oar joints (deg)')
    ax[1].plot(grid,candidate[:,6],label='H2 candidate')
    ax[1].plot(grid,[targets(p)[0] for p in grid],'--',label='Scaled seat prior')
    ax[1].set_ylabel('Seat joint (m)');ax[1].legend()
    ax[2].plot(grid,np.rad2deg(candidate[:,7:]),label=['Left H2','Right H2'])
    ax[2].plot(grid,np.rad2deg([targets(p)[1][0] for p in grid]),'--',label='Elbow prior in H2 coordinates')
    ax[2].set_ylabel('Elbow joint (deg)');ax[2].set_xlabel('Cycle phase');ax[2].legend()
    fig.suptitle('Nine-channel candidate: inspect conversion_report before use')
    fig.tight_layout();fig.savefig(QA/'reference9_candidate.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
