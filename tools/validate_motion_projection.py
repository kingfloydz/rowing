"""Polish global projection on the export grid and audit the complete motion.

This is kinematic conversion only; collisions and dynamic feasibility can reject
the output for training even if the attachments close accurately.
"""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.convert_motion_reference import OARS
from rowing_mjlab.asset import build_spec
OUT=ROOT/'data/reference9'
QA=ROOT/'results/reference9'


def main():
    a=np.load(OUT/'projected_nodes.npz')
    names=a['joint_names'].tolist();qa=a['qpos_indices'];k=len(qa)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    m.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    d=mujoco.MjData(m)
    ji=[m.joint(name).id for name in names];va=m.jnt_dofadr[ji]
    lo,hi=m.jnt_range[ji].T.copy();lo+=1e-5;hi-=1e-5
    eidx=[names.index(s+'_elbow_joint') for s in ('left','right')]
    ext=np.deg2rad(json.loads((OUT/'conversion_report.json').read_text())['elbow_extension_coordinates_deg'])
    lo[eidx]=ext
    seat_index=names.index('seat_slide')
    mapping=json.loads((OUT/'seat_mapping.json').read_text())
    source=np.loadtxt(OUT/'aligned_priors.csv',delimiter=',',skiprows=1)
    seat_shape=CubicSpline(source[:,0],source[:,1],bc_type='periodic')
    p=a['phase'];v=a['q']
    initial=CubicSpline(np.r_[p,1.],np.vstack([v,v[0]]),bc_type='periodic')
    grid=np.linspace(0,1,401)
    X=initial(grid)
    cached=None
    def evaluate(q):
        nonlocal cached
        if cached is None or not np.array_equal(cached,q):
            d.qpos[:]=m.qpos0;d.qpos[qa]=q;mujoco.mj_fwdPosition(m,d);cached=q.copy()
        mask=d.efc_type==0
        return d.efc_pos[mask],d.efc_J.reshape(d.nefc,m.nv)[mask][:,va]
    for i in range(len(grid)-1):
        target=X[i].copy()
        # Preserve the globally fitted offset/span, not the arbitrary reset pose.
        seat_target=mapping['offset_m']+mapping['span_m']*seat_shape(grid[i])
        def residual(q):return np.r_[evaluate(q)[0]*10000,q-target,10000*(q[seat_index]-seat_target)]
        def jac(q):return np.vstack([evaluate(q)[1]*10000,np.eye(k),np.eye(k)[seat_index]*10000])
        fit=least_squares(residual,np.clip(target,lo+1e-8,hi-1e-8),jac=jac,
                          bounds=(lo,hi),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        X[i]=fit.x
        if i%80==0:print('polish',i,'closure',float(abs(evaluate(fit.x)[0]).max()),flush=True)
    X[-1]=X[0]
    full=np.tile(m.qpos0,(401,1));full[:,qa]=X
    # Cubic interpolation represents the intended runtime interpolation. Audit
    # all exported samples and their midpoints, including periodic derivatives.
    spline=CubicSpline(grid,full,bc_type='periodic')
    dense=np.linspace(0,1,801);F=spline(dense)
    cm=build_spec().compile();cd=mujoco.MjData(cm)
    errors=[];contact_rows=[]
    for t,q in zip(dense,F):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        er=[]
        for eid in range(m.neq):
            aa,bb=int(m.eq_obj1id[eid]),int(m.eq_obj2id[eid])
            ra=d.xmat[aa].reshape(3,3);rb=d.xmat[bb].reshape(3,3)
            rq=Rotation.from_quat(m.eq_data[eid,6:10][[1,2,3,0]]).as_matrix()
            er += [np.linalg.norm(d.xpos[aa]+ra@m.eq_data[eid,3:6]-d.xpos[bb]),
                   np.linalg.norm(Rotation.from_matrix((ra@rq).T@rb).as_rotvec())]
        errors.append(er)
        cd.qpos[:]=q;mujoco.mj_fwdPosition(cm,cd)
        for c in cd.contact:
            if c.dist < -.001:
                contact_rows.append((float(t),cm.geom(c.geom1).name,cm.geom(c.geom2).name,float(c.dist)))
    errors=np.array(errors)
    lower,upper=m.jnt_range[ji].T
    violation=np.maximum(np.maximum(lower-F[:,qa],F[:,qa]-upper),0)
    o=np.loadtxt(ROOT/'data/rowing_reference.csv',delimiter=',',skiprows=1)
    outnames=OARS+['seat_slide','left_elbow_joint','right_elbow_joint']
    outqa=np.array([m.joint(s).qposadr[0] for s in outnames])
    Y=full[:,outqa]
    delta=Y[:,:6]-o[:,1:]
    period=60/26
    np.savetxt(OUT/'rowing_reference9_projected.csv',np.c_[grid,Y],delimiter=',',comments='',
               header=','.join(['phase']+outnames),fmt='%.12g')
    np.savez_compressed(OUT/'h2_trajectory_projected.npz',phase=grid,qpos=full,
                        qpos_derivative=spline(grid,1)/period,joint_names=np.array(names),qpos_indices=qa)
    np.savetxt(QA/'projected_closure_errors.csv',np.c_[dense,errors],delimiter=',',comments='',
               header='phase,'+','.join(m.equality(i).name+'_'+s for i in range(m.neq) for s in ('position_m','rotation_rad')))
    import csv
    with (QA/'projected_contacts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['phase','geom1','geom2','signed_distance_m']);w.writerows(contact_rows)
    attachment_ok=bool(errors[:,::2].max()<.001 and np.rad2deg(errors[:,1::2]).max()<.5)
    lim_ok=bool(violation.max()<1e-6)
    worst=sorted(contact_rows,key=lambda x:x[-1])[:12]
    report={
        'status':'converted_candidate_not_activated',
        'data_kind':'H2 kinematic projection of cross-study priors, NOT synchronized experimental data',
        'scene_sha256':hashlib.sha256((ROOT/'scene.xml').read_bytes()).hexdigest(),
        'period_s':period,'grid_samples':401,'audit_samples':801,
        'reference_order':outnames,'units':['rad']*6+['m','rad','rad'],
        'oar_deviation_deg':{name:dict(max_abs=float(np.rad2deg(abs(delta[:,i])).max()),
                                      rms=float(np.rad2deg(np.sqrt(np.mean(delta[:,i]**2))))) for i,name in enumerate(OARS)},
        'seat_range_m':[float(Y[:,6].min()),float(Y[:,6].max())],
        'seat_selected_start_phase':mapping['reference_start_phase_information_only'],
        'seat_at_reference_start_m':float(spline(mapping['reference_start_phase_information_only'])[outqa[6]]),
        'seat_shape_error_m':float(abs(Y[:,6]-(mapping['offset_m']+mapping['span_m']*seat_shape(grid))).max()),
        'seat_origin':'fitted using full-cycle geometry; reset is not an anchor',
        'elbow_range_deg':np.rad2deg(np.array([Y[:,7:].min(0),Y[:,7:].max(0)]).T).tolist(),
        'max_attachment_position_error_m':float(errors[:,::2].max()),
        'max_attachment_rotation_error_deg':float(np.rad2deg(errors[:,1::2]).max()),
        'attachment_gate_passed':attachment_ok,'joint_limit_gate_passed':lim_ok,
        'max_joint_limit_violation':float(violation.max()),
        'max_robot_joint_speed_rad_s':float(abs(spline(dense,1)[:,qa[1:]]/period).max()),
        'max_oar_step_deg':float(np.rad2deg(abs(np.diff(Y[:,:6],axis=0))).max()),
        'collision_gate_passed':not bool(contact_rows),'contact_threshold_m':-.001,
        'penetration_count_across_audit_frames':len(contact_rows),'worst_contacts':worst,
        'training_ready':False,'dynamics_check':'not performed; torque and waterline interaction not validated',
        'start_transition':'not generated; existing 0.3s independent blending is not a closure guarantee',
        'note':'Source oar angles were allowed to change during global periodic optimization; see deviations',
    }
    (OUT/'projection_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(3,3,figsize=(15,10))
    for i,ax in enumerate(axes.flat):
        vals=Y[:,i] if i==6 else np.rad2deg(Y[:,i])
        ax.plot(grid,vals,label='H2 projected')
        if i<6:ax.plot(grid,np.rad2deg(o[:,i+1]),'--',label='Source reference')
        ax.set_title(outnames[i]);ax.set_ylabel('m' if i==6 else 'deg');ax.grid(alpha=.2)
        ax.set_xlabel('Phase');ax.legend(fontsize=8)
    fig.suptitle('Converted nine-channel candidate (not activated; see quality report)')
    fig.tight_layout();fig.savefig(QA/'reference9_projected.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
