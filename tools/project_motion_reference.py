"""Global periodic kinematic projection of nine motion priors onto H2.

Unlike the fixed-oar diagnostic, this explicitly permits deviations from the
six source oar angles. Output is an engineered motion, not experimental data.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.sparse import block_diag, diags, eye, kron, vstack, hstack, lil_matrix, csr_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.convert_motion_reference import OARS
OUT=ROOT/'data/reference9'
QA=ROOT/'results/reference9'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nodes',type=int,default=80)
    ap.add_argument('--max-nfev',type=int,default=150)
    ap.add_argument('--collision-weight',type=float,default=0.)
    args=ap.parse_args();n=args.nodes
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    if args.collision_weight:
        from rowing_mjlab.asset import build_spec
        m=build_spec().compile()
        m.geom_margin[:]=.002
    m.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    d=mujoco.MjData(m)
    ji=[i for i in range(m.njnt) if m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE
        and not m.joint(i).name.startswith('head_')]
    names=[m.joint(i).name for i in ji];qa=m.jnt_qposadr[ji];va=m.jnt_dofadr[ji]
    k=len(ji);oi=[names.index(s) for s in OARS]
    si=names.index('seat_slide');ei=[names.index(s+'_elbow_joint') for s in ('left','right')]
    lo,hi=m.jnt_range[ji].T.copy();lo+=1e-4;hi-=1e-4
    ext=np.deg2rad(json.loads((OUT/'conversion_report.json').read_text())['elbow_extension_coordinates_deg'])
    lo[ei]=ext
    phi=np.arange(n)/n
    o=np.loadtxt(ROOT/'data/rowing_reference.csv',delimiter=',',skiprows=1)
    p=np.loadtxt(OUT/'aligned_priors.csv',delimiter=',',skiprows=1)
    op=CubicSpline(o[:,0],o[:,1:],bc_type='periodic')(phi)
    pp=CubicSpline(p[:,0],p[:,1:],bc_type='periodic')(phi)
    target=np.tile(m.qpos0[qa],(n,1));target[:,oi]=op
    mapping=json.loads((OUT/'seat_mapping.json').read_text())
    target[:,si]=mapping['offset_m']+mapping['span_m']*pp[:,0]
    target[:,ei]=np.minimum(ext+pp[:,1,None],hi[ei])
    w=np.full(k,.025)
    w[oi]=np.sqrt(.75/6)/np.deg2rad([10,4,15,10,4,15])
    w[si]=0.;w[ei]=np.sqrt(.05/2)/.4
    for name in ('waist_yaw_joint','waist_roll_joint'):
        w[names.index(name)]=.2
    D=lil_matrix((n,n))
    for i in range(n):D[i,i]=-1;D[i,(i+1)%n]=1
    D=D.tocsr()
    D1=hstack([kron(D,eye(k),format='csr')*(.01*n),csr_matrix((n*k,2))],format='csr')
    D2=hstack([kron(D.T@D,eye(k),format='csr')*(.0003*n*n),csr_matrix((n*k,2))],format='csr')
    W=hstack([kron(eye(n),diags(w),format='csr'),csr_matrix((n*k,2))],format='csr')
    tail=vstack([W,D1,D2],format='csr')
    # Fit the seat offset and stroke jointly. There is no reset-phase anchor.
    shape=lil_matrix((n,n*k+2))
    for i in range(n):
        shape[i,i*k+si]=1.;shape[i,-2]=-1.;shape[i,-1]=-pp[i,0]
    shape=shape.tocsr()
    x=np.r_[np.tile(np.clip(m.qpos0[qa],lo+1e-8,hi-1e-8),(n,1)).ravel(),mapping['offset_m'],mapping['span_m']]
    target_flat=np.r_[target.ravel(),0.,0.]
    bodypairs={(i,j):idx for idx,(i,j) in enumerate((i,j) for i in range(m.nbody) for j in range(i+1,m.nbody))}
    cached=None;E=None;J=None;CE=None;CJ=None;last=time.monotonic();calls=0
    scale=10.
    def evaluate(x):
        nonlocal cached,E,J,CE,CJ,calls,last
        if cached is not None and np.array_equal(cached,x):return
        eq=[];jacs=[];ces=[];cjs=[]
        for q in x[:-2].reshape(n,k):
            d.qpos[:]=m.qpos0;d.qpos[qa]=q;mujoco.mj_fwdPosition(m,d)
            mask=d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY
            eq.append(d.efc_pos[mask].copy())
            jacs.append(d.efc_J.reshape(d.nefc,m.nv)[mask][:,va].copy())
            if args.collision_weight:
                ce=np.zeros(len(bodypairs));cj=np.zeros((len(bodypairs),k))
                for contact in d.contact:
                    if contact.dist>=.002:continue
                    aa,bb=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2])
                    if aa==bb:continue
                    row=bodypairs[tuple(sorted((aa,bb)))]
                    depth=.002-contact.dist
                    if depth<=ce[row]:continue
                    ja=np.zeros((3,m.nv));jb=np.zeros((3,m.nv))
                    mujoco.mj_jac(m,d,ja,None,contact.pos,aa)
                    mujoco.mj_jac(m,d,jb,None,contact.pos,bb)
                    ce[row]=depth
                    cj[row]=-contact.frame[:3]@(jb-ja)[:,va]
                ces.append(ce);cjs.append(csr_matrix(cj))
        E=np.array(eq).ravel()
        J=hstack([block_diag([csr_matrix(j) for j in jacs],format='csr'),csr_matrix((len(E),2))],format='csr')
        if args.collision_weight:
            CE=np.array(ces).ravel()
            CJ=hstack([block_diag(cjs,format='csr'),csr_matrix((len(CE),2))],format='csr')
        cached=x.copy();calls+=1
        if time.monotonic()-last>20:
            print(f'constraint_weight={scale} eval={calls} closure={abs(E).max():.6g} contact={max(CE) if CE is not None else 0:.6g}',flush=True);last=time.monotonic()
    def fun(x):
        evaluate(x)
        base=np.r_[scale*E,W@(x-target_flat),D1@x,D2@x,scale*(shape@x)]
        return np.r_[base,args.collision_weight*CE] if args.collision_weight else base
    def jac(x):
        evaluate(x)
        blocks=[scale*J,tail,scale*shape]
        if args.collision_weight:blocks.append(args.collision_weight*CJ)
        return vstack(blocks,format='csr')
    logs=[]
    for scale in (10.,100.,1000.):
        fit=least_squares(fun,x,jac=jac,bounds=(np.r_[np.tile(lo,n),-.24,0.],np.r_[np.tile(hi,n),.24,.48]),
                          tr_solver='lsmr',max_nfev=args.max_nfev,ftol=1e-7,xtol=1e-8,gtol=1e-7,
                          tr_options={'maxiter':500,'atol':1e-7,'btol':1e-7})
        x=fit.x;evaluate(x)
        print('stage',scale,'closure',float(abs(E).max()),'cost',fit.cost,'nfev',fit.nfev,flush=True)
        logs.append(dict(weight=scale,cost=float(fit.cost),closure=float(abs(E).max()),nfev=fit.nfev))
    vals=x[:-2].reshape(n,k)
    mapping.update(offset_m=float(x[-2]),span_m=float(x[-1]),
                   calibration_status='whole_cycle_kinematic_fit_pending_collision_validation',
                   fitting_method='Joint constrained optimization of pose sequence, seat offset and seat span; no reset anchor',
                   offset_bounds_m=[-.24,.24],span_bounds_m=[0.,.48],collision_penalty_weight=args.collision_weight)
    mapping['range_m']=[float(vals[:,si].min()),float(vals[:,si].max())]
    mapping['physical_distance_range_m']=[mapping['distance_at_joint_zero_m']+q for q in mapping['range_m']]
    dense_priors=np.loadtxt(OUT/'rowing_reference9_priors.csv',delimiter=',',skiprows=1)
    shape_data=np.loadtxt(OUT/'aligned_priors.csv',delimiter=',',skiprows=1)
    dense_priors[:,7]=x[-2]+x[-1]*shape_data[:,1]
    np.savetxt(OUT/'rowing_reference9_priors.csv',dense_priors,delimiter=',',comments='',fmt='%.12g',
               header='phase,'+','.join(OARS+['seat_slide','left_elbow_joint','right_elbow_joint']))
    mapping['position_at_selected_start_m']=float(CubicSpline(shape_data[:,0],dense_priors[:,7],bc_type='periodic')(mapping['reference_start_phase_information_only']))
    (OUT/'seat_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    meta=json.loads((OUT/'source_provenance.json').read_text());meta['model_seat_mapping']=mapping
    (OUT/'source_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('fitted seat offset/span',x[-2:],flush=True)
    np.savez_compressed(OUT/'projected_nodes.npz',phase=phi,q=vals,joint_names=np.array(names),qpos_indices=qa)
    (OUT/'projection_optimization.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')


if __name__=='__main__':main()
