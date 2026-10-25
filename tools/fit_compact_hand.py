"""Install sleeve hands and solve closed-chain reset pose; use --apply to save.

Preserve anatomical joint coordinates by rebasing body frames AND hinge axes.
The original elbow world axes are recorded in pose_baseline.json.
"""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'designs/compact_hand'
PAIRS=(('left','starboard',1),('right','port',-1))
ARM=('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')
fmt=lambda a:' '.join(f'{v:.14g}' for v in np.atleast_1d(a))


def compile_root(root):
    import copy
    temp=copy.deepcopy(root)
    temp.find('compiler').set('meshdir','')
    for mesh in temp.findall('./asset/mesh'):
        mesh.set('file',str(ROOT/mesh.get('file')))
    return mujoco.MjModel.from_xml_string(ET.tostring(temp,encoding='unicode'))


def install(root):
    root.set('model','H2_compact_hand_single_scull')
    asset=root.find('asset')
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name','').startswith('Dex1_') or mesh.get('name')=='compact_hand':asset.remove(mesh)
    ET.SubElement(asset,'mesh',name='compact_hand',file='designs/compact_hand/compact_hand.stl')
    mesh=trimesh.load_mesh(OUT/'compact_hand.stl');mesh.density=2700.
    mass=float(mesh.mass);com=mesh.center_mass;inertia=mesh.moment_inertia
    for h,_,_ in PAIRS:
        body=root.find(f'.//body[@name="{h}_dex1_base_link"]')
        for child in list(body):body.remove(child)
        ET.SubElement(body,'inertial',mass=fmt(mass),pos=fmt(com),
                      fullinertia=fmt([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
        ET.SubElement(body,'geom',name=h+'_compact_hand_visual',type='mesh',mesh='compact_hand',
                      rgba='.7 .7 .7 1',contype='0',conaffinity='0',group='1')
        ET.SubElement(body,'site',name=h+'_gripper_center',pos='.029 0 0',size='.004',rgba='0 0 0 0')
    return mass


def solve(root):
    m=compile_root(root);d=mujoco.MjData(m)
    baseline=json.loads((OUT/'pose_baseline.json').read_text())
    names=[h+'_'+a+'_joint' for h,_,_ in PAIRS for a in ARM]
    names += [o+'_'+a for _,o,_ in PAIRS for a in ('sweep','elevation','feather')]
    jids=np.array([m.joint(n).id for n in names]);qa=m.jnt_qposadr[jids]
    # Grip mesh extends from 0.74425 to 0.86216 m along either oar.
    # Keep the complete 70 mm sleeve on the grip, with end clearance.
    lo=np.r_[m.jnt_range[jids,0],.785];hi=np.r_[m.jnt_range[jids,1],.82]
    x0=np.r_[[baseline['joint_angles_rad'][n] for n in names],.80]
    bdata=[]
    for h,o,sign in PAIRS:
        bdata.append(dict(h=h,o=o,sign=sign,hand=m.body(h+'_dex1_base_link').id,
            oar=m.body(o+'_oar').id,elbow=m.joint(h+'_elbow_joint').id,
            shoulder=m.joint(h+'_shoulder_yaw_joint').id,wrist=m.joint(h+'_wrist_roll_joint').id,
            wrist_end=m.joint(h+'_wrist_yaw_joint').id,
            axis=np.array(baseline['elbow_axes_world'][h])))
    def forward(x):
        d.qpos[:]=m.qpos0;d.qpos[qa]=x[:-1];mujoco.mj_kinematics(m,d)
    def segment_angle(b):
        a=d.xanchor[b['elbow']]-d.xanchor[b['shoulder']]
        c=d.xanchor[b['wrist']]-d.xanchor[b['elbow']]
        return np.arccos(np.clip(a@c/np.linalg.norm(a)/np.linalg.norm(c),-1,1))
    targets=[]
    for b in bdata:
        idx=names.index(b['h']+'_elbow_joint')
        def bending(q):
            x=x0.copy();x[idx]=q;forward(x);return segment_angle(b)
        opt=minimize_scalar(bending,bounds=(lo[idx],hi[idx]),method='bounded')
        targets.append(float(opt.x))
        # Stay on the slightly flexed side of maximal extension; no large fold.
        lo[idx]=opt.x;hi[idx]=min(hi[idx],opt.x+np.deg2rad(5))
    x0=np.clip(x0,lo+1e-8,hi-1e-8)
    def constraints(x):
        forward(x);eq=[];axes=[];wrists=[]
        for b in bdata:
            rh=d.xmat[b['hand']].reshape(3,3);ro=d.xmat[b['oar']].reshape(3,3)
            center=d.xpos[b['hand']]+rh@np.array([.029,0,0])
            target=d.xpos[b['oar']]+ro@np.array([0,b['sign']*x[-1],0])
            eq.extend((center-target)*10)
            eq.extend(rh[:,:2].T@(-ro[:,1]))
            axes.append(d.xaxis[b['elbow']]@b['axis']-np.cos(np.deg2rad(4.95)))
            fore=d.xanchor[b['wrist_end']]-d.xanchor[b['elbow']];fore/=np.linalg.norm(fore)
            wrists.extend(rh[:,0]-fore)
        return np.array(eq),np.array(axes),np.array(wrists)
    def objective(x):
        _,_,w=constraints(x)
        elbows=np.array([x[names.index(b['h']+'_elbow_joint')]-t for b,t in zip(bdata,targets)])
        return float(10*w@w+20*elbows@elbows+.004*np.sum((x[:-1]-x0[:-1])**2)+.5*(x[-1]-.785)**2)
    def residual(x):
        eq,ax,w=constraints(x)
        return np.r_[eq*100,np.minimum(ax,0)*300,w*2,(x[:-1]-x0[:-1])*.01]
    best=None
    rng=np.random.default_rng(31)
    for k in range(8):
        initial=x0 if k==0 else np.clip(x0+rng.normal(0,.18,len(x0)),lo+1e-8,hi-1e-8)
        warm=least_squares(residual,initial,bounds=(lo,hi),max_nfev=350,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        result=minimize(objective,warm.x,method='SLSQP',bounds=list(zip(lo,hi)),
            constraints=[{'type':'eq','fun':lambda x:constraints(x)[0]},
                         {'type':'ineq','fun':lambda x:constraints(x)[1]}],
            options={'maxiter':700,'ftol':1e-11})
        eq,ax,_=constraints(result.x)
        print('attempt',k,'cost',result.fun,'closure',max(abs(eq)),'axis_margin',min(ax),flush=True)
        if max(abs(eq))<1e-6 and min(ax)>-1e-8:
            if best is None or result.fun<best.fun:best=result
            if result.fun<.2:break
    if best is None:raise RuntimeError('No feasible pose found; source XML left unchanged')
    forward(best.x)
    report=dict(grip_distance_from_oar_origin_m=float(best.x[-1]),objective=float(best.fun),
                joint_angles_deg=dict(zip(names,np.rad2deg(best.x[:-1]).tolist())),hands={})
    for b,t in zip(bdata,targets):
        rh=d.xmat[b['hand']].reshape(3,3);ro=d.xmat[b['oar']].reshape(3,3)
        center=d.xpos[b['hand']]+rh@np.array([.029,0,0]);target=d.xpos[b['oar']]+ro@np.array([0,b['sign']*best.x[-1],0])
        fore=d.xanchor[b['wrist_end']]-d.xanchor[b['elbow']];fore/=np.linalg.norm(fore)
        report['hands'][b['h']]=dict(position_error_mm=float(np.linalg.norm(center-target)*1000),
          bore_alignment_deg=float(np.rad2deg(np.arccos(np.clip(abs(rh[:,2]@ro[:,1]),-1,1)))),
          elbow_axis_change_deg=float(np.rad2deg(np.arccos(np.clip(d.xaxis[b['elbow']]@b['axis'],-1,1)))),
          wrist_deflection_deg=float(np.rad2deg(np.arccos(np.clip(rh[:,0]@fore,-1,1)))),
          geometric_arm_bend_deg=float(np.rad2deg(segment_angle(b))),
          elbow_flexion_from_max_extension_deg=float(np.rad2deg(d.qpos[m.jnt_qposadr[b['elbow']]]-t)))
    return m,d,names,report


def bake(root,m,d,names):
    bodies={b.get('name'):b for b in root.iter('body')}
    changed={int(m.jnt_bodyid[m.joint(n).id]) for n in names}
    for bid in changed:
        body=bodies[m.body(bid).name];pid=int(m.body_parentid[bid])
        rp=d.xmat[pid].reshape(3,3);rb=d.xmat[bid].reshape(3,3)
        body.set('pos',fmt(rp.T@(d.xpos[bid]-d.xpos[pid])))
        body.set('quat',fmt(Rotation.from_matrix(rp.T@rb).as_quat()[[3,0,1,2]]))
        for node in body.findall('joint'):
            jid=m.joint(node.get('name')).id
            node.set('ref',fmt(d.qpos[m.jnt_qposadr[jid]]))
            node.set('axis',fmt(rb.T@d.xaxis[jid]))
            node.set('pos',fmt(rb.T@(d.xanchor[jid]-d.xpos[bid])))
    for h,o,_ in PAIRS:
        a=m.body(h+'_dex1_base_link').id;b=m.body(o+'_oar').id;ra=d.xmat[a].reshape(3,3);rb=d.xmat[b].reshape(3,3)
        eq=root.find(f'./equality/weld[@name="{h}_hand_fixed"]')
        eq.set('relpose',fmt(np.r_[ra.T@(d.xpos[b]-d.xpos[a]),Rotation.from_matrix(ra.T@rb).as_quat()[[3,0,1,2]]]))
    for eq in root.findall('./equality/weld'):
        eq.set('solref','.004 1');eq.set('solimp','.99 .999 .001 .5 2')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=ET.parse(ROOT/'scene.xml').getroot();mass=install(root)
    m,d,names,report=solve(root);bake(root,m,d,names)
    check=compile_root(root);dc=mujoco.MjData(check);mujoco.mj_forward(check,dc)
    assert np.max(abs(dc.efc_pos[dc.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY]))<1e-8
    # Rebasing must preserve the physical kinematic map, including 2-hinge oars.
    rng=np.random.default_rng(7);maxerr=0.
    for _ in range(5):
        delta=rng.normal(0,.01,m.nq);delta[:7]=0
        old=mujoco.MjData(m);old.qpos[:]=d.qpos+delta;mujoco.mj_kinematics(m,old)
        new=mujoco.MjData(check);new.qpos[:]=check.qpos0+delta;mujoco.mj_kinematics(check,new)
        maxerr=max(maxerr,float(np.max(abs(old.xpos-new.xpos))),float(np.max(abs(old.xmat-new.xmat))))
    assert maxerr<1e-8,maxerr
    report.update(hand_mass_kg=mass,density_assumption_kg_m3=2700,kinematic_rebase_max_error=maxerr)
    (OUT/'pose_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if args.apply:
        ET.indent(root,space='  ');ET.ElementTree(root).write(ROOT/'scene.xml',encoding='utf-8',xml_declaration=True)
        preview=ET.parse(ROOT/'preview.xml').getroot();install(preview);bake(preview,m,d,names)
        ET.indent(preview,space='  ');ET.ElementTree(preview).write(ROOT/'preview.xml',encoding='utf-8',xml_declaration=True)


if __name__=='__main__':main()
