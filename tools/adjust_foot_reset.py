"""Rebase ankle-roll reset and foot weld poses without changing weld solref/solimp."""
from pathlib import Path
import argparse, hashlib, json, re
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/foot_reset_adjustment'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 reports={};prepared={}
 for filename in ['scene.xml','preview.xml']:
  path=ROOT/filename;text=path.read_text(encoding='utf-8');root=ET.fromstring(text)
  m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);d.qpos[:]=m.qpos0
  original_angles={};deltas={}
  for side,sign in [('left',1),('right',-1)]:
   j=m.joint(side+'_ankle_roll_joint').id;new=np.deg2rad(sign*9.5)
   original_angles[side]=float(np.rad2deg(m.qpos0[m.jnt_qposadr[j]]));deltas[side]=new-m.qpos0[m.jnt_qposadr[j]]
   d.qpos[m.jnt_qposadr[j]]=new
  mujoco.mj_fwdPosition(m,d);expected_pos=d.xpos.copy();expected_rot=d.xmat.copy()
  replacement={};footinfo={}
  for side,sign in [('left',1),('right',-1)]:
   j=m.joint(side+'_ankle_roll_joint').id;bid=m.jnt_bodyid[j]
   quat=np.empty(4);deltaquat=np.empty(4)
   mujoco.mju_axisAngle2Quat(deltaquat,m.jnt_axis[j],deltas[side]);mujoco.mju_mulQuat(quat,m.body_quat[bid],deltaquat)
   replacement[('body',side+'_ankle_roll_link')]={'quat':' '.join(f'{v:.15g}' for v in quat)}
   replacement[('joint',side+'_ankle_roll_joint')]={'ref':f'{np.deg2rad(sign*9.5):.15g}'}
   eid=m.equality(side+'_foot_fixed').id;a,b=m.eq_obj1id[eid],m.eq_obj2id[eid]
   ra=d.xmat[a].reshape(3,3);rb=d.xmat[b].reshape(3,3)
   pos=ra.T@(d.xpos[b]-d.xpos[a]);q=Rotation.from_matrix(ra.T@rb).as_quat()[[3,0,1,2]]
   if q[0]<0:q=-q
   replacement[('weld',side+'_foot_fixed')]={'relpose':' '.join(f'{v:.15g}' for v in np.r_[pos,q]),'torquescale':'0.5'}
   element=root.find(f"./equality/weld[@name='{side}_foot_fixed']")
   footinfo[side]={'solref_unchanged':element.get('solref'),'solimp_unchanged':element.get('solimp'),
    'old_torquescale':float(element.get('torquescale','1')),'new_torquescale':.5,'new_relpose':np.r_[pos,q].tolist()}
  updated=text
  for (tag,name),attrs in replacement.items():
   pattern=rf'<{tag}\b[^>]*\bname="{re.escape(name)}"[^>]*>'
   matches=list(re.finditer(pattern,updated));assert len(matches)==1,(tag,name,len(matches))
   old=matches[0].group();new=old
   for key,value in attrs.items():
    if re.search(rf'\b{key}="[^"]*"',new):new=re.sub(rf'\b{key}="[^"]*"',f'{key}="{value}"',new)
    else:new=new.replace(' />',f' {key}="{value}" />')
   updated=updated[:matches[0].start()]+new+updated[matches[0].end():]
  compile_text=updated.replace('meshdir="."',f'meshdir="{ROOT.as_posix()}"')
  nm=mujoco.MjModel.from_xml_string(compile_text);nd=mujoco.MjData(nm);mujoco.mj_fwdPosition(nm,nd)
  assert np.max(abs(nd.xpos-expected_pos))<1e-9
  assert np.max(abs(nd.xmat-expected_rot))<1e-9
  eqmask=nd.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY
  assert np.max(abs(nd.efc_pos[eqmask]))<1e-8
  for s in ['left','right']:
   oi=m.equality(s+'_foot_fixed').id;ni=nm.equality(s+'_foot_fixed').id
   assert np.array_equal(m.eq_solref[oi],nm.eq_solref[ni])
   assert np.array_equal(m.eq_solimp[oi],nm.eq_solimp[ni])
  prepared[path]=updated
  backup=OUT/(filename+'.before')
  if not backup.exists():backup.write_text(text,encoding='utf-8')
  reports[filename]={'before_sha256':hashlib.sha256(text.encode()).hexdigest(),
   'after_sha256':hashlib.sha256(updated.encode()).hexdigest(),'old_ankle_roll_deg':original_angles,
   'new_ankle_roll_deg':{'left':9.5,'right':-9.5},'foot_welds':footinfo,
   'max_reset_equality_residual':float(np.max(abs(nd.efc_pos[eqmask]))),
   'geometry_rebase_position_error_m':float(np.max(abs(nd.xpos-expected_pos)))}
 if args.apply:
  for path,updated in prepared.items():path.write_text(updated,encoding='utf-8')
 reports['applied']=args.apply
 (OUT/'reset_change_report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
 print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
