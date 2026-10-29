"""Install the checked physical reset without replacing ten-channel targets."""
import sys,json,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from hydrodynamics.models import ROOT


def main():
    folder=ROOT/'renders/reference10_literature';report=json.loads((folder/'reset_fit.json').read_text())
    if report['max_equality_error']>1e-7 or report['contacts']:raise RuntimeError('Physical reset is not checked')
    m=mujoco.MjModel.from_binary_path(str(folder/'reset_fit.mjb'));d=mujoco.MjData(m)
    positions=report['reset_q'];qi=[m.joint(n).qposadr[0] for n in positions];reset=np.array(list(positions.values()));trees={}
    for file in ('scene.xml','preview.xml'):
        root=ET.parse(ROOT/file).getroot()
        for side,g in report['grip'].items():
            b=root.find(f".//body[@name='{side}_dex1_base_link']")
            b.set('pos',' '.join(map(str,g['position_m'])));b.set('quat',' '.join(map(str,g['quaternion_wxyz'])))
        for body in root.iter('body'):
            joints=body.findall('joint');selected=[j for j in joints if j.get('name') in positions]
            if not selected:continue
            pos=np.fromstring(body.get('pos','0 0 0'),sep=' ');rot=Rotation.from_quat(np.fromstring(body.get('quat','1 0 0 0'),sep=' ')[[1,2,3,0]])
            for j in selected:
                value=positions[j.get('name')];old=float(j.get('ref','0'));axis=np.fromstring(j.get('axis','0 0 1'),sep=' ');axis/=np.linalg.norm(axis)
                if j.get('type','hinge')=='slide':pos+=rot.apply(axis*(value-old))
                else:
                    delta=Rotation.from_rotvec(axis*(value-old));anchor=np.fromstring(j.get('pos','0 0 0'),sep=' ')
                    pos+=rot.apply(anchor-delta.apply(anchor));rot=rot*delta
                    for earlier in joints[:joints.index(j)]:
                        aa=np.fromstring(earlier.get('axis','0 0 1'),sep=' ');earlier.set('axis',' '.join(map(str,delta.inv().apply(aa))))
                j.set('ref',str(value))
            body.set('pos',' '.join(map(str,pos)));body.set('quat',' '.join(map(str,rot.as_quat()[[3,0,1,2]])))
        if file=='scene.xml':
            root.find('compiler').set('meshdir',str(ROOT));cm=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));cd=mujoco.MjData(cm)
            for values in (reset,reset+.005):
                d.qpos[:]=m.qpos0;d.qpos[qi]=values;mujoco.mj_forward(m,d)
                cd.qpos[:]=cm.qpos0;cd.qpos[:7]=m.qpos0[:7];cd.qpos[[cm.joint(n).qposadr[0] for n in positions]]=values;mujoco.mj_forward(cm,cd)
                for b in range(1,cm.nbody):
                    old=m.body(cm.body(b).name).id
                    np.testing.assert_allclose(cd.xpos[b],d.xpos[old],atol=1e-9);np.testing.assert_allclose(cd.xmat[b],d.xmat[old],atol=1e-9)
            root.find('compiler').set('meshdir','.')
        trees[file]=root
    for file,root in trees.items():
        ET.indent(root);ET.ElementTree(root).write(ROOT/file,encoding='utf-8',xml_declaration=True)
    (ROOT/'data/reset_joint_pose.json').write_text(json.dumps(dict(note='Closed near-extension reset for literature-derived ten-channel targets; constant coaxial grip constraints.',joint_positions_rad=positions),indent=2)+'\n')
    path=ROOT/'data/reference20/metadata.json';meta=json.loads(path.read_text());meta['reset_phase']=report['reset_phase'];meta['physical_reset_calibration']=report
    meta.pop('grip_adjustment_rotation_vector_deg',None);meta['grip_adjustment_type']='Fixed wrist-to-gripper mounts calibrated at the closed reset; sleeve and oar remain coaxial.'
    path.write_text(json.dumps(meta,indent=2)+'\n')
    path=ROOT/'data/rig_geometry.json';geometry=json.loads(path.read_text());geometry['grip']=report['grip'];geometry['reset_calibration']='renders/reference10_literature/reset_fit.json';path.write_text(json.dumps(geometry,indent=2)+'\n')
    print('Installed reset and fixed gripper mounts; ten target channels unchanged.')


if __name__=='__main__':main()
