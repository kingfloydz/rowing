"""Training-only scene edits; preserve the saved assembly and all closures."""
from functools import lru_cache
import re
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

from hydrodynamics.models import ROOT

OAR_NAMES = tuple(f"{s}_{j}" for s in ("port","starboard") for j in ("sweep","elevation","feather"))


@lru_cache(maxsize=1)
def base_model():
    return mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))


def h2_names():
    m = base_model()
    return tuple(m.joint(i).name for i in range(m.njnt)
                 if m.joint(i).name.endswith('_joint'))


def gains(name):
    """Engineering initial gains, not identified H2 motor bandwidths."""
    if name.startswith('head_'):
        return 20.0, 2.0
    if 'wrist' in name:
        return 20.0, 1.0
    if 'shoulder' in name or 'elbow' in name:
        return 60.0, 3.0
    if 'ankle' in name:
        return 40.0, 2.0
    return 120.0, 6.0


def reset_float_pose():
    pose=json.loads((ROOT/'data/reset_float_pose.json').read_text())
    return tuple(pose['position_m']),tuple(pose['quaternion_wxyz'])


def build_spec():
    root = ET.parse(ROOT/'scene.xml').getroot()
    pos,quat=reset_float_pose()
    boat=root.find("./worldbody/body[@name='boat']")
    boat.set('pos',' '.join(map(str,pos)))
    boat.set('quat',' '.join(map(str,quat)))
    # mjlab applies simulation options on the assembled scene, not this entity.
    root.remove(root.find('option'))
    for weld in root.findall('./equality/weld'):
        if weld.get('name','') in ('left_hand_fixed','right_hand_fixed',
                                  'left_foot_fixed','right_foot_fixed'):
            # Six-dimensional hand/foot attachments with stiff enforcement.
            # Two physics timesteps is MuJoCo's positive-solref lower bound.
            weld.set('solref','.004 1')
            weld.set('solimp','.99 .999 .001 .5 2')
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file'):
            mesh.set('file',str(ROOT/mesh.get('file')))
    root.find('compiler').set('meshdir','')
    # H2 collision meshes use the existing CAD convex hulls. Do not turn on
    # collisions for the boat/seat: foot and seat attachments are constraints.
    bodies = {b.get('name'):b for b in root.iter('body')}
    parents = {child.get('name'):parent.get('name') for parent in root.iter('body')
               for child in parent.findall('body')}
    robot = set()
    def visit(body):
        robot.add(body.get('name'))
        for child in body.findall('body'):
            visit(child)
    visit(bodies['pelvis'])
    for name in robot:
        for index, geom in enumerate(bodies[name].findall('geom')):
            if geom.get('type') == 'mesh':
                if geom.get('mesh') == 'compact_hand':
                    # A single convex hull would fill the 34 mm sleeve bore.
                    # Tangent boxes preserve the bore; rounded rims/root are
                    # approximated for contact, while the visual mesh is exact.
                    for k in range(16):
                        angle=2*np.pi*k/16
                        pos=(.029+.021*np.cos(angle),.021*np.sin(angle),0.)
                        ET.SubElement(bodies[name],'geom',name=f'h2_collision_{name}_ring_{k}',
                            type='box',pos=' '.join(map(str,pos)),
                            quat=f'{np.cos(angle/2)} 0 0 {np.sin(angle/2)}',
                            size=f'.004 {.025*np.tan(np.pi/16)} .035',
                            contype='1',conaffinity='3',group='3',rgba='0 0 0 0',condim='3',density='0')
                    ET.SubElement(bodies[name],'geom',name=f'h2_collision_{name}_connector',
                        type='cylinder',pos='.006 0 0',quat='.707106781186548 0 .707106781186548 0',
                        size='.025 .006',contype='1',conaffinity='3',group='3',
                        rgba='0 0 0 0',condim='3',density='0')
                    continue
                attrs = dict(geom.attrib)
                attrs.update(name=f'h2_collision_{name}_{index}',contype='1',conaffinity='3',
                             group='3',rgba='0.6 0.6 0.6 0',condim='3',density='0')
                ET.SubElement(bodies[name],'geom',attrs)
    # Oar shaft/grip capsules and blade convex mesh. Original visual meshes
    # remain untouched. Bit 2 also enables contacts between the two oars.
    for side in ('port','starboard'):
        for part in ('shaft','grip'):
            mesh = trimesh.load_mesh(ROOT/f'boat_meshes/{side}_{part}.stl')
            if part == 'grip':
                # The grip STL also contains a distant, much wider collar.
                # A single capsule incorrectly made the handle ~114 mm wide,
                # causing false wrist contacts after shortening the hands.
                for k,piece in enumerate(mesh.split(only_watertight=False)):
                    low,high=piece.bounds;center=(low+high)/2
                    a,b=center.copy(),center.copy();a[1],b[1]=low[1],high[1]
                    radius=float(np.linalg.norm(piece.vertices[:,[0,2]]-center[[0,2]],axis=1).max())
                    ET.SubElement(bodies[side+'_oar'],'geom',name=f'oar_collision_{side}_{part}_{k}',
                        type='cylinder',fromto=' '.join(map(str,np.r_[a,b])),size=str(radius),
                        contype='2',conaffinity='3',group='3',rgba='0 0 0 0',density='0')
                continue
            verts = mesh.vertices
            low, high = verts.min(0), verts.max(0)
            center = (low+high)/2
            a, b = center.copy(), center.copy()
            radius = float(max(high[0]-low[0], high[2]-low[2])/2)
            a[1], b[1] = low[1]+radius, high[1]-radius
            if b[1] <= a[1]:
                a[1], b[1] = low[1], high[1]
            ET.SubElement(bodies[side+'_oar'],'geom',name=f'oar_collision_{side}_{part}',
                          type='capsule',fromto=' '.join(map(str,np.r_[a,b])),size=str(radius),
                          contype='2',conaffinity='3',group='3',rgba='0 0 0 0',density='0')
        ET.SubElement(bodies[side+'_oar'],'geom',name=f'oar_collision_{side}_blade',
                      type='mesh',mesh=side+'_blade',contype='2',conaffinity='3',
                      group='3',rgba='0 0 0 0',density='0')
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root,'contact')
    # Adjacent joint housings overlap in CAD. Exclude at most two ancestry edges;
    # non-neighbor links (including opposing limbs) still collide.
    pairs = set()
    for name in robot:
        ancestor = name
        for _ in range(2):
            ancestor = parents.get(ancestor)
            if ancestor in robot:
                pairs.add(tuple(sorted((name,ancestor))))
    # Rigidly held hand geometry intersects its own handle by design. Body-level
    # exclusions affect that hand only, not forearm/torso against the same oar.
    for hand,side in (('left','starboard'),('right','port')):
        for name in robot:
            if name.startswith(hand+'_dex1_'):
                pairs.add(tuple(sorted((name,side+'_oar'))))
    for i,(a,b) in enumerate(sorted(pairs)):
        ET.SubElement(contact,'exclude',name=f'rowing_exclude_{i}',body1=a,body2=b)
    return mujoco.MjSpec.from_string(ET.tostring(root,encoding='unicode'))


def entity_cfg():
    model = base_model()
    positions = {re.escape(model.joint(i).name):float(model.qpos0[model.jnt_qposadr[i]])
                 for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE}
    actuators = []
    for name in h2_names():
        kp,kd = gains(name)
        limit = float(model.jnt_actfrcrange[model.joint(name).id,1])
        actuators.append(BuiltinPositionActuatorCfg(target_names_expr=(name,),
            stiffness=kp,damping=kd,effort_limit=limit))
    pos,quat=reset_float_pose()
    return EntityCfg(spec_fn=build_spec,
        init_state=EntityCfg.InitialStateCfg(pos=pos,
            rot=quat,joint_pos=positions),
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators)))
