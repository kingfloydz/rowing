"""Rebase training reset joints without changing their physical kinematics."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from hydrodynamics.models import ROOT


def reset_joint_positions():
    return json.loads((ROOT/'data/reset_joint_pose.json').read_text())['joint_positions_rad']


def apply_reset_joints(root):
    """Update body transforms and joint refs together (scene uses radians/quats)."""
    positions = reset_joint_positions()
    for body in root.iter('body'):
        for joint in body.findall('joint'):
            name = joint.get('name')
            if name not in positions:
                continue
            old = float(joint.get('ref', '0'))
            if abs(positions[name]-old)<1e-12:
                continue
            if len(body.findall('joint')) != 1 or joint.get('type', 'hinge') != 'hinge':
                raise ValueError(f'Reset rebasing requires a single hinge: {name}')
            axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
            delta = Rotation.from_rotvec(axis/np.linalg.norm(axis)*(positions[name]-old))
            quat = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
            rotation = Rotation.from_quat(quat[[1,2,3,0]])
            anchor = np.fromstring(joint.get('pos', '0 0 0'), sep=' ')
            pos = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
            pos += rotation.apply(anchor-delta.apply(anchor))
            quat = (rotation*delta).as_quat()[[3,0,1,2]]
            body.set('pos', ' '.join(map(str, pos)))
            body.set('quat', ' '.join(map(str, quat)))
            joint.set('ref', str(positions[name]))
