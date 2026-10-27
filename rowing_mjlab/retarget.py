"""Catch-aligned soft priors, not a dynamically feasible demonstration.

The elbow angle is defined in the plane normal to its hinge axis. This avoids
calling the non-collinear CAD shoulder/elbow/wrist anchors an anatomical zero.
"""
import numpy as np


def adjusted_priors(prior):
    """Already catch-aligned source data; do not alter its timing or shape."""
    return prior[:,1].copy(), prior[:,2].copy()


def elbow_calibration(model, prefix=''):
    """Identify projected extension zero and select the outward branch at qpos0.

    Outward is the elbow's lateral displacement from the shoulder-wrist chord
    in the torso frame. Subsequent shoulder motion requires its own posture
    constraint; this calibration alone cannot certify a whole-body trajectory.
    """
    import mujoco
    data = mujoco.MjData(model)
    torso = model.body(prefix+'torso_link').id
    result = []
    for side, outward_sign in [('left', 1), ('right', -1)]:
        ids = [model.joint(prefix+side+'_'+part+'_joint').id
               for part in ('shoulder_yaw', 'elbow', 'wrist_roll')]
        jid = ids[1]
        qa = model.jnt_qposadr[jid]
        def geometry(q):
            data.qpos[:] = model.qpos0
            data.qpos[qa] = q
            mujoco.mj_kinematics(model, data)
            s, e, w = data.xanchor[ids].copy()
            a, b = e-s, w-e
            bend = np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b), -1, 1))
            axis = data.xaxis[jid]
            a -= axis*(a@axis)
            b -= axis*(b@axis)
            angle = np.arctan2(axis@np.cross(a, b), a@b)
            chord = w-s
            projection = s+chord*((e-s)@chord)/(chord@chord)
            outward = outward_sign*((e-projection)@data.xmat[torso].reshape(3,3))[1]
            return angle, outward, bend
        q = model.qpos0[qa]
        angle = geometry(q)[0]
        zero = q-angle
        candidates = [sign for sign in (-1, 1)
                      if model.jnt_range[jid,0] < zero+sign*.6 < model.jnt_range[jid,1]]
        sign = max(candidates, key=lambda s: geometry(zero+s*.6)[1])
        if geometry(zero+sign*.6)[1] <= 0:
            raise ValueError(f'{side}: no outward elbow branch at calibration pose')
        result.append(dict(side=side, zero_rad=float(zero), direction=sign,
                           minimum_anchor_bend_deg=float(np.rad2deg(geometry(zero)[2])),
                           qpos_index=int(qa), joint_id=int(jid)))
    return result


def retarget_seat_elbows(prior, seat_calibration, model, prefix=''):
    seat, human_angle = adjusted_priors(prior)
    seat = (seat_calibration['finish_seat_joint_m']
            + seat_calibration['source_seat_span_m']*(seat-1))
    calibration = elbow_calibration(model, prefix)
    elbows = np.column_stack([c['zero_rad']+c['direction']*human_angle for c in calibration])
    values = np.column_stack([seat, elbows])
    ids = [model.joint(prefix+'seat_slide').id] + [c['joint_id'] for c in calibration]
    limits = model.jnt_range[ids]
    if np.any(values < limits[:,0]-1e-8) or np.any(values > limits[:,1]+1e-8):
        raise ValueError('Retargeted seat/elbow reference exceeds joint limits; recalibrate')
    return values, calibration
