"""Coaching-informed soft priors, not a dynamically feasible demonstration.

The elbow angle is defined in the plane normal to its hinge axis. This avoids
calling the non-collinear CAD shoulder/elbow/wrist anchors an anatomical zero.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


def adjusted_priors(prior):
    """Preserve source shapes, with explicit designer-selected event times.

    Seat: catch at 0, finish of leg drive at .36, return begins at .58.
    Elbows: peak at .40, source late-recovery (.85) reached at .58.
    These times are initial coordination settings, not measured synchrony.
    """
    phase, seat, elbow = prior.T
    catch, finish = phase[seat.argmin()], phase[seat.argmax()]
    seat = (seat-seat.min()) / np.ptp(seat)
    source = PchipInterpolator(np.r_[phase[:-1]-1, phase, phase[1:]+1],
                               np.r_[seat[:-1], seat, seat[1:]])
    # Zero time-map velocity at segment endpoints makes the dwell and seam C1.
    def ease(x):
        return x*x*(3-2*x)
    time = np.empty_like(phase)
    drive, recovery = phase < .36, phase > .58
    time[drive] = catch+(finish-catch)*ease(phase[drive]/.36)
    time[~(drive | recovery)] = finish
    time[recovery] = finish+(1+catch-finish)*ease((phase[recovery]-.58)/.42)
    adjusted_seat = source(time)
    # Remove tiny digitization reversals, independently on each stroke half.
    adjusted_seat[drive] = np.maximum.accumulate(adjusted_seat[drive])
    adjusted_seat[recovery] = np.minimum.accumulate(adjusted_seat[recovery])
    adjusted_seat[0] = adjusted_seat[-1] = 0.
    elbow_time = PchipInterpolator([0, .20, .40, .58, 1], [0, .20, .40, .85, 1])(phase)
    adjusted_elbow = PchipInterpolator(phase, elbow)(elbow_time)
    adjusted_elbow[-1] = adjusted_elbow[0]
    return adjusted_seat, adjusted_elbow


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
