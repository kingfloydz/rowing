"""Soft bilateral coordination around the assembled grip pose.

Compare physical joint coordinates, not normalized policy actions: left/right
joint travel and assembly offsets differ. This is anatomical joint-space
symmetry, not exact Cartesian mirroring of the hands.
"""
import math
import torch

MIRROR_PAIRS = (
    ('hip_pitch', 1), ('hip_roll', -1), ('hip_yaw', -1), ('knee', 1),
    ('ankle_roll', -1), ('ankle_pitch', 1),
    ('shoulder_pitch', 1), ('shoulder_roll', -1), ('shoulder_yaw', -1),
    ('elbow', 1), ('wrist_roll', -1), ('wrist_pitch', 1), ('wrist_yaw', -1),
)


class BilateralSymmetry:
    def __init__(self, model, command_names, device, prefix=''):
        names = [[side+'_'+part+'_joint' for part,_ in MIRROR_PAIRS] for side in ('left','right')]
        self.sign = torch.tensor([sign for _,sign in MIRROR_PAIRS],device=device,dtype=torch.float32)
        self.q, self.v, self.commands, self.rest = [], [], [], []
        for side_names in names:
            ids = [model.joint(prefix+name).id for name in side_names]
            self.q.append(torch.tensor(model.jnt_qposadr[ids].copy(),device=device,dtype=torch.long))
            self.v.append(torch.tensor(model.jnt_dofadr[ids].copy(),device=device,dtype=torch.long))
            self.commands.append(torch.tensor([command_names.index(n) for n in side_names],device=device,dtype=torch.long))
            self.rest.append(torch.tensor(model.qpos0[model.jnt_qposadr[ids]].copy(),device=device,dtype=torch.float32))
        self.position_scale = math.radians(10)
        self.velocity_scale = math.radians(60)
        self.position_tolerance = math.radians(3)
        self.velocity_tolerance = math.radians(5)

    def errors(self, qpos, qvel, commands):
        position = (qpos[...,self.q[0]]-self.rest[0])-self.sign*(qpos[...,self.q[1]]-self.rest[1])
        velocity = qvel[...,self.v[0]]-self.sign*qvel[...,self.v[1]]
        command = (commands[...,self.commands[0]]-self.rest[0])-self.sign*(commands[...,self.commands[1]]-self.rest[1])
        return position,velocity,command

    def cost(self, qpos, qvel, commands):
        position,velocity,command = self.errors(qpos,qvel,commands)
        def mse(error, tolerance, scale):
            return (torch.relu(error.abs()-tolerance)/scale).square().mean(dim=-1)
        return (0.5*mse(position,self.position_tolerance,self.position_scale)
                +0.2*mse(velocity,self.velocity_tolerance,self.velocity_scale)
                +0.3*mse(command,self.position_tolerance,self.position_scale))
