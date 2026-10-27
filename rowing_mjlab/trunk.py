"""T6-inspired task-space trunk angle, not a waist encoder coordinate."""
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from hydrodynamics.models import ROOT


def trunk_vertical_angle(torso_rotation, boat_rotation):
    """Signed sagittal tilt from world vertical, positive toward rower's back.

    torso_link +Z is its longitudinal axis. In this assembly boat +X points
    toward the rower's back. Use its horizontal heading so boat pitch does not
    redefine vertical. Both arguments have shape (..., 3, 3), local-to-world.
    Roll alone is not counted as sagittal flexion; yaw rotates the plane.
    """
    up = torso_rotation[..., :, 2]
    heading = boat_rotation[..., :2, 0]
    heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1e-8)
    backward = (up[..., :2]*heading).sum(dim=-1)
    return torch.atan2(backward, up[..., 2])


def angular_error(actual, target):
    delta = actual-target
    return torch.atan2(torch.sin(delta), torch.cos(delta))


class TrunkReference:
    def __init__(self, device, path=None):
        table = np.loadtxt(path or ROOT/'data/reference9/trunk_vertical_prior.csv', delimiter=',', skiprows=1)
        phase, angle = table.T
        if (not np.isfinite(table).all() or not np.allclose(phase, np.linspace(0,1,len(phase)))
                or not np.isclose(angle[0], angle[-1])):
            raise ValueError('Trunk reference requires finite periodic samples on a uniform phase grid')
        self.n = len(phase)-1
        self.coefficients = torch.as_tensor(PchipInterpolator(phase, angle).c,
                                             dtype=torch.float32, device=device)
        self.sigma = float(np.deg2rad(10))  # Initial tuning value, not a paper SD.

    def __call__(self, phase):
        x = phase.remainder(1)*self.n
        index = x.long().clamp(0, self.n-1)
        delta = (x-index)/self.n
        c = self.coefficients[:, index]
        return ((c[0]*delta+c[1])*delta+c[2])*delta+c[3]

    def target(self, phase, blend, initial):
        return initial+blend*angular_error(self(phase), initial)
