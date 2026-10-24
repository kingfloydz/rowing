"""Periodic joint-space reference; all runtime queries stay on the device."""
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from hydrodynamics.models import ROOT

PERIOD = 60.0/26.0
SIGMA_DEG = (10,4,15,10,4,15)


class OarReference:
    def __init__(self, device, path=None):
        values = np.loadtxt(path or ROOT/'data/rowing_reference.csv',delimiter=',',skiprows=1)
        if not np.allclose(values[0,1:],values[-1,1:]):
            raise ValueError('Reference must be periodic')
        self.n = len(values)-1
        if not np.allclose(values[:,0],np.linspace(0,1,self.n+1)):
            raise ValueError('Reference needs a uniform phase grid')
        self.coefficients = torch.as_tensor(CubicSpline(values[:,0],values[:,1:],bc_type='periodic').c,
                                            dtype=torch.float32,device=device)
        self.sigma = torch.deg2rad(torch.tensor(SIGMA_DEG,device=device,dtype=torch.float32))

    def __call__(self, phase):
        x = phase.remainder(1.0)*self.n
        index = x.long().clamp(0,self.n-1)
        delta = ((x-index)/self.n)[...,None]
        c = self.coefficients[:,index]
        return ((c[0]*delta+c[1])*delta+c[2])*delta+c[3]

    def nearest(self, angles):
        # Exact saved pose is common across all worlds; compute once at startup.
        grid = torch.arange(4096,device=angles.device,dtype=torch.float32)/4096
        cost = ((self(grid)-angles)/self.sigma).square().mean(-1)
        return grid[cost.argmin()]

    def target(self, elapsed, phase0, reset_angles):
        cycles = torch.where(elapsed < 2*PERIOD, elapsed/(2*PERIOD),
                             1+(elapsed-2*PERIOD)/PERIOD)
        phase = (phase0+cycles).remainder(1.0)
        rate = torch.where(elapsed < 2*PERIOD,0.5,1.0)
        u = (elapsed/0.3).clamp(0,1)
        blend = 10*u**3-15*u**4+6*u**5
        target = reset_angles+blend[...,None]*(self(phase)-reset_angles)
        return phase,rate,blend,target
