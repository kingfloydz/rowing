"""Shared periodic whole-body targets; metadata declares hard or soft fitting."""
import json
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from hydrodynamics.models import ROOT

DIRECTORY=ROOT/'data/reference20'


def available():
    return (DIRECTORY/'trajectory.npz').is_file()


class CoordinatedReference:
    def __init__(self,device):
        data=np.load(DIRECTORY/'trajectory.npz')
        self.names=list(data['names'])
        self.metadata=json.loads((DIRECTORY/'metadata.json').read_text())
        self.n=len(data['phase'])-1
        values=np.column_stack([data['q'],data['trunk']])
        if (not np.isfinite(values).all() or self.n<4
            or not np.allclose(data['phase'],np.linspace(0,1,self.n+1))
            or not np.allclose(values[0],values[-1],atol=1e-9)):
            raise ValueError('Coordinated reference must be finite, periodic and uniformly sampled')
        self.coefficients=torch.as_tensor(CubicSpline(data['phase'],values,bc_type='periodic').c,dtype=torch.float32,device=device)

    def __call__(self,phase):
        x=phase.remainder(1)*self.n
        index=x.long().clamp(0,self.n-1);delta=((x-index)/self.n)[...,None]
        c=self.coefficients[:,index]
        return ((c[0]*delta+c[1])*delta+c[2])*delta+c[3]

    def joints(self,phase,names):
        return self(phase)[...,[self.names.index(n) for n in names]]
