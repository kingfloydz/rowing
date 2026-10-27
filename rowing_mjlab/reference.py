"""Periodic joint-space reference; all runtime queries stay on the device."""
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from hydrodynamics.models import ROOT

PERIOD = 60.0/26.0
SIGMA_DEG = (10,4,15,10,4,15)
OAR_REFERENCE_PATH = ROOT/'data/rowing_reference.csv'
# Interior of the digitized drive, excluding catch/extraction transitions.
# Verified at the repository waterline: both blades >70% wetted surface area
# over this interval; sweep advances and the blade remains approximately square.
DRIVE_START_PHASE = (0.05, 0.35)


class OarReference:
    def __init__(self, device, path=None):
        import json
        offsets = json.loads((ROOT/'data/reference_phase_offsets.json').read_text())
        self.phase_offset = ((json.loads((ROOT/'data/oar_finish_alignment.json').read_text())['phase_delay']
                              - offsets['oar_phase_advance']) if path is None else 0.0)
        self.seat_phase_offset = offsets['seat_phase_delay']
        values = np.loadtxt(path or OAR_REFERENCE_PATH,delimiter=',',skiprows=1)
        if not np.allclose(values[0,1:],values[-1,1:]):
            raise ValueError('Reference must be periodic')
        self.n = len(values)-1
        if not np.allclose(values[:,0],np.linspace(0,1,self.n+1)):
            raise ValueError('Reference needs a uniform phase grid')
        self.coefficients = torch.as_tensor(CubicSpline(values[:,0],values[:,1:],bc_type='periodic').c,
                                            dtype=torch.float32,device=device)
        self.sigma = torch.deg2rad(torch.tensor(SIGMA_DEG,device=device,dtype=torch.float32))

    def __call__(self, phase):
        return self._evaluate((phase-self.phase_offset).remainder(1.0))

    def _evaluate(self, phase):
        x = phase.remainder(1.0)*self.n
        index = x.long().clamp(0,self.n-1)
        delta = ((x-index)/self.n)[...,None]
        c = self.coefficients[:,index]
        return ((c[0]*delta+c[1])*delta+c[2])*delta+c[3]

    def nearest(self, angles):
        """One shared bilateral phase, nearest in sweep only, inside drive."""
        grid = torch.arange(4096,device=angles.device,dtype=torch.float32)/4096
        sweep = torch.tensor([0,3],device=angles.device)
        cost = (self(grid).index_select(-1,sweep)-angles.index_select(-1,sweep)).square().mean(-1)
        start,end = ((p+self.phase_offset)%1 for p in DRIVE_START_PHASE)
        valid = ((grid >= start)&(grid <= end) if start <= end
                 else (grid >= start)|(grid <= end))
        cost = cost.masked_fill(~valid,float('inf'))
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


IMITATION_NAMES = tuple(s+'_'+j for s in ('port','starboard')
                        for j in ('sweep','elevation','feather')) + (
                            'seat_slide','left_elbow_joint','right_elbow_joint')


class RowingReference(OarReference):
    """Nine synchronized soft targets; not a certified closed-chain trajectory.

    Base curves use Catch/Finish alignment with independent manual phase offsets. Elbow flexion is
    mapped to the outward branch in each hinge plane using the model geometry.
    These are soft priors, not certified closed-chain human demonstrations.
    """
    def __init__(self, device, model, prefix=''):
        import json
        from scipy.interpolate import PchipInterpolator
        from .retarget import retarget_seat_elbows

        super().__init__(device)
        calibration = json.loads((ROOT/'data/seat_calibration.json').read_text(encoding='utf-8'))
        prior = np.loadtxt(ROOT/'data/reference9/aligned_priors.csv',delimiter=',',skiprows=1)
        phase = prior[:,0]
        oars = np.loadtxt(OAR_REFERENCE_PATH,delimiter=',',skiprows=1)
        values = np.empty((len(phase),9))
        values[:,:6] = CubicSpline(oars[:,0],oars[:,1:],bc_type='periodic')(phase)
        values[:,6:], self.elbow_calibration = retarget_seat_elbows(prior, calibration, model, prefix)
        values[-1] = values[0]
        self.n = len(phase)-1
        if not np.allclose(phase,np.linspace(0,1,self.n+1)):
            raise ValueError('Reference needs a uniform phase grid')
        # Shape-preserving interpolation avoids overshooting seat/elbow limits.
        self.coefficients = torch.as_tensor(PchipInterpolator(phase,values).c,
                                             dtype=torch.float32,device=device)
        # Reuse the same shifted oar spline as OarReference.
        self.oars = OarReference(device)
        self.seat_sigma = 0.05  # metres
        self.elbow_sigma = float(np.deg2rad(15))

    def __call__(self, phase):
        # Manual shifts are independent; elbows keep their source phase.
        values = self._evaluate(phase)
        seat = self._evaluate((phase-self.seat_phase_offset).remainder(1.0))[...,6:7]
        return torch.cat((self.oars(phase),seat,values[...,7:]),dim=-1)
