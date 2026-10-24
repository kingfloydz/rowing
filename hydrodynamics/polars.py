"""Angle-resolved blade forces and rotating-tank response surfaces.

Static and dynamic models are alternative closures: do not sum their forces.
No automatic blending with depth/history factors whose joint calibration is absent.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class BladePolar:
    angle_deg: np.ndarray
    cd: np.ndarray
    cl: np.ndarray

    def coefficients(self, alpha):
        """alpha in [0,pi], using Caplan's chord/free-stream convention."""
        x = np.degrees(np.asarray(alpha))
        if np.any((x < -1e-8) | (x > 180+1e-8)):
            raise ValueError('Polar is measured on 0..180 degrees only')
        x = np.clip(x, self.angle_deg[0], self.angle_deg[-1])
        return np.maximum(PchipInterpolator(self.angle_deg, self.cd)(x), 0), PchipInterpolator(self.angle_deg, self.cl)(x)

    @classmethod
    def caplan_flat(cls, path=None, training_only=True):
        p=Path(path or Path(__file__).resolve().parents[1]/'data/caplan2007_flat_polar.csv')
        a=np.genfromtxt(p,delimiter=',',names=True)
        if training_only: a=a[a['angle_deg']%10 == 0]
        return cls(a['angle_deg'], a['cd'], a['cl'])

    def force(self, relative_velocity, normal, chord_direction, area, rho=1000.):
        """Planar force on blade. Relative velocity is blade minus water.

        alpha=atan2(|v.n|, v.chord). Reverse-side normal flow mirrors a flat
        blade; that extension is an assumption, not curved-blade validation.
        Spanwise velocity is excluded, as it is absent from measured polars.
        Lift is exactly perpendicular to planar v; drag is dissipative.
        """
        v=np.asarray(relative_velocity); n=np.asarray(normal); c=np.asarray(chord_direction)
        vn=v@n; vt=v@c; u=np.hypot(vn,vt)
        alpha=np.arctan2(np.abs(vn),vt)
        cd,cl=self.coefficients(alpha)
        e=np.divide(vt[...,None]*c+vn[...,None]*n,u[...,None],out=np.zeros_like(v),where=u[...,None]>1e-12)
        # lift = (|vn| c - sign(vn) vt n)/u; positive CL then resists normal motion.
        lift=np.divide(np.abs(vn)[...,None]*c-(np.sign(vn)*vt)[...,None]*n,
                       u[...,None],out=np.zeros_like(v),where=u[...,None]>1e-12)
        return (.5*rho*np.asarray(area)*u*u)[...,None]*(-cd[...,None]*e+cl[...,None]*lift)


def barre_published_maxima(reduced_frequency, rotation_rate):
    """Fig.11 published regressions. SI K range 3..4 rad/s, scale 0.7 only.

    fr=c/(Le*(1-eta0)); this expression holds for their theta_dot=K*sin(theta).
    These are dynamic coefficients already containing unsteady effects.
    """
    if not 3 <= rotation_rate <= 4:
        raise ValueError('Barre K outside 3..4 rad/s measured range')
    f=np.asarray(reduced_frequency)
    k=np.array([3.,3.5,3.75,4.])
    bounds=np.array([[.55,1.76],[.64,1.15],[.64,1.15],[.55,1.]])
    matches=np.where(k==rotation_rate)[0]
    if len(matches):
        lo,hi=bounds[matches[0]]
    else:
        right=np.searchsorted(k,rotation_rate)
        lo=max(bounds[right-1,0],bounds[right,0])
        hi=min(bounds[right-1,1],bounds[right,1])
    if np.any((f<lo)|(f>hi)):
        raise ValueError('Reduced frequency outside support for this rotation rate')
    # Explicit association checked against the symbols in Fig.11.
    d_s=np.array([1.7224,1.8472,1.9614,1.804])
    d_i=np.array([.7014,.5044,.284,.3218])
    l_s=np.array([.6359,.6628,.5774,.4454])
    l_i=np.array([.3982,.226,.2167,.2845])
    return (np.interp(rotation_rate,k,d_s)*f+np.interp(rotation_rate,k,d_i),
            np.interp(rotation_rate,k,l_s)*f+np.interp(rotation_rate,k,l_i))


def harmonic_polar(alpha, cd_max, cl_max):
    """Barre2010 Eq10; retains tangential force when CDmax != 2CLmax."""
    return cd_max*np.sin(alpha)**2, cl_max*np.sin(2*alpha)
