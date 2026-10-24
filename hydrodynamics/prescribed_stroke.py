"""Restricted Barré/Leroyer rectangular-blade towing-tank model.

This is a dynamic empirical closure for theta_dot=K*sin(theta), fixed immersion,
and the reference 0.300 x 0.140 m flat blade. It is NOT an instantaneous polar
for arbitrary robot trajectories. Its coefficients already contain unsteady
and free-surface effects; do not add another added-mass/history force to them.
"""
from dataclasses import dataclass
import numpy as np
from .polars import barre_published_maxima, harmonic_polar


@dataclass(frozen=True)
class PrescribedTankStroke:
    rotation_rate: float = 3.
    advance_number: float = 2.91 / (3. * 1.308)
    length_scale: float = 1.

    def __post_init__(self):
        if not (np.isfinite(self.rotation_rate) and self.rotation_rate > 0
                and np.isfinite(self.length_scale) and self.length_scale > 0
                and 0 < self.advance_number < 1):
            raise ValueError('Finite positive rate/scale and 0 < advance < 1 required')

    @property
    def lever(self): return 1.308 * self.length_scale

    @property
    def chord(self): return .300 * self.length_scale

    @property
    def span(self): return .140 * self.length_scale

    @property
    def area(self): return .042 * self.length_scale**2

    @property
    def boat_speed(self): return self.advance_number * self.lever * self.rotation_rate

    @property
    def reduced_frequency(self): return self.chord / (self.lever * (1-self.advance_number))

    def kinematics(self, theta):
        """SI arrays; theta in radians, t=0 at 90 deg, boat moves toward -x.

        Blade centre = (-Vb*t-Le*cos(theta), -Le*sin(theta)). The analytical
        position, velocity and acceleration refer to the same physical point.
        'angle' uses trajectory.py's normal/chord convention, not shaft angle.
        """
        th=np.asarray(theta,float)
        if th.ndim != 1 or np.any(~np.isfinite(th)) or np.any((th<=0)|(th>=np.pi)):
            raise ValueError('One-dimensional finite angles strictly between 0 and pi required')
        if np.any(np.diff(th)<=0): raise ValueError('Angles must be strictly increasing')
        omega=self.rotation_rate*np.sin(th)
        angular_acc=self.rotation_rate**2*np.sin(th)*np.cos(th)
        time=np.log(np.tan(th/2))/self.rotation_rate
        normal=np.column_stack((-np.sin(th),np.cos(th)))
        tangent=np.column_stack((np.cos(th),np.sin(th)))
        position=-self.lever*tangent-np.column_stack((self.boat_speed*time,np.zeros_like(th)))
        velocity=-self.lever*omega[:,None]*normal
        velocity[:,0]-=self.boat_speed
        acceleration=self.lever*(omega[:,None]**2*tangent-angular_acc[:,None]*normal)
        incidence=np.arctan2((1-self.advance_number)*self.lever*omega,
                            self.boat_speed*np.cos(th))
        return dict(time=time,position=position,velocity=velocity,acceleration=acceleration,
                    angle=th+np.pi/2,omega=omega,incidence=incidence,
                    normal_froude=(1-self.advance_number)*self.lever*omega/np.sqrt(9.81*self.chord))

    def force(self, theta, rho=1000.):
        """Published harmonic dynamic coefficients, no fit to Leroyer's curves.

        Froude scaling maps K to the reference-scale K*sqrt(length_scale).
        Empirical support checks are inherited from barre_published_maxima.
        This scaling is a similarity assumption, not a new full-scale validation.
        """
        if not np.isfinite(rho) or rho<=0: raise ValueError('Positive finite density required')
        kin=self.kinematics(theta)
        cdmax,clmax=barre_published_maxima(self.reduced_frequency,
                                        self.rotation_rate*np.sqrt(self.length_scale))
        cd,cl=harmonic_polar(kin['incidence'],cdmax,clmax)
        v=kin['velocity'];speed=np.linalg.norm(v,axis=1)
        # vn is negative throughout this prescribed motion. Lift direction is
        # therefore -z cross v; drag opposes the blade velocity in still water.
        force=.5*rho*self.area*speed[:,None]*(-cd[:,None]*v
                      -cl[:,None]*np.column_stack((-v[:,1],v[:,0])))
        return dict(force=force,cd=cd,cl=cl,cd_max=float(cdmax),cl_max=float(clmax),
                    reduced_frequency=self.reduced_frequency,
                    reference_rotation_rate=self.rotation_rate*np.sqrt(self.length_scale))

    def moment_from_force(self, theta, force, centre_of_pressure_shift=0.):
        """Mz about the oarlock; positive shift points from centre toward shaft.

        No pressure-centre prediction is implied. With the default zero shift,
        this is a centre-application baseline to audit against measured moments.
        A supplied shift must be independently measured or modelled.
        """
        th=np.asarray(theta);f=np.asarray(force)
        return (self.lever-np.asarray(centre_of_pressure_shift))*(f[:,0]*np.sin(th)-f[:,1]*np.cos(th))
