"""Conservative-in-energy TRAINING approximations, not calibrated CAD truth.

Day residuals are priors. Extrapolation and episode ranges are engineering
assumptions; positivity is a numerical/energetic constraint, NOT an error bound.
The scoped experimental references and legacy adapter are intentionally separate.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from .models import ROOT, G


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def skin_friction(reynolds):
    """Laminar/ITTC blend; Re<1 regularized, 1e5..1e6 transition assumed.

    Plate-mean Blasius coefficient at low Re and ITTC at Re>=1e6.
    This transition interval is NOT measured for the rowing hull.
    """
    re = np.asarray(reynolds, float)
    if np.any(~np.isfinite(re)) or np.any(re < 0):
        raise ValueError('Reynolds number must be finite and nonnegative')
    r = np.maximum(re, 1.)
    lam = 1.328/np.sqrt(r)
    turb = .075/(np.log10(np.maximum(r, 1e5))-2.)**2
    w = smoothstep((np.log10(r)-5.))
    return (1-w)*lam+w*turb


@dataclass(frozen=True)
class TrainingParameters:
    friction_scale: float = 1.0
    residual_scale: float = 1.0
    residual_tilt: float = 0.0
    high_fn_power: float = 2.0
    hull_crossflow_cd: float = 1.0
    blade_cd_max: float = 2.0
    blade_cl_max: float = .9
    blade_skin_cd: float = .01
    blade_scale: float = 1.0
    normal_blend: float = .5
    fin_efficiency: float = .8
    fin_normal_cd: float = 1.2
    viscosity: float = 1e-6

    def __post_init__(self):
        if not all(np.isfinite(x) for x in asdict(self).values()):
            raise ValueError('Non-finite training parameters')
        for name in ('friction_scale','residual_scale','hull_crossflow_cd',
                     'blade_cd_max','blade_scale','fin_efficiency',
                     'fin_normal_cd','viscosity'):
            if getattr(self,name) <= 0:
                raise ValueError(name+' must be positive')
        if self.blade_cl_max < 0 or self.blade_skin_cd < 0:
            raise ValueError('Blade coefficients must be nonnegative')
        if not 0 <= self.normal_blend <= 1 or not 0 < self.fin_efficiency <= 1:
            raise ValueError('Invalid blend/efficiency')
        if not 0 <= self.high_fn_power <= 4 or abs(self.residual_tilt) > 1:
            raise ValueError('Outside supported engineering extrapolation family')
        if not 0.5e-6 <= self.viscosity <= 2e-6:
            raise ValueError('Viscosity outside freshwater training profile')


# Suggested sensitivity ranges, NOT confidence intervals from experiments.
EPISODE_RANGES = {
    'friction_scale': (.8,1.2), 'residual_scale': (.5,2.),
    'residual_tilt': (-.3,.3), 'high_fn_power': (0.,4.),
    'hull_crossflow_cd': (.6,1.6), 'blade_cd_max': (1.5,2.5),
    'blade_cl_max': (.6,1.2), 'blade_skin_cd': (.005,.025),
    'blade_scale': (.8,1.2), 'normal_blend': (0.,1.),
    'fin_efficiency': (.6,.95), 'fin_normal_cd': (.8,1.8),
    'viscosity': (.8e-6,1.3e-6),
}


def sample_episode(seed):
    """Reproducible immutable sample; caller samples ONCE at episode reset.

    Scales are log-uniform, remaining scalars uniform. These independent priors
    are a robustness experiment, not a joint posterior for the real vessel.
    """
    rng = np.random.default_rng(seed)
    return TrainingParameters(**{k: float(np.exp(rng.uniform(np.log(a),np.log(b))))
        if k.endswith('_scale') else float(rng.uniform(a,b))
        for k,(a,b) in EPISODE_RANGES.items()})


class TrainingHullResistance:
    """C1 positive Day-residual continuation, intended for 0..4 m/s training.

    Speed is NEVER clipped: target limits must not cap physical drag on overshoot.
    Same-Fn transfer does not imply hull-shape similarity.
    """
    def __init__(self, parameters=None, path=None):
        self.p = parameters or TrainingParameters()
        h = json.loads((path or ROOT/'data/parameters.json').read_text())['hull']
        self.fn = np.asarray(h['fn_nodes']);self.cr = np.asarray(h['residual_nodes'])
        if np.any(self.cr <= 0) or np.any(np.diff(self.fn)<=0):
            raise ValueError('Positive ordered reference data required')
        self.form_factor = h['form_factor']
        self.interpolator = PchipInterpolator(self.fn,self.cr)
        self.end_slopes = np.array([self.fn[i]*self.interpolator.derivative()(self.fn[i])/self.cr[i]
                                   for i in (0,-1)])

    def residual(self, fn):
        f = np.asarray(fn,float)
        if np.any(~np.isfinite(f)) or np.any(f<0):
            raise ValueError('Fn must be finite and nonnegative')
        lo,hi = self.fn[[0,-1]]
        # Low-Fn C_R~Fn^4 is a smooth engineering closure, not a wave theorem.
        ratio = np.minimum(f/lo,1.)
        low = self.cr[0]*ratio**4*np.exp((4-self.end_slopes[0])*(1-ratio))
        # Logarithmic slope transitions continuously from measured endpoint to p.
        x = np.log(np.maximum(f/hi,1.));tau=.1
        exponent = self.end_slopes[1]*x+(self.p.high_fn_power-self.end_slopes[1])*(x-tau*(-np.expm1(-x/tau)))
        high = self.cr[-1]*np.exp(exponent)
        middle = self.interpolator(np.clip(f,lo,hi))
        cr = np.where(f<lo,low,np.where(f>hi,high,middle))
        # Smooth positive shape uncertainty, bounded multiplier at all Fn.
        shape = np.exp(self.p.residual_tilt*np.tanh((f-.33)/.15))
        return self.p.residual_scale*shape*cr

    def components(self,speed,length,wetted_area,rho=1000.):
        u = np.asarray(speed,float)
        if (np.any(~np.isfinite(u)) or not np.isfinite(length) or length<=0
            or not np.isfinite(wetted_area) or wetted_area<0 or not np.isfinite(rho) or rho<=0):
            raise ValueError('Invalid hull inputs')
        fn=np.abs(u)/np.sqrt(G*length);re=np.abs(u)*length/self.p.viscosity
        cf=self.p.friction_scale*(1+self.form_factor)*skin_friction(re)
        cr=self.residual(fn);dynamic=.5*rho*wetted_area*u*u
        return {'friction_N':dynamic*cf,'residual_N':dynamic*cr,
                'total_N':dynamic*(cf+cr),'Fn':fn,'Re':re,
                'outside_reference':(fn<self.fn[0])|(fn>self.fn[-1]),
                'outside_target_speed':np.abs(u)>4.}

    def magnitude(self,speed,length,wetted_area,rho=1000.):
        return self.components(speed,length,wetted_area,rho)['total_N']


def blade_traction(velocity,normal,chord,area,p=None,rho=1000.,mode='harmonic'):
    """Local planar quasi-steady traction, with explicit spanwise skin drag.

    Harmonic CD/CL structure follows Barre, but constants are engineering priors,
    not its K/fr regression. Triangle-wise use is an additional approximation.
    No additional depth gain, history force or added mass is superposed.
    """
    p=p or TrainingParameters()
    v=np.asarray(velocity,float);n=np.asarray(normal,float);c=np.asarray(chord,float)
    if not (np.isclose(n@n,1) and np.isclose(c@c,1) and abs(n@c)<1e-8):
        raise ValueError('Orthonormal blade directions required')
    vn=v@n;vt=v@c;u=np.hypot(vn,vt)
    if mode=='normal':
        scalar=(1-p.normal_blend)*abs(vn)*vn+p.normal_blend*u*vn
        f=-.5*rho*np.asarray(area)[...,None]*p.blade_cd_max*scalar[...,None]*n
        tangent=v-vn[...,None]*n
    elif mode=='harmonic':
        alpha=np.arctan2(abs(vn),vt)
        cd=p.blade_cd_max*np.sin(alpha)**2+p.blade_skin_cd
        cl=p.blade_cl_max*np.sin(2*alpha)
        e=np.divide(vt[...,None]*c+vn[...,None]*n,u[...,None],out=np.zeros_like(v),where=u[...,None]>0)
        el=np.divide(abs(vn)[...,None]*c-(np.sign(vn)*vt)[...,None]*n,
                     u[...,None],out=np.zeros_like(v),where=u[...,None]>0)
        f=(.5*rho*np.asarray(area)*u*u)[...,None]*(-cd[...,None]*e+cl[...,None]*el)
        tangent=v-vt[...,None]*c-vn[...,None]*n
    else:
        raise ValueError('Choose harmonic or normal blade closure')
    f-= (.5*rho*np.asarray(area)*p.blade_skin_cd*np.linalg.norm(tangent,axis=-1))[...,None]*tangent
    return p.blade_scale*f


def fin_force(velocity,chord,normal,area,aspect_ratio,p=None,rho=1000.):
    """Passive small-incidence finite-wing approximation blended to plate drag.

    Lift slope a=2pi/(1+2/(e AR)), CDi=CL^2/(pi e AR). Blend 15..35 deg
    is an engineering stall regularization, NOT measured fin polar. Parasitic
    axial skin drag is left in the transferred total hull prior to avoid counting
    another complete axial drag law. Uses local water-relative point velocity.
    """
    p=p or TrainingParameters();v=np.asarray(velocity,float)
    c=np.asarray(chord,float);n=np.asarray(normal,float)
    if not (np.isclose(n@n,1) and np.isclose(c@c,1) and abs(n@c)<1e-8):
        raise ValueError('Orthonormal fin directions required')
    if area<0 or aspect_ratio<=0: raise ValueError('Invalid fin geometry')
    u=float(v@c);vn=float(v@n);speed=np.hypot(u,vn)
    if speed==0 or area==0:return np.zeros(3)
    cs=u/speed;sn=vn/speed
    a=2*np.pi/(1+2/(p.fin_efficiency*aspect_ratio))
    cl=a*sn*cs;cdi=cl*cl/(np.pi*p.fin_efficiency*aspect_ratio)
    direction=cs*c+sn*n;lift=sn*c-cs*n
    q=.5*rho*area*speed*speed
    attached=q*(cl*lift-cdi*direction)
    separated=-q*p.fin_normal_cd*abs(sn)*sn*n
    angle=np.arctan2(abs(vn),abs(u))
    mix=smoothstep((angle-np.deg2rad(15))/np.deg2rad(20))
    return (1-mix)*attached+mix*separated
