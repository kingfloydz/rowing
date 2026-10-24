"""Dode et al. JFM 2022, doi:10.1017/jfm.2022.592, Appendix E.

Deep-water Lorentzian moving-pressure benchmark, not the actual CAD hull.
Functions return beta^(4/3) times coefficients, so the unknown pressure amplitude
cancels from the normalized mean correction. No fit to unsteady measurements.
"""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.special import kv
from scipy.optimize import minimize_scalar


def steady_coefficient(fn):
    f=np.asarray(fn,float)
    if np.any(f<=0):raise ValueError('Positive Froude number required')
    a=1/(2*f*f)
    return np.exp(-a)/(2*f**8)*(kv(0,a)+(1+f*f)*kv(1,a))


@lru_cache(maxsize=1)
def maximum_steady_coefficient():
    fit=minimize_scalar(lambda f:-float(steady_coefficient(f)),bounds=(.2,1.5),method='bounded')
    return float(fit.x),float(-fit.fun)


def quasi_static_mean_correction(fn):
    # Differentiate Eq5.10 directly, avoiding the apparent missing F^-4 factor
    # in printed EqD3. Verified independently by finite-amplitude phase integral.
    h=fn*1e-4
    c0=steady_coefficient(fn);cm=steady_coefficient(fn-h);cp=steady_coefficient(fn+h)
    first=(cp-cm)/(2*h);second=(cp-2*c0+cm)/(h*h)
    return float(.5*c0+fn*first+.25*fn*fn*second)


def mean_wave_correction(fn,omega,tolerance=1e-9):
    """A0 * beta^(4/3), EqE2--E4, to order epsilon².

    Integrate in q=sqrt(K); split all Heaviside-support boundaries. A sine
    substitution resolves endpoint square-root singularities. The ideal inviscid
    resonance fn*omega=1/4 is singular, and is explicitly rejected.
    """
    if fn<=0 or omega<0:raise ValueError('Positive Fn, nonnegative frequency required')
    if omega==0:return quasi_static_mean_correction(fn)
    if abs(fn*omega-.25)<1e-9:raise ValueError('Inviscid resonance; finite-time/damping model required')
    limit=12.
    terms=[]
    for sign in [1,-1]:
        roots=np.roots([fn*fn,0,-1,-2*sign*omega,-omega*omega])
        bounds=sorted([0.,limit]+[float(x.real) for x in roots if abs(x.imag)<1e-8 and 0<x.real<limit])
        integral=0.
        for left,right in zip(bounds[:-1],bounds[1:]):
            mid=(left+right)/2;half=(right-left)/2
            if fn*fn*mid**4-(mid+sign*omega)**2<=0:continue
            def integrand(phi):
                q=mid+half*np.sin(phi)
                denom=fn*fn*q**4-(q+sign*omega)**2
                if denom<=0:return 0.
                return 2*q**4*np.exp(-q*q)*(q+sign*omega)**3/np.sqrt(denom)*half*np.cos(phi)
            integral+=quad(integrand,-np.pi/2,np.pi/2,epsabs=tolerance,epsrel=tolerance,limit=200)[0]
        terms.append(integral/(4*fn**3*omega**2))
    # K=1/F²+z² removes the only square-root singularity of E4.
    def integrand(z):
        k=1/fn**2+z*z
        return 2*k**2.5*np.exp(-k)/fn
    j2=quad(integrand,0,limit,epsabs=tolerance,epsrel=tolerance)[0]/(2*fn**3*omega**2)
    return float(sum(terms)-j2)


def normalized_mean_wave_correction(fn,omega):
    return mean_wave_correction(fn,omega)/maximum_steady_coefficient()[1]


def trajectory_wave_force(time,position_nd,fn_reference,k_points=256,k_max=35.):
    """Dimensionless Eq5.5 kernel for any 1-D trajectory from rest.

    Position is x/L, time is t*sqrt(g/L); returns beta^(4/3)*Cw normalized
    by the fixed reference speed, not instantaneous speed. O(N²*Nk) offline
    reference solver. Convergence must be checked for each chosen time window.
    For online use, precompute a state-space/spectral surrogate, not this loop.
    """
    from scipy.special import j1
    t=np.asarray(time,float);x=np.asarray(position_nd,float)
    if t.shape!=x.shape or len(t)<2 or np.any(np.diff(t)<=0) or fn_reference<=0:
        raise ValueError('Invalid trajectory')
    k=np.linspace(1e-6,k_max,k_points);weight=k**2.5*np.exp(-k)/fn_reference**2
    out=np.zeros_like(t)
    for i in range(1,len(t)):
        age=t[i]-t[:i+1];distance=x[i]-x[:i+1]
        memory=np.sin(np.sqrt(k[:,None])*age)*j1(k[:,None]*distance)
        out[i]=np.trapezoid(weight*np.trapezoid(memory,t[:i+1],axis=1),k)
    return out
