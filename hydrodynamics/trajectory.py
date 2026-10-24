"""Prescribed-motion blade benchmark and passive hull memory building block.

The blade closure is a candidate reduced model, not a CFD replacement validated
for arbitrary strokes. Parameters must be calibrated and tested on whole cases.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class BladeTrajectoryParameters:
    drag_scale: float = 1.
    lift_scale: float = 1.
    circulation_lag_chords: float = 0.
    added_mass_scale: float = 1.


def blade_trajectory_force(time, velocity, acceleration, angle, omega, polar,
                           area, chord, span, parameters=None, rho=1000.,
                           initial_circulation=0.):
    """2-D force at the prescribed reference point; no dry-body inertia.

    angle is CCW from outward +y chord, velocity is body minus still water.
    Circulation follows a first-order convective lag. The added-mass tensor
    rotates with the blade, so both normal and tangential inertia terms appear.
    Return the intrinsic added-mass torque needed for energy consistency too.
    Area/chord choice and reference-point location must be explicit at call site.
    This function never infers parameters from measured forces.
    """
    p=parameters or BladeTrajectoryParameters()
    t=np.asarray(time,float);v=np.asarray(velocity,float);a=np.asarray(acceleration,float)
    angle=np.asarray(angle,float);omega=np.asarray(omega,float)
    if len(t)<2 or np.any(np.diff(t)<=0):raise ValueError('Strictly increasing time required')
    if min(area,chord,span,rho)<=0:raise ValueError('Positive geometry and density required')
    if min(p.drag_scale,p.lift_scale,p.circulation_lag_chords,p.added_mass_scale)<0:
        raise ValueError('Nonnegative parameters required')
    n=np.column_stack((np.cos(angle),np.sin(angle)))
    c=np.column_stack((-np.sin(angle),np.cos(angle)))
    vn=np.sum(v*n,axis=1);vt=np.sum(v*c,axis=1);u=np.linalg.norm(v,axis=1)
    cd,cl=polar.coefficients(np.arctan2(np.abs(vn),vt))
    drag=-.5*rho*area*(p.drag_scale*cd*u)[:,None]*v
    # A signed circulation avoids a discontinuous lift-direction flip on reversal.
    gamma_eq=.5*chord*u*p.lift_scale*cl*np.sign(vn)
    gamma=np.empty(len(t));gamma[0]=initial_circulation
    if p.circulation_lag_chords==0:
        gamma=gamma_eq.copy()
    else:
        for i,dt in enumerate(np.diff(t),1):
            rate=.5*(u[i]+u[i-1])/(p.circulation_lag_chords*chord)
            weight=-np.expm1(-rate*dt)
            gamma[i]=gamma[i-1]+weight*(.5*(gamma_eq[i]+gamma_eq[i-1])-gamma[i-1])
    # z cross v; span here is effective area/chord for the circulation force.
    lift=(rho*area/chord*gamma)[:,None]*np.column_stack((-v[:,1],v[:,0]))
    ma=p.added_mass_scale*.84*np.pi/4*rho*span*chord**2
    ndot=omega[:,None]*c
    inertia=-ma*(n*(np.sum(a*n,axis=1)+np.sum(ndot*v,axis=1))[:,None]+vn[:,None]*ndot)
    torque=ma*vn*(n[:,0]*v[:,1]-n[:,1]*v[:,0])
    return {'force':drag+lift+inertia,'drag':drag,'lift':lift,'inertia':inertia,
            'circulation':gamma,'added_mass_kg':ma,'intrinsic_torque':torque}


def passive_surge_memory(time, velocity, acceleration, added_mass, damping, decay_rates):
    """Passive Prony realization of a linear radiation-memory force.

    zdot_j=-lambda_j*z_j+u; F=-ma*udot-sum(b_j*z_j).
    Positive b,lambda imply nonnegative dissipation after stored-energy change.
    Supply perturbation velocity about a fixed reference advance speed, not the
    total speed, when steady wave resistance is already in R(U). No fitted hull
    coefficients are supplied: mean towing data cannot identify this kernel.
    """
    t=np.asarray(time,float);u=np.asarray(velocity,float);a=np.asarray(acceleration,float)
    b=np.asarray(damping,float);rates=np.asarray(decay_rates,float)
    if len(t)<2 or np.any(np.diff(t)<=0) or added_mass<0 or np.any(b<0) or np.any(rates<=0) or b.shape!=rates.shape:
        raise ValueError('Invalid passive memory parameters or time')
    z=np.zeros((len(t),len(b)))
    # Exact update for linearly varying u on each timestep.
    for i,dt in enumerate(np.diff(t),1):
        e=np.exp(-rates*dt);h=-np.expm1(-rates*dt)
        z[i]=e*z[i-1]+u[i-1]*h/rates+(u[i]-u[i-1])/dt*(dt/rates-h/rates**2)
    force=-added_mass*a-z@b
    energy=.5*added_mass*u*u+.5*(z*z)@b
    dissipation=(z*z)@(b*rates)
    return {'force':force,'state':z,'stored_energy':energy,'dissipation':dissipation}


@dataclass
class WakeMomentumParameters:
    drag_scale: float = 1.
    lift_scale: float = 1.
    wake_mass_coefficient: float = 1.
    wake_decay_chords: float = 1.
    added_mass_scale: float = 0.


def wake_momentum_force(time, velocity, acceleration, angle, omega, polar,
                        area, chord, span, parameters=None, rho=1000.,substeps=4):
    """Candidate passive entrained-fluid closure, with a world-frame wake state.

    mw * wdot = -Fqs(v-w) - mw * lambda * w; lambda=|v|/(Lwake*c).
    Unlike an algebraic force gain, this permits the moving wake to push on a
    decelerating blade. The closure is proposed here based on the wake mechanism
    observed by Grift2021; it is NOT an equation claimed from that publication.
    With nonnegative polar CD, the blade+wake dissipates relative drag power
    plus mw*lambda*|w|². Rotating added mass is the separate potential term.
    No partial-immersion, camber, or spatial vortex-convection state is inferred.
    """
    p=parameters or WakeMomentumParameters()
    if min(p.drag_scale,p.lift_scale,p.added_mass_scale)<0 or min(p.wake_mass_coefficient,p.wake_decay_chords)<=0:
        raise ValueError('Invalid wake parameters')
    t=np.asarray(time,float);v=np.asarray(velocity,float);a=np.asarray(acceleration,float)
    if len(t)<2 or np.any(np.diff(t)<=0) or min(area,chord,span,rho)<=0 or substeps<1:
        raise ValueError('Invalid time, geometry, or substeps')
    # Interpolators are prepared once; no PDF/data access during integration.
    from scipy.interpolate import PchipInterpolator
    cd_interp=PchipInterpolator(polar.angle_deg,polar.cd)
    cl_interp=PchipInterpolator(polar.angle_deg,polar.cl)
    mw=p.wake_mass_coefficient*rho*area*chord
    def force(vv,theta,ww):
        n=np.array([np.cos(theta),np.sin(theta)]);c=np.array([-n[1],n[0]])
        rel=vv-ww;vn=rel@n;vt=rel@c;speed=np.linalg.norm(rel)
        alpha=np.degrees(np.arctan2(abs(vn),vt))
        cd=max(0.,float(cd_interp(alpha)))*p.drag_scale
        cl=float(cl_interp(alpha))*p.lift_scale
        return .5*rho*area*speed*(-cd*rel+cl*np.sign(vn)*np.array([-rel[1],rel[0]]))
    w=np.zeros_like(v);f=np.zeros_like(v)
    f[0]=force(v[0],angle[0],w[0])
    for i,dt in enumerate(np.diff(t),1):
        wi=w[i-1].copy();h=dt/substeps
        def derivative(ww,fraction):
            vv=v[i-1]+fraction*(v[i]-v[i-1]);theta=angle[i-1]+fraction*(angle[i]-angle[i-1])
            decay=np.linalg.norm(vv)/(p.wake_decay_chords*chord)
            return -force(vv,theta,ww)/mw-decay*ww
        for j in range(substeps):
            u=j/substeps
            k1=derivative(wi,u);k2=derivative(wi+h*k1/2,u+.5/substeps)
            k3=derivative(wi+h*k2/2,u+.5/substeps);k4=derivative(wi+h*k3,u+1/substeps)
            wi+=h*(k1+2*k2+2*k3+k4)/6
        w[i]=wi;f[i]=force(v[i],angle[i],wi)
    potential=blade_trajectory_force(t,v,a,angle,omega,polar,area,chord,span,
                                    BladeTrajectoryParameters(0,0,0,p.added_mass_scale),rho)
    return {'force':f+potential['inertia'],'quasi_steady_relative_force':f,'wake_velocity':w,
            'wake_mass_kg':mw,'wake_energy':.5*mw*np.sum(w*w,axis=1),
            'wake_dissipation':mw*np.linalg.norm(v,axis=1)/(p.wake_decay_chords*chord)*np.sum(w*w,axis=1),
            'inertia':potential['inertia'],'intrinsic_torque':potential['intrinsic_torque']}
