"""Published Grift trajectory reconstruction and experimental force coordinates.

Shared by calibration and independent no-fit checks; contains no model fitting.
"""
import numpy as np
from scipy.signal import savgol_filter


def trace(frame,variable,s,**where):
    a=frame[frame.variable==variable]
    for k,v in where.items():a=a[a[k]==v]
    a=a.groupby('phase',as_index=False).value.mean().sort_values('phase')
    return np.interp(s,a.phase,a.value)


def kinematics(frame,beta,kappa,s,window=15):
    xy=np.column_stack([trace(frame,n,s,beta_deg=beta) for n in ['x_m','y_m']])
    v=np.column_stack([trace(frame,n,s,beta_deg=beta) for n in ['vx_m_s','vy_m_s']])
    ds=s[1]-s[0]
    dxy=savgol_filter(xy,window,3,deriv=1,delta=ds,axis=0)
    core=(s>.05)&(s<.95)
    duration=float(np.sum(dxy[core]*v[core])/np.sum(v[core]**2))
    v_res=float(np.sqrt(np.mean((dxy[core]/duration-v[core])**2)))
    angle=np.deg2rad(trace(frame,'theta_deg',s,beta_deg=beta)+beta)
    angle=savgol_filter(angle,window,3)
    time=s*duration/kappa
    vel=v*kappa
    acc=savgol_filter(vel,window,3,deriv=1,delta=time[1]-time[0],axis=0)
    omega=savgol_filter(angle,window,3,deriv=1,delta=time[1]-time[0])
    vn=vel[:,0]*np.cos(angle)+vel[:,1]*np.sin(angle)
    vt=-vel[:,0]*np.sin(angle)+vel[:,1]*np.cos(angle)
    alpha=np.degrees(np.unwrap(np.arctan2(-vn,vt)))
    alpha_observed=trace(frame,'alpha_deg',s,beta_deg=beta)
    return dict(time=time,velocity=vel,acceleration=acc,angle=angle,omega=omega),dict(
        beta_deg=beta,kappa=kappa,duration_kappa1_s=duration,velocity_compatibility_rmse_m_s=v_res,
        alpha_compatibility_rmse_deg=float(np.sqrt(np.mean((alpha-alpha_observed)**2))))


def force_observation(frame,beta,kappa,s):
    fig=9 if beta==0 else 22
    fn=trace(frame,'fn_N',s,beta_deg=beta,kappa=kappa,figure=fig)
    ft=trace(frame,'ft_N',s,beta_deg=beta,kappa=kappa,figure=fig)
    return np.column_stack((fn,ft))


def to_nt(f,angle):
    n=np.column_stack((np.cos(angle),np.sin(angle)))
    c=np.column_stack((-np.sin(angle),np.cos(angle)))
    return np.column_stack((-np.sum(f*n,axis=1),-np.sum(f*c,axis=1)))


def to_xy(f,angle):
    n=np.column_stack((np.cos(angle),np.sin(angle)))
    c=np.column_stack((-np.sin(angle),np.cos(angle)))
    return -f[:,0,None]*n-f[:,1,None]*c


