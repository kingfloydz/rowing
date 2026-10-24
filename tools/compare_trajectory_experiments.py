"""Whole-trajectory calibration/test split; never fit validation trajectories."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear, least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hydrodynamics.polars import BladePolar
from hydrodynamics.trajectory import (BladeTrajectoryParameters,blade_trajectory_force,
                                     WakeMomentumParameters,wake_momentum_force)
from hydrodynamics.models import HullResistance
OUT=ROOT/'results/trajectories'


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


def evaluate(kin,polar,params):
    # The source reports physical area approximately 0.02 m². Effective span
    # area/chord preserves this area; the finite-span added mass is uncertain.
    fun=wake_momentum_force if isinstance(params,WakeMomentumParameters) else blade_trajectory_force
    return fun(**kin,polar=polar,area=.02,chord=.275,span=.02/.275,parameters=params)


def fit(kin,obs,polar,lag_options,with_mass):
    best=None
    for lag in lag_options:
        raw=evaluate(kin,polar,BladeTrajectoryParameters(1,1,float(lag),1))
        columns=[to_nt(raw[name],kin['angle']).ravel() for name in ['drag','lift','inertia']]
        if not with_mass:columns=columns[:2]
        design=np.column_stack(columns)
        result=lsq_linear(design,obs.ravel(),bounds=(0,4))
        error=float(np.mean((design@result.x-obs.ravel())**2))
        coeff=list(result.x)+([] if with_mass else [0.])
        p=BladeTrajectoryParameters(coeff[0],coeff[1],float(lag),coeff[2])
        if best is None or error<best[0]:best=error,p
    return best[1]


def hull_comparison():
    points=pd.read_csv(ROOT/'data/day2011_rowing_excursion.csv')
    phase=points.phase.to_numpy();exc=points.excursion_m.to_numpy()
    # Periodic Fourier representation gives exactly zero net displacement and
    # avoids spline end-condition artifacts. Six harmonics for the 55 vertices.
    n=6
    basis=np.column_stack([np.ones(len(phase))]+[fun(2*np.pi*j*phase) for j in range(1,n+1) for fun in [np.cos,np.sin]])
    coef=np.linalg.lstsq(basis,exc,rcond=None)[0]
    s=np.linspace(0,1,2001);t=s*1.84
    vrow=np.zeros_like(s)
    for j in range(1,n+1):vrow+=2*np.pi*j/1.84*(-coef[2*j-1]*np.sin(2*np.pi*j*s)+coef[2*j]*np.cos(2*np.pi*j*s))
    vsine=.24*2*np.pi/1.84*np.cos(2*np.pi*s)
    hp=json.loads((ROOT/'data/parameters.json').read_text())['hull']
    h=HullResistance(np.array(hp['fn_nodes']),np.array(hp['residual_nodes']),hp['form_factor'],hp['nu'])
    rows=[];outside={}
    for name,dv in [('sinusoidal',vsine),('rowing',vrow)]:
        u=4+dv
        outside[name]=float(np.mean(u>hp['fn_nodes'][-1]*np.sqrt(9.81*7.937)))
        for closure in ['quadratic','ittc_residual']:
            if closure=='quadratic':ratio=u*u/16
            else:ratio=h.magnitude(u,7.937,2.34)/h.magnitude(4.,7.937,2.34)
            pred=(np.trapezoid(ratio,s)-1)*100
            for surface,observed in [('free_transition',1.8 if name=='sinusoidal' else 3.),('studs_400mm',1.9 if name=='sinusoidal' else 2.4)]:
                rows.append(dict(motion=name,surface=surface,closure=closure,predicted_increase_percent=pred,
                                 measured_increase_percent=observed,error_percentage_points=pred-observed))
        pd.DataFrame(dict(phase=s,time_s=t,speed_m_s=u)).to_csv(OUT/f'hull_{name}_motion.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'hull_mean_resistance.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    ax[0].plot(s,4+vrow,label='Reconstructed rowing motion')
    ax[0].plot(s,4+vsine,label='Sinusoidal motion')
    ax[0].axhline(4,color='gray',ls=':');ax[0].set(xlabel='Stroke phase',ylabel='Hull speed [m/s]')
    ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    for i,motion in enumerate(['sinusoidal','rowing']):
        rs=[r for r in rows if r['motion']==motion]
        vals=[next(r['predicted_increase_percent'] for r in rs if r['closure']==c) for c in ['quadratic','ittc_residual']]
        ax[1].bar(np.array([i-.18,i+.18]),vals,width=.32,color=['tab:blue','tab:orange'])
        for surface,marker in [('free_transition','o'),('studs_400mm','s')]:
            value=next(r['measured_increase_percent'] for r in rs if r['surface']==surface)
            ax[1].plot(i,value,marker,color='black',ms=7,label=surface if i==0 else None)
    from matplotlib.patches import Patch
    handles,labels=ax[1].get_legend_handles_labels()
    ax[1].legend([Patch(color='tab:blue'),Patch(color='tab:orange')]+handles,['Quadratic QS','ITTC + clamped residual']+labels,fontsize=7)
    ax[1].set(xticks=[0,1],xticklabels=['Sinusoidal','Rowing'],ylabel='Mean resistance increase [%]',ylim=(0,3.8));ax[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'hull_real_motion_comparison.png',dpi=170);plt.close(fig)
    return dict(excursion_fit_rmse_m=float(np.sqrt(np.mean((basis@coef-exc)**2))),
                rowing_speed_perturbation_min=float(vrow.min()),rowing_speed_perturbation_max=float(vrow.max()),
                fraction_above_steady_calibration=outside,
                caveat='No hull memory coefficient fitted from mean data; ITTC residual above calibrated speed clamps.',rows=rows)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    k=pd.read_csv(ROOT/'data/grift2021_kinematics.csv')
    f=pd.read_csv(ROOT/'data/grift2021_force_trajectories.csv')
    s=np.linspace(.002,.998,201);polar=BladePolar.caplan_flat()
    curved_data=pd.read_csv(ROOT/'data/caplan2007_curved_polar.csv')
    curved=BladePolar(curved_data.angle_deg.to_numpy(),curved_data.cd.to_numpy(),curved_data.cl.to_numpy())
    train,_=kinematics(k,0,.75,s)
    obs=force_observation(f,0,.75,s)
    models={'static_transfer':BladeTrajectoryParameters(1,1,0,0),
            'static_calibrated':fit(train,obs,polar,[0.],False),
            'inertia_calibrated':fit(train,obs,polar,[0.],True),
            'lag_inertia_calibrated':fit(train,obs,polar,np.r_[0.,np.geomspace(.02,3,60)],True)}
    # All optimization uses the selected beta0/kappa0.75 calibration case.
    # Bounds constrain the proposed closure; held-out forces are never passed.
    fit_log=[]
    def residual(x):
        p=WakeMomentumParameters(*x)
        pred=evaluate(train,polar,p)['force']
        return (to_nt(pred,train['angle'])-obs).ravel()
    best=None
    for initial in [[2.,1.,1.,1.,.1],[3.,1.,.3,2.,.2]]:
        result=least_squares(residual,initial,bounds=([.1,0,.05,.02,0],[8,5,8,10,2]),
                             max_nfev=100,ftol=2e-5,xtol=2e-5,gtol=2e-5)
        fit_log.append(dict(cost=float(result.cost),parameters=result.x.tolist(),nfev=result.nfev,success=bool(result.success)))
        if best is None or result.cost<best.cost:best=result
    models['wake_momentum_calibrated']=WakeMomentumParameters(*best.x)
    models['curved_static_transfer']=BladeTrajectoryParameters(1,1,0,0)
    metrics=[];audits=[];cases=[(0,.75,'calibration'),(0,.5,'speed_holdout'),(0,1.,'speed_holdout'),(15,1.,'mounting_holdout')]
    fig,axs=plt.subplots(4,2,figsize=(12,13),sharex=True)
    for row,(beta,kap,split) in enumerate(cases):
        kin,audit=kinematics(k,beta,kap,s);audits.append(audit)
        # Fig22 Fn,Ft remain in the shaft/transducer basis theta, not theta+beta.
        measurement_angle=kin['angle']-np.deg2rad(beta)
        observed_nt=force_observation(f,beta,kap,s);observed_xy=to_xy(observed_nt,measurement_angle)
        out=pd.DataFrame(dict(phase=s,time_s=kin['time'],fn_observed=observed_nt[:,0],ft_observed=observed_nt[:,1],fx_observed=observed_xy[:,0],fy_observed=observed_xy[:,1]))
        if kap==1:
            check=np.column_stack([trace(f,n,s,beta_deg=beta,kappa=kap,figure=10 if beta==0 else 22) for n in ['fx_N','fy_N']])
            audit['basis_check_force_rmse_N']=float(np.sqrt(np.mean((observed_xy-check)**2)))
        for name,p in models.items():
            pred=evaluate(kin,curved if name=='curved_static_transfer' else polar,p);nt=to_nt(pred['force'],measurement_angle);xy=pred['force']
            for domain,selector in [('whole_drive',np.ones(len(s),bool)),('interior_drive',(s>=.1)&(s<=.9))]:
                error=nt[selector]-observed_nt[selector]
                metrics.append(dict(beta_deg=beta,kappa=kap,split=split,model=name,domain=domain,
                                    force_component_rmse_N=float(np.sqrt(np.mean(error**2))),
                                    force_nrmse_rms_percent=float(100*np.linalg.norm(error)/np.linalg.norm(observed_nt[selector])),
                                    propulsion_impulse_error_percent=float(100*(np.trapezoid(xy[selector,0],kin['time'][selector])/np.trapezoid(observed_xy[selector,0],kin['time'][selector])-1))))
            for j,component in enumerate(['fn','ft']):out[f'{component}_{name}']=nt[:,j]
            if name in ['static_transfer','curved_static_transfer','wake_momentum_calibrated']:
                for j in range(2):axs[row,j].plot(s,nt[:,j],label=name,lw=1.5)
        for j in range(2):
            axs[row,j].plot(s,observed_nt[:,j],'k',lw=2,label='Experiment (digitized)')
            axs[row,j].set_title(f'beta={beta} deg, kappa={kap}: {split}; '+['Fn','Ft'][j])
            axs[row,j].set_ylabel('Force [N]');axs[row,j].grid(alpha=.25)
        out.to_csv(OUT/f'blade_beta{beta}_kappa{kap}.csv',index=False)
    axs[0,0].legend(fontsize=8);axs[-1,0].set_xlabel('Drive phase');axs[-1,1].set_xlabel('Drive phase')
    fig.tight_layout();fig.savefig(OUT/'real_trajectory_forces.png',dpi=160);plt.close(fig)
    pd.DataFrame(metrics).to_csv(OUT/'blade_metrics.csv',index=False)
    result=dict(training='Grift2021 Fig9b beta0 kappa0.75 whole drive',
                parameters={name:p.__dict__ for name,p in models.items()},kinematics_audits=audits,wake_fit_log=fit_log)
    result['hull']=hull_comparison()
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
    print(pd.DataFrame(metrics).query("domain=='whole_drive'").to_string(index=False))


if __name__=='__main__':main()
