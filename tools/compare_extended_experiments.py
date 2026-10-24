"""Run separate calibration, held-condition and cross-study comparisons."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hydrodynamics.polars import BladePolar,harmonic_polar,barre_published_maxima
from hydrodynamics.unsteady import history_excess_force,prescribed_motion
from hydrodynamics.models import plate_added_mass,load_parameters,start_plate_force

OUT=ROOT/'results/extended';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22,'figure.dpi':140})
METRICS=[]


def metric(study,model,measured,predicted,unit,scope,condition='all'):
    y=np.asarray(measured);p=np.asarray(predicted)
    rmse=float(np.sqrt(np.mean((p-y)**2)))
    METRICS.append(dict(study=study,model=model,scope=scope,condition=condition,n=len(y),
                        rmse=rmse,mae=float(np.mean(abs(p-y))),
                        nrmse_rms_percent=float(100*rmse/np.sqrt(np.mean(y*y))),unit=unit))


def caplan():
    d=pd.read_csv(ROOT/'data/caplan2007_flat_polar.csv')
    train=d.angle_deg%10==0;test=~train
    p=BladePolar.caplan_flat();angles=np.deg2rad(d.angle_deg.values)
    # Fair baseline: fit one normal-force Cd to the same training subset.
    x=np.concatenate([np.sin(angles[train])**2,.5*np.sin(2*angles[train])])
    y=np.concatenate([d.cd[train],d.cl[train]])
    cd0=float(x@y/(x@x));normal=harmonic_polar(angles,cd0,cd0/2)
    polar=p.coefficients(angles)
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,field,i in zip(axs,['cd','cl'],[0,1]):
        for name,pred in [('normal_only_fitted',normal[i]),('angle_polar',polar[i])]:
            metric('Caplan2007',name,d[field][test],pred[test],'coefficient','held_out_angles',field)
        ax.plot(d.angle_deg,d[field],'o',ms=3,label='Published markers')
        ax.plot(d.angle_deg,normal[i],label='One normal coefficient')
        ax.plot(d.angle_deg,polar[i],label='10-degree polar nodes')
        ax.scatter(d.angle_deg[test],d[field][test],facecolors='none',edgecolors='k',s=45,label='Held-out angles')
        ax.set(xlabel='Angle of attack (deg)',ylabel=field.upper())
        d[field+'_normal']=normal[i];d[field+'_polar']=polar[i]
    axs[0].legend(fontsize=8);fig.suptitle('Static flat Big-Blade outline: interpolation test, same experiment')
    fig.tight_layout();fig.savefig(OUT/'caplan_polar.png');plt.close(fig)
    d['split']=np.where(train,'train','held_out');d.to_csv(OUT/'caplan_predictions.csv',index=False)
    return {'normal_coefficient_fitted':cd0,'training_angles':d.angle_deg[train].tolist(),'held_out_angles':d.angle_deg[test].tolist()}


def design(d):
    # Affine dependence on fr and K; interaction allows different slopes.
    f=d.reduced_frequency.to_numpy();k=d.rotation_rate_rad_s.to_numpy()-3.5
    return np.column_stack([np.ones(len(d)),f,k,f*k])


def barre():
    d=pd.read_csv(ROOT/'data/barre2010_maxima.csv');rows=[];fits={}
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,coef in zip(axs,['cd_max','cl_max']):
        q=d[d.coefficient==coef].copy();fits[coef]=np.linalg.lstsq(design(q),q.value,rcond=None)[0].tolist()
        for k in [3.,3.5,3.75,4.]:
            train=q[q.rotation_rate_rad_s!=k];test=q[q.rotation_rate_rad_s==k].copy()
            # No fr extrapolation in the held-K test. Edge K3/K4 still K extrapolate.
            test=test[test.reduced_frequency.between(train.reduced_frequency.min(),train.reduced_frequency.max())]
            b=np.linalg.lstsq(design(train),train.value,rcond=None)[0]
            pred=design(test)@b;constant=np.full(len(test),train.value.mean())
            metric('Barre2010', 'fr_K_response',test.value,pred,'coefficient','leave_rotation_rate_out',f'{coef}_K{k}')
            metric('Barre2010', 'constant_response',test.value,constant,'coefficient','leave_rotation_rate_out',f'{coef}_K{k}')
            measured_line,=ax.plot(test.reduced_frequency,test.value,'o',ms=4,label=f'Measured K={k:g}')
            ax.plot(test.reduced_frequency,pred,'--',lw=1,color=measured_line.get_color())
            test['prediction']=pred;test['constant_prediction']=constant;rows.append(test)
        ax.set(xlabel='Reduced frequency',ylabel=coef)
    axs[0].legend(fontsize=8,ncol=2)
    fig.suptitle('Rotating blade: leave one rotation rate out; dashed = prediction')
    fig.tight_layout();fig.savefig(OUT/'barre_maxima_holdout.png');plt.close(fig)
    pred=pd.concat(rows);pred.to_csv(OUT/'barre_maxima_predictions.csv',index=False)
    for coef in ['cd_max','cl_max']:
        q=pred[pred.coefficient==coef]
        for name,col in [('fr_K_response','prediction'),('constant_response','constant_prediction')]:
            metric('Barre2010',name,q.value,q[col],'coefficient','leave_rotation_rate_out_pooled',coef)
    # Dynamic polar shape: estimate only amplitudes using alternating 10-degree nodes.
    # It is a separate within-condition check, not an independent force prediction.
    d=pd.read_csv(ROOT/'data/barre2010_dynamic_polars.csv');polar=BladePolar.caplan_flat()
    fig,axs=plt.subplots(1,2,figsize=(10,3.7));rows=[]
    for (k,eta,field),q in d.groupby(['rotation_rate_rad_s','advance_number','coefficient']):
        q=q.groupby('angle_deg',as_index=False).value.mean().sort_values('angle_deg')
        x=np.arange(10,171,5);y=np.interp(x,q.angle_deg,q.value)
        train=x%10==0;test=~train;rad=np.deg2rad(x)
        basis=np.sin(rad)**2 if field=='cd' else np.sin(2*rad)
        amp=float(basis[train]@y[train]/(basis[train]@basis[train]));yp=amp*basis
        static=polar.coefficients(rad)[field=='cl']
        for name,z in [('within_condition_harmonic',yp),('Caplan_static_transfer',static)]:
            metric('Barre2010',name,y[test],z[test],'coefficient','held_out_angles',f'{field}_K{k}_eta{eta}')
        ax=axs[int(field=='cl')];ax.plot(x,y,lw=1,label=f'K{k:g}, eta{eta:g}')
        ax.plot(x,yp,'--',lw=.7,color=ax.lines[-1].get_color())
        rows.append(pd.DataFrame(dict(K=k,eta=eta,coefficient=field,angle=x,measured=y,harmonic=yp,static_transfer=static,split=np.where(train,'train','test'))))
    for ax,field in zip(axs,['CD','CL']):ax.set(xlabel='Pseudo-incidence (deg)',ylabel=field)
    axs[0].legend(fontsize=8);fig.suptitle('Dynamic polars: solid = measured, dashed = fitted harmonic')
    fig.tight_layout();fig.savefig(OUT/'barre_dynamic_polars.png');plt.close(fig)
    pd.concat(rows).to_csv(OUT/'barre_dynamic_polar_predictions.csv',index=False)
    # Zero-torque point: avoid low-incidence singularity and clipped source portions.
    d=pd.read_csv(ROOT/'data/barre2010_zero_torque_point.csv');rows=[]
    fig,ax=plt.subplots(figsize=(7,3.7))
    for (k,eta),q in d.groupby(['rotation_rate_rad_s','advance_number']):
        q=q.groupby('angle_deg',as_index=False).agg({'dip':'mean','clipped':'max'}).sort_values('angle_deg')
        angle=np.arange(30,151,5);dip=np.interp(angle,q.angle_deg,q.dip)
        for name,p in [('centre',np.zeros_like(dip)),('quarter_chord_offset',np.full_like(dip,.5))]:
            metric('Barre2010',name,dip,p,'dip','published_approximation_30_to_150deg',f'K{k}_eta{eta}')
        ax.plot(angle,dip,label=f'K{k:g}, eta{eta:g}')
        rows.append(pd.DataFrame(dict(K=k,eta=eta,angle=angle,dip=dip)))
    ax.axhline(0,color='k',ls=':',label='Centre');ax.axhline(.5,color='k',ls='--',label='Offset c/4')
    ax.set(xlabel='Pseudo-incidence (deg)',ylabel='dip = 2 Lip / chord',title='Force application point is not fixed at the blade centre')
    ax.legend(fontsize=8,ncol=3);fig.tight_layout();fig.savefig(OUT/'barre_centre_of_pressure.png');plt.close(fig)
    pd.concat(rows).to_csv(OUT/'barre_zero_torque_comparison.csv',index=False)
    return {'response_design':['1','fr','K-3.5','fr*(K-3.5)'],'all_data_response_fits':fits,
            'warning':'All-data fits are for deployment in source domain only; held-K metrics fit anew without each held-out K.'}


def history_comparisons():
    d=pd.read_csv(ROOT/'data/reijtenbagh2026_excess_force.csv');rows=[]
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for (kind,param),q in d.groupby(['motion','parameter']):
        visible_intervals=[(s.time_s.min(),s.time_s.max()) for _,s in q.groupby('segment')]
        q=q[q.time_s>=0].groupby('time_s',as_index=False).measured_excess_force_N.mean().sort_values('time_s')
        # Uniform resampling avoids weighting curves by their PDF point density.
        t=np.arange(max(.01,q.time_s.min()),q.time_s.max(),.005)
        # The publisher clips some initial Fa<0 portions; do not invent their data.
        visible=np.any([(t>=lo)&(t<=hi) for lo,hi in visible_intervals],axis=0)
        t=t[visible]
        y=np.interp(t,q.time_s,q.measured_excess_force_N)
        v,a,stop=prescribed_motion(t,kind,param,.6)
        fa=history_excess_force(t,kind,param,.6)
        ma=plate_added_mass(.2,.1,rho=998)*a
        early=(t>=.15)&(t<=.6*stop)
        for scope,mask in [('full_published_window',np.ones(len(t),bool)),('early_nominal_motion',early)]:
            for name,p in [('QS_only',np.zeros_like(y)),('inviscid_added_mass',ma),('history_nominal_motion',fa)]:
                metric('Reijtenbagh2026',name,y[mask],p[mask],'N_excess',scope,f'{kind}_{param}')
        rows.append(pd.DataFrame(dict(motion=kind,parameter=param,time_s=t,measured=y,history=fa,added_mass=ma,early=early)))
        if (kind=='constant_jerk' and param in [.2,.6]) or (kind=='sqrt_velocity' and param in [.25,.64]):
            ax=axs[0 if kind=='constant_jerk' else 1,0 if param in [.2,.25] else 1]
            ax.plot(t,y,label='Published experiment',lw=1);ax.plot(t,fa,label='History, nominal motion')
            ax.plot(t,ma,label='Inviscid added mass');ax.axvspan(.15,.6*stop,color='gray',alpha=.12)
            ax.set(title=f'{kind}, parameter={param:g}',xlabel='Time (s)',ylabel='Excess force (N)',ylim=(-.2,4.5))
    axs[0,0].legend(fontsize=8);fig.suptitle('2026 plate: literature C=1.5, no fit to these extracted curves')
    fig.tight_layout();fig.savefig(OUT/'history_2026.png');plt.close(fig)
    z=pd.concat(rows);z.to_csv(OUT/'history_2026_predictions.csv',index=False)
    for scope,mask in [('full_published_window',np.ones(len(z),bool)),('early_nominal_motion',z.early)]:
        for name,col in [('inviscid_added_mass','added_mass'),('history_nominal_motion','history')]:
            metric('Reijtenbagh2026',name,z.measured[mask],z[col][mask],'N_excess',scope+'_pooled')
    # Cross-study transfer to 2019: same nominal plate dimensions but different depth,
    # thickness and force definition. Add dry inertia only here, per 2019 Fig10.
    old=pd.read_csv(ROOT/'data/grift2019_acceleration_curves.csv')
    _,blade,params=load_parameters();cd=float(blade.cd(.1,.1));rows=[]
    for trial,q in old.groupby('trial'):
        if trial=='fig10b_2':continue # Fig10a_0 repeats this motion.
        q=q.sort_values('time_s');a=float(q.acceleration_m_s2.iloc[0]);vmax=float(q.target_speed_m_s.iloc[0])
        t=np.arange(max(.005,q.time_s.min()),min(3.,q.time_s.max()),.005)
        y=np.interp(t,q.time_s,q.measured_total_force_N)
        v,acc,stop=prescribed_motion(t,'constant_acceleration',a,vmax)
        qs=.5*1000*.02*cd*v*v+.4*acc
        hist=qs+history_excess_force(t,'constant_acceleration',a,vmax,rho=1000)
        inviscid=qs+plate_added_mass(.2,.1)*acc
        early=(t>=.07)&(t<=stop-.07)
        for scope,mask in [('full_window',np.ones(len(t),bool)),('acceleration_core',early)]:
            for name,pred in [('inviscid_added_mass',inviscid),('history_2026_transfer',hist)]:
                metric('Grift2019',name,y[mask],pred[mask],'N_transducer',scope,trial)
        rows.append(pd.DataFrame(dict(trial=trial,t=t,measured=y,history=hist,added_mass=inviscid,early=early)))
    z=pd.concat(rows);z.to_csv(OUT/'history_transfer_2019.csv',index=False)
    for scope,mask in [('full_window',np.ones(len(z),bool)),('acceleration_core',z.early)]:
        for name,col in [('inviscid_added_mass','added_mass'),('history_2026_transfer','history')]:
            metric('Grift2019',name,z.measured[mask],z[col][mask],'N_transducer',scope+'_pooled')
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,trial in zip(axs,['fig10a_0','fig10b_4']):
        q=z[z.trial==trial];ax.plot(q.t,q.measured,label='2019 experiment',lw=1)
        ax.plot(q.t,q.history,label='2026 history transfer');ax.plot(q.t,q.added_mass,label='Inviscid added mass')
        ax.set(xlabel='Time (s)',ylabel='Transducer force (N)',title=trial)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'history_transfer_2019.png');plt.close(fig)


if __name__=='__main__':
    settings={'caplan':caplan(),'barre':barre()}
    history_comparisons()
    pd.DataFrame(METRICS).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'parameters_extended.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    result=pd.DataFrame(METRICS)
    print(result[(result.scope.str.contains('pooled')) | (result.study=='Caplan2007')].to_string(index=False))
