"""No refitting: transfer existing closures to two Nantes towing-tank strokes.

These cases share a source database with Barre2010 regressions. Their evaluation
is an additional whole-curve check, not independent out-of-campaign validation.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hydrodynamics.prescribed_stroke import PrescribedTankStroke
from hydrodynamics.polars import BladePolar
from hydrodynamics.trajectory import (blade_trajectory_force,BladeTrajectoryParameters,
                                     wake_momentum_force,WakeMomentumParameters)
OUT=ROOT/'results/leroyer';OUT.mkdir(parents=True,exist_ok=True)


def vector_series(frame,variable,theta):
    d=frame[frame.variable==variable].groupby('theta_deg').value.mean().sort_index()
    if np.min(theta)<d.index.min() or np.max(theta)>d.index.max():raise ValueError('No source extrapolation')
    return np.interp(theta,d.index,d)


def observations(year,theta):
    if year==2010:
        f=pd.read_csv(ROOT/'data/leroyer2010_force_trajectory.csv')
        return np.column_stack([np.interp(theta,f[f.component==name].theta_deg,
                                          f[f.component==name].force_N) for name in ['fx','fy']])
    f=pd.read_csv(ROOT/'data/leroyer2008_force_moment.csv')
    # Positive fy_plot is the physical +Fy despite the Fig19 printed -Fy label.
    # The measured moment / pressure-centre identity below independently checks it.
    return np.column_stack((-vector_series(f,'minus_fx',theta),vector_series(f,'fy_plot',theta)))


def metrics(obs,pred,theta,stroke):
    time=stroke.kinematics(np.deg2rad(theta))['time'];e=pred-obs
    return dict(force_component_rmse_N=float(np.sqrt(np.mean(e**2))),
                time_weighted_component_rmse_N=float(np.sqrt(np.trapezoid(np.sum(e**2,axis=1),time)/(2*(time[-1]-time[0])))),
                force_nrmse_percent=float(100*np.linalg.norm(e)/np.linalg.norm(obs)),
                propulsion_impulse_error_percent=float(100*(np.trapezoid(-pred[:,0],time)/np.trapezoid(-obs[:,0],time)-1)),
                observed_propulsion_impulse_Ns=float(np.trapezoid(-obs[:,0],time)))


def main():
    polar=BladePolar.caplan_flat()
    prior=json.loads((ROOT/'results/trajectories/verification.json').read_text())
    wp=WakeMomentumParameters(**prior['parameters']['wake_momentum_calibrated'])
    angles=np.arange(15.,166.,5.);dense=np.linspace(.1,179.9,2401)
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.8),sharex=True)
    rows=[];audit={};predictions={}
    labels={'caplan_static':'Caplan static transfer','barre_dynamic':'Barre dynamic, fixed coefficients',
            'grift_wake_transfer':'Grift-calibrated wake transfer'}
    for row,(year,speed) in enumerate([(2008,2.31),(2010,2.91)]):
        stroke=PrescribedTankStroke(advance_number=speed/(3*1.308))
        kin=stroke.kinematics(np.deg2rad(dense))
        args={k:kin[k] for k in ['time','velocity','acceleration','angle','omega']}
        base=dict(**args,polar=polar,area=stroke.area,chord=stroke.chord,span=stroke.span)
        dynamic=stroke.force(np.deg2rad(dense))
        series={
            'caplan_static':blade_trajectory_force(**base,parameters=BladeTrajectoryParameters(1,1,0,0))['force'],
            'barre_dynamic':dynamic['force'],
            'grift_wake_transfer':wake_momentum_force(**base,parameters=wp)['force'],
        }
        obs=observations(year,angles)
        export=pd.DataFrame(dict(theta_deg=angles,fx_observed_N=obs[:,0],fy_observed_N=obs[:,1]))
        for name,force in series.items():
            sampled=np.column_stack([np.interp(angles,dense,force[:,j]) for j in range(2)])
            predictions[(year,name)]=sampled
            for domain,select in [('15_165deg',np.ones(len(angles),bool)),('30_150deg',(angles>=30)&(angles<=150)),
                                  ('70_110deg',(angles>=70)&(angles<=110))]:
                result=metrics(obs[select],sampled[select],angles[select],stroke)
                rows.append(dict(year=year,model=name,domain=domain,**result))
            for j,comp in enumerate(['fx','fy']):export[f'{comp}_{name}_N']=sampled[:,j]
            for j in range(2):
                plotted=(dense>=15)&(dense<=165)
                axes[row,j].plot(dense[plotted],force[plotted,j],label=labels[name],lw=1.6)
        for j in range(2):
            axes[row,j].plot(angles,obs[:,j],'ko',ms=3,label='Experiment, digitized')
            axes[row,j].set(title=f'Leroyer {year}, Vb={speed:.2f} m/s, K=3 rad/s',
                            ylabel=['Fx [N], propulsion toward -x','Fy [N]'][j],xlim=(15,165))
            axes[row,j].grid(alpha=.2)
        export.to_csv(OUT/f'force_comparison_{year}.csv',index=False)
        audit[str(year)]=dict(advance_number=stroke.advance_number,reduced_frequency=stroke.reduced_frequency,
                              max_normal_froude=float(kin['normal_froude'].max()),cd_max=dynamic['cd_max'],cl_max=dynamic['cl_max'],
                              wake_initialization='Zero wake at theta=0.1 deg; precursor flow is not measured.')
        refined=wake_momentum_force(**base,parameters=wp,substeps=8)['force']
        relevant=(dense>=30)&(dense<=150)
        audit[str(year)]['wake_substeps_4_vs_8_component_rmse_N']=float(np.sqrt(np.mean(
            (refined[relevant]-series['grift_wake_transfer'][relevant])**2)))
    for ax in axes[-1]:ax.set_xlabel('Shaft angle theta [deg]')
    axes[0,0].legend(fontsize=7.3,loc='lower left')
    fig.suptitle('Two simplified strokes: no new force-parameter fitting',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'whole_stroke_comparison.png',dpi=175);plt.close(fig)

    # Torque-only audit: use measured force to isolate application-point error.
    earlier=pd.read_csv(ROOT/'data/leroyer2008_force_moment.csv')
    deg=np.arange(30.,151.,5.);theta=np.deg2rad(deg)
    stroke=PrescribedTankStroke(advance_number=2.31/(3*1.308))
    force=observations(2008,deg);mz=-vector_series(earlier,'minus_mz',deg)
    shift=.15*vector_series(earlier,'dip_over_half_chord',deg)
    reconstructed=stroke.moment_from_force(theta,force,shift)
    reversed_fy=force.copy();reversed_fy[:,1]*=-1
    moment_cases={'measured_force_at_centre':stroke.moment_from_force(theta,force),
                  'measured_force_at_quarter_toward_shaft':stroke.moment_from_force(theta,force,.075),
                  'measured_force_at_quarter_away_from_shaft':stroke.moment_from_force(theta,force,-.075),
                  'measured_force_measured_cop_identity':reconstructed,
                  'barre_force_at_centre':stroke.moment_from_force(theta,stroke.force(theta)['force'])}
    moment_metrics={name:dict(rmse_Nm=float(np.sqrt(np.mean((pred-mz)**2))),
                              max_abs_error_Nm=float(np.max(abs(pred-mz)))) for name,pred in moment_cases.items()}
    audit['moment']=dict(domain='30..150 deg',metrics=moment_metrics,
        sign_check_wrong_minus_fy_rmse_Nm=float(np.sqrt(np.mean((stroke.moment_from_force(theta,reversed_fy,shift)-mz)**2))),
        note='Measured CoP is derived from force and moment, so its reconstruction is a sign/unit consistency check, NOT validation of a CoP predictor.',
        measured_cop_shift_range_m=[float(shift.min()),float(shift.max())])
    moment_frame=pd.DataFrame(dict(theta_deg=deg,measured_mz_Nm=mz,measured_cop_shift_m=shift,**moment_cases))
    moment_frame.to_csv(OUT/'moment_comparison_2008.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.2))
    ax[0].plot(deg,mz,'ko',ms=4,label='Measured torque')
    for name in ['measured_force_at_centre','measured_force_at_quarter_toward_shaft','barre_force_at_centre']:
        ax[0].plot(deg,moment_cases[name],label=name.replace('_',' '))
    ax[0].set(xlabel='Shaft angle [deg]',ylabel='Moment about oarlock [N m]');ax[0].legend(fontsize=7)
    ax[1].plot(deg,shift*1000,'ko-',ms=4,label='Measured equivalent CoP')
    ax[1].axhline(0,color='tab:blue',ls='--',label='Blade centre')
    ax[1].axhline(75,color='tab:orange',ls=':',label='Quarter-chord toward shaft')
    ax[1].set(xlabel='Shaft angle [deg]',ylabel='CoP shift toward shaft [mm]');ax[1].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'moment_and_pressure_centre.png',dpi=175);plt.close(fig)

    # Conservative ordinate-reading bounds; no randomized confidence claims.
    audit['digitization_2010']=dict(fx_bound_N=3/4.1,fy_bound_N=3/3.08,
        component_rmse_perturbation_bound_N=float(np.sqrt(((3/4.1)**2+(3/3.08)**2)/2)),
        note='Triangle-inequality bound from +/-3 native pixels; excludes x-axis and experimental uncertainty.')
    audit['model_scope']='No new fitting. Barre shares the source experiment family; Caplan/Grift transfers also change shape and protocol. No autonomous robot or catch/release validation.'
    audit['free_surface_interaction_at_90deg']=dict(
        static_deep_cd=1.15,static_surface_cd=1.55,dynamic_deep_cd=2.7,dynamic_surface_cd=2.2,experiment_cd=2.05,
        additive_prediction_cd=1.55+(2.7-1.15),
        nonadditive_interaction_cd=2.2-2.7-1.55+1.15,
        source='Leroyer2010 text around Figs7-11; approximate CFD ablations, not five experiments.')
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query("domain=='30_150deg'").to_string(index=False))
    print(json.dumps(audit['moment'],indent=2))


if __name__=='__main__':main()
