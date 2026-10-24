"""Fit designated training conditions; report held-out figure-data comparisons."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hydrodynamics.models import HullResistance, BladeDrag, ittc57, start_plate_force, G, plate_added_mass

OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})


def scores(y,p):
    y,p=np.asarray(y),np.asarray(p)
    return {'rmse':float(np.sqrt(np.mean((p-y)**2))),
            'mae':float(np.mean(np.abs(p-y))),
            'nrmse_mean_percent':float(100*np.sqrt(np.mean((p-y)**2))/np.mean(np.abs(y))),
            'mape_percent':float(100*np.mean(np.abs(p-y)/np.maximum(np.abs(y),1e-6)))}


def main():
    results={}
    d=pd.read_csv(ROOT/'data/day2011_hull_resistance.csv')
    u=d.speed_m_s.to_numpy(); ct=d.ct.to_numpy(); n=len(d)
    train=np.arange(n)%2==0; train[-1]=True
    length=7.937; area=2.340; nu=1e-6; k=.012
    fn=u/np.sqrt(G*length); cr=ct-(1+k)*ittc57(u,length,nu)
    hull=HullResistance(fn[train],np.maximum(cr[train],0),k,nu)
    predct=hull.coefficient(u,length)
    measured=.5*1000*area*u*u*ct; predicted=.5*1000*area*u*u*predct
    d['split']=np.where(train,'calibration','held_out_speed'); d['resistance_N']=measured
    d['prediction_N']=predicted; d['prediction_ct']=predct
    d.to_csv(OUT/'hull_comparison.csv',index=False)
    results['hull_held_out']={**scores(measured[~train],predicted[~train]),'n':int(sum(~train)),
                              'ct_rmse':scores(ct[~train],predct[~train])['rmse'],
                              'interpretation':'Held-out speed interpolation on literature hull; not validation of repository CAD.'}
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    fine=np.linspace(u.min(),u.max(),400)
    for a,y,yp,title in [(ax[0],ct,hull.coefficient(fine,length),'Total resistance coefficient'),
                          (ax[1],measured,hull.magnitude(fine,length,area),'Resistance (N)')]:
        a.plot(fine,yp,color='#1b7387',label='Reduced model')
        a.scatter(u[train],y[train],marker='o',facecolors='none',edgecolors='#1b7387',label='Calibration speeds')
        a.scatter(u[~train],y[~train],marker='D',color='#d46d24',label='Held-out speeds')
        a.set(xlabel='Speed (m/s)',ylabel=title);a.grid(alpha=.2)
    ax[0].legend(fontsize=8);fig.suptitle('Day et al. (2011), Fig. 11b — digitized markers')
    fig.tight_layout();fig.savefig(OUT/'hull_validation.png');plt.close(fig)

    s=pd.read_csv(ROOT/'data/grift2019_steady_markers.csv')
    # Nominal experimental depths are integral mm; vector marker/grid offset is
    # about 0.24 mm. Snap BEFORE filtering, so h=0 surface trials are retained.
    s['depth_mm']=np.round(s.h_m*1000).astype(int)
    s=s[s.depth_mm>=0].copy()
    grouped=s.groupby('depth_mm').cd.mean(); depths=grouped.index.to_numpy()/1000
    st=np.arange(len(depths))%2==0;st[-1]=True
    blade=BladeDrag(depths[st]/.1,grouped.to_numpy()[st])
    s['split']=np.where(s.depth_mm.isin(grouped.index[st]),'calibration_depth','held_out_depth')
    s['prediction_cd']=blade.cd(s.depth_mm.to_numpy()/1000,.1)
    s.to_csv(OUT/'blade_depth_comparison.csv',index=False)
    test=s['split']=='held_out_depth'
    results['blade_depth_held_out']={**scores(s.loc[test,'cd'],s.loc[test,'prediction_cd']),
                                     'n_markers':int(test.sum()),'n_depths':int((~st).sum()),
                                     'interpretation':'Interpolation across omitted depths; broadside plate, not arbitrary angle.'}
    fig,ax=plt.subplots(figsize=(8,3.8));xx=np.linspace(0,.2,501)
    ax.plot(xx*1000,blade.cd(xx,.1),color='#1b7387',label='PCHIP depth law (training depths only)')
    ax.scatter(s.loc[~test,'h_m']*1000,s.loc[~test,'cd'],s=13,facecolors='none',edgecolors='#1b7387',label='Calibration')
    ax.scatter(s.loc[test,'h_m']*1000,s.loc[test,'cd'],s=14,marker='x',color='#d46d24',label='Held-out depths')
    ax.axhline(1.3,ls='--',color='gray',label='Deep-water reference 1.3')
    ax.set(xlabel='Top-edge depth h (mm)',ylabel='Broadside drag coefficient',title='Grift et al. (2019), Fig. 4 — digitized markers')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'blade_depth_validation.png');plt.close(fig)

    raw=pd.read_csv(ROOT/'data/grift2019_acceleration_curves.csv')
    # Equal 0.005 s samples prevent PDF path decimation from weighting errors.
    trials={}
    for name,g in raw.groupby('trial',sort=False):
        if name=='fig10b_2':continue # same condition as fig10a_0, avoid double counting
        g=g.groupby('time_s',as_index=False).mean(numeric_only=True).sort_values('time_s')
        t=np.arange(.005,2.996,.005)
        t=t[(t>=g.time_s.min())&(t<=g.time_s.max())]
        trials[name]=(t,np.interp(t,g.time_s,g.measured_total_force_N),float(g.acceleration_m_s2.iloc[0]),float(g.target_speed_m_s.iloc[0]))
    t,y,a,v=trials['fig10b_4'];cd=float(blade.cd(.1,.1));tc=v/a
    valid_acc=(t>=.09)&(t<tc-.04)
    base=start_plate_force(t,a,v,cd)
    basis=a*np.maximum(t-.07,0)
    mu=max(0,float(np.dot(basis[valid_acc],(y-base)[valid_acc])/np.dot(basis[valid_acc],basis[valid_acc])))
    valid_tail=t>=tc+.07
    fit=least_squares(lambda x:(start_plate_force(t,a,v,cd,mu,x[0],x[1])-y)[valid_tail],
                      [1.0,3.0],bounds=([0,.1],[10,30]))
    amplitude,scale=fit.x
    params={'hull':{'fn_nodes':fn[train].tolist(),'residual_nodes':np.maximum(cr[train],0).tolist(),
                    'form_factor':k,'nu':nu,'calibration_reference_length_m':length,
                    'note':'nu=1e-6 nominal freshwater assumption; PCHIP residual, clamp outside range.'},
            'blade':{'relative_depth':blade.relative_depth.tolist(),'cd_nodes':blade.cd_nodes.tolist(),
                     'note':'Fitted to alternate depths from Fig4; full-wet broadside flat plate.'},
            'start_from_rest_benchmark':{'training_trial':'fig10b_4','entrainment_kg_s':mu,
                    'tail_amplitude':float(amplitude),'tail_scale':float(scale),'cd_at_h_0_1':cd,
                    'added_mass_kg':float(plate_added_mass(.2,.1)),
                    'note':'Start-from-rest benchmark only. Tail is a fitted envelope, not paper equation for h=0.1.'}}
    (ROOT/'data/parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    rows=[]; times=[]; pools={m:[[],[]] for m in ['quasi_steady','constant_added_mass','entrainment_tail']}
    fig,axes=plt.subplots(3,4,figsize=(14,9),sharex=True,sharey=True)
    for ax,(name,(t,y,a,v)) in zip(axes.flat,trials.items()):
        tc=v/a;valid=(t>=.07)&(np.abs(t-tc)>=.07)
        predictions={'quasi_steady':start_plate_force(t,a,v,cd,added_mass=False),
                     'constant_added_mass':start_plate_force(t,a,v,cd),
                     'entrainment_tail':start_plate_force(t,a,v,cd,mu,amplitude,scale)}
        split='calibration' if name=='fig10b_4' else 'held_out_condition'
        ax.plot(t,y,color='#262626',lw=1,label='Published filtered trace')
        for model,color,ls in [('quasi_steady','#999999',':'),('constant_added_mass','#d46d24','--'),('entrainment_tail','#137b91','-')]:
            yp=predictions[model]
            ax.plot(t,yp,color=color,ls=ls,lw=1,label=model.replace('_',' '))
            rows.append({'trial':name,'split':split,'a_m_s2':a,'v_m_s':v,'model':model,
                         **scores(y[valid],yp[valid]),'full_window_rmse_N':scores(y,yp)['rmse'],
                         'peak_force_relative_error_percent':float(100*(yp.max()-y.max())/y.max()),
                         'samples':int(valid.sum())})
            if split=='held_out_condition':pools[model][0].extend(y[valid]);pools[model][1].extend(yp[valid])
        for j in range(len(t)):
            times.append({'trial':name,'split':split,'time_s':t[j],'measured_N':y[j],
                          'metric_mask':int(valid[j]),**{m:p[j] for m,p in predictions.items()}})
        ax.axvspan(0,.07,color='gray',alpha=.12);ax.axvspan(max(0,tc-.07),tc+.07,color='gray',alpha=.12)
        ax.set_title(f'{"Calibration" if split=="calibration" else "Held-out"}: a={a:.2f}, V={v:.2f}',fontsize=9);ax.grid(alpha=.15)
        ax.set(xlim=(0,3),ylim=(0,5.5))
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',fontsize=10)
    for ax in axes[-1,:3]:ax.set_xlabel('Time (s)')
    for ax in axes[:,0]:ax.set_ylabel('Transducer force (N)')
    fig.suptitle('Grift et al. (2019), Fig. 10 — one training condition, ten held-out conditions\nShaded intervals excluded from primary metrics; plate inertia is included',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'unsteady_validation.png');plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'unsteady_metrics.csv',index=False)
    pd.DataFrame(times).to_csv(OUT/'unsteady_predictions.csv',index=False)
    results['unsteady_held_out']={m:scores(*p) for m,p in pools.items()}
    full=pd.DataFrame(times);full=full[full.split=='held_out_condition']
    results['unsteady_held_out_full_window']={m:scores(full.measured_N,full[m]) for m in pools}
    results['unsteady_calibration']=params['start_from_rest_benchmark']
    results['notes']=['Figure-digitized data, not raw sensor records. No experimental uncertainty inferred from pixel precision.',
       'Primary dynamic metrics exclude first 0.07 s and +/-0.07 s about acceleration stop; full-window RMSE also reported.',
       'Fig10b_2 excluded because it shares the a=.82,V=.30 condition with Fig10a_0.',
       'Dynamic forces include plate mass 0.4 kg times imposed acceleration. Online hydrodynamics must not add robot/oar dry mass again.']
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
