"""Unsteady wave benchmark: equation model and explicitly scoped empirical ROM."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hydrodynamics.waves import normalized_mean_wave_correction,maximum_steady_coefficient,quasi_static_mean_correction
OUT=ROOT/'results/trajectories'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/dode2022_wave_experiments.csv')
    # Fit two single-amplitude series, chosen as moderate/high signal amplitude.
    # Hull B is completely held out. This table applies only to these hull shapes
    # and Fn values; it cannot be assigned directly to the rowing CAD hull.
    data['split']=np.where(((data.hull=='A')&(data.epsilon==.25))|((data.hull=='C')&(data.epsilon==.3)),'calibration','holdout')
    data['equation_prediction']=[normalized_mean_wave_correction(f,w) for f,w in zip(data.fn,data.omega_nd)]
    data['qs_prediction']=[quasi_static_mean_correction(f)/maximum_steady_coefficient()[1] for f in data.fn]
    data['response_table_prediction']=np.nan
    tables={}
    for fn,group in data.groupby('fn'):
        train=group[group.split=='calibration'].sort_values('omega_nd')
        tables[str(fn)]={'hull':str(train.hull.iloc[0]),'epsilon':float(train.epsilon.iloc[0]),
                         'omega_nd':train.omega_nd.tolist(),'a0_normalized':train.a0_normalized.tolist()}
        ok=(group.omega_nd>=train.omega_nd.min())&(group.omega_nd<=train.omega_nd.max())
        data.loc[group[ok].index,'response_table_prediction']=PchipInterpolator(train.omega_nd,train.a0_normalized)(group[ok].omega_nd)
    rows=[]
    for name,mask in [('all_visible',data.plot_clipped==0),('heldout_visible',(data.plot_clipped==0)&(data.split=='holdout')),
                      ('heldout_eps_ge_20',(data.plot_clipped==0)&(data.split=='holdout')&(data.epsilon>=.2)),
                      ('heldout_hull_B',(data.hull=='B')&(data.plot_clipped==0)),
                      ('common_holdout_support',(data.plot_clipped==0)&(data.split=='holdout')&data.response_table_prediction.notna())]:
        for model in ['qs_prediction','equation_prediction','response_table_prediction']:
            a=data[mask&data[model].notna()]
            error=a[model]-a.a0_normalized
            rows.append(dict(subset=name,model=model,n=len(a),a0_rmse=float(np.sqrt(np.mean(error**2))),
                             normalized_wave_increment_rmse=float(np.sqrt(np.mean((error*a.epsilon**2)**2)))))
    pd.DataFrame(rows).to_csv(OUT/'wave_metrics.csv',index=False)
    data.to_csv(OUT/'wave_predictions.csv',index=False)
    (OUT/'wave_response_tables.json').write_text(json.dumps(tables,indent=2),encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,fn in zip(axes,[.3,.57]):
        a=data[(data.fn==fn)&(data.plot_clipped==0)]
        for (hull,epsilon),g in a.groupby(['hull','epsilon']):
            ax.scatter(g.omega_nd,g.a0_normalized,s=25,marker='D' if hull=='B' else 'o',
                       label=f'{hull}: eps={epsilon:g}'+(' fit' if (g.split=='calibration').all() else ' test'))
        ws=np.linspace(.05,4.4,160);theory=np.array([normalized_mean_wave_correction(fn,w) for w in ws])
        theory[abs(fn*ws-.25)<.004]=np.nan
        ax.plot(ws,theory,'k-',lw=1.3,label='Appendix E calculation')
        tab=tables[str(fn)];xs=np.linspace(min(tab['omega_nd']),max(tab['omega_nd']),200)
        ax.plot(xs,PchipInterpolator(tab['omega_nd'],tab['a0_normalized'])(xs),'g--',label='Calibrated response table')
        ax.axhline(quasi_static_mean_correction(fn)/maximum_steady_coefficient()[1],color='gray',ls=':',label='QS limit')
        ax.set(xlabel='Dimensionless frequency',ylabel='Normalized mean wave correction A0',title=f'Fn = {fn}',ylim=((-1,5) if fn==.3 else (-7,1)))
        ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'unsteady_wave_comparison.png',dpi=170);plt.close(fig)
    # Compare against published theory separately: a reproduction audit, not data.
    published=pd.read_csv(ROOT/'data/dode2022_published_theory.csv');audit=[]
    for fn,a in published.groupby('fn'):
        a=a[(a.omega_nd>.1)&(a.omega_nd<4)&(abs(fn*a.omega_nd-.25)>.02)]
        calculated=np.array([normalized_mean_wave_correction(fn,w) for w in a.omega_nd])
        audit.append(dict(fn=fn,n=len(a),published_theory_curve_rmse=float(np.sqrt(np.mean((calculated-a.a0_normalized)**2)))))
    result=dict(steady_maximum_fn=maximum_steady_coefficient()[0],n_experimental_points=len(data),n_clipped=int(data.plot_clipped.sum()),
                formula_curve_audit=audit,
                note='Appendix E calculation does not exactly reproduce the plotted theory, particularly Fn0.3. Cause unresolved. No quantitative CFD-level accuracy claim.',
                response_table_scope='Only geometrically similar Dode NACA0017 hulls at the two measured Fn; no extrapolation, no transfer to rowing CAD.',metrics=rows)
    (OUT/'wave_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False))
    print('theory audit',audit)


if __name__=='__main__':main()
