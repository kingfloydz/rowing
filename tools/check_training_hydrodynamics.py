"""Generate training-profile diagnostics, no source assets or reference fits edited."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import asdict
from hydrodynamics.training import TrainingHullResistance,TrainingParameters,sample_episode,EPISODE_RANGES
from hydrodynamics.models import load_parameters


def main():
    out=ROOT/'results/training';out.mkdir(exist_ok=True)
    scene=json.loads((ROOT/'results/scene_audit.json').read_text())
    length=scene['waterline_length_m'];area=scene['hull_wetted_area_m2']
    speed=np.linspace(0,4,401);nominal=TrainingHullResistance()
    parts=nominal.components(speed,length,area)
    draws=np.array([TrainingHullResistance(sample_episode(i)).magnitude(speed,length,area) for i in range(512)])
    old,_,_=load_parameters()
    p05,p50,p95=np.quantile(draws,[.05,.5,.95],axis=0)
    cols=np.column_stack([speed,parts['Fn'],parts['friction_N'],parts['residual_N'],parts['total_N'],
        old.magnitude(speed,length,area),p05,p50,p95,draws.min(axis=0),draws.max(axis=0),parts['outside_reference']])
    np.savetxt(out/'hull_0_4.csv',cols,delimiter=',',comments='',
        header='speed_m_s,Fn,friction_N,residual_N,nominal_N,legacy_N,prior_p05_N,prior_p50_N,prior_p95_N,sample_min_N,sample_max_N,outside_day_Fn')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].fill_between(speed,p05,p95,alpha=.2,label='5-95% of assumed prior (512 samples)')
    axes[0].plot(speed,parts['total_N'],label='New nominal training model')
    axes[0].plot(speed,old.magnitude(speed,length,area),'--',label='Legacy clamped-tail model')
    axes[1].plot(speed,parts['friction_N'],label='Skin + nominal form contribution')
    axes[1].plot(speed,parts['residual_N'],label='Transferred Day residual')
    limits=nominal.fn[[0,-1]]*np.sqrt(9.81*length)
    for ax in axes:
        ax.axvspan(0,limits[0],color='grey',alpha=.10)
        ax.axvspan(limits[1],4,color='orange',alpha=.12)
        ax.set(xlabel='Boat-relative axial speed (m/s)',ylabel='Resistance (N)',xlim=(0,4))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Training prior: current H2 scene geometry, not measured CAD resistance')
    fig.tight_layout();fig.savefig(out/'hull_0_4.png',dpi=160);plt.close(fig)
    summary={'target_speed_m_s':[0,4],'geometry_source':'results/scene_audit.json',
        'robot':'H2 + Dex1, not final G1','length_m':length,'wetted_area_m2':area,
        'day_Fn_mapped_speed_m_s':limits.tolist(),'nominal_parameters':asdict(TrainingParameters()),
        'randomization_ranges':EPISODE_RANGES,'randomization_interpretation':'Engineering priors, NOT confidence bounds',
        'samples':512,'current_CAD_experimental_validation':False,'rows':[]}
    for u in [0,.5,1,2,3,3.5,4]:
        i=int(round(u/.01));summary['rows'].append({'speed_m_s':u,'nominal_N':float(parts['total_N'][i]),
            'friction_N':float(parts['friction_N'][i]),'residual_N':float(parts['residual_N'][i]),
            'prior_p05_N':float(p05[i]),'prior_p95_N':float(p95[i])})
    (out/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['rows'],indent=2))


if __name__=='__main__':main()
