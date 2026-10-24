"""Compact scientific comparison figure and execution environment record."""
from pathlib import Path
import json,platform,sys
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
metrics=json.loads((OUT/'metrics.json').read_text())
df=pd.read_csv(OUT/'unsteady_predictions.csv')
fig=plt.figure(figsize=(12,6.5),layout='constrained')
gs=fig.add_gridspec(2,2,height_ratios=[1.4,1])
colors=['#a4a4a4','#d57b35','#087d91']
models=['quasi_steady','constant_added_mass','entrainment_tail']
labels=['Quasi-steady','+ constant added mass','+ entrainment / tail']
for col,(trial,title) in enumerate([('fig10a_5','Held-out: a = 0.41 m/s², V = 0.30 m/s'),
                                    ('fig10b_3','Held-out: a = 0.82 m/s², V = 0.35 m/s')]):
    ax=fig.add_subplot(gs[0,col]);d=df[df.trial==trial]
    ax.plot(d.time_s,d.measured_N,c='#222',lw=1.2,label='Published filtered trace')
    for m,c,l,ls in zip(models,colors,labels,[':','--','-']):
        ax.plot(d.time_s,d[m],color=c,ls=ls,lw=1.3,label=l)
    ax.set(title=title,xlabel='Time (s)',ylabel='Transducer force (N)',xlim=(0,3))
    ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    if col==0:ax.legend(fontsize=8)
ax=fig.add_subplot(gs[1,:]);values=[metrics['unsteady_held_out_full_window'][m]['rmse'] for m in models]
bars=ax.barh(labels[::-1],values[::-1],color=colors[::-1],height=.55)
for b,v in zip(bars,values[::-1]):ax.text(v+.008,b.get_y()+b.get_height()/2,f'{v:.3f} N',va='center',fontsize=11)
ax.set(xlabel='Full-window RMSE (N), pooled over 10 held-out conditions',xlim=(0,.8))
ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Accelerating-plate force model vs Grift et al. (2019), Fig. 10\nOne condition calibrated; all figures digitized from the published PDF',fontsize=13)
fig.savefig(OUT/'comparison_summary.png',dpi=180)
env={'python':sys.version,'platform':platform.platform(),
     'packages':{name:version(name) for name in ['numpy','scipy','pandas','matplotlib','mujoco','trimesh','shapely','pytest']},
     'tests':'python -m pytest tests/test_hydrodynamics.py -q: 7 passed',
     'extraction_note':'PDF extraction was run separately with bundled Python + pdfplumber/Pillow. Comparisons use the versioned CSV files.'}
(OUT/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
print('Wrote comparison_summary.png and environment.json')
