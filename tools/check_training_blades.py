"""No-fit transfer test of the two training force laws on digitized Grift2021.

Whole-blade representative-point evaluation (not current CAD strip integration).
No trial's measured forces are used to choose or fit parameters here.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hydrodynamics.training import TrainingParameters,blade_traction
from tools.trajectory_data import kinematics,force_observation,to_nt,to_xy


def main():
    out=ROOT/'results/training';out.mkdir(exist_ok=True)
    k=pd.read_csv(ROOT/'data/grift2021_kinematics.csv')
    f=pd.read_csv(ROOT/'data/grift2021_force_trajectories.csv')
    phase=np.linspace(.002,.998,201);p=TrainingParameters();rows=[]
    cases=[(0,.5),(0,.75),(0,1.),(15,1.)]
    fig,axes=plt.subplots(4,2,figsize=(11,12),sharex=True)
    for row,(beta,kappa) in enumerate(cases):
        kin,audit=kinematics(k,beta,kappa,phase)
        angle=kin['angle'];measurement_angle=angle-np.deg2rad(beta)
        obs=force_observation(f,beta,kappa,phase);obsxy=to_xy(obs,measurement_angle)
        for mode in ['normal','harmonic']:
            xy=np.array([blade_traction(np.r_[v,0.],np.array([np.cos(a),np.sin(a),0.]),
                np.array([-np.sin(a),np.cos(a),0.]),.02,p,mode=mode)[:2]
                for v,a in zip(kin['velocity'],angle)])
            nt=to_nt(xy,measurement_angle)
            impulse=float(np.trapezoid(xy[:,0],kin['time']))
            measured=float(np.trapezoid(obsxy[:,0],kin['time']))
            rows.append(dict(beta_deg=beta,kappa=kappa,model=mode,
                component_RMSE_N=float(np.sqrt(np.mean((nt-obs)**2))),
                impulse_error_percent=100*(impulse/measured-1),
                fit_to_this_data=False,interpretation='Uncalibrated force-law transfer, NOT CAD validation'))
            for j in range(2):axes[row,j].plot(phase,nt[:,j],label=mode)
        for j in range(2):
            axes[row,j].plot(phase,obs[:,j],'k',lw=1.5,label='Digitized experiment')
            axes[row,j].set_title(f'beta={beta}, kappa={kappa}: '+['Fn','Ft'][j])
            axes[row,j].set_ylabel('Force (N)');axes[row,j].grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Drive phase')
    axes[0,0].legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'blade_no_fit_comparison.png',dpi=150);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'blade_no_fit_comparison.csv',index=False)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
