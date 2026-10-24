"""Reproducible theory-core audit; writes only results/core/*.

New calculations: full fixed-posture hydrostatic trim/stiffness, dimensional
support screening, and condition-balanced experimental metrics. No fitting.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import mujoco
from scipy.optimize import least_squares,brentq
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hydrodynamics.models import hydrostatic_wrench,wet_triangles,load_parameters
from hydrodynamics.mujoco_adapter import RowingHydrodynamics
from hydrodynamics.reference_models import (DayReferenceResistance,caplan_static_force,
                                           grift_steady_normal_force,reijtenbagh_start_force)

OUT=ROOT/'results/core'


def trim_audit():
    model=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    adapter=RowingHydrodynamics(model)
    boat=adapter.boat;origin=data.xpos[boat].copy();rotation=data.xmat[boat].reshape(3,3)
    surfaces=[adapter.hull]
    for body,_,volume,_ in adapter.blades:
        world=adapter.world(data,body,volume)
        surfaces.append((world-origin)@rotation)
    mass=float(model.body_mass.sum());weight=mass*9.81
    com_body=(data.subtree_com[boat]-origin)@rotation

    def loads(q):
        z,roll,pitch=q
        rot=Rotation.from_euler('xyz',[roll,pitch,0]).as_matrix()
        position=np.array([origin[0],origin[1],z])
        com=position+rot@com_body
        force=np.array([0.,0.,-weight]);moment=np.cross(com-position,force)
        for surface in surfaces:
            f,m,_=hydrostatic_wrench(surface@rot.T+position,position)
            force+=f;moment+=m
        # Generalized Euler-angle torques about boat origin, including gravity.
        roll_axis=Rotation.from_euler('y',pitch).apply([1.,0.,0.])
        pitch_axis=np.array([0.,1.,0.])
        return np.array([force[2],moment@roll_axis,moment@pitch_axis])

    initial=np.array([.1193943,0.,0.])
    solution=least_squares(lambda q:loads(q)/[weight,.3*weight,3*weight],
        initial,bounds=([-.15,-.65,-.25],[.35,.65,.25]),
        xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
    q=solution.x;residual=loads(q)
    if not solution.success or np.max(np.abs(residual))>1e-5:
        raise RuntimeError(f'No near-upright fixed-posture trim found: {solution.message}; {residual}')
    stiffness=[]
    for step in [1e-5,5e-6]:
        k=-np.column_stack([(loads(q+step*e)-loads(q-step*e))/(2*step) for e in np.eye(3)])
        stiffness.append(k)
    k=stiffness[-1]
    reduced=k[1:,1:]-np.outer(k[1:,0],k[0,1:])/k[0,0]
    eigen=np.linalg.eigvalsh(.5*(reduced+reduced.T))
    rows=[]
    for delta in np.linspace(-.20,.20,81):
        candidate=q.copy();candidate[1]+=delta
        candidate[0]=brentq(lambda z:loads([z,candidate[1],candidate[2]])[0],-.15,.35)
        rows.append({'roll_deg':np.degrees(candidate[1]),'heave_m':candidate[0],
                     'generalized_roll_torque_Nm':loads(candidate)[1]})
    curve=pd.DataFrame(rows);curve.to_csv(OUT/'hydrostatic_roll_curve.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(curve.roll_deg,curve.generalized_roll_torque_Nm)
    ax.axhline(0,color='black',lw=.8);ax.axvline(np.degrees(q[1]),color='gray',ls='--')
    ax.set(xlabel='Roll angle (deg)',ylabel='Net roll torque (N m)',
           title='Fixed robot posture: heave-equilibrated hydrostatic torque')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'hydrostatic_roll_curve.png',dpi=160);plt.close(fig)
    return {'pose_z_roll_pitch_SI':q.tolist(),'roll_pitch_deg':np.degrees(q[1:]).tolist(),
        'residual_Fz_Mroll_Mpitch_SI':residual.tolist(),
        'stiffness_coordinates':['heave_m','roll_rad','pitch_rad'],
        'stiffness_matrix':k.tolist(),'stiffness_antisymmetry_max':float(np.max(np.abs(k-k.T))),
        'stiffness_step_difference_max':float(np.max(np.abs(stiffness[0]-k))),
        'heave_relaxed_rotational_stiffness_Nm':reduced.tolist(),
        'rotational_stiffness_eigenvalues_Nm':eigen.tolist(),
        'locally_hydrostatically_stable_at_fixed_posture':bool(k[0,0]>0 and eigen.min()>0),
        'limitations':['fixed robot and oar posture, not actively balanced',
                      'same separate closed CAD components; overlap union not resolved',
                      'flat-water hydrostatics only; shaft/fin displacement omitted',
                      'no experimental trim or stability validation'],
        'total_mass_kg':mass}


def reference_recheck():
    """Independent calls through the new restricted APIs; original holdouts."""
    rows=[]
    hull=pd.read_csv(ROOT/'results/hull_comparison.csv')
    depth=pd.read_csv(ROOT/'results/blade_depth_comparison.csv')
    def add(study,quantity,predicted,measured,old_prediction):
        rows.append({'study':study,'quantity':quantity,'n':len(measured),
            'rmse':float(np.sqrt(np.mean((np.asarray(predicted)-np.asarray(measured))**2))),
            'max_change_from_previous_prediction':float(np.max(np.abs(np.asarray(predicted)-np.asarray(old_prediction))))})
    h=hull[hull.split=='held_out_speed']
    predicted=-DayReferenceResistance().force(h.speed_m_s.to_numpy(),
        geometry_id='day2011_reference_hull',wetted_area=2.34)
    add('Day2011','resistance_N',predicted,h.resistance_N,h.prediction_N)
    d=depth[depth.split=='held_out_depth']
    predicted=-grift_steady_normal_force(.3,d.depth_mm.to_numpy()/1000)/(.5*1000*.02*.3**2)
    add('Grift2019','Cd',predicted,d.cd,d.prediction_cd)
    c=pd.read_csv(ROOT/'results/extended/caplan_predictions.csv')
    c=c[c.split=='held_out']
    if len(c)==0:raise ValueError('No Caplan holdout rows found')
    predicted=caplan_static_force(np.radians(c.angle_deg.to_numpy()))/(.5*999*.007742*.75**2)
    add('Caplan2007','Cd',predicted[:,0],c.cd,c.cd_polar)
    add('Caplan2007','Cl',predicted[:,1],c.cl,c.cl_polar)
    history=pd.read_csv(ROOT/'results/extended/history_2026_predictions.csv')
    predicted=[];measured=[];previous=[]
    for (motion,parameter),q in history.groupby(['motion','parameter']):
        t=np.r_[0.,q.time_s.to_numpy()]
        force=reijtenbagh_start_force(t,motion,parameter,.6,
            full_wet_fixed_depth=True,straight_normal_motion=True,
            quiescent_start=True,free_surface_negligible=True)[1:]
        predicted.extend(force);measured.extend(q.measured);previous.extend(q.history)
    add('Reijtenbagh2026','excess_force_N',predicted,measured,previous)
    pd.DataFrame(rows).to_csv(OUT/'reference_revalidation.csv',index=False)
    return rows


def metric_audit():
    source=pd.read_csv(ROOT/'results/extended/metrics.csv')
    rows=[]
    for scope in ['full_window','acceleration_core']:
        selected=source[(source.study=='Grift2019')&(source.scope==scope)]
        for model,q in selected.groupby('model'):
            rows.append({'study':'Grift2019_transfer','scope':scope,'model':model,
                'n_conditions':len(q),'equal_condition_mean_rmse_N':float(q.rmse.mean()),
                'equal_condition_rms_rmse_N':float(np.sqrt(np.mean(q.rmse**2))),
                'pooled_rmse_N':float(np.sqrt(np.sum(q.n*q.rmse**2)/q.n.sum()))})
    pd.DataFrame(rows).to_csv(OUT/'condition_balanced_metrics.csv',index=False)
    return rows


def support_audit():
    law,_,_=load_parameters()
    scene=json.loads((ROOT/'results/scene_audit.json').read_text(encoding='utf8'))
    length=scene['waterline_length_m']
    rows=[]
    # Frequencies are scenarios, NOT measurements of this robot's stroke.
    for speed in [2.,3.,4.,5.]:
        for frequency in [.3,.5,.75,1.]:
            rows.append({'speed_m_s':speed,'stroke_frequency_Hz':frequency,
                'Fn':speed/np.sqrt(9.81*length),
                'Omega_nd':2*np.pi*frequency*np.sqrt(length/9.81),
                'Fn_times_Omega':2*np.pi*frequency*speed/9.81,
                'Wehausen_resonance_frequency_Hz':9.81/(8*np.pi*speed),
                'inside_reference_Fn_range':bool(law.fn_nodes[0]<=speed/np.sqrt(9.81*length)<=law.fn_nodes[-1])})
    pd.DataFrame(rows).to_csv(OUT/'dimensionless_support.csv',index=False)
    return {'CAD_reference_wet_length_m':length,
            'source_Fn_range':[float(law.fn_nodes.min()),float(law.fn_nodes.max())],
            'same_Fn_speed_range_m_s':(law.fn_nodes[[0,-1]]*np.sqrt(9.81*length)).tolist(),
            'note':'Fn overlap alone does not validate transfer to this CAD geometry. Frequency scenarios are not observed motions.'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result={'hydrostatic_trim':trim_audit(),'support':support_audit(),
            'balanced_experiment_metrics':metric_audit(),
            'scoped_reference_revalidation':reference_recheck(),
            'no_experimental_parameters_refitted':True}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
