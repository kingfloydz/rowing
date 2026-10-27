"""Align full blade emergence to the body figure's Finish at fixed hull pose.

Retains original oar CSV and applies one constant phase delay to all six
channels. Catch and Finish both shift; cycle-internal durations do not change.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
import trimesh
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]


def main():
    model=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    data=mujoco.MjData(model)
    source=np.loadtxt(ROOT/'data/rowing_reference.csv',delimiter=',',skiprows=1)
    curve=CubicSpline(source[:,0],source[:,1:],bc_type='periodic')
    meta=json.loads((ROOT/'data/rowing_reference_provenance.json').read_text())
    alignment=json.loads((ROOT/'data/reference9/catch_alignment.json').read_text())
    # The two panels depict one Finish; small fractions differ with digitization.
    finish=(alignment['elbow_finish_source_fraction']+alignment['trunk_finish_source_fraction'])/2
    water=meta['water_level_m_at_repository_reset']
    phase=source[:,0]
    values=np.empty_like(source[:,1:])
    ids=[model.joint(s+'_'+j).qposadr[0] for s in ('port','starboard')
         for j in ('sweep','elevation','feather')]
    reports=[]
    geometries=[]
    def clearance(phi, spline, body, vertices):
        data.qpos[:]=model.qpos0
        data.qpos[ids]=spline(phi)
        mujoco.mj_kinematics(model,data)
        return float((vertices@data.xmat[body].reshape(3,3).T+data.xpos[body])[:,2].min()-water)
    for i,side in enumerate(('port','starboard')):
        vertices=np.asarray(trimesh.load(ROOT/'boat_meshes'/f'{side}_blade_3mm.stl',process=False).vertices)
        body=model.body(side+'_oar').id
        geometries.append((body,vertices))
        grid=np.linspace(.2,.7,501)
        height=np.array([clearance(p,curve,body,vertices) for p in grid])
        exits=np.flatnonzero((height[:-1]<0)&(height[1:]>=0))
        if len(exits)!=1:
            raise ValueError(f'{side}: expected exactly one full-emergence event')
        k=exits[0]
        source_exit=brentq(lambda p:clearance(p,curve,body,vertices),grid[k],grid[k+1],xtol=1e-13)
        reports.append(dict(side=side, source_exit_phase=source_exit))
    delay = finish-np.mean([r['source_exit_phase'] for r in reports])
    values = curve((phase-delay)%1)
    values[-1]=values[0]
    spline=CubicSpline(phase,values,bc_type='periodic')
    for report,(body,vertices) in zip(reports,geometries):
        exit_phase=brentq(lambda p:clearance(p,spline,body,vertices),finish-.01,finish+.01,xtol=1e-13)
        report['resampled_exit_phase']=exit_phase
        report['clearance_at_finish_m']=clearance(finish,spline,body,vertices)
        if abs(exit_phase-(report['source_exit_phase']+delay))>1e-5:
            raise ValueError('Resampling moved emergence event beyond tolerance')
    header=(ROOT/'data/rowing_reference.csv').read_text().splitlines()[0]
    np.savetxt(ROOT/'data/rowing_reference_finish_aligned.csv',np.c_[phase,values],delimiter=',',comments='',header=header,fmt='%.12g')
    report=dict(target_finish_phase=finish,
        finish_definition='mean of printed elbow and T6 Finish fractions; mean bilateral emergence aligned to this event',
        event_definition='lowest vertex of full 3 mm blade mesh crosses water plane upward (fully out)',
        water_level_m=water,hull_pose='scene qpos0, held fixed',
        mapping='source_phase = (phase-delay) modulo 1; one constant delay for all six channels',
        phase_delay=float(delay), phase_delay_seconds=float(delay*meta['period_s']),
        original_data='data/rowing_reference.csv',runtime_data='original spline evaluated at shifted phase', sampled_preview='data/rowing_reference_finish_aligned.csv',
        unchanged=['source angle trajectories','within-cycle durations','bilateral relative timing','seat','elbow','trunk'],sides=reports,
        limitation='geometric reference event at fixed hull, not guaranteed emergence under learned boat motion')
    (ROOT/'data/oar_finish_alignment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
