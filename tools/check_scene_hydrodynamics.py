"""Geometry audit, static force balance, and short unconstrained-model smoke run.

This does NOT train or demonstrate a rowing controller. No source XML is edited.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,time
import numpy as np
import mujoco
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hydrodynamics.models import hydrostatic_wrench,wet_triangles,triangle_geometry,G
from hydrodynamics.mujoco_adapter import RowingHydrodynamics


def main():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));d=mujoco.MjData(m)
    hydro=RowingHydrodynamics(m);mujoco.mj_forward(m,d)
    j=int(m.joint('boat_free').id);qadr=int(m.jnt_qposadr[j]);vadr=int(m.jnt_dofadr[j])
    mass=float(m.body_mass.sum());weight=mass*G
    def at_height(z):
        d.qpos[qadr+2]=z;mujoco.mj_forward(m,d)
        _,loads,_=hydro.force_terms(d)
        return sum(l['force_N'][2] for l in loads)-weight
    z=brentq(at_height,-.3,.4,xtol=1e-10);at_height(z)
    q,loads,terms=hydro.force_terms(d)
    netf=sum((x['force_N'] for x in loads),np.zeros(3))
    com=d.subtree_com[hydro.boat].copy()
    netmoment=sum((x['torque_Nm']+np.cross(x['point_m']-com,x['force_N']) for x in loads),np.zeros(3))
    hull=hydro.world(d,hydro.boat,hydro.hull);wet=wet_triangles(hull)
    area,_,_=triangle_geometry(wet);lwl=np.ptp(wet[:,:,0]);draft=-hull[:,:,2].min()
    speeds=np.array([1.,2.,3.,4.]);drag=hydro.hull_law.magnitude(speeds,lwl,area.sum())
    # Fixed-pose reference only: residual trim/roll moments are explicitly retained.
    info={'repository_commit':'aaa31478ccb7eac62993865e3c4aac64313eeccf',
          'robot':'H2 + Dex1 (not G1)','nq':m.nq,'nv':m.nv,'nu':m.nu,'total_mass_kg':mass,
          'water_level_world_z_m':0,'level_pose_heave_m':z,'draft_m':float(draft),
          'waterline_length_m':float(lwl),'hull_wetted_area_m2':float(area.sum()),
          'total_buoyancy_N':float(netf[2]),'weight_N':weight,'heave_balance_error_N':float(netf[2]-weight),
          'buoyancy_moment_about_system_com_Nm':netmoment.tolist(),
          'system_com_m':com.tolist(),'surge_speed_m_s':speeds.tolist(),
          'transferred_hull_resistance_N':drag.tolist(),'geometry':hydro.metadata,
          'warning':'Heave balance at original level attitude only, not full trim/roll equilibrium. CAD hull transfer is unvalidated.'}
    fig,axes=plt.subplots(2,1,figsize=(11,5))
    pts=hull.reshape(-1,3)
    for ax,coord,title in zip(axes,[0,1],['Hull side projection','Hull transverse projection']):
        ax.scatter(pts[:,coord],pts[:,2],s=.8,c=np.where(pts[:,2]<0,'#16829b','#aaa'))
        ax.axhline(0,color='#16829b',ls='--');ax.set(ylabel='World z (m)',xlabel='x (m)' if coord==0 else 'y (m)',title=title)
        ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle(f'Original CAD at level heave balance: mass {mass:.2f} kg, draft {draft:.3f} m')
    fig.tight_layout();fig.savefig(ROOT/'results/geometry_waterline.png');plt.close(fig)
    # Moving initial state exercises hydrodynamic signs and implicit solve with all
    # original robot joints/welds. 0.1 s only; unactuated posture will not persist.
    d.qvel[vadr]=1.0;mujoco.mj_forward(m,d)
    maxres=0;start=time.perf_counter()
    for i in range(50):
        r=hydro.step(d);maxres=max(maxres,r['added_mass_residual_N'])
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):raise RuntimeError('Nonfinite scene state')
    info['smoke_run']={'steps':50,'sim_time_s':float(d.time),'wall_time_s':time.perf_counter()-start,
                       'max_implicit_added_mass_residual_N':maxres,'final_boat_position_m':d.qpos[qadr:qadr+3].tolist(),
                       'finite_state':True,'meaning':'Short numerical integration check, not controller or long-term stability validation.'}
    (ROOT/'results/scene_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
