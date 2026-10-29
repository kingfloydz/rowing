"""Audit source lineage, ten-channel changes, and like-for-like IK diagnostics."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,mujoco,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from hydrodynamics.models import ROOT
from rowing_mjlab.asset import build_spec
from tools.literature_reference10 import NAMES


def main():
    out=ROOT/'renders/reference10_literature';meta=json.loads((ROOT/'data/reference20/metadata.json').read_text());m=build_spec().compile();d=mujoco.MjData(m)
    for source in meta['literature_lineage']['sources']:
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
    water=json.loads((ROOT/'data/reset_float_pose.json').read_text())['water_level_m'];boat=m.body('boat').id;torso=m.body('torso_link').id
    verts=[trimesh.load_mesh(ROOT/f'boat_meshes/{s}_blade_3mm.stl').vertices for s in ('port','starboard')];bodies=[m.body(s+'_oar').id for s in ('port','starboard')]
    def evaluate(path):
        a=np.load(path);phase=np.linspace(0,1,1601);q=CubicSpline(a['phase'],a['q'],bc_type='periodic')(phase);trunk=CubicSpline(a['phase'],a['trunk'],bc_type='periodic')(phase)
        names=list(a['names']);qi=[m.joint(n).qposadr[0] for n in names];hands=[];handangles=[];feet=[];heights=[];elbows=[];chord_offsets=[];penetration=[];gaps=[];handgaps=[];near=[];trunkerr=[]
        for angles,target in zip(q,trunk):
            d.qpos[:]=m.qpos0;d.qpos[qi]=angles;mujoco.mj_forward(m,d);r=d.xmat.reshape(-1,3,3)
            grasp=[d.site_xpos[m.site(s+'_grasp').id].copy() for s in ('starboard','port')];points=[];angles_error=[];foot=[];ee=[];offsets=[]
            for side in ('left','right'):
                b=m.body(side+'_dex1_base_link').id;points.append(d.xpos[b]+r[b]@np.array([.029,0,0]))
                he=m.equality(side+'_hand_fixed').id;ha,hb=m.eq_obj1id[he],m.eq_obj2id[he];rel=Rotation.from_quat(m.eq_data[he,6:10][[1,2,3,0]]).as_matrix()
                angles_error.append(Rotation.from_matrix((r[ha]@rel).T@r[hb]).magnitude())
                e=m.equality(side+'_foot_fixed').id;aa,bb=m.eq_obj1id[e],m.eq_obj2id[e];foot.append(np.linalg.norm(d.xpos[aa]+r[aa]@m.eq_data[e,3:6]-d.xpos[bb]))
                s,el,w=d.xanchor[[m.joint(side+'_'+part+'_joint').id for part in ('shoulder_yaw','elbow','wrist_roll')]]
                ee.append((el-s)@r[boat]);v=w-s;projection=s+v*((el-s)@v)/(v@v);offsets.append((el-projection)@r[boat])
            hands.append(np.linalg.norm(np.array(points)-grasp,axis=1));handangles.append(angles_error);feet.append(foot);elbows.append(ee);chord_offsets.append(offsets)
            heights.append([[(v@r[b].T+d.xpos[b])[:,2].min()-water,(v@r[b].T+d.xpos[b])[:,2].max()-water] for v,b in zip(verts,bodies)])
            penetration.append(max([0,*[-c.dist for c in d.contact]]));gaps.append(grasp[0][2]-grasp[1][2]);handgaps.append(points[0][2]-points[1][2]);near.append(np.linalg.norm(grasp[0][:2]-grasp[1][:2])<.15)
            trunkerr.append(np.arctan2(r[torso,0,2],r[torso,2,2])-target)
        hands,heights,elbows,chord_offsets,near=map(np.array,(hands,heights,elbows,chord_offsets,near));drive=(phase>=.11)&(phase<=.41);recovery=(phase>=.56)&(phase<=.96);finish=(phase>=.45)&(phase<=.55)
        limits=m.jnt_range[[m.joint(n).id for n in names]]
        return dict(hand_position_rmse_m=np.sqrt(np.mean(hands**2,axis=0)).tolist(),hand_position_max_m=hands.max(0).tolist(),
            hand_orientation_rmse_deg=np.rad2deg(np.sqrt(np.mean(np.array(handangles)**2,axis=0))).tolist(),hand_orientation_max_deg=np.rad2deg(np.max(handangles,axis=0)).tolist(),
            foot_position_max_m=np.max(feet,axis=0).tolist(),trunk_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(np.array(trunkerr)**2)))),
            drive_top_max_m=heights[drive,:,1].max(0).tolist(),recovery_bottom_min_m=heights[recovery,:,0].min(0).tolist(),
            full_cycle_bottom_min_m=heights[:,:,0].min(0).tolist(),full_cycle_top_max_m=heights[:,:,1].max(0).tolist(),
            crossing_target_height_min_m=float(np.min(np.array(gaps)[near])),crossing_actual_hand_height_min_m=float(np.min(np.array(handgaps)[near])),
            max_penetration_m=float(max(penetration)),collision_phase_fraction=float(np.mean(np.array(penetration)>0)),
            finish_elbow_relative_shoulder_mean_m=elbows[finish].mean(0).tolist(),finish_elbow_chord_offset_mean_m=chord_offsets[finish].mean(0).tolist(),
            maximum_joint_limit_violation=float(max(0,np.max(limits[:,0]-q),np.max(q-limits[:,1]))))
    adjusted=evaluate(ROOT/'data/reference20/trajectory.npz');baseline=evaluate(out/'baseline_diagnostic/trajectory.npz')
    report=dict(samples=1601,original_source_hashes_unchanged=True,adjusted=adjusted,literature_baseline_same_ik=baseline,
        comparison_note='Same current installation, pose-fitting objective, solver and phase grid. Local IK diagnostics, not a proof of global feasibility or dynamics. Baseline water metrics use adjusted interior windows and are contextual only.')
    tables={label:np.loadtxt(out/(label+'_ten_channels.csv'),delimiter=',',skiprows=1) for label in ('baseline','affine','adjusted')}
    report['maximum_post_affine_change']={name:float(np.max(abs(tables['adjusted'][:,i+1]-tables['affine'][:,i+1]))) for i,name in enumerate(NAMES)}
    titles=['Right sweep','Right elevation','Right feather','Left sweep','Left elevation','Left feather','Seat','Left elbow (mapped joint)','Right elbow (mapped joint)','Trunk vertical']
    fig,axes=plt.subplots(5,2,figsize=(14,16),sharex=True)
    for i,ax in enumerate(axes.flat):
        scale=1 if i==6 else 180/np.pi
        for label,color,style in [('baseline','#777777','--'),('affine','#d78b27',':'),('adjusted','#167c91','-')]:
            table=tables[label];ax.plot(table[:,0]*100,table[:,i+1]*scale,color=color,ls=style,lw=1.6,label={'baseline':'Literature baseline (robot units)','affine':'Amplitude + phase adjustment','adjusted':'Final reference'}[label])
        ax.set_title(titles[i]);ax.set_ylabel('m' if i==6 else 'deg');ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Cycle (%) — 20 strokes/min')
    axes[0,0].legend(fontsize=8);fig.suptitle('Ten channels: retained literature curves and explicit adjustments',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(out/'ten_channel_comparison.png',dpi=160);fig.savefig(out/'ten_channel_comparison.svg');plt.close(fig)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');meta['literature_validation']=report;(ROOT/'data/reference20/metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
