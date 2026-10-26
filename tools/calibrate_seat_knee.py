"""Calibrate a 172-degree sagittal knee target and map seat travel through H2 IK.

Standalone diagnostic. Does not change the scene, reset, or training reference.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/seat_knee_172'


def seat_positions_to_knee(seat_positions_m, map_path=OUT/'seat_to_knee_map.csv'):
 """Vectorized saved-map conversion; angles in radians, invalid inputs yield NaN.

 Column order in `angles_rad`: left/right geometric included angle, then
 left/right motor coordinate. Only the contiguous interval from finish is used.
 This is a geometry/posture-dependent lookup, not a universal knee formula.
 """
 map_path=Path(map_path)
 metadata=json.loads((map_path.parent/'calibration.json').read_text(encoding='utf-8'))
 if metadata['scene_sha256']!=hashlib.sha256((ROOT/'scene.xml').read_bytes()).hexdigest():
  raise ValueError('Seat/knee map belongs to a different scene; regenerate or pass the current map_path')
 table=np.loadtxt(map_path,delimiter=',',skiprows=1)
 failed=np.flatnonzero(table[:,-1]<.5)
 table=table[:int(failed[0])] if len(failed) else table
 if len(table)<2:raise ValueError('No usable contiguous seat/knee calibration')
 q=np.asarray(seat_positions_m,dtype=float)
 angles=PchipInterpolator(table[::-1,0],np.deg2rad(table[::-1,2:6]),extrapolate=False)(q)
 return {'angles_rad':angles,'valid':np.isfinite(angles).all(axis=-1)}


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));d=mujoco.MjData(m)
 names=['seat_slide','seat_pelvis_hinge']+[s+'_'+j+'_joint' for s in ('left','right')
  for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_roll','ankle_pitch')]
 ids=[m.joint(n).id for n in names];qa=m.jnt_qposadr[ids]
 lo,hi=m.jnt_range[ids].T.copy();lo[0]=-.7;hi[0]=.7
 base=m.qpos0[qa].copy();footids=[m.equality(s+'_foot_fixed').id for s in ('left','right')]
 kneeidx=[names.index(s+'_knee_joint') for s in ('left','right')]
 def evaluate(x):
  d.qpos[:]=m.qpos0;d.qpos[qa]=x;mujoco.mj_fwdPosition(m,d)
  mask=(d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&np.isin(d.efc_id,footids)
  R=d.xmat[m.body('boat').id].reshape(3,3)
  included=[]
  for s in ('left','right'):
   p=[R.T@d.xanchor[m.joint(s+'_'+j+'_joint').id] for j in ('hip_pitch','knee','ankle_pitch')]
   h,k,a=[v[[0,2]] for v in p];u=h-k;v=a-k
   included.append(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
  return d.efc_pos[mask],np.array(included)
 def physical_errors():
  pos=[];rot=[]
  for i in footids:
   a,b=int(m.eq_obj1id[i]),int(m.eq_obj2id[i]);ra=d.xmat[a].reshape(3,3);rb=d.xmat[b].reshape(3,3)
   rel=m.eq_data[i];rr=Rotation.from_quat(rel[6:10][[1,2,3,0]]).as_matrix()
   pos.append(np.linalg.norm(d.xpos[a]+ra@rel[3:6]-d.xpos[b]))
   rot.append(np.rad2deg(Rotation.from_matrix((ra@rr).T@rb).magnitude()))
  return max(pos),max(rot)
 target=np.deg2rad(172.)
 def anchor_res(v):
  eq,ang=evaluate(v)
  return np.r_[1000*eq,10*(ang-target),.003*(v[1]-base[1]),.000001*(v-base)]
 seeds=[]
 for p in [-30,-20,-10,10,30]:
  seed=base.copy();seed[0]=.25;seed[1]=np.deg2rad(p);seed[kneeidx]=.15;seeds.append(seed)
 fits=[least_squares(anchor_res,np.clip(s,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=500,
  ftol=1e-11,xtol=1e-11,gtol=1e-11) for s in seeds]
 fit=min(fits,key=lambda f:np.linalg.norm(f.fun));finish=fit.x.copy();eq,ang=evaluate(finish)
 finish_errors=physical_errors()
 assert max(abs(np.rad2deg(ang)-172))<.1 and finish_errors[0]<1e-4 and finish_errors[1]<.1
 print('finish',finish[0],'included angles',np.rad2deg(ang),'motor angles',np.rad2deg(finish[kneeidx]),flush=True)
 raw=np.loadtxt(ROOT/'data/reference9/wirms_stage4_raw.csv',delimiter=',',skiprows=1)
 span=float(np.ptp(raw[:,1]));requested_min=float(finish[0]-span)
 seats=np.linspace(finish[0],requested_min,301);x=finish.copy();rows=[];poses=[]
 for i,seat in enumerate(seats):
  prev=x.copy()
  def res(v):
   xx=np.r_[seat,v];eq,_=evaluate(xx)
   return np.r_[1000*eq,.003*(v[0]-finish[1]),.00001*(v-prev[1:])]
  fit=least_squares(res,np.clip(x[1:],lo[1:]+1e-9,hi[1:]-1e-9),bounds=(lo[1:],hi[1:]),
   max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  x=np.r_[seat,fit.x];eq,ang=evaluate(x);ep,er=physical_errors()
  if ep>1e-4 or er>.1:
   for p in [-30,-10,10,30]:
    seed=x[1:].copy();seed[0]=np.deg2rad(p)
    trial=least_squares(res,np.clip(seed,lo[1:]+1e-9,hi[1:]-1e-9),bounds=(lo[1:],hi[1:]),
     max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    if np.linalg.norm(trial.fun)<np.linalg.norm(fit.fun):fit=trial
   x=np.r_[seat,fit.x];eq,ang=evaluate(x);ep,er=physical_errors()
  valid=(ep<1e-4 and er<.1)
  rows.append([seat,seat-finish[0],*np.rad2deg(ang),*np.rad2deg(x[kneeidx]),np.rad2deg(x[1]),ep,er,int(valid)])
  poses.append(x.copy())
  if i%60==0:print('seat',round(seat,4),'knee',np.round(np.rad2deg(ang),2),'foot m/deg',ep,er,flush=True)
 rows=np.array(rows);poses=np.array(poses);valid=rows[:,-1].astype(bool)
 # Use the contiguous feasible interval connected to the calibrated finish.
 invalid=np.flatnonzero(~valid);nvalid=int(invalid[0]) if len(invalid) else len(rows)
 accepted=rows[:nvalid];catch=poses[nvalid-1].copy();feasible_min=float(catch[0])
 np.savetxt(OUT/'seat_to_knee_map.csv',rows,delimiter=',',comments='',header=
  'seat_joint_m,seat_relative_finish_m,left_included_angle_deg,right_included_angle_deg,left_motor_angle_deg,right_motor_angle_deg,pelvis_hinge_deg,max_foot_error_m,max_foot_error_deg,valid')
 np.savez_compressed(OUT/'mapping_poses.npz',joint_names=names,poses=poses,valid=valid)
 # Conversion is only defined on the connected validated interval, with no clipping.
 interp=PchipInterpolator(accepted[::-1,0],accepted[::-1,2:6],extrapolate=False)
 seat_ref=finish[0]+raw[:,1]-raw[:,1].max();angles_ref=interp(seat_ref)
 np.savetxt(OUT/'wirms_seat_to_knee.csv',np.c_[raw[:,0],raw[:,1],seat_ref,angles_ref,np.isfinite(angles_ref[:,0])],
  delimiter=',',comments='',header='source_phase,source_seat_m,seat_joint_m,left_included_angle_deg,right_included_angle_deg,left_motor_angle_deg,right_motor_angle_deg,within_valid_map')
 # Actual compiled CAD geometry, including MuJoCo mesh frame transforms.
 def triangles(gname,pose):
  evaluate(pose);gid=m.geom(gname).id;mid=m.geom_dataid[gid]
  va=m.mesh_vertadr[mid];nv=m.mesh_vertnum[mid];fa=m.mesh_faceadr[mid];nf=m.mesh_facenum[mid]
  verts=m.mesh_vert[va:va+nv]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
  return verts[m.mesh_face[fa:fa+nf]]
 def seat_origin(pose):
  evaluate(pose);return d.xpos[m.body('seat').id].copy()
 pts={k:seat_origin(v) for k,v in [('finish',finish),('feasible',catch),('source',np.r_[requested_min,finish[1:]])]}
 evaluate(finish)
 ankle=np.mean([d.xanchor[m.joint(s+'_ankle_pitch_joint').id] for s in ('left','right')],axis=0)
 meshes={n:triangles(n+'_visual',finish) for n in ['shell_white','deck_blue','cockpit','rails','rigger']}
 seatmeshes={k:triangles('seat_visual',v) for k,v in [('finish',finish),('feasible',catch),('source',np.r_[requested_min,finish[1:]])]}
 colors={'finish':'#198655','feasible':'#d18b22','source':'#c83d45'}
 plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
 fig=plt.figure(figsize=(13,10));gs=fig.add_gridspec(3,2,height_ratios=[1.5,1,1]);ax=fig.add_subplot(gs[0,:],projection='3d')
 for n,tris in meshes.items():
  tr=tris[(tris[:,:,0].max(axis=1)>-.95)&(tris[:,:,0].min(axis=1)<.8)]
  ax.add_collection3d(Poly3DCollection(tr,facecolors={'shell_white':'#c5ced4','deck_blue':'#227bb2','cockpit':'#9ba5ad','rails':'#303a42','rigger':'#aab3bb'}[n],edgecolors='none',alpha=.12 if n in ['shell_white','cockpit','deck_blue'] else .5))
 for k,tr in seatmeshes.items():ax.add_collection3d(Poly3DCollection(tr,facecolors=colors[k],edgecolors='none',alpha=.75))
 start,end=pts['source'],pts['finish'];ax.plot([start[0],end[0]],[.3,.3],[.15,.15],color='#555555',lw=2)
 ax.scatter([ankle[0]],[ankle[1]],[ankle[2]],color='#1d589b',s=35)
 ax.set(xlim=(-.9,.8),ylim=(-.9,.9),zlim=(-.2,.35));ax.set_box_aspect((1.7,1.8,.55),zoom=1.9);ax.view_init(35,-65);ax.set_axis_off()
 ax.set_title('实际船体 CAD：伸腿端172°标定后的滑座位置（船体半透明）',fontsize=14)
 side=fig.add_subplot(gs[1,:])
 for n in ['shell_white','cockpit','rails']:
  side.add_collection(PolyCollection(meshes[n][:,:,[0,2]],facecolors='#a5adb5' if n!='rails' else '#555555',edgecolors='none',alpha=.1 if n!='rails' else .5))
 labels={'finish':'伸腿端：172°','feasible':'当前连续可行区间的收腿端','source':'WiRMS 原行程的收腿端'+('（下肢约束通过）' if nvalid==len(rows) else '（未通过约束）')}
 for k,tr in seatmeshes.items():
  side.add_collection(PolyCollection(tr[:,:,[0,2]],facecolors=colors[k],edgecolors='none',alpha=.7))
  side.axvline(pts[k][0],color=colors[k],ls='--',lw=.8)
  side.plot([],[],color=colors[k],lw=5,label=labels[k])
 for y,k,col in [(.19,'source','#c83d45'),(.13,'feasible','#b67514')]:
  side.annotate('',xy=(pts['finish'][0],y),xytext=(pts[k][0],y),arrowprops={'arrowstyle':'<->','color':col})
  side.text((pts['finish'][0]+pts[k][0])/2,y+.012,f'{(pts["finish"][0]-pts[k][0])*1000:.0f} mm',ha='center',color=col)
 side.scatter([ankle[0]],[ankle[2]],color='#1d589b',s=20);side.text(ankle[0]-.06,ankle[2]+.04,'固定踝轴中心',ha='right',fontsize=8)
 side.set(xlim=(-.85,.7),ylim=(-.12,.25),xlabel='船体纵向 x / m',ylabel='船体高度 z / m');side.legend(fontsize=8,loc='lower right');side.grid(alpha=.15)
 plot=fig.add_subplot(gs[2,0])
 plot.plot(accepted[:,0],accepted[:,2],color='#198655',label='左膝几何夹角')
 plot.plot(accepted[:,0],accepted[:,4],'--',color='#315d9e',label='左膝电机关节角')
 plot.axvspan(requested_min,feasible_min,color='#c83d45',alpha=.12,label='未验证可行：不外推')
 plot.scatter([finish[0]],[172],color='#198655');plot.set(xlabel='滑座关节坐标 q / m',ylabel='角度 / °');plot.legend(fontsize=8);plot.grid(alpha=.15)
 time=fig.add_subplot(gs[2,1]);time.plot(raw[:,0],angles_ref[:,0],color='#198655',label='WiRMS位置 → 膝几何夹角')
 time.plot(raw[:,0],angles_ref[:,2],'--',color='#315d9e',label='对应膝电机关节角');time.set(xlabel='WiRMS 原始周期相位',ylabel='角度 / °');time.legend(fontsize=8);time.grid(alpha=.15)
 fig.text(.06,.018,'绿色：伸腿端；橙色：本次下肢 IK 连续通过区间边界；红色：保留578 mm源行程后的座椅位置。\n172°是矢状面髋—膝—踝夹角，不是电机关节指令。仅检查下肢连接与关节限位，尚未验证碰撞、双手—桨连接及动力学。',fontsize=9)
 fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'seat_travel_on_boat.png',dpi=180);plt.close(fig)
 report={'target_included_angle_deg':172,'target_flexion_deg':8,
  'definition':'sagittal hip_pitch-anchor / knee-anchor / ankle_pitch-anchor included angle; proxy for human landmarks',
  'scene_sha256':hashlib.sha256((ROOT/'scene.xml').read_bytes()).hexdigest(),
  'finish_seat_joint_m':float(finish[0]),'finish_motor_angles_deg':np.rad2deg(finish[kneeidx]).tolist(),
  'finish_pelvis_hinge_deg':float(np.rad2deg(finish[1])),
  'finish_foot_position_error_m':finish_errors[0],'finish_foot_rotation_error_deg':finish_errors[1],
  'source_seat_span_m':span,'source_seat_joint_range_m':[requested_min,float(finish[0])],
  'continuous_valid_seat_joint_range_m':[feasible_min,float(finish[0])],
  'continuous_valid_span_m':float(finish[0]-feasible_min),
  'boundary_sampling_resolution_m':float(span/300),'valid_catch_included_angle_deg':accepted[-1,2:4].tolist(),
  'seat_body_origin_boat_coordinates_m':{k:v.tolist() for k,v in pts.items()},
  'mean_fixed_ankle_center_m':ankle.tolist(),
  'source_curve_valid_fraction':float(np.isfinite(angles_ref[:,0]).mean()),
  'mapping_rule':'fixed feet, calibrated finish pelvis hinge preferred; other solutions may exist',
  'conversion_rule':'q=finish_q+source_seat-source_seat_max; PCHIP map; out-of-domain returns NaN',
  'validation':'foot position <0.1 mm and orientation <0.1 deg; joint bounds retained; seat limit relaxed only in solver',
  'training_ready':False}
 (OUT/'calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
