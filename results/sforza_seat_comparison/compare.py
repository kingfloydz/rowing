"""Figure digitization and diagnostic lower-body IK; does not modify training assets."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
im=np.array(Image.open(OUT/'figure4_native.png').convert('L'))
# Hand-selected guide points on the solid mean, refined using dark pixel runs.
guide=np.array([[110,575],[150,535],[198,484],[247,400],[285,318],[317,235],
 [350,154],[390,113],[430,89],[470,89],[510,91],[550,98],[590,113],
 [617,154],[643,195],[670,235],[696,277],[722,318],[753,360],
 [783,400],[818,443],[852,484],[886,527],[923,569],[945,592]],float)
pred=PchipInterpolator(guide[:,0],guide[:,1])
xs=np.arange(110,946); ys=[]
grid=np.array([69,154,235,318,400,484,569,649])
for x in xs:
 y0=float(pred(x)); rr=np.arange(max(0,int(y0-14)),min(im.shape[0],int(y0+15)))
 rr=rr[(im[rr,x]<125)&(np.min(abs(rr[:,None]-grid),axis=1)>4)]
 groups=np.split(rr,np.where(np.diff(rr)>1)[0]+1) if len(rr) else []
 groups=[g for g in groups if len(g)>=3]
 ys.append(float(np.mean(min(groups,key=lambda g:abs(np.mean(g)-y0)-2*len(g)))) if groups else np.nan)
ys=np.array(ys);good=np.isfinite(ys)
ys=np.interp(xs,xs[good],ys[good]);ys=gaussian_filter1d(ys,2)
# Image has a nonuniform printed x scale; calibrate using all visible tick centers.
phase=np.interp(xs,[95,292,470,646,814,953],[0,.2,.4,.6,.8,1])
angle=np.interp(ys,grid,[180,160,140,120,100,80,60,40])
flex=180-angle
np.savetxt(OUT/'sforza_fig4_digitized.csv',np.c_[phase,angle,flex,xs,ys],delimiter=',',
 header='source_phase,knee_included_angle_deg,knee_flexion_deg,pixel_x,pixel_y',comments='')
fig,ax=plt.subplots(figsize=(10,7));ax.imshow(im,cmap='gray',vmin=0,vmax=255)
ax.plot(xs,ys,color='red',lw=1,label='Extracted solid mean');ax.legend();ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'digitization_check.png',dpi=150);plt.close(fig)

m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));d=mujoco.MjData(m)
names=['seat_slide','seat_pelvis_hinge']+[s+'_'+j+'_joint' for s in ['left','right']
 for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_roll','ankle_pitch']]
ids=[m.joint(n).id for n in names];qa=m.jnt_qposadr[ids]
lo,hi=m.jnt_range[ids].T.copy();lo[0]=-.7;hi[0]=.7
footids=[m.equality(s+'_foot_fixed').id for s in ['left','right']]
base=m.qpos0[qa].copy()
def evaluate(x):
 d.qpos[:]=m.qpos0;d.qpos[qa]=x;mujoco.mj_fwdPosition(m,d)
 mask=(d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&np.isin(d.efc_id,footids)
 bends=[]
 for s in ['left','right']:
  pts=[d.xanchor[m.joint(s+'_'+j+'_joint').id] for j in ['hip_pitch','knee','ankle_pitch']]
  # Boat is unrotated at qpos0. Sforza reports sagittal-plane landmark angles.
  h,k,a=[p[[0,2]] for p in pts];u=k-h;v=a-k
  bends.append(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
 return d.efc_pos[mask],np.array(bends)

# Single-valued diagnostic map requires a pelvis selection rule: closest to saved
# hinge angle, with a tiny continuation tie-breaker. No seat/reset-position anchor.
targets=np.linspace(float(flex.min()),float(flex.max()),151)
x=base.copy(); rows=[]; poses=[]
for i,target in enumerate([] if '--plot-only' in sys.argv else targets):
 previous=x.copy()
 def residual(v):
  eq,b=evaluate(v)
  return np.r_[1000*eq,10*(b-np.deg2rad(target)),.003*(v[1]-base[1]),.00001*(v-previous)]
 seeds=[x]
 if i==0: seeds += [np.r_[.25,p,base[2:]] for p in np.deg2rad([-30,-10,10,30])]
 fits=[least_squares(residual,np.clip(s,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=350,
  ftol=1e-10,xtol=1e-10,gtol=1e-10) for s in seeds]
 fit=min(fits,key=lambda f:np.linalg.norm(f.fun))
 eq0,b0=evaluate(fit.x)
 if max(abs(np.rad2deg(b0)-target))>.1:
  for p in np.deg2rad([-30,-10,10,30]):
   seed=fit.x.copy();seed[1]=p
   trial=least_squares(residual,np.clip(seed,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=350,
    ftol=1e-10,xtol=1e-10,gtol=1e-10)
   if np.linalg.norm(trial.fun)<np.linalg.norm(fit.fun):fit=trial
 x=fit.x
 eq,b=evaluate(x);err=float(max(abs(eq)));angleerr=float(max(abs(np.rad2deg(b)-target)))
 rows.append([target,x[0],np.rad2deg(x[1]),*np.rad2deg(b),err,angleerr]);poses.append(x.copy())
 if i%30==0: print('mapping',i,'angle',round(target,2),'seat',round(x[0],4),'error',err,angleerr,flush=True)
rows=np.loadtxt(OUT/'h2_knee_seat_map.csv',delimiter=',',skiprows=1) if '--plot-only' in sys.argv else np.array(rows)
targets=rows[:,0];valid=(rows[:,-2]<1e-5)&(rows[:,-1]<.1)
np.savetxt(OUT/'h2_knee_seat_map.csv',rows,delimiter=',',comments='',
 header='target_flexion_deg,seat_joint_m,pelvis_hinge_deg,left_actual_flexion_deg,right_actual_flexion_deg,max_foot_equality_residual,max_knee_error_deg')
if '--plot-only' not in sys.argv: np.savez_compressed(OUT/'h2_mapping_poses.npz',joint_names=names,poses=poses,valid=valid)
mapped=np.interp(flex,targets,rows[:,1]);isvalid=np.interp(flex,targets,valid.astype(float))>.999
finish=int(np.argmin(flex));qfinish=float(mapped[finish]);rel=mapped-qfinish
sfphase=phase-phase[finish]
raw=np.loadtxt(ROOT/'data/reference9/wirms_stage4_raw.csv',delimiter=',',skiprows=1)
wfinish=int(np.argmax(raw[:,1]));wp=raw[:,0]-raw[wfinish,0];wy=raw[:,1]-raw[wfinish,1]
np.savetxt(OUT/'sforza_h2_seat.csv',np.c_[sfphase,phase,flex,mapped,rel,isvalid],delimiter=',',comments='',
 header='phase_relative_finish,source_phase,knee_flexion_deg,seat_joint_m,seat_relative_finish_m,ik_valid')
np.savetxt(OUT/'wirms_finish_aligned.csv',np.c_[wp,wy],delimiter=',',comments='',
 header='phase_relative_finish,seat_relative_finish_m')
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(2,1,figsize=(11,8),gridspec_kw={'height_ratios':[2,1]},sharex=True)
ax=axs[0];ax.plot(wp,wy*1000,label='WiRMS Stage 4：实验滑座位移',color='#1967a3',lw=2)
ax.plot(sfphase,np.where(isvalid,rel*1000,np.nan),label='Sforza Fig.4 → H2：满足下肢闭链约束',color='#cf5728',lw=2)
if np.any(~isvalid): ax.plot(sfphase,np.where(~isvalid,rel*1000,np.nan),'--',color='#cf5728',label='IK未满足约束（仅显示最小残差解）')
ax.axvline(0,color='gray',lw=.8);ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('滑座相对伸腿端的位置 / mm');ax.legend(fontsize=8,loc='lower center');ax.grid(alpha=.2)
ax.set_title('伸腿端对齐：Sforza 膝角经 H2 模型转换，与 WiRMS 对比',fontsize=14)
axs[1].plot(sfphase,flex,color='#cf5728',label='Figure 4 提取的膝屈曲角')
axs[1].plot(targets*0+np.nan,targets*0+np.nan)
axs[1].axvline(0,color='gray',lw=.8);axs[1].set_ylabel('膝屈曲角 / °');axs[1].set_xlabel('归一化周期相位 − 各自伸腿端相位（无时间拉伸）');axs[1].grid(alpha=.2);axs[1].legend(fontsize=9)
fig.text(.08,.015,'仅平移对齐，不缩放位移；不同实验，不代表同一划手。H2采用矢状面几何角，骨盆角按最接近原姿态选择。\n仅核验下肢连接与关节范围；未验证手—桨连接、碰撞与动力学。虚线不能用作可行参考轨迹。',fontsize=9)
fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'seat_comparison.png',dpi=180);plt.close(fig)
report={'scene_sha256':hashlib.sha256((ROOT/'scene.xml').read_bytes()).hexdigest(),
 'source':'Sforza 2012 Fig4 solid mean, image digitization; WiRMS participant3 Stage4 vector extraction',
 'angle_definition':'180 minus sagittal hip_pitch-anchor/knee-anchor/ankle_pitch-anchor included angle; robot proxy for human landmarks',
 'seat_limit_relaxed_in_solver_only_m':[-.7,.7],
 'pelvis_rule':'nearest original hinge angle with weak regularization; hinge and leg limits retained',
 'figure_flexion_range_deg':[float(flex.min()),float(flex.max())],
 'sforza_finish_source_phase':float(phase[finish]),'wirms_finish_source_phase':float(raw[wfinish,0]),
 'h2_finish_joint_m':qfinish,'wirms_span_m':float(np.ptp(raw[:,1])),
 'map_valid_count':int(valid.sum()),'map_total':len(valid),
 'valid_flexion_range_deg':[float(targets[valid].min()),float(targets[valid].max())] if valid.any() else None,
 'mapped_curve_valid_fraction':float(isvalid.mean()),
 'h2_valid_seat_span_m':float(np.ptp(mapped[isvalid])) if isvalid.any() else None,
 'h2_all_best_fit_span_m_NOT_validated':float(np.ptp(mapped)),
 'limitations':['Not a unique map without pelvis rule','No human landmark registration','No oar/hand collision or dynamics checks',
 'Figure4 differs from Table2; no endpoint replacement or cyclic endpoint forcing','Image trace has about 1-2 degree reading uncertainty; not sensor data']}
(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
