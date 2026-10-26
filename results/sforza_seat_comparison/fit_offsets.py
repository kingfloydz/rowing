"""Fit two constant offsets on fixed, feasible comparison samples; no extrapolation."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
sf=np.loadtxt(out/'sforza_h2_seat.csv',delimiter=',',skiprows=1)
wi=np.loadtxt(out/'wirms_finish_aligned.csv',delimiter=',',skiprows=1)
mp=np.loadtxt(out/'h2_knee_seat_map.csv',delimiter=',',skiprows=1)
ok=(mp[:,-2]<1e-5)&(mp[:,-1]<.1)
aa,qq=mp[ok,0],mp[ok,1]
mapping=PchipInterpolator(aa,qq,extrapolate=False)
# Equal phase spacing avoids digitized-image sampling density weighting.
t=np.linspace(max(sf[0,0],wi[0,0]),min(sf[-1,0],wi[-1,0]),1501)
theta=PchipInterpolator(sf[:,0],sf[:,2])(t)
target=PchipInterpolator(wi[:,0],wi[:,1])(t)
# Fixed interior domain allows meaningful offset freedom without changing the
# samples evaluated per candidate. End ranges are validation-only, not clipped.
fitmask=(theta>=20)&(theta<=90)
lower=float(aa[0]-theta[fitmask].min())
upper=float(aa[-1]-theta[fitmask].max())
qfinish=float(json.loads((out/'report.json').read_text())['h2_finish_joint_m'])
def evaluate(delta):
 pred=mapping(theta[fitmask]+delta)-qfinish
 shift=float(np.mean(target[fitmask]-pred))
 err=pred+shift-target[fitmask]
 return float(np.mean(err**2)),shift
grid=np.linspace(lower,upper,2001)
vals=np.array([evaluate(v)[0] for v in grid]);k=int(np.argmin(vals))
opt=minimize_scalar(lambda x:evaluate(x)[0],bounds=(grid[max(0,k-1)],grid[min(len(grid)-1,k+1)]),method='bounded')
delta=float(opt.x);mse,shift=evaluate(delta)
before=mapping(theta)-qfinish
after=mapping(theta+delta)-qfinish+shift
common=np.isfinite(before)&np.isfinite(after)
rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
report={
 'definition':'theta_new=theta_Sforza+delta_deg; fitted_position=q_H2(theta_new)-old_q_finish+shift_m',
 'time_alignment':'previous maximum-extension phase alignment retained; no phase shift or time warp',
 'delta_knee_flexion_deg':delta,'delta_included_angle_deg':-delta,'seat_vertical_shift_m':shift,
 'fit_domain_original_flexion_deg':[20,90],'offset_search_bounds_deg':[lower,upper],
 'fit_sample_count':int(fitmask.sum()),'comparison_sample_count':len(t),
 'fit_rmse_before_m':rmse(before[fitmask],target[fitmask]),'fit_rmse_after_m':float(np.sqrt(mse)),
 'validation_common_valid_count':int(common.sum()),
 'validation_common_rmse_before_m':rmse(before[common],target[common]),
 'validation_common_rmse_after_m':rmse(after[common],target[common]),
 'after_valid_fraction_in_overlap':float(np.isfinite(after).mean()),
 'mapped_angle_domain_deg':[float(aa[0]),float(aa[-1])],
 'full_trace_angle_span_deg':float(np.ptp(sf[:,2])),
 'at_search_boundary':bool(min(delta-lower,upper-delta)<.05),
 'limitations':['Optimum is conditional on fixed 20-90 degree fitting domain, not a full-cycle optimum',
 'Seat shift is comparison-datum offset, not an IK-feasible translation of robot poses',
 'No clipping or extrapolation of missing IK map; missing sections cannot be training targets',
 'Previous pelvis selection rule and human landmark proxy still apply']}
(out/'offset_fit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savetxt(out/'offset_fit_comparison.csv',np.c_[t,theta,target,before,after,fitmask],delimiter=',',comments='',
 header='phase_relative_finish,original_knee_flexion_deg,wirms_relative_seat_m,h2_before_m,h2_after_offset_m,is_fit_sample')
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(2,1,figsize=(11,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
ax=axs[0]
ax.plot(t,target*1000,label='WiRMS',color='#1967a3',lw=2)
ax.plot(t,before*1000,'--',label='调整前：Sforza → H2（可行部分）',color='#999999',lw=1.5)
ax.plot(t,after*1000,label='调整零点后：Sforza → H2（可行部分）',color='#d05b26',lw=2)
ax.set_ylabel('相对原伸腿端的位置 / mm');ax.legend(fontsize=9,loc='lower center');ax.grid(alpha=.2)
ax.set_title('膝角零点与滑座零点的两参数拟合（不改变时间轴和幅值）',fontsize=13)
axs[1].plot(t,(before-target)*1000,'--',color='#999999',label='调整前误差')
axs[1].plot(t,(after-target)*1000,color='#d05b26',label='调整后误差')
axs[1].fill_between(t,0,1,where=fitmask,transform=axs[1].get_xaxis_transform(),color='#8ac28a',alpha=.18,label='固定拟合区间')
axs[1].axhline(0,color='black',lw=.7);axs[1].set_ylabel('与 WiRMS 的差 / mm');axs[1].set_xlabel('相对伸腿端的归一化周期相位');axs[1].legend(fontsize=8);axs[1].grid(alpha=.2)
fig.text(.08,.018,f'膝屈曲角偏移 {delta:+.2f}°；滑座比较零点偏移 {shift*1000:+.1f} mm。拟合区间 RMSE：{report["fit_rmse_before_m"]*1000:.1f} → {report["fit_rmse_after_m"]*1000:.1f} mm。\n仅在固定原始膝屈曲角 20°–90° 区间拟合；缺失曲线表示超出已验证映射范围。位置偏移不是机器人姿态的可行平移。',fontsize=9)
fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(out/'seat_comparison_offsets.png',dpi=180);plt.close(fig)
print(json.dumps(report,indent=2))
