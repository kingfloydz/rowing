"""Reproducible identification from a prescribed trajectory and raw CFD ZIP."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from hydrodynamics.cfd_calibration import (read_archive, interpolate_trajectory,
    align_forces, oar_basis, sha256)


def fit(basis,target,t,mask,indices=(0,1),inertia_scale=1.):
    # Fixed friction and an explicit nuisance inertia prior; never fit offsets.
    fixed=basis[:,:,:,2]+inertia_scale*basis[:,:,:,3]
    if 3 in indices:
        fixed=basis[:,:,:,2]
    x=basis[mask][:,:,:,indices]
    y=(target-fixed)[mask]
    # Moments / 2 m = equivalent force; nonuniform output cadence is weighted.
    scale=np.array([1,1,1,.5,.5,.5])
    dt=np.gradient(t)[mask]; weight=np.sqrt(dt/np.median(dt))[:,None,None]
    def residual(c):
        r=(np.einsum('nsck,k->nsc',x,c)-y)*scale*weight
        return np.r_[r.ravel()/25., .2*(c-1.)]
    lower=[0.25 if k==0 else 0. for k in indices]
    upper=[3. if k==0 else 2. for k in indices]
    opt=least_squares(residual,np.ones(len(indices)),bounds=(lower,upper),loss='soft_l1',f_scale=1.)
    c=np.array([1.,1.,1.,inertia_scale]);c[list(indices)]=opt.x
    return c


def metrics(pred,target,t,mask):
    dt=np.gradient(t)[mask]
    def rmse(x):return np.sqrt(np.average(x*x,axis=0,weights=dt))
    error=pred[mask]-target[mask]
    values={}
    for side,name in enumerate(('port','starboard')):
        values[name]=dict(force_rmse_N=float(np.sqrt(np.average(np.sum(error[:,side,:3]**2,axis=1),weights=dt))),
                         moment_rmse_Nm=float(np.sqrt(np.average(np.sum(error[:,side,3:]**2,axis=1),weights=dt))),
                         component_rmse=rmse(error[:,side]).tolist())
    return values


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trajectory',type=Path,default=ROOT/'renders/policy_replay/final_policy_1100/policy_trajectory.csv')
    ap.add_argument('--forces',type=Path,default=ROOT/'force_data/rowing_20261108/rowing_forces_raw.zip')
    ap.add_argument('--output',type=Path,default=ROOT/'reports/cfd_calibration_20261108')
    ap.add_argument('--reuse-basis',action='store_true')
    ap.add_argument('--install-profile',action='store_true',help='Install the bounded experimental oar profile after validation')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    ale=args.trajectory.with_name('ale_motion.csv')
    t,segment,raw,manifest,configs=read_archive(args.forces)
    states=interpolate_trajectory(args.trajectory,t)
    target,alignment=align_forces(raw,states,ale,t)
    level=float(dict(line.split(maxsplit=1) for line in configs[0].splitlines() if line.strip() and not line.startswith('#'))['water_level_m'])
    provenance=dict(trajectory_sha256=sha256(args.trajectory),forces_sha256=sha256(args.forces),ale_sha256=sha256(ale),
                    basis_code_sha256=sha256(ROOT/'hydrodynamics/cfd_calibration.py'),
                    geometry_sha256={s:sha256(ROOT/'boat_meshes'/f'{s}_blade_3mm.stl') for s in ('port','starboard')},
                    inertia_sha256=sha256(ROOT/'data/blade_inertia.json'))
    cache=out/'basis.npz';cache_meta=out/'basis_provenance.json'
    if args.reuse_basis and cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())==provenance:
        b=np.load(cache);basis=b['basis'];diag={k:b[k] for k in ('wet_area_m2','speed_ms','acceleration_ms2')}
    else:
        basis,diag=oar_basis(states,level)
        np.savez_compressed(cache,basis=basis,t=t,**diag)
        cache_meta.write_text(json.dumps(provenance,indent=2))
    # One trajectory, time-block split. Later changed-scheme output is an audit
    # only, never silently pooled into a nominally homogeneous training set.
    seams=np.array([s['start_s'] for s in manifest['source_segments'][1:]])
    clean=np.all(abs(t[:,None]-seams[None,:])>.06,axis=1)
    masks=dict(train=(t>=.5)&(t<6)&clean,
               validation=(t>=6)&(t<7.44)&clean,
               changed_solver_audit=(t>=7.56)&(t<=10.4))
    y=target[:,1:]
    coef=fit(basis,y,t,masks['train'])
    baseline=basis.sum(axis=3);pred=np.einsum('nsck,k->nsc',basis,coef)
    sensitivity=[]
    for value in (0.,.5,1.,1.5,2.):
        c=fit(basis,y,t,masks['train'],inertia_scale=value)
        sensitivity.append(dict(inertia_scale=value,coefficients=c.tolist(),
            validation=metrics(np.einsum('nsck,k->nsc',basis,c),y,t,masks['validation'])))
    joint=fit(basis,y,t,masks['train'],indices=(0,1,3))
    result=dict(schema_version=1,status='provisional_raw_ib_fit',installed_profile=args.install_profile,
        coefficient_order=['blade_drag_scale','blade_lift_scale','fixed_skin_scale','inertia_scale'],
        coefficients=coef.tolist(),joint_fit_diagnostic=joint.tolist(),inertia_sensitivity=sensitivity,
        source=provenance,force_definition=manifest['force_definition'],
        alignment_max_m=float(alignment.max()),alignment_rms_m=float(np.sqrt(np.mean(alignment**2))),
        frames=len(t),time_s=[float(t[0]),float(t[-1])],water_level_m=level,
        scope='Shared left/right effective oar coefficients, fixed prescribed motion; no hull or hydrostatic identification.',
        metrics={k:dict(frames=int(m.sum()),baseline=metrics(baseline,y,t,m),candidate=metrics(pred,y,t,m)) for k,m in masks.items()},
        force_basis='World axes, torque about body origin. Dynamic oar proxy compared to uncorrected IBM reaction; no claim of total exterior hydrodynamic force.',
        state_range={k:dict(min=np.min(v[masks['train']],axis=0).tolist(),max=np.max(v[masks['train']],axis=0).tolist()) for k,v in diag.items()},
        limitations=['Internal fictitious-fluid momentum and external hydrostatic loads are not recovered by these aggregate files.',
                    'Oar CFD assembly includes shaft; reduced model basis includes blade only.',
                    'Last-stage IBM reaction is not an independently timestamped RK-averaged load.',
                    'Last source segment switches numerical algorithm; terminal CFL failure at 10.57025 s.',
                    'Time-block validation is not validation on an independently generated trajectory.'])
    validation=result['metrics']['validation']
    result['validation_improves_both_oars']=all(validation['candidate'][s]['force_rmse_N']<validation['baseline'][s]['force_rmse_N'] and validation['candidate'][s]['moment_rmse_Nm']<validation['baseline'][s]['moment_rmse_Nm'] for s in ('port','starboard'))
    result['split_definition_s']={'train':[.5,6.],'validation':[6.,7.44],'changed_solver_audit':[7.56,10.4],'restart_exclusion_radius':.06}
    profile=dict(schema_version=1,status='experimental_prescribed_motion_cfd',default_enabled=False,
        disabled_reason='Source audit: velocity-only last-stage IBM reaction omits phase-remap momentum and prescribed-body interior correction.',
        blade_cd_max=float(2*coef[0]),blade_cl_max=float(.9*coef[1]),blade_skin_cd=.01,
        left_right_shared=True,source=provenance,
        required_blade_inertia={k:json.loads((ROOT/'data/blade_inertia.json').read_text())[k] for k in
            ('full_normal_mass_kg_per_blade','wetting_depth_m','strips_per_blade','cells_per_strip')},
        identification='Effective dynamic blade law conditional on existing inertia; raw IBM proxy, not total exterior load.',
        limitations=result['limitations'],validation=result['metrics']['validation'],
        state_range=result['state_range'])
    (out/'force_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    if args.install_profile:
        if not result['validation_improves_both_oars']:
            raise ValueError('Held-out validation failed; refusing to install')
        (ROOT/'data/cfd_force_calibration.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    (out/'calibration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Human-readable aligned records, including the hull which is excluded from fit.
    import csv
    with (out/'aligned_forces.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time_s','source_segment','body']+[prefix+c for prefix in ('cfd_','baseline_','candidate_') for c in ('fx_N','fy_N','fz_N','mx_Nm','my_Nm','mz_Nm')])
        for i in range(len(t)):
            for s in range(2):writer.writerow([t[i],segment[i],('port_oar','starboard_oar')[s],*y[i,s],*baseline[i,s],*pred[i,s]])
    np.savez_compressed(out/'aligned_all_bodies.npz',time=t,segment=segment,wrench_world_origin=target,pose_error_m=alignment)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for s in range(2):
        for k in range(3):
            ax=axs[k,s];ax.plot(t,y[:,s,k],color='.5',lw=.7,label='CFD raw IBM reaction')
            ax.plot(t,baseline[:,s,k],lw=1,label='Original dynamic blade model')
            ax.plot(t,pred[:,s,k],lw=1,label='Fitted candidate')
            ax.axvline(6,color='k',ls=':',lw=.8);ax.axvline(7.5,color='r',ls=':',lw=.8)
            ax.set_ylabel('F'+'xyz'[k]+' (N)');ax.grid(alpha=.2)
        axs[0,s].set_title(('Port','Starboard')[s]);axs[2,s].set_xlabel('Time (s)')
    axs[0,0].legend(fontsize=8);fig.suptitle('Prescribed-motion audit: raw IBM reaction versus dynamic blade proxy')
    fig.tight_layout();fig.savefig(out/'calibrated_oar_forces.png',dpi=160);plt.close(fig)
    lines=['# CFD 固定轨迹校准（2026-11-08）','',
        '本次只辨识双桨共享的准定常阻力/升力系数。模型按当前浸水几何、相对速度计算，未引入时间或相位查表；船体、静水浮力、桨叶附加质量及辐射记忆参数不改。','',
        f'- CDmax: 2 → {profile["blade_cd_max"]:.8f}；CLmax: 0.9 → {profile["blade_cl_max"]:.8f}；摩擦系数仍为 0.01。',
        f'- 从原始 ZIP 读取并验证 {len(t)} 帧的 SHA256。ALE 反变换后位置最大残差 {alignment.max()*1000:.3f} mm。',
        '- CFD 是旋转 ALE 基下各组件原点的最后阶段 IBM 反力；两类向量均转至世界轴，基函数力矩也统一到组件原点。',
        '- 轨迹按 CFD 相同线性插值/四元数 nlerp 对齐；使用实际重建输出时刻，存在一个子步量级的评估时刻不确定性。',
        '- 训练：0.5–6 s；连续留出：6–7.44 s；续算点两侧 0.06 s 排除。7.56–10.4 s 仅作数值算法切换后的审计，不参与拟合或选参。',
        '- 采用时间间隔加权的稳健有界拟合，力矩除以固定 2 m 力臂与力共同拟合；阻力倍率范围 [0.25,3]，升力倍率 [0,2]。',
        '- 附加惯性按规定轨迹加速度计算，包括法向旋转与入水质量招募；惯性参数固定为原先验。','',
        '| 区间 | 部件 | 力向量 RMSE 原值 → 校准 (N) | 力矩向量 RMSE 原值 → 校准 (N·m) |',
        '|---|---|---:|---:|']
    for key,m in result['metrics'].items():
        for side in ('port','starboard'):
            a=m['baseline'][side];b=m['candidate'][side]
            lines.append(f'| {key} | {side} | {a["force_rmse_N"]:.2f} → {b["force_rmse_N"]:.2f} | {a["moment_rmse_Nm"]:.2f} → {b["moment_rmse_Nm"]:.2f} |')
    lines+=['','## 可辨识性与适用范围','',
        '这些误差比较的是动态桨叶代理模型与原始 IBM 反力，不能解释为对真实总水动力的误差。CFD 整个桨组件含桨杆，模型只显式计算桨叶；附着水、自由面历史与桨杆残差没有通过时序查表强行补偿。',
        '内部虚拟流体保留了与外部相同的初始相分布。聚合 CSV 缺少内部动量变化、重力及应力贡献，无法唯一恢复外部载荷。船体排水体积大，尤其不能用其 raw Fz 替换浮力或据此重标定附加质量。',
        '这条轨迹是外部规定运动，既不是新力模型下的自由航行，也不是独立策略验证；不从位置二阶差分反推机器人总水动力。',
        '惯性倍率从 0 改至 2 的敏感性结果（仅诊断，不写入运行参数）：','',
        '| 惯性倍率 | 阻力倍率 | 升力倍率 |','|---:|---:|---:|']
    for item in sensitivity:lines.append(f'| {item["inertia_scale"]} | {item["coefficients"][0]:.4f} | {item["coefficients"][1]:.4f} |')
    lines+=['','升力系数与惯性模型明显耦合，因此这是一组有条件的有效参数，不是独立辨识的桨叶极曲线。左右共用系数；左右受力差异由状态产生。',
        '后续应输出经过内部虚拟流体修正的外部力/力矩、明确 RK 受力评估时刻，并增加不同桨频、船速及 feather 的独立规定轨迹，验证后再校准船体和非定常项。','',
        '## 使用','',
        '重新拟合：`python tools/calibrate_cfd_forces.py --reuse-basis --install-profile`。',
        '源码审计后候选不再默认启用，训练、微调和回放默认使用原始力律。仅做实验时显式传入 `--force-profile data/cfd_force_calibration.json`；详见 `source_audit.md`。',
        '关闭桨叶惯性时自动回到原力律，显式指定有惯性依赖的校准配置会报错。参数只在构造时读取，GPU 回调不读取 CSV、不按相位播放载荷。','',
        '图中虚线 6 s 为留出起点，红线 7.5 s 为数值算法切换点。','',
        '![力对比](calibrated_oar_forces.png)','']
    (out/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('coefficients','joint_fit_diagnostic','alignment_max_m','validation_improves_both_oars','metrics')},indent=2))


if __name__=='__main__': main()
