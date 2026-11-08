"""Record deterministic checkpoint playback without hiding episode endings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# TensorBoard's optional TensorFlow backend is incompatible with local NumPy;
# inference does not need TensorFlow. Use TensorBoard's supported stub backend.
sys.modules.setdefault('tensorboard.compat.notf', types.ModuleType('tensorboard.compat.notf'))
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from rowing_mjlab.config import env_cfg
from rowing_mjlab.play import load_actor
from rowing_mjlab import mdp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=40)
    parser.add_argument('--noise',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--force-profile',help='CFD profile JSON; use prior for original hydrodynamics')
    parser.add_argument('--reference-controller',action='store_true',help='Track the coordinated whole-body reference, without a trained actor')
    args=parser.parse_args()
    if not args.reference_controller and args.checkpoint is None:
        parser.error('--checkpoint is required unless --reference-controller is selected')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    cfg=env_cfg(1,play=not args.noise);cfg.seed=args.seed;cfg.auto_reset=False
    cfg.actions['h2'].force_profile=args.force_profile
    env=ManagerBasedRlEnv(cfg,device='cpu')
    records=[];started=time.perf_counter()
    def arr(x):return x.detach().cpu().numpy().copy()[0]
    try:
        obs,_=env.reset()
        # Preserve the actual reset state; CFD must not infer it from qpos0.
        np.savez_compressed(args.output/'initial_state.npz',time=np.array(0.),
                            qpos=arr(env.sim.data.qpos),qvel=arr(env.sim.data.qvel))
        actor=None if args.reference_controller else load_actor(args.checkpoint,obs,len(env.rowing.policy_names),'cpu')
        m=env.sim.mj_model;s=env.rowing
        mujoco.mj_saveModel(m,str(args.output/'model.mjb'))
        reward_names=list(cfg.rewards)
        meta=dict(checkpoint=str(args.checkpoint) if actor is not None else None,
                  sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if actor is not None else None,
                  seed=args.seed,observation_noise=args.noise,step_dt=env.step_dt,reference_controller=args.reference_controller,
                  reward_names=reward_names,reward_weights={k:v.weight for k,v in cfg.rewards.items()},
                  water_level_m=float(s.water_level[0]),policy_joint_names=list(s.policy_names),
                  force_profile=s.hydro.force_profile,
                  imitation_q=s.imitation_q.tolist(),policy_q=s.policy_q.tolist(),policy_v=s.policy_v.tolist(),
                  phase_offset=s.ref.phase_offset,phase0=float(s.phase0),
                  settings_available=(args.checkpoint.parent/'settings.json').exists() if actor is not None else False,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  note=('Open-loop reference PD tracking without an actor or roll feedback. ' if args.reference_controller else 'Deterministic actor. ')
                  +'Current local environment; auto-reset disabled; records include terminal state.')
        with torch.inference_mode():
            for i in range(round(args.seconds/env.step_dt)):
                if args.reference_controller:
                    phase,_,_,_=s.reference()
                    target=s.ref.coordinated.joints(phase,s.policy_names)
                    zero=s.q0[s.policy_q];limits=s.limits[s.controlled_local]
                    action=(target-zero)/torch.where(target>=zero,limits[:,1]-zero,zero-limits[:,0]).clamp_min(1e-5)
                else:action=actor(obs,stochastic_output=False)
                obs,reward,terminated,truncated,extras=env.step(action.clamp(-1,1))
                phase,rate,blend,target=s.reference()
                actual_trunk,target_trunk=mdp.trunk_angles(env)
                rec=dict(time=(i+1)*env.step_dt,qpos=arr(env.sim.data.qpos),qvel=arr(env.sim.data.qvel),
                         action=arr(action),command=arr(s.commanded_targets),
                         torque=arr(env.sim.data.qfrc_actuator),target=arr(target),
                         phase=float(phase[0]),rate=float(rate[0]),blend=float(blend[0]),
                         forward=float(mdp.boat_forward_speed(env)[0]),velocity=arr(mdp.boat_world_velocity(env)),
                         lateral_sq=float(mdp.boat_lateral_velocity_cost(env)[0]),
                         omega=arr(env.sim.data.cvel[:,s.boat,:3]),roll=float(mdp.boat_roll(env)[0]),
                         trunk=float(actual_trunk[0]),trunk_target=float(target_trunk[0]),
                         collision=float(mdp.collision_cost(env)[0]),saturation=float(mdp.torque_saturation(env)[0]),
                         loads=arr(s.loads),blade_area=arr(s.blade_area.actual(env.sim.data,s.water_level)),
                         reward=float(reward[0]),reward_terms=np.array([float(v.func(env)[0])*v.weight for v in cfg.rewards.values()]),
                         terminated=bool(terminated[0]),truncated=bool(truncated[0]))
                records.append(rec)
                if (i+1)%100==0:
                    print(f'{(i+1)*env.step_dt:.1f}s simulated; speed={rec["forward"]:.3f} m/s; roll={np.rad2deg(rec["roll"]):.2f} deg; wall={time.perf_counter()-started:.0f}s',flush=True)
                if rec['terminated'] or rec['truncated']:break
        np.savez_compressed(args.output/'rollout.npz',**{k:np.array([r[k] for r in records]) for k in records[0]})
        meta.update(simulated_seconds=records[-1]['time'],wall_seconds=time.perf_counter()-started,
                    terminated=records[-1]['terminated'],truncated=records[-1]['truncated'])
        (args.output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print(json.dumps(meta,indent=2),flush=True)
    finally:env.close()


if __name__=='__main__':main()
