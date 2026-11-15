"""Read-only substep recorder around the pinned repository's play.py path."""
from pathlib import Path
import os, sys, types, json, hashlib, time
HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('ROWING_SOURCE', str(HERE/'source_snapshot'))).resolve()
OUT = HERE/'replay'
sys.path.insert(0, str(SOURCE))
os.environ['MUJOCO_GL'] = 'disable'
sys.modules.setdefault('tensorboard.compat.notf', types.ModuleType('tensorboard.compat.notf'))
import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from rowing_mjlab.config import env_cfg
from rowing_mjlab.play import load_actor

def main():
    OUT.mkdir(exist_ok=True)
    checkpoint = Path(os.environ['ROWING_POLICY']).resolve()
    configure_torch_backends()
    torch.set_num_threads(4)
    cfg = env_cfg(1, play=True)
    cfg.seed = 42
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device='cpu')
    start = time.perf_counter()
    records, control = [], []
    def arr(x): return x.detach().cpu().numpy().copy()[0]
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs, _ = wrapped.reset()
        actor = load_actor(checkpoint, obs, wrapped.num_actions, 'cpu')
        model = env.sim.mj_model
        mujoco.mj_saveModel(model, str(OUT/'model.mjb'))
        np.savez_compressed(OUT/'initial_state.npz', qpos=arr(env.sim.data.qpos), qvel=arr(env.sim.data.qvel))
        original_step = env.sim.step
        def capture_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            records.append((len(records)*env.physics_dt+env.physics_dt,
                            arr(env.sim.data.qpos), arr(env.sim.data.qvel)))
            return result
        env.sim.step = capture_step
        done = False
        with torch.inference_mode():
            for step in range(round(11.6/env.step_dt)):
                action = actor(obs, stochastic_output=False)
                obs, reward, terminal, _ = wrapped.step(action)
                if not torch.isfinite(action).all() or not torch.isfinite(env.sim.data.qpos).all():
                    raise RuntimeError('Non-finite policy state')
                control.append((step*env.step_dt+env.step_dt, arr(action)))
                done = bool(terminal[0])
                if (step+1)%25 == 0 or done:
                    print(f't={(step+1)*env.step_dt:.3f}s, samples={len(records)}, wall={time.perf_counter()-start:.1f}s, done={done}', flush=True)
                if done: break
        np.savez_compressed(OUT/'rollout.npz', time=np.array([r[0] for r in records]),
                            qpos=np.array([r[1] for r in records]), qvel=np.array([r[2] for r in records]))
        np.savez_compressed(OUT/'actions.npz', time=np.array([r[0] for r in control]), action=np.array([r[1] for r in control]))
        meta = dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source_commit='82e68ab63d02336b43daeb59cd407e2671c132e0',
                    device='cpu', seed=42, step_dt=env.physics_dt, control_dt=env.step_dt,
                    simulated_seconds=float(records[-1][0]), terminated=done,
                    samples=len(records), water_level_m=float(env.rowing.water_level[0]),
                    force_profile=env.rowing.hydro.force_profile,
                    settings_available=(checkpoint.parent/'settings.json').exists(),
                    note='Pinned new repository; deterministic play.py wrapper; no auto-reset; read-only qpos/qvel capture after each 2ms physical step. Missing training settings means full training provenance is not certified.',
                    wall_seconds=time.perf_counter()-start)
        (OUT/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
        print(json.dumps(meta, indent=2), flush=True)
        if done or abs(records[-1][0]-11.6)>1e-8: raise RuntimeError('Replay did not complete 11.6 seconds')
    finally:
        env.close()

if __name__ == '__main__': main()
