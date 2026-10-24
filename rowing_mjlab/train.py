"""python -m rowing_mjlab.train --num-envs 2048 --device cuda:0"""
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import time
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=2048)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--iterations',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--log-dir',type=Path)
    parser.add_argument('--compile',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--benchmark',action='store_true',help='Full Actor+environment throughput, no training')
    parser.add_argument('--benchmark-steps',type=int,default=1000)
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu --num-envs 1 for correctness smoke tests only.')
    if args.num_envs < 1:
        parser.error('--num-envs must be positive')
    if args.benchmark_steps < 1 or args.iterations < 1:
        parser.error('Iteration and benchmark step counts must be positive')
    torch.set_num_threads(4)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.utils.torch import configure_torch_backends
    from .config import env_cfg,rl_cfg
    configure_torch_backends()
    cfg = env_cfg(args.num_envs)
    cfg.seed = args.seed
    agent = rl_cfg(args.num_envs)
    agent.seed = args.seed
    settings = asdict(agent)
    settings['torch_compile_mode'] = 'default' if args.compile and args.device.startswith('cuda') else None
    # check_nan performs host synchronizations every rollout step. Simulator
    # numerical-state termination stays active; enable checks in CPU smoke tests.
    settings['check_for_nan'] = not args.device.startswith('cuda')
    log_dir = args.log_dir or Path('logs/rowing')/datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True,exist_ok=True)
    env = ManagerBasedRlEnv(cfg,device=args.device)
    wrapped = RslRlVecEnvWrapper(env,clip_actions=1.0)
    try:
        runner = MjlabOnPolicyRunner(wrapped,settings,str(log_dir),args.device)
        runner.add_git_repo_to_log(__file__)
        if args.resume:
            runner.load(str(args.resume))
        metadata = {'num_envs':args.num_envs,'device':args.device,'seed':args.seed,
                    'physics_dt':env.physics_dt,'control_dt':env.step_dt,
                    'history_frames':121,'latent_dim':24,'privileged_dim':27,
                    'hydro_faces':env.rowing.hydro.face_count,'water_level_m':float(env.rowing.hydro.water.numpy()[0,0]),
                    'settings':asdict(agent)}
        (log_dir/'settings.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
        if args.benchmark:
            obs = wrapped.get_observations()
            with torch.inference_mode():
                for _ in range(20):
                    obs,*_ = wrapped.step(runner.alg.actor(obs,stochastic_output=False))
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(args.benchmark_steps):
                    obs,*_ = wrapped.step(runner.alg.actor(obs,stochastic_output=False))
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
            report = dict(metadata,benchmark_seconds=elapsed,
                          env_steps_per_second=args.num_envs*args.benchmark_steps/elapsed,
                          peak_torch_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                          note='Torch peak excludes Warp allocations; measure total GPU memory externally.')
            (log_dir/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf8')
            print(json.dumps({k:report[k] for k in ('env_steps_per_second','benchmark_seconds','peak_torch_bytes')},indent=2))
        else:
            # Random episode ages would skip the mandatory reset/half-speed cycle.
            runner.learn(args.iterations,init_at_random_ep_len=False)
    finally:
        env.close()


if __name__ == '__main__':
    main()
