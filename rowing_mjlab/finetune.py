"""Fine tune the existing rowing configuration with torchrun gradient synchronization."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist


def synchronize_normalizers(models, world):
    """Use the same global observation batch on every rank, including buffers."""
    if world == 1:
        return
    for model in models:
        for name in ('obs_normalizer', 'priv_normalizer'):
            normalizer = getattr(model, name, None)
            if normalizer is None or not hasattr(normalizer, 'update'):
                continue
            original = normalizer.update

            @torch.no_grad()
            def update(x, original=original):
                x = x.contiguous()
                batches = [torch.empty_like(x) for _ in range(world)]
                dist.all_gather(batches, x)
                original(torch.cat(batches, dim=0))

            normalizer.update = update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force-profile',help='CFD profile JSON; use prior for the original force law')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=1024, help='Environments per GPU')
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1:
        parser.error('Environment and iteration counts must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Missing checkpoint: {args.checkpoint}')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = f'cuda:{local_rank}' if world > 1 else args.device
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .runner import RowingRunner
    from mjlab.utils.torch import configure_torch_backends
    from .config import env_cfg, rl_cfg
    configure_torch_backends()
    cfg = env_cfg(args.num_envs)
    cfg.actions['h2'].force_profile = args.force_profile
    cfg.seed = 42 + rank
    settings = asdict(rl_cfg(args.num_envs))
    settings['seed'] = 42 + rank
    # Conservative continuation from model_17700: keep the policy close to the
    # checkpoint while still allowing small task-specific corrections.
    settings['algorithm']['learning_rate'] = 1e-5
    settings['algorithm']['schedule'] = 'fixed'
    settings['algorithm']['entropy_coef'] = 5e-4
    settings['torch_compile_mode'] = 'default' if args.compile else None
    settings['check_for_nan'] = True
    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        runner = RowingRunner(RslRlVecEnvWrapper(env, clip_actions=1.0),
                                     settings, str(args.log_dir), device)
        if world > 1:
            assert dist.is_initialized() and dist.get_world_size() == world
        runner.load(str(args.checkpoint),
                    load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False),
                    strict=True, map_location=device)
        models = (runner.alg._raw_actor, runner.alg._raw_critic)
        base = torch.load(args.checkpoint, map_location=device, weights_only=True)
        for label, model in zip(('actor', 'critic'), models):
            for name, value in model.state_dict().items():
                assert torch.equal(value, base[label + '_state_dict'][name]), (label, name)
        initial = torch.cat([p.detach().flatten() for p in models[0].parameters()]).clone()
        synchronize_normalizers(models, world)
        if rank == 0:
            metadata = dict(checkpoint=str(args.checkpoint),
                            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                            parent_iteration=base['iter'], optimizer_reset=True,
                            world_size=world, environments_per_rank=args.num_envs,
                            total_environments=args.num_envs * world,
                            additional_iterations=args.iterations, settings=settings,
                            force_profile=env.rowing.hydro.force_profile,
                            source_directory=str(Path(__file__).resolve().parent.parent),
                            normalization='global_batch_all_gather')
            (args.log_dir / 'finetune_settings.json').write_text(json.dumps(metadata, indent=2))
        del base
        print(f'FINETUNE rank={rank}/{world} device={device} envs={args.num_envs}', flush=True)
        runner.learn(args.iterations, init_at_random_ep_len=False)
        current = torch.cat([p.detach().flatten() for p in models[0].parameters()])
        changed = not torch.equal(initial, current)
        valid = torch.tensor(int(changed), device=device)
        max_difference = torch.zeros((), device=device)
        for model in models:
            for value in model.state_dict().values():
                valid *= torch.isfinite(value).all().to(valid.dtype)
                if world > 1:
                    reference = value.detach().clone()
                    dist.broadcast(reference, src=0)
                    difference = (value.to(torch.float64) - reference.to(torch.float64)).abs().max()
                    max_difference = torch.maximum(max_difference, difference)
                    valid *= torch.allclose(value, reference, atol=1e-6, rtol=1e-6)
        if world > 1:
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_difference, op=dist.ReduceOp.MAX)
        assert valid.item(), 'Nonfinite, unchanged, or inconsistent policy/critic state'
        health = dict(rank=rank, world_size=world, actor_updated=changed,
                      finite=True, max_state_difference=max_difference.item(),
                      synchronized=world > 1, completed_iterations=args.iterations)
        (args.log_dir / f'health_rank{rank}.json').write_text(json.dumps(health, indent=2))
        print('FINETUNE_HEALTH ' + json.dumps(health), flush=True)
    finally:
        env.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
