"""Viser playback: python -m rowing_mjlab.play --checkpoint-file model_100.pt."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path


def load_actor(checkpoint, obs, num_actions, device):
    """Load only Actor and its normalizer; no PPO storage/optimizer allocation."""
    import torch
    from .config import rl_cfg
    from .networks import TemporalModel, HISTORY_LENGTH
    checkpoint = Path(checkpoint)
    settings = asdict(rl_cfg(1).actor)
    metadata = checkpoint.parent/'settings.json'
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding='utf8'))
        if saved.get('history_frames', HISTORY_LENGTH) != HISTORY_LENGTH:
            raise ValueError(f'Checkpoint history is {saved["history_frames"]} frames; '
                             f'this playback requires {HISTORY_LENGTH}. Use matching training code.')
        settings.update(saved.get('settings',{}).get('actor',{}))
    else:
        print('[WARN] No settings.json next to checkpoint; using current 125-frame task configuration.')
    settings.pop('class_name',None)
    for key in ('cnn_cfg','rnn_type','rnn_hidden_dim','rnn_num_layers'):
        settings.pop(key,None)
    actor = TemporalModel(obs,{'actor':['actor']},'actor',num_actions,**settings).to(device)
    state = torch.load(checkpoint,map_location=device,weights_only=True)
    actor.load_state_dict(state['actor_state_dict'],strict=True)
    actor.eval()
    actor.requires_grad_(False)
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-file',type=Path,required=True)
    parser.add_argument('--device',default=None)
    parser.add_argument('--num-envs',type=int,default=1)
    parser.add_argument('--host',default='0.0.0.0')
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--smoke-steps',type=int,default=0,
                        help='Run bounded headless inference instead of starting the viewer')
    args = parser.parse_args()
    if not args.checkpoint_file.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint_file}')
    if args.num_envs < 1 or args.smoke_steps < 0 or not 1 <= args.port <= 65535:
        parser.error('Invalid environment count, smoke step count or port')
    # Viser renders in the browser; do this BEFORE importing mjlab/MuJoCo.
    os.environ['MUJOCO_GL'] = 'disable'
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from .config import env_cfg
    configure_torch_backends()
    torch.set_num_threads(4)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            parser.error('CUDA unavailable; use --device cpu')
        torch.cuda.set_device(device)
    cfg = env_cfg(args.num_envs,play=True)
    cfg.seed = args.seed
    cfg.viewer.body_name = 'rowing/boat'
    cfg.viewer.distance = 8.0
    env = ManagerBasedRlEnv(cfg,device=device)
    server = None
    try:
        wrapped = RslRlVecEnvWrapper(env,clip_actions=1.0)
        obs,_ = wrapped.reset()
        actor = load_actor(args.checkpoint_file,obs,wrapped.num_actions,device)

        @torch.inference_mode()
        def policy(observations):
            return actor(observations,stochastic_output=False)

        if args.smoke_steps:
            for _ in range(args.smoke_steps):
                action = policy(obs)
                obs,reward,_,_ = wrapped.step(action)
                if not torch.isfinite(action).all() or not torch.isfinite(reward).all():
                    raise RuntimeError('Non-finite playback output')
            print(f'Playback smoke passed: {args.smoke_steps} steps, {args.num_envs} environments.')
            return
        import viser
        from mjlab.viewer import ViserPlayViewer
        server = viser.ViserServer(host=args.host,port=args.port,label='H2 rowing policy')
        print(f'Viser port {args.port}; open the forwarded port in your browser. Ctrl+C to stop.')
        ViserPlayViewer(wrapped,policy,viser_server=server).run()
    finally:
        if server is not None:
            server.stop()
        env.close()


if __name__ == '__main__':
    main()
