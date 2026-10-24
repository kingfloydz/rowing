"""RSL-RL PPO with lossless history reconstruction instead of repeated windows."""
import copy
import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_callable
from .networks import HISTORY_LENGTH, SEQUENCE_LENGTH, TemporalModel


class HistoryStorage(RolloutStorage):
    def __init__(self,num_envs,steps,obs,actions_shape,device):
        compact = TensorDict({'actor':obs['actor'][:,-1],
                              'privileged':obs['privileged']},batch_size=[num_envs])
        super().__init__('rl',num_envs,steps,compact,actions_shape,device)
        self.timeline = torch.zeros(HISTORY_LENGTH+steps,num_envs,obs['actor'].shape[-1],device=device)
        self.start_indices = torch.zeros(steps,num_envs,dtype=torch.long,device=device)
        self.offsets = torch.arange(SEQUENCE_LENGTH,device=device)

    def add_transition(self,transition):
        t = self.step
        original = transition.observations
        if t == 0:
            self.timeline[:HISTORY_LENGTH].copy_(original['actor'][:,:-1].transpose(0,1))
            self.start_indices[0].zero_()
        else:
            self.start_indices[t] = torch.where(self.dones[t-1,:,0].bool(),
                                                HISTORY_LENGTH+t,self.start_indices[t-1])
        self.timeline[HISTORY_LENGTH+t].copy_(original['actor'][:,-1])
        transition.observations = TensorDict({'actor':original['actor'][:,-1],
            'privileged':original['privileged']},batch_size=[self.num_envs])
        try:
            super().add_transition(transition)
        finally:
            transition.observations = original

    def reconstruct(self,flat_indices):
        t = flat_indices//self.num_envs
        env = flat_indices%self.num_envs
        indices = t[:,None]+self.offsets
        indices = torch.maximum(indices,self.start_indices[t,env,None])
        history = self.timeline[indices,env[:,None]]
        return TensorDict({'actor':history,'privileged':self.observations['privileged'][t,env]},
                          batch_size=[len(flat_indices)])

    def mini_batch_generator(self,num_mini_batches,num_epochs=8):
        count = self.num_envs*self.num_transitions_per_env
        if num_mini_batches > count:
            raise ValueError('More minibatches than rollout samples')
        for _ in range(num_epochs):
            # tensor_split also includes the remainder; no dropped environments.
            for ids in torch.randperm(count,device=self.device).tensor_split(num_mini_batches):
                yield RolloutStorage.Batch(
                    observations=self.reconstruct(ids),actions=self.actions.flatten(0,1)[ids],
                    values=self.values.flatten(0,1)[ids],advantages=self.advantages.flatten(0,1)[ids],
                    returns=self.returns.flatten(0,1)[ids],
                    old_actions_log_prob=self.actions_log_prob.flatten(0,1)[ids],
                    old_distribution_params=tuple(p.flatten(0,1)[ids] for p in self.distribution_params),
                )


class HistoryPPO(PPO):
    @staticmethod
    def construct_algorithm(obs,env,cfg,device):
        # Deliberately scoped to this task: no RND, symmetry or recurrent state.
        cfg['algorithm'].setdefault('rnd_cfg',None)
        cfg['algorithm'].setdefault('symmetry_cfg',None)
        algorithm = copy.deepcopy(cfg['algorithm'])
        algorithm.pop('class_name')
        for key in ('rnd_cfg','symmetry_cfg','share_cnn_encoders'):
            if algorithm.pop(key,None):
                raise ValueError(f'{key} is not supported by compact history storage')
        groups = {'actor':['actor'],'critic':['actor','privileged']}
        cfg['obs_groups'] = groups
        models = []
        for name,out_dim in (('actor',env.num_actions),('critic',1)):
            settings = copy.deepcopy(cfg[name])
            cls = resolve_callable(settings.pop('class_name'))
            if cls is not TemporalModel:
                raise ValueError('HistoryPPO expects the rowing temporal model')
            models.append(cls(obs,groups,name,out_dim,**settings).to(device))
        storage = HistoryStorage(env.num_envs,cfg['num_steps_per_env'],obs,[env.num_actions],device)
        alg = HistoryPPO(*models,storage,device=device,multi_gpu_cfg=cfg.get('multi_gpu'),**algorithm)
        alg.compile(cfg.get('torch_compile_mode'))
        return alg
