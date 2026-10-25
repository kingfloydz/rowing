"""Causal residual TCN structure from kingfloydz/head_transport, with 125-frame history and 16-dimensional latent.

Actor consumes onboard history only. Critic has its own encoder/normalizers and
27 current privileged features; no world position is consumed by either model.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization

HISTORY_LENGTH = 125
SEQUENCE_LENGTH = HISTORY_LENGTH+1
LATENT_DIM = 16
TCN_CHANNELS = 48
PRIVILEGED_DIM = 27


class CausalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.left_pad = 4*dilation
        self.conv = nn.Conv1d(TCN_CHANNELS,TCN_CHANNELS,5,dilation=dilation)
        self.mix = nn.Conv1d(TCN_CHANNELS,TCN_CHANNELS,1)

    def forward(self,x):
        return F.elu(x+self.mix(F.elu(self.conv(F.pad(x,(self.left_pad,0))))))


class HistoryEncoder(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        self.input = nn.Conv1d(obs_dim,TCN_CHANNELS,1)
        self.blocks = nn.Sequential(*(CausalBlock(d) for d in (1,2,4,8,16)))
        self.context = nn.Sequential(nn.Linear(2*TCN_CHANNELS,TCN_CHANNELS),nn.ELU(),nn.Linear(TCN_CHANNELS,LATENT_DIM))

    def forward(self,history):
        x = self.blocks(self.input(history.transpose(1,2)))
        return self.context(torch.cat((x[:,:,-1],x.mean(-1)),dim=-1))


class TemporalModel(MLPModel):
    def _get_obs_dim(self,obs,obs_groups,obs_set):
        self.privileged = obs_set == 'critic'
        if obs['actor'].shape[-2] != SEQUENCE_LENGTH:
            raise ValueError('Expected 125 past frames plus the current frame')
        if self.privileged and obs['privileged'].shape[-1] != PRIVILEGED_DIM:
            raise ValueError('Critic requires exactly 27 privileged scalars')
        return ['actor'],obs['actor'].shape[-1]

    def _get_latent_dim(self):
        return self.obs_dim+LATENT_DIM+(PRIVILEGED_DIM if self.privileged else 0)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.encoder = HistoryEncoder(self.obs_dim)
        if self.privileged:
            self.priv_normalizer = EmpiricalNormalization(PRIVILEGED_DIM)

    def get_latent(self,obs,masks=None,hidden_state=None):
        sequence = self.obs_normalizer(obs['actor'])
        parts = [sequence[:,-1],self.encoder(sequence[:,:-1])]
        if self.privileged:
            parts.append(self.priv_normalizer(obs['privileged']))
        return torch.cat(parts,dim=-1)

    def update_normalization(self,obs):
        if self.obs_normalization:
            self.obs_normalizer.update(obs['actor'][:,-1])
        if self.privileged:
            self.priv_normalizer.update(obs['privileged'])

    def as_onnx(self,verbose=False):
        if self.privileged:
            raise ValueError('Export the onboard Actor, not the privileged Critic')
        return PolicyExport(self)

    def as_jit(self):
        return self.as_onnx()


class PolicyExport(nn.Module):
    is_recurrent = False

    def __init__(self,model):
        super().__init__()
        self.obs_dim = model.obs_dim
        self.normalizer = deepcopy(model.obs_normalizer)
        self.encoder = deepcopy(model.encoder)
        self.mlp = deepcopy(model.mlp)
        self.output = model.distribution.as_deterministic_output_module()

    def forward(self,sequence):
        x = self.normalizer(sequence)
        return self.output(self.mlp(torch.cat((x[:,-1],self.encoder(x[:,:-1])),dim=-1)))

    def get_dummy_inputs(self):
        return (torch.zeros(1,SEQUENCE_LENGTH,self.obs_dim),)

    @property
    def input_names(self):
        return ['onboard_history_and_current']

    @property
    def output_names(self):
        return ['h2_position_actions']
