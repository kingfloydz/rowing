"""Reduced episode-wise randomization for the nominal 567-parameter water model."""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class LocalModes(nn.Module):
    def __init__(self,archive,prefix,mapping):
        super().__init__()
        for name in ['center','scale','anchors','bandwidth']:
            self.register_buffer(name,torch.as_tensor(archive[prefix+name],dtype=torch.float32))
        indices=np.flatnonzero(np.any(mapping!=0,axis=1))
        self.register_buffer('indices',torch.tensor(indices))
        projected=archive[prefix+'modes']@mapping[indices].T
        self.register_buffer('coefficients',torch.as_tensor(projected,dtype=torch.float32))

    def forward(self,x,z):
        x=(x-self.center)/self.scale
        distance=(x.unsqueeze(-2)-self.anchors).square().mean(-1)
        w=torch.softmax(-.5*distance/self.bandwidth.square(),dim=-1)
        local=(w@self.coefficients.flatten(1)).unflatten(-1,(6,len(self.indices)))
        return (local@z.index_select(-1,self.indices).unsqueeze(-1)).squeeze(-1)


class ReducedWaterDR(nn.Module):
    def __init__(self,directory=None):
        super().__init__()
        root=Path(directory) if directory else Path(__file__).resolve().parent
        with np.load(root/'reduced_bank.npz') as data:
            self.register_buffer('bank',torch.as_tensor(data['theta'],dtype=torch.float32))
            self.names=data['names'].tolist()
            expansion=data['expand']
        modes=root/'motion_modes.npz'
        with np.load(modes if modes.exists() else root.parent/'motion_modes.npz') as archive:
            self.hull=LocalModes(archive,'hull',expansion[:,7:13])
            self.oars=LocalModes(archive,'oars',expansion[:,13:])
        self.register_buffer('physics',torch.tensor(expansion[:,:7],dtype=torch.float32))
        self.register_buffer('h_ids',torch.tensor([0,1,3,4,5,6,8,9,10,12,14]))
        self.register_buffer('o_ids',torch.tensor([0,1,2,3,5,6,8,9,10,11,12,14,15,20]))

    def sample(self,n,generator=None):
        idx=torch.randint(len(self.bank),(n,),generator=generator,device=self.bank.device)
        return self.bank[idx]

    def forward(self,nominal,columns,hull_motion,oar_motion,world_maps,theta):
        h=self.hull(hull_motion.index_select(-1,self.h_ids),theta[:,None])
        o=self.oars(oar_motion.index_select(-1,self.o_ids),theta[:,None])
        mix=((-oar_motion[...,1]-.05)/.1).clamp(0,1)
        speed=torch.linalg.vector_norm(oar_motion[...,8:11],dim=-1)
        local=torch.cat([h,o*(1-mix+mix*(speed/.5).clamp(0,1)).unsqueeze(-1)],dim=1)
        extra=world_maps@local.unsqueeze(-1)+columns@(theta@self.physics)[:,None,:,None]
        return nominal+extra.squeeze(-1)
