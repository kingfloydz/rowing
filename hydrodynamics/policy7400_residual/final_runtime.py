"""500 Hz causal correction for one trajectory of the policy7400 fast-water solver."""
from pathlib import Path
import json
import numpy as np
import torch


class WaterForces:
    def __init__(self,directory=None):
        path=Path(directory) if directory else Path(__file__).resolve().parent
        self.config={name:json.loads((path/f'config_{name}.json').read_text()) for name in ['hull','oars']}
        self.network={name:torch.jit.load(str(path/f'{name}.pt'),map_location='cpu').eval() for name in self.config}
        self.reset()

    def reset(self):
        """Clear all causal filter states at the start of an independent replay."""
        self.state={}

    def filter(self,name,value,tau):
        if name not in self.state:
            self.state[name]=np.array(value,dtype=float,copy=True)
        else:
            self.state[name]+=(-np.expm1(-.002/tau))*(value-self.state[name])
        return self.state[name]

    @torch.inference_mode()
    def components(self,hull_motion,oar_motion,hull_map,oar_map,previous_physics,friction,quasi_local,oar_rotation):
        """Advance once. Return the physical and residual world wrenches separately."""
        physical=np.array(previous_physics,dtype=float,copy=True)
        physical[0]+=(self.config['hull']['shear']-1)*self.filter('friction',friction,.008)
        transient=quasi_local-self.filter('quasi',quasi_local,.08)
        world=np.einsum('bij,bkj->bki',oar_rotation,transient).reshape(2,6)
        physical[1:]+=self.config['oars']['history_gain']*self.filter('unsteady',world,.008)
        residual=[]
        for name,motion,mapping in [('hull',hull_motion,hull_map),('oars',oar_motion,oar_map)]:
            cfg=self.config[name]
            x=motion[...,cfg['ids']]
            if cfg['tau']:
                x=np.concatenate([x,x-self.filter(name,x,cfg['tau'])],axis=-1)
            amplitudes=self.network[name](torch.as_tensor(x,dtype=torch.float32)).numpy()
            residual.append((mapping[...,cfg['columns']]@amplitudes[...,None]).squeeze(-1))
        return physical,np.concatenate(residual,axis=0)

    def step(self,hull_motion,oar_motion,hull_map,oar_map,previous_physics,friction,quasi_local,oar_rotation):
        """Return total fluid-on-body world forces and body-origin moments, shape (3,6)."""
        physical,residual=self.components(hull_motion,oar_motion,hull_map,oar_map,previous_physics,friction,quasi_local,oar_rotation)
        return physical+residual
