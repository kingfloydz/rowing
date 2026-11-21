"""500 Hz coupling from the most recent MuJoCo forward evaluation."""
import json
import math
import torch
import warp as wp
from hydrodynamics.models import ROOT
from hydrodynamics.policy7400_residual.reduced_runtime import ReducedWaterDR


def rotate(r, x):
    return (r @ x.unsqueeze(-1)).squeeze(-1)


class LiveWater:
    def __init__(self,state,randomize):
        self.s=state
        env=state.env
        self.device=env.device
        self.dt=env.physics_dt
        if not math.isclose(self.dt,.002):
            raise ValueError("The calibrated water response requires 500 Hz physics")
        self.randomize=randomize
        root=ROOT/'hydrodynamics/policy7400_residual'
        self.dr=ReducedWaterDR(root).to(self.device).eval()
        self.models=[torch.jit.load(str(root/f'{b}.pt'),map_location=self.device).eval() for b in ('hull','oars')]
        self.ids=[json.loads((root/f'config_{b}.json').read_text())['ids'] for b in ('hull','oars')]
        self.bodies=[state.boat,*state.blade_area.bodies]
        self.roots=env.sim.mj_model.body_rootid[self.bodies].tolist()
        self.centers=torch.stack([(tri.mean(1)*a[:,None]).sum(0)/a.sum() for tri,a in zip(state.blade_area.triangles,state.blade_area.areas)])
        self.vertices=[tri.reshape(-1,3).unique(dim=0) for tri in state.blade_area.triangles]
        self.reflect=torch.tensor([[1,1,1],[1,-1,1]],device=self.device)
        self.sign=self.reflect[:,1:2]
        self.cache={}
        self.fresh=torch.ones(env.num_envs,dtype=torch.bool,device=self.device)
        self.theta=torch.zeros(env.num_envs,len(self.dr.names),device=self.device)
        self.out=wp.to_torch(state.hydro.correction)
        self.explicit=wp.to_torch(state.hydro.explicit)
        self.chunks=wp.to_torch(state.hydro.chunks)
        self.reset(None)

    def reset(self,ids):
        ids=slice(None) if ids is None else ids
        self.fresh[ids]=True
        self.out[ids]=0
        self.theta[ids]=self.dr.sample(len(self.theta[ids])) if self.randomize else 0

    def history(self,name,x,tau):
        if name not in self.cache:
            self.cache[name]=torch.zeros_like(x)
        state=self.cache[name]
        first=self.fresh.reshape((-1,)+(1,)*(x.ndim-1))
        old=torch.where(first,x,state)
        state.copy_(old+(-math.expm1(-self.dt/tau))*(x-old) if tau else x)
        return state if tau else old

    @torch.no_grad()
    def update(self):
        s=self.s
        d=s.env.sim.data
        r=d.xmat[:,self.bodies].reshape(-1,3,3,3)
        rt=r.transpose(-1,-2)
        pos=d.xpos[:,self.bodies]
        com=d.xipos[:,self.bodies]
        cv=d.cvel[:,self.bodies]
        omega=cv[...,:3]
        vel=cv[...,3:]+torch.cross(omega,pos-d.subtree_com[:,self.roots],dim=-1)
        acc=(vel-self.history('prev_v',vel,0))/self.dt
        alpha=(omega-self.history('prev_w',omega,0))/self.dt
        u=rotate(rt,vel)
        w=rotate(rt,omega)
        arm=rotate(r[:,1:],self.centers)
        blade_v=vel[:,1:]+torch.cross(omega[:,1:],arm,dim=-1)
        blade_a=acc[:,1:]+torch.cross(alpha[:,1:],arm,dim=-1)+torch.cross(omega[:,1:],torch.cross(omega[:,1:],arm,dim=-1),dim=-1)
        sums=torch.stack([self.chunks[:,sl].sum(1) for sl in s.hydro.blade_chunks],1)
        wet=(sums[...,6]/s.blade_area.full_area).clamp(0,1)
        depth=torch.stack([s.water_level-pos[:,i+1,2]-(r[:,i+1,2]@v.T).amin(-1) for i,v in enumerate(self.vertices)],1)
        near=(1+depth/.12).clamp(0,1)
        gate=torch.maximum(near,self.history('gate',near,.08))
        h=torch.cat([pos[:,:1,2:3],r[:,:1,2],u[:,:1],w[:,:1],
                     self.history('ha',rotate(rt[:,:1],acc[:,:1])-torch.cross(w[:,:1],u[:,:1],dim=-1),.012),
                     self.history('halpha',rotate(rt[:,:1],alpha[:,:1]),.012)],-1)
        o=torch.cat([wet[...,None],depth[...,None],r[:,1:,2]*self.reflect,
                     rotate(rt[:,1:],r[:,:1,:,0])*self.reflect,rotate(rt[:,1:],blade_v)*self.reflect,
                     w[:,1:]*self.reflect*self.sign,self.history('oa',rotate(rt[:,1:],blade_a)*self.reflect,.012),
                     self.history('oalpha',rotate(rt[:,1:],alpha[:,1:])*self.reflect*self.sign,.012),
                     u[:,:1,0:1].expand(-1,2,-1)],-1)
        amps=[]
        for i,x in enumerate((h,o)):
            x=x[...,self.ids[i]]
            amps.append(self.models[i](torch.cat([x,x-self.history(f'motion{i}',x,.08)],-1)))
        batch=len(h)
        maps=h.new_zeros(batch,3,6,6)
        maps[:,0,:3,:3]=maps[:,0,3:,3:]=r[:,0]
        axes=r[:,1:]*self.reflect[:,None,:]
        maps[:,1:,:3,:3]=axes
        maps[:,1:,3:,3:]=axes*self.sign[:,None,:]
        maps[:,1:,3:,:3]=torch.cross(arm.unsqueeze(-2).expand_as(axes),axes.transpose(-1,-2),dim=-1).transpose(-1,-2)
        local=h.new_zeros(batch,3,6)
        local[:,0,[0,2,4]]=amps[0][:,0]
        local[:,1:,:2]=amps[1]*gate[...,None]
        neural=rotate(maps,local)
        def origin(load,shift):
            return torch.cat([load[...,:3],load[...,3:]+torch.cross(shift,load[...,:3],dim=-1)],-1)
        def wrench(rot,load):
            return torch.cat([rotate(rot,load[...,:3]),rotate(rot,load[...,3:])],-1)
        raw=origin(wrench(r,self.explicit),com-pos)
        quasi=origin(sums[...,:6],com[:,1:]-pos[:,1:])
        drag=origin(sums[...,12:18],com[:,1:]-pos[:,1:])
        def transient(key,load):
            body=wrench(rt[:,1:],load)
            return wrench(r[:,1:],body-self.history(key,body,.08))
        correction=self.history('base',raw,.008)-raw+neural
        correction[:,1:]+=self.history('transient',transient('quasi',quasi),.008)
        if not self.randomize:
            self.out.copy_(correction)
            self.fresh.fill_(False)
            return
        cols=h.new_zeros(batch,3,6,7)
        cols[:,1:,:,2]=self.history('drag',drag+transient('qdrag',drag),.008)
        cols[:,:1,:,5]=neural[:,:1]
        cols[:,1:,:,6]=neural[:,1:]
        self.out.copy_(self.dr(correction,cols,h,o,maps,self.theta))
        self.fresh.fill_(False)
