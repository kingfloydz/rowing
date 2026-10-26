"""Wetted blade planform projected along the boat longitudinal axis (not flow)."""
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from hydrodynamics.models import ROOT, projected_blade, triangle_geometry, wet_triangles


def wet_fraction(heights):
    """Exact fraction of a planar triangle below z=0, including intersections."""
    h = heights.sort(dim=-1).values
    a,b,c = h.unbind(-1)
    one = a.square()/((b-a)*(c-a)).clamp_min(1e-20)
    two = 1-c.square()/((c-a)*(c-b)).clamp_min(1e-20)
    return torch.where(c<=0,1.,torch.where(a>=0,0.,torch.where(b>=0,one,two))).clamp(0,1)


class BladeAreaReference:
    def __init__(self, model, reference, water_level, device, prefix=''):
        import mujoco
        self.boat = model.body(prefix+'boat').id
        self.bodies = [model.body(prefix+s+'_oar').id for s in ('port','starboard')]
        geometry = [projected_blade(ROOT/'boat_meshes'/f'{s}_blade_3mm.stl')[0]
                    for s in ('port','starboard')]
        self.triangles = [torch.tensor(t,dtype=torch.float32,device=device) for t in geometry]
        self.areas = [torch.tensor(triangle_geometry(t)[0],dtype=torch.float32,device=device) for t in geometry]
        self.full_area = torch.stack([a.sum() for a in self.areas])
        self.sigma = .15  # fraction of each blade's full planform area
        data = mujoco.MjData(model)
        qa = [model.joint(prefix+s+'_'+j).qposadr[0] for s in ('port','starboard')
              for j in ('sweep','elevation','feather')]
        phase = np.linspace(0,1,401)
        angles = reference(torch.tensor(phase,dtype=torch.float32,device=device)).cpu().numpy()[:,:6]
        def area_at_pose():
            mujoco.mj_kinematics(model,data)
            axis = data.xmat[self.boat].reshape(3,3)[:,0]
            result = []
            for body,tri in zip(self.bodies,geometry):
                rot = data.xmat[body].reshape(3,3)
                wet = wet_triangles(tri@rot.T+data.xpos[body],water_level)
                result.append(triangle_geometry(wet)[0].sum()*abs(rot[:,0]@axis))
            return result
        data.qpos[:] = model.qpos0
        self.initial = torch.tensor(area_at_pose(),dtype=torch.float32,device=device)
        values = []
        for q in angles:
            data.qpos[qa] = q
            values.append(area_at_pose())
        values = np.array(values); values[-1] = values[0]
        self.n = len(phase)-1
        self.coefficients = torch.tensor(PchipInterpolator(phase,values).c,dtype=torch.float32,device=device)

    def target(self,phase,blend):
        x = phase.remainder(1)*self.n
        index = x.long().clamp(0,self.n-1)
        delta = ((x-index)/self.n)[...,None]
        c = self.coefficients[:,index]
        area = ((c[0]*delta+c[1])*delta+c[2])*delta+c[3]
        return self.initial+blend[...,None]*(area-self.initial)

    def actual(self,data,water_level):
        axis = data.xmat[:,self.boat].reshape(-1,3,3)[:,:,0]
        result = []
        for body,tri,areas in zip(self.bodies,self.triangles,self.areas):
            rot = data.xmat[:,body].reshape(-1,3,3)
            heights = torch.einsum('nj,fvj->nfv',rot[:,2,:],tri)
            heights = heights+(data.xpos[:,body,2]-water_level)[:,None,None]
            wetted = (wet_fraction(heights)*areas).sum(-1)
            projection = (rot[:,:,0]*axis).sum(-1).abs()
            result.append(wetted*projection)
        return torch.stack(result,dim=-1)
