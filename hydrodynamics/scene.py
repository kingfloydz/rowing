"""Shared scene geometry and force assembly; no hydrodynamic closure selection."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mujoco
from .models import ROOT, RHO, load_closed_surface, projected_blade


@dataclass
class Water:
    level: float = 0.0
    density: float = RHO
    current: tuple = (0.,0.,0.)
    hull_crossflow_cd: float = 1.0  # uncalibrated damping closure
    fin_normal_cd: float = 1.0     # uncalibrated damping closure


class SceneGeometry:
    def __init__(self, model, repository=ROOT, water=None):
        self.model=model; self.repo=Path(repository); self.water=water or Water()
        self.boat=int(model.body('boat').id)
        self.hull,self.hull_metadata=load_closed_surface(self.repo/'boat_meshes/shell_white.stl')
        self.fin,_=load_closed_surface(self.repo/'boat_meshes/fin.stl')
        self.blades=[]
        for side in ['port','starboard']:
            surface,meta=projected_blade(self.repo/f'boat_meshes/{side}_blade_3mm.stl')
            volume,_=load_closed_surface(self.repo/f'boat_meshes/{side}_blade_3mm.stl')
            self.blades.append((int(model.body(side+'_oar').id),surface,volume,meta))
        self.metadata={'hull':self.hull_metadata,'blades':{side:entry[3] for side,entry in zip(['port','starboard'],self.blades)}}

    def world(self,data,body,triangles):
        return triangles@data.xmat[body].reshape(3,3).T+data.xpos[body]

    def jacobian(self,data,body,point):
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(self.model,data,jp,jr,np.asarray(point),body)
        return jp,jr

    def motion(self,data,body):
        jp,jr=self.jacobian(data,body,data.xpos[body])
        return jp@data.qvel,jr@data.qvel

    def step(self,data,external_qforce=None,added_mass=True):
        if self.model.opt.integrator==mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError('Use Euler/implicitfast; external hydro forces are updated once per timestep.')
        result=self.forward(data,external_qforce,added_mass)
        mujoco.mj_step2(self.model,data)
        return result


class ForceAccumulator:
    """Convert world-space point loads to generalized forces and retain diagnostics."""
    def __init__(self, model, data):
        self.model=model; self.data=data
        self.q=np.zeros(model.nv); self.loads=[]

    def __call__(self, body, force, torque, point, label):
        force,torque,point=(np.asarray(x) for x in (force,torque,point))
        mujoco.mj_applyFT(self.model,self.data,force,torque,point,body,self.q)
        self.loads.append({'body':body,'label':label,'force_N':force,
                           'torque_Nm':torque,'point_m':point})

    def distributed(self, body, points, forces, origin, label):
        self(body,forces.sum(axis=0),np.cross(points-origin,forces).sum(axis=0),origin,label)
