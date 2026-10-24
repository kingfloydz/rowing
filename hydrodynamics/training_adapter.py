"""Opt-in training adapter for 0..4 m/s; source XML/STL stays unchanged.

No fictitious roll controller, acceleration-feedback mass or positive wake gain.
The legacy adapter and strict experimental references remain reproducible.
"""
from dataclasses import asdict
import numpy as np
import mujoco
from .models import ROOT, wet_triangles, triangle_geometry, hydrostatic_wrench
from .scene import SceneGeometry, ForceAccumulator
from .training import (TrainingParameters,TrainingHullResistance,sample_episode,
                       blade_traction,fin_force)


class TrainingRowingHydrodynamics(SceneGeometry):
    def __init__(self,model,repository=ROOT,water=None,parameters=None,blade_mode='harmonic'):
        super().__init__(model,repository,water)
        if blade_mode not in ('harmonic','normal'):raise ValueError('Unknown blade mode')
        self.blade_mode=blade_mode
        self.set_parameters(parameters or TrainingParameters())
        self.metadata['model_status']={
            'profile':'training_0_4_m_s','validated_for_current_CAD':False,
            'target_speed_m_s':[0.,4.],
            'unresolved':['hull_shape_and_loading_transfer','free_surface_and_wake_history',
                          'added_inertia_and_radiation_coefficients','entry_exit_slamming',
                          'hull_component_volume_overlap','shaft_hydrodynamics'],
            'conservative_meaning':'passive resistance; NOT a true-force bound',
            'recommended_reference':'HYDRODYNAMICS_TRAINING.md'}

    def set_parameters(self,p):
        """Set explicitly at episode reset; no resampling during force evaluation."""
        if not isinstance(p,TrainingParameters):raise TypeError('TrainingParameters required')
        self.training_parameters=p
        self.hull_law=TrainingHullResistance(p,self.repo/'data/parameters.json')
        self.metadata['training_parameters']=asdict(p)

    def reset_episode(self,seed):
        self.set_parameters(sample_episode(seed))
        return self.training_parameters

    def force_terms(self,data,rotational_added_mass=False):
        if rotational_added_mass:
            raise ValueError('No calibrated added-inertia closure in training profile')
        w=self.water;p=self.training_parameters
        if (not np.isfinite(w.level) or not np.isfinite(w.density) or w.density<=0
                or np.asarray(w.current).shape!=(3,) or not np.all(np.isfinite(w.current))):
            raise ValueError('Invalid water configuration')
        load=ForceAccumulator(self.model,data)
        self.last_diagnostics={'hull_outside_reference':False,'hull_outside_target_speed':False,
            'relative_dissipative_power_W':0.,'partial_blades':0,
            'model_status':'uncalibrated_training_approximation'}
        def power(forces,velocity):
            self.last_diagnostics['relative_dissipative_power_W']+=float(np.sum(forces*velocity))

        origin=data.xpos[self.boat];rot=data.xmat[self.boat].reshape(3,3)
        hull=self.world(data,self.boat,self.hull)
        f,t,_=hydrostatic_wrench(hull,origin,w.level,w.density)
        load(self.boat,f,t,origin,'hull_buoyancy')
        wet=wet_triangles(hull,w.level)
        if len(wet):
            area,n,c=triangle_geometry(wet);v,om=self.motion(data,self.boat)
            s=float(area.sum());length=float(np.ptp((wet.reshape(-1,3)-origin)@rot,axis=0)[0])
            point=np.average(c,axis=0,weights=area)
            vp=v+np.cross(om,point-origin)-w.current;u=float(vp@rot[:,0])
            if length>1e-8:
                parts=self.hull_law.components(u,length,s,w.density)
                force=-np.sign(u)*parts['total_N']*rot[:,0]
                load(self.boat,force,np.zeros(3),point,'hull_axial_resistance')
                power(force,vp)
                self.last_diagnostics.update(hull_outside_reference=bool(parts['outside_reference']),
                    hull_outside_target_speed=bool(parts['outside_target_speed']),
                    hull_Fn=float(parts['Fn']),hull_friction_N=float(parts['friction_N']),
                    hull_residual_N=float(parts['residual_N']))
            vel=v+np.cross(om,c-origin)-w.current
            # Transverse projected normals: no extra surge resistance. Factor 1/2
            # allocates projected area across both sides of the closed mesh.
            normal=n-(n@rot[:,0])[:,None]*rot[:,0]
            vn=np.sum(vel*normal,axis=1)
            ff=(-.25*w.density*p.hull_crossflow_cd*area*abs(vn)*vn)[:,None]*normal
            load.distributed(self.boat,c,ff,origin,'hull_crossflow');power(ff,vel)

        fin_world=self.world(data,self.boat,self.fin)
        f,t,_=hydrostatic_wrench(fin_world,origin,w.level,w.density)
        load(self.boat,f,t,origin,'fin_buoyancy')
        wet_fin=wet_triangles(fin_world,w.level)
        if len(wet_fin):
            area,n,c=triangle_geometry(wet_fin)
            weights=.5*area*abs(n@rot[:,1]);s=float(weights.sum())
            span=float(np.ptp((wet_fin.reshape(-1,3)-origin)@rot,axis=0)[2])
            if s>1e-10 and span>1e-8:
                point=np.average(c,axis=0,weights=weights)
                v,om=self.motion(data,self.boat);vp=v+np.cross(om,point-origin)-w.current
                ff=fin_force(vp,rot[:,0],rot[:,1],s,span*span/s,p,w.density)
                load(self.boat,ff,np.zeros(3),point,'fin_lift_drag');power(ff,vp)
        for body,surface,volume,meta in self.blades:
            origin=data.xpos[body];rot=data.xmat[body].reshape(3,3)
            full=self.world(data,body,surface)
            f,t,_=hydrostatic_wrench(self.world(data,body,volume),origin,w.level,w.density)
            load(body,f,t,origin,'blade_buoyancy')
            wet=wet_triangles(full,w.level)
            if not len(wet):continue
            if full[:,:,2].min()<w.level<full[:,:,2].max():
                self.last_diagnostics['partial_blades']+=1
            area,_,c=triangle_geometry(wet);v,om=self.motion(data,body)
            vel=v+np.cross(om,c-origin)-w.current
            chord=rot[:,1]*np.sign(surface[:,:,1].mean())
            ff=blade_traction(vel,rot[:,0],chord,area,p,w.density,self.blade_mode)
            load.distributed(body,c,ff,origin,'blade_'+self.blade_mode);power(ff,vel)
        return load.q,load.loads,[]

    def forward(self,data,external_qforce=None,added_mass=False):
        if added_mass:raise ValueError('Training profile does not include validated added mass')
        if external_qforce is not None:
            ext=np.asarray(external_qforce)
            if ext.shape!=(self.model.nv,) or not np.all(np.isfinite(ext)):
                raise ValueError('Invalid external generalized force')
        mujoco.mj_forward(self.model,data)
        q,loads,_=self.force_terms(data)
        data.qfrc_applied[:]=q if external_qforce is None else q+external_qforce
        mujoco.mj_forward(self.model,data)
        result={'loads':loads,'added_mass_residual_N':0.,'added_mass_forces_N':[]}
        result['diagnostics']=self.last_diagnostics.copy()
        return result

    def step(self,data,external_qforce=None,added_mass=False):
        return super().step(data,external_qforce,added_mass)
