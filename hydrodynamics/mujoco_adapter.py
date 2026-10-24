"""Exploratory CPU coupling for the unmodified repository scene.

Online model: geometry-based buoyancy + resistance + strip blade loads + optional
implicit centre-lumped blade added mass. Caplan static polars and orientation
terms for added mass are opt-in extensions. Scalar start-from-rest history laws
are benchmark closures, not validated for cyclic strokes.
This legacy scene adapter is NOT the validated hydrodynamic core. In particular
CAD-transferred polars/mass and Cd=1 lateral closures remain exploratory.
Use reference_models.py for scoped empirical references and core.py for theory.
"""
import numpy as np
import mujoco
from scipy.optimize import root
from .scene import Water, SceneGeometry, ForceAccumulator
from .polars import BladePolar
from .unsteady import rotating_added_mass_wrench
from .models import (ROOT, load_parameters,
                     wet_triangles, triangle_geometry, hydrostatic_wrench,
                     normal_plate_force, plate_added_mass)


def solve_normal_added_mass(model,data,base,B,mass,bias):
    """Solve f + diag(m)*(B*qacc(f)+bias)=0, retaining MuJoCo constraints."""
    return solve_coupled_added_inertia(model,data,base,B,np.diag(mass),bias)


def solve_coupled_added_inertia(model,data,base,B,inertia,bias):
    """Solve w=-A(B qacc+bias) simultaneously with rigid-body constraints.

    B maps generalized acceleration into one consistent fluid-coordinate basis.
    Caller includes frame/velocity-dependent terms in base and bias. A can be
    a block or full matrix, including off-diagonal interactions when independently
    supplied. This solver does not generate hydrodynamic coefficients.
    Residual uses the caller's SI coordinates (mixed N and Nm for a full wrench).
    """
    B=np.asarray(B,float);inertia=np.asarray(inertia,float);bias=np.asarray(bias,float)
    n=len(bias)
    if (B.shape!=(n,model.nv) or inertia.shape!=(n,n)
            or not all(np.all(np.isfinite(x)) for x in (B,inertia,bias))):
        raise ValueError('Invalid added-inertia coupling arrays')
    if not np.allclose(inertia,inertia.T,rtol=1e-10,atol=1e-12):
        raise ValueError('Added inertia must be symmetric')
    if np.linalg.eigvalsh(inertia).min() < -1e-10*max(np.linalg.norm(inertia,2),1e-30):
        raise ValueError('Added inertia must have nonnegative kinetic energy')
    warm=data.qacc_warmstart.copy()
    def acceleration(f):
        data.qfrc_applied[:]=base+B.T@f
        data.qacc_warmstart[:]=warm
        mujoco.mj_forward(model,data)
        return B@data.qacc+bias
    a0=acceleration(np.zeros(n))
    H=np.column_stack([acceleration(e)-a0 for e in np.eye(n)])
    f=np.linalg.solve(np.eye(n)+inertia@H,-inertia@a0)
    residual=lambda force: force+inertia@acceleration(force)
    if np.linalg.norm(residual(f),ord=np.inf)>1e-5:
        sol=root(residual,f);f=sol.x
    error=float(np.linalg.norm(residual(f),ord=np.inf))
    if error>1e-4: raise RuntimeError(f'Implicit added-inertia solve failed: residual {error} in SI coordinates')
    return f,error


def couple_unbounded_inertia(model,data,external_qforce,body,point,inertia,current=(0.,0.,0.)):
    """Apply a supplied constant Kirchhoff tensor at a body-fixed point.

    Explicit unbounded-fluid benchmark interface, NOT automatically attached to
    the surface-piercing CAD boat/oars. Assumes steady uniform current; derivatives
    below are derivatives of relative BODY velocity components.
    """
    from .core import UnboundedPotentialInertia
    if not isinstance(inertia,UnboundedPotentialInertia):
        raise TypeError('An explicit UnboundedPotentialInertia with provenance is required')
    mujoco.mj_forward(model,data)
    rotation=data.xmat[body].reshape(3,3)
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    dp=np.zeros_like(jp);dr=np.zeros_like(jp)
    mujoco.mj_jac(model,data,jp,jr,np.asarray(point),body)
    mujoco.mj_jacDot(model,data,dp,dr,np.asarray(point),body)
    B=np.vstack((rotation.T@jp,rotation.T@jr))
    nu=B@data.qvel-np.r_[rotation.T@np.asarray(current),np.zeros(3)]
    bias=np.r_[rotation.T@dp@data.qvel-np.cross(nu[3:],nu[:3]),
               rotation.T@dr@data.qvel]
    convective=-inertia.coriolis_wrench(nu)
    base=np.asarray(external_qforce)+B.T@convective
    acceleration_wrench,error=solve_coupled_added_inertia(model,data,base,B,inertia.matrix,bias)
    return {'body_wrench':acceleration_wrench+convective,'residual_SI':error,
            'fluid_domain':'unbounded_potential_flow','geometry_id':inertia.geometry_id}


class RowingHydrodynamics(SceneGeometry):
    def __init__(self, model, repository=ROOT, water=None, blade_model='normal', added_mass_rotation=False):
        super().__init__(model,repository,water)
        if blade_model not in ('normal','caplan_flat'):
            raise ValueError('blade_model must be normal or caplan_flat')
        self.blade_model=blade_model
        self.polar=BladePolar.caplan_flat(self.repo/'data/caplan2007_flat_polar.csv') if blade_model=='caplan_flat' else None
        self.added_mass_rotation=added_mass_rotation
        self.hull_law,self.blade_law,self.parameters=load_parameters(self.repo/'data/parameters.json')
        self.metadata['model_status']={
            'profile':'exploratory_legacy_scene', 'validated_for_current_CAD':False,
            'unresolved':['hull_shape_transfer','blade_free_surface_history',
                          'lateral_and_fin_coefficients','variable_wetted_added_mass'],
            'recommended_reference':'HYDRODYNAMICS_CORE.md'}

    def force_terms(self,data,rotational_added_mass=False):
        """Call after mj_forward or mj_step1. Does not modify qfrc_applied."""
        w=self.water; added=[]
        load=ForceAccumulator(self.model,data)
        origin=data.xpos[self.boat]; rot=data.xmat[self.boat].reshape(3,3)
        hull=self.world(data,self.boat,self.hull)
        f,t,s=hydrostatic_wrench(hull,origin,w.level,w.density)
        load(self.boat,f,t,origin,'hull_buoyancy')
        wet=wet_triangles(hull,w.level)
        if len(wet):
            area,n,centres=triangle_geometry(wet)
            v,omega=self.motion(data,self.boat)
            length=float(np.ptp((wet.reshape(-1,3)-origin)@rot,axis=0)[0])
            point=np.average(centres,axis=0,weights=area)
            # Use velocity at the force application point, so surge drag remains
            # dissipative even when the hull is pitching/rotating.
            local_v=(v+np.cross(omega,point-origin)-np.asarray(w.current))@rot
            u=float(local_v[0]);drag=self.hull_law.magnitude(u,max(length,.1),s,w.density)
            load(self.boat,-np.sign(u)*drag*rot[:,0],np.zeros(3),point,'hull_axial_resistance')
            # Transverse surface-pressure closure; zero axial component to avoid
            # double counting calibrated surge resistance. No measured lateral data.
            velocity=(v+np.cross(omega,centres-origin)-w.current)@rot
            normal=n@rot;normal[:,0]=0;velocity[:,0]=0
            vn=np.sum(velocity*normal,axis=1)
            forces=(-.25*w.density*w.hull_crossflow_cd*area*abs(vn)*vn)[:,None]*normal@rot.T
            load.distributed(self.boat,centres,forces,origin,'hull_crossflow')
        # Fin normal drag only. Axial skin drag already absorbed into hull law.
        fin=wet_triangles(self.world(data,self.boat,self.fin),w.level)
        if len(fin):
            area,n,c=triangle_geometry(fin);v,om=self.motion(data,self.boat)
            vel=v+np.cross(om,c-origin)-w.current
            n=n-(n@rot[:,0])[:,None]*rot[:,0]
            vn=np.sum(vel*n,axis=1)
            ff=(-.25*w.density*w.fin_normal_cd*area*abs(vn)*vn)[:,None]*n
            load.distributed(self.boat,c,ff,origin,'fin_transverse_drag')
        for body,surface,volume,meta in self.blades:
            origin=data.xpos[body];rot=data.xmat[body].reshape(3,3)
            world=self.world(data,body,surface)
            f,t,_=hydrostatic_wrench(self.world(data,body,volume),origin,w.level,w.density)
            load(body,f,t,origin,'blade_buoyancy')
            wet=wet_triangles(world,w.level)
            if not len(wet):continue
            area,_,centres=triangle_geometry(wet);normal=rot[:,0]
            v,omega=self.motion(data,body)
            velocity=v+np.cross(omega,centres-origin)-w.current
            top_depth=w.level-world[:,:,2].max()
            # h<0 is partial immersion: use h=0 Cd and explicitly scale wet area.
            # This extrapolation is not a calibrated entry/exit/slamming model.
            cd=self.blade_law.cd(max(top_depth,0),meta['max_vertical_chord_m'])
            if self.polar is None:
                forces=normal_plate_force(velocity,normal,area,cd,w.density)
                label='blade_normal_drag'
            else:
                # Local y runs along the shaft; mesh centroid identifies the tip side.
                # Caplan's static top-edge-at-waterline experiment is an alternative
                # closure: do not multiply it by Grift's independent depth Cd again.
                chord=rot[:,1]*np.sign(surface[:,:,1].mean())
                forces=self.polar.force(velocity,normal,chord,area,w.density)
                label='blade_caplan_static_polar'
            load.distributed(body,centres,forces,origin,label)
            centre=np.average(centres,axis=0,weights=area)
            jp,_=self.jacobian(data,body,centre)
            jd=np.zeros_like(jp);mujoco.mj_jacDot(self.model,data,jd,None,centre,body)
            ma=plate_added_mass(meta['span_m'],meta['equivalent_chord_m'],w.density)*area.sum()/meta['area_m2']
            if rotational_added_mass:
                # Centre-lumped anisotropic mass: orientation terms only. Distributed
                # rotational added inertia and wet-mass derivatives remain unmodelled.
                vp=v+np.cross(omega,centre-origin)-w.current
                fa,ta=rotating_added_mass_wrench(vp,np.zeros(3),normal,omega,ma)
                load(body,fa,ta,centre,'blade_added_mass_rotation')
            added.append({'B':normal@jp,'mass':ma,'bias':float(normal@jd@data.qvel)})
        return load.q,load.loads,added

    def forward(self,data,external_qforce=None,added_mass=True):
        """Solve blade normal acceleration/force simultaneously, with constraints.

        Default: frozen-direction F_A=-m_A*n*(n.a_point). Optional orientation
        terms follow fixed-mass anisotropic fluid kinetic energy. Neither mode
        includes changing wet mass, distributed angular added inertia, slamming
        or wake history. No delayed acceleration feedback.
        Preserves caller's supplied generalized forces via external_qforce.
        """
        mujoco.mj_forward(self.model,data)
        q,loads,terms=self.force_terms(data,rotational_added_mass=added_mass and self.added_mass_rotation)
        external=np.zeros(self.model.nv) if external_qforce is None else np.asarray(external_qforce)
        base=q+external
        if not added_mass or not terms:
            data.qfrc_applied[:]=base;mujoco.mj_forward(self.model,data)
            return {'loads':loads,'added_mass_residual_N':0.,'added_mass_forces_N':[]}
        B=np.array([a['B'] for a in terms]);mass=np.array([a['mass'] for a in terms]);bias=np.array([a['bias'] for a in terms])
        f,error=solve_normal_added_mass(self.model,data,base,B,mass,bias)
        return {'loads':loads,'added_mass_residual_N':error,'added_mass_forces_N':f.tolist()}
