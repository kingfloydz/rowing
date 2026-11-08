"""Prescribed-motion CFD audit and oar force-law identification, in SI units.

Raw IBM reactions are NOT exterior fluid loads. The identification below is
deliberately a provisional effective oar calibration; hull coefficients cannot
be identified without the fictitious-interior momentum correction.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from .models import ROOT, clip_underwater, projected_blade, triangle_geometry
from .blade_inertia import load_parameters, repository_strips

PARTS = ('hull', 'port_oar', 'starboard_oar')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_archive(path):
    """Read only canonical manifest-selected records, checking every SHA256."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        rows = []
        for record in manifest['records']:
            raw = archive.read(record['path'])
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError('Source checksum mismatch: '+record['path'])
            frame = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
            if len(frame) != 3 or {r['kind'] for r in frame} != set(PARTS):
                raise ValueError('Expected exactly three bodies in '+record['path'])
            ordered = {r['kind']: r for r in frame}
            values = np.array([[float(ordered[k][col]) for col in
                               ('x_m','y_m','z_m','fx_N','fy_N','fz_N','mx_Nm','my_Nm','mz_Nm')]
                               for k in PARTS])
            rows.append((record['time_output_reconstructed_s'], record['segment'], values))
        rows.sort(key=lambda x: x[0])
        t = np.array([r[0] for r in rows])
        values = np.stack([r[2] for r in rows])
        if not np.all(np.isfinite(values)) or not np.all(np.diff(t) > 0):
            raise ValueError('Invalid force values or timestamps')
        configs = [archive.read(f'raw/segment_{i}/case.cfg').decode()
                   for i in range(len(manifest['source_segments']))]
    return t, np.array([r[1] for r in rows]), values, manifest, configs


def interpolate_trajectory(path, t):
    """Match the CFD's linear interpolation and shortest-path quaternion nlerp."""
    table = np.genfromtxt(path, delimiter=',', names=True)
    times = table['time']
    if np.any(np.diff(times) <= 0) or min(t) < times[0] or max(t) > times[-1]:
        raise ValueError('Trajectory times invalid or do not cover forces')
    result = []
    for part in PARTS:
        source = np.column_stack([table[part+'_'+s] for s in
                                  ('px','py','pz','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz')])
        q = source[:,3:7].copy()
        for i in range(1,len(q)):
            if q[i]@q[i-1] < 0:
                q[i] *= -1
        source[:,3:7] = q
        state = np.column_stack([np.interp(t,times,c) for c in source.T])
        state[:,3:7] /= np.linalg.norm(state[:,3:7],axis=1)[:,None]
        rotation = Rotation.from_quat(state[:,[4,5,6,3]]).as_matrix()
        # Derivative of recorded velocity, not of the prescribed force.
        accel = np.gradient(source[:,7:13], times, axis=0, edge_order=2)
        accel = np.column_stack([np.interp(t,times,c) for c in accel.T])
        result.append(dict(position=state[:,:3], rotation=rotation,
                           velocity=state[:,7:10], omega=state[:,10:13],
                           acceleration=accel[:,:3], alpha=accel[:,3:]))
    return result


def align_forces(raw, trajectory, ale_path, t):
    """ALE -> inertial axes. Moments remain about each body's origin, NOT COM."""
    ale = np.genfromtxt(ale_path, delimiter=',', names=True)
    if t[0] < ale['t_s'][0] or t[-1] > ale['t_s'][-1]:
        raise ValueError('ALE motion does not cover force data')
    yaw = np.interp(t, ale['t_s'], ale['yaw_rad'])
    r = Rotation.from_euler('z',yaw[:,None]).as_matrix()
    center = np.column_stack([np.interp(t,ale['t_s'],ale[k]) for k in ('Cx_m','Cy_m')])
    center = np.column_stack([center,np.zeros(len(t))])
    position = np.einsum('nij,nbj->nbi',r,raw[:,:,:3])+center[:,None,:]
    errors = np.stack([np.linalg.norm(position[:,b]-s['position'],axis=1)
                       for b,s in enumerate(trajectory)],axis=1)
    wrench = np.concatenate([np.einsum('nij,nbj->nbi',r,raw[:,:,3:6]),
                             np.einsum('nij,nbj->nbi',r,raw[:,:,6:9])],axis=2)
    # Printed positions have six significant digits and may be one RK stage old.
    if errors.max() > .01:
        raise ValueError(f'ALE pose alignment mismatch: {errors.max():.6g} m')
    return wrench, errors


def smoothstep(x):
    a = np.clip(x,0.,1.)
    return a*a*(3-2*a)


def cell_wet(a,b,adot,bdot):
    """NumPy equivalent of rowing_mjlab.blade_inertia.cell_wet."""
    def integral(x):
        c=np.clip(x,0,1)
        return c**3-.5*c**4+np.maximum(x-1,0)
    diff=b-a
    near=np.abs(diff)<1e-3
    safe=np.where(near,1,diff)
    fraction=(integral(b)-integral(a))/safe
    rate=(smoothstep(b)*bdot-smoothstep(a)*adot-fraction*(bdot-adot))/safe
    mid=np.clip(.5*(a+b),0,1)
    fraction=np.where(near,smoothstep(mid),fraction)
    rate=np.where(near,6*mid*(1-mid)*.5*(adot+bdot),rate)
    return np.clip(fraction,0,1),rate


def oar_basis(states, level, repository=ROOT):
    """World wrenches about origin: drag, lift, fixed friction, added inertia.

    The first two columns multiply the EXISTING CDmax=2, CLmax=.9. The last
    column multiplies the existing 8 kg strip inertia. No phase/time lookup.
    Hydrostatics deliberately excluded: the source is a raw IBM reaction.
    """
    p=load_parameters(Path(repository)/'data/blade_inertia.json')
    strips=repository_strips(repository,p)
    nt=len(states[0]['position']); ns=p['strips_per_blade']
    basis=np.zeros((nt,2,6,4)); wet_area=np.zeros((nt,2))
    speed=np.zeros((nt,2)); acceleration=np.zeros((nt,2))
    for side,name in enumerate(('port','starboard')):
        triangles=projected_blade(Path(repository)/'boat_meshes'/f'{name}_blade_3mm.stl')[0]
        state=states[side+1]
        endpoints=strips.endpoints[side*ns:(side+1)*ns].astype(float)
        nominal=strips.full_mass[side*ns:(side+1)*ns].astype(float)
        for i in range(nt):
            R=state['rotation'][i]; pos=state['position'][i]
            v0=state['velocity'][i]; omega=state['omega'][i]
            normal=R[:,0]; chord=R[:,1]*(1 if side==0 else -1)
            world=triangles@R.T+pos
            depths=level-world[:,:,2].mean(axis=1)
            full=np.all(world[:,:,2]<=level,axis=1)
            partial=(world[:,:,2].min(axis=1)<level)&(world[:,:,2].max(axis=1)>level)
            wet=list(world[full]); factors=list(smoothstep((depths[full]+p['wetting_depth_m'])/(2*p['wetting_depth_m'])))
            for tri,depth in zip(world[partial],depths[partial]):
                poly=clip_underwater(tri,level)
                for j in range(1,len(poly)-1):
                    wet.append(poly[[0,j,j+1]])
                    factors.append(smoothstep((depth+p['wetting_depth_m'])/(2*p['wetting_depth_m'])))
            if wet:
                area,_,center=triangle_geometry(np.asarray(wet)); arm=center-pos
                v=v0+np.cross(omega,arm)
                vn=v@normal;vt=v@chord;u=np.sqrt(vn**2+vt**2);safe=np.maximum(u,1e-12)
                e=(vt[:,None]*chord+vn[:,None]*normal)/safe[:,None]
                el=(abs(vn)[:,None]*chord-np.sign(vn)[:,None]*vt[:,None]*normal)/safe[:,None]
                q=.5*1000*area*np.asarray(factors)
                drag=-(q*2*vn**2)[:,None]*e
                lift=(q*1.8*abs(vn)*vt)[:,None]*el
                tangent=v-vt[:,None]*chord-vn[:,None]*normal
                skin=-(q*.01*u*u)[:,None]*e-(q*.01*np.linalg.norm(tangent,axis=1))[:,None]*tangent
                for k,f in enumerate((drag,lift,skin)):
                    basis[i,side,:,k]=np.r_[f.sum(0),np.cross(arm,f).sum(0)]
                wet_area[i,side]=area.sum();speed[i,side]=np.max(np.linalg.norm(v,axis=1))
            # Normal strip added inertia and entry recruitment, evaluated on
            # prescribed acceleration rather than omitted qacc matrix terms.
            ep=endpoints@R.T+pos; ev=v0+np.cross(omega,ep-pos)
            wet,rate=cell_wet((level-ep[:,:-1,2])/p['wetting_depth_m'],
                              (level-ep[:,1:,2])/p['wetting_depth_m'],
                              -ev[:,:-1,2]/p['wetting_depth_m'], -ev[:,1:,2]/p['wetting_depth_m'])
            fraction=wet.mean(1);frate=rate.mean(1)
            mass=.25*nominal[:,None]*wet*fraction[:,None]
            mdot=.25*nominal[:,None]*(rate*fraction[:,None]+wet*frate[:,None])
            arm=(.5*(endpoints[:,:-1]+endpoints[:,1:]))@R.T
            v=v0+np.cross(omega,arm)
            a=state['acceleration'][i]+np.cross(state['alpha'][i],arm)+np.cross(omega,np.cross(omega,arm))
            ndot=np.cross(omega,normal);vn=v@normal
            f=-mass[:,:,None]*((a@normal+v@ndot)[:,:,None]*normal+vn[:,:,None]*ndot)
            f-=np.maximum(mdot,0)[:,:,None]*vn[:,:,None]*normal
            torque=np.cross(arm,f)+mass[:,:,None]*vn[:,:,None]*np.cross(normal,v)
            basis[i,side,:,3]=np.r_[f.sum((0,1)),torque.sum((0,1))]
            acceleration[i,side]=np.max(np.linalg.norm(a,axis=2))
    return basis,dict(wet_area_m2=wet_area,speed_ms=speed,acceleration_ms2=acceleration)
