"""New geometry + recorded trajectory; old layout with envelope-delta edits."""
from pathlib import Path
import json, csv, importlib.util, hashlib
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPLAY = HERE/'replay'

def main():
    meta = json.loads((REPLAY/'metadata.json').read_text())
    assert not meta['terminated'] and abs(meta['simulated_seconds']-11.6)<1e-8
    model = mujoco.MjModel.from_binary_path(str(REPLAY/'model.mjb')); data = mujoco.MjData(model)
    r = np.load(REPLAY/'rollout.npz'); reset = np.load(REPLAY/'initial_state.npz')
    times = np.r_[0., r['time']]; qpos = np.vstack([reset['qpos'], r['qpos']]); qvel = np.vstack([reset['qvel'], r['qvel']])
    bodies = [model.body('rowing/'+s).id for s in ('boat','port_oar','starboard_oar')]
    names = ['shell_white_visual','port_blade_visual','starboard_blade_visual','port_shaft_visual','starboard_shaft_visual']
    parts = [0,1,2,1,2]; local = {}; centers = []
    data.qpos[:] = qpos[0]; mujoco.mj_kinematics(model,data)
    for name, part in zip(names, parts):
        g=model.geom('rowing/'+name).id; mesh=model.geom_dataid[g]; body=bodies[part]
        v=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]].astype(float)
        f=model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh]+model.mesh_facenum[mesh]]
        p=(v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]-data.xpos[body])@data.xmat[body].reshape(3,3)
        local[name]=p[ConvexHull(p).vertices]
        if 'blade' in name:
            tri=p[f]; area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            centers.append((tri.mean(1)*area[:,None]).sum(0)/area.sum())
    poses = np.zeros((len(times),40)); poses[:,0]=times; blade_centers=[]
    for k in range(len(times)):
        data.qpos[:]=qpos[k]; data.qvel[:]=qvel[k]
        mujoco.mj_kinematics(model,data); mujoco.mj_comPos(model,data); mujoco.mj_comVel(model,data)
        for part, b in enumerate(bodies):
            off=1+part*13; poses[k,off:off+3]=data.xpos[b]; poses[k,off+3:off+7]=data.xquat[b]
            # Jacobian gives velocity at the body origin, not its COM.
            jp=np.zeros((3,model.nv)); jr=np.zeros_like(jp)
            mujoco.mj_jacBody(model,data,jp,jr,b)
            poses[k,off+7:off+10]=jp@qvel[k]; poses[k,off+10:off+13]=jr@qvel[k]
        blade_centers.append([data.xpos[bodies[j+1]]+data.xmat[bodies[j+1]].reshape(3,3)@centers[j] for j in range(2)])
    blade_centers=np.asarray(blade_centers); center=blade_centers.mean(1)
    line=blade_centers[:,1,:2]-blade_centers[:,0,:2]; forward=np.c_[-line[:,1],line[:,0]]
    data.qpos[:]=qpos[0]; mujoco.mj_kinematics(model,data)
    if forward[0]@data.xmat[bodies[0]].reshape(3,3)[:2,0]<0: forward=-forward
    yaw=np.unwrap(np.arctan2(forward[:,1],forward[:,0]))
    ale=np.c_[times,center[:,:2],np.zeros(len(times)),yaw,np.gradient(center[:,:2],times,axis=0,edge_order=2),np.zeros(len(times)),np.gradient(yaw,times,edge_order=2)]
    headers=['time']
    for n in ('hull','port_oar','starboard_oar'): headers += [n+s for s in ('_px','_py','_pz','_qw','_qx','_qy','_qz','_vx','_vy','_vz','_wx','_wy','_wz')]
    np.savetxt(REPLAY/'policy_trajectory.csv',poses,delimiter=',',header=','.join(headers),comments='')
    np.savetxt(REPLAY/'ale_motion.csv',ale,delimiter=',',header='t_s,Cx_m,Cy_m,Cz_m,yaw_rad,Vx_ms,Vy_ms,Vz_ms,yaw_rate_rad_s',comments='')
    # Every SSPRK half-step time for fixed 50us. Interpolation matches current solver.
    ts=np.linspace(0,11.6,464001); envelope={}; wet_shaft={}
    for name,part in zip(names,parts):
        low=np.full(3,np.inf); high=-low; off=1+13*part
        wetlow=np.full(3,np.inf); wethigh=-wetlow
        if 'shaft' in name:
            faces=ConvexHull(local[name]).simplices
            edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        for start in range(0,len(ts),2048):
            t=ts[start:start+2048]; k=np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2); f=((t-times[k])/(times[k+1]-times[k]))[:,None]
            p=poses[k,off:off+3]*(1-f)+poses[k+1,off:off+3]*f
            qa=poses[k,off+3:off+7]; qb=poses[k+1,off+3:off+7]; qb=qb*np.where((qa*qb).sum(1)<0,-1.,1.)[:,None]
            q=qa*(1-f)+qb*f; q/=np.linalg.norm(q,axis=1)[:,None]
            xyz=np.einsum('tij,vj->tvi',Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix(),local[name])+p[:,None,:]
            af=ale[k,1:5]*(1-f)+ale[k+1,1:5]*f; x=xyz[:,:,0]-af[:,None,0]; y=xyz[:,:,1]-af[:,None,1]; c=np.cos(af[:,3])[:,None]; s=np.sin(af[:,3])[:,None]
            xyz[:,:,0]=c*x+s*y; xyz[:,:,1]=-s*x+c*y
            low=np.minimum(low,xyz.min((0,1))); high=np.maximum(high,xyz.max((0,1)))
            if 'shaft' in name:
                # Clip the convex shaft support at water+5cm, including edge/plane
                # intersections. This conservatively contains the wetted CAD.
                level=meta['water_level_m']+.05
                inside=xyz[:,:,2]<=level
                wetlow=np.minimum(wetlow,np.where(inside[:,:,None],xyz,np.inf).min((0,1)))
                wethigh=np.maximum(wethigh,np.where(inside[:,:,None],xyz,-np.inf).max((0,1)))
                va=xyz[:,edges[:,0]]; vb=xyz[:,edges[:,1]]
                cross=(va[:,:,2]<level)&(vb[:,:,2]>level)|(vb[:,:,2]<level)&(va[:,:,2]>level)
                va=va[cross];vb=vb[cross]
                if len(va):
                    hit=va+(vb-va)*((level-va[:,2])/(vb[:,2]-va[:,2]))[:,None]
                    wetlow=np.minimum(wetlow,hit.min(0));wethigh=np.maximum(wethigh,hit.max(0))
        envelope[name]=dict(low_m=low.tolist(),high_m=high.tolist(),support_vertices=len(local[name]))
        if 'shaft' in name:wet_shaft[name]=dict(low_m=wetlow.tolist(),high_m=wethigh.tolist(),clip_z_m=meta['water_level_m']+.05)
        print(name, low, high, flush=True)
    (HERE/'measured_envelope.json').write_text(json.dumps(dict(envelope=envelope,wet_shaft_envelope=wet_shaft),indent=2))

if __name__=='__main__':main()
