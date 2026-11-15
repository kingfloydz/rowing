"""Remove labelled interior/dry-handle IB geometry; preserve retained quadrature.

This changes the hydrodynamic surface selection, not the rendering model.
The original marker CSV and replay are read-only. Geometry identities and the
original area-centroid quadrature are verified before any row is selected.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from export_policy_ib_assembly import consistent_winding

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'designs/ale_mesh_policy1100_blade_focus_20261101/policy_markers_oriented.csv'
REPLAY=ROOT/'renders/policy_replay/final_policy_1100'
OUT=ROOT/'designs/ib_exterior_policy1100_20261106'
WATER=-.12027640640735626

def clearance_lower_bound(points,part,trajectory):
    """Lower bound over continuous linear-translation/short-arc nlerp replay.

    Each time lies within half an interval of one endpoint. Bound translation
    by |delta z|/2 and the rotated radius by 2*r*sin(theta/4). This deliberately
    overestimates motion toward the water between the stored policy samples.
    """
    off=1+13*part;q=trajectory[:,off+3:off+7]
    q=q/np.linalg.norm(q,axis=1)[:,None]
    r=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    theta=2*np.arccos(np.clip(abs(np.sum(q[:-1]*q[1:],axis=1)),0,1))
    z=trajectory[:,off+2]
    answer=np.empty(len(points))
    for start in range(0,len(points),2048):
        p=points[start:start+2048];height=r[:,2,:]@p.T+z[:,None]
        bound=np.minimum(height[:-1],height[1:])-.5*abs(np.diff(z))[:,None]
        bound-=2*np.sin(theta[:,None]/4)*np.linalg.norm(p,axis=1)[None,:]
        answer[start:start+len(p)]=bound.min(axis=0)-WATER
    return answer

def local_surface(model,data,name,part):
    g=model.geom(name).id;mid=model.geom_dataid[g]
    va,vn=model.mesh_vertadr[mid],model.mesh_vertnum[mid];fa,fn=model.mesh_faceadr[mid],model.mesh_facenum[mid]
    root=model.body(['rowing/boat','rowing/port_oar','rowing/starboard_oar'][part]).id
    vertices=model.mesh_vert[va:va+vn]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
    vertices=(vertices-data.xpos[root])@data.xmat[root].reshape(3,3)
    faces,orientation=consistent_winding(vertices,model.mesh_face[fa:fa+fn].astype(np.int64))
    return vertices,faces,orientation

def interior_triangles(triangles,shell,occluders):
    """Conservative inner-layer candidates on the transverse hull envelope.

    A face is omitted only if its normal points into the hull at all seven
    vertex/edge/centroid probes AND another CAD surface hides every probe
    from the transverse exterior. Ambiguous rim/end faces remain in IB.
    This is a sampled CAD classification, independently audited by rays.
    """
    edges=np.concatenate([shell[:,[0,1]],shell[:,[1,2]],shell[:,[2,0]]])
    e0,e1=edges[:,0],edges[:,1];dx=e1[:,0]-e0[:,0]
    a=occluders[:,0];u=occluders[:,1]-a;v=occluders[:,2]-a
    def centre(x):
        selected=(abs(dx)>1e-12)&(np.minimum(e0[:,0],e1[:,0])<=x)&(np.maximum(e0[:,0],e1[:,0])>=x)
        if selected.sum()<2:return None
        cross=e0[selected]+((x-e0[selected,0])/dx[selected])[:,None]*(e1[selected]-e0[selected])
        lo,hi=cross.min(0),cross.max(0)
        if np.ptp(cross[:,1])<1e-6 or np.ptp(cross[:,2])<1e-6:return None
        return (lo+hi)/2
    removed=np.zeros(len(triangles),dtype=bool)
    for i,t in enumerate(triangles):
        normal=np.cross(t[1]-t[0],t[2]-t[0]);size=np.linalg.norm(normal)
        if size<1e-14:continue
        normal/=size
        probes=np.concatenate([t,(t+np.roll(t,1,axis=0))/2,t.mean(0)[None,:]])
        hidden=True
        for point in probes:
            origin=centre(point[0])
            if origin is None:hidden=False;break
            direction=point-origin;length=np.linalg.norm(direction)
            if length<1e-8:hidden=False;break
            direction/=length
            if normal@direction>=-1e-8:hidden=False;break
            h=np.cross(np.broadcast_to(direction,v.shape),v);det=np.einsum('ij,ij->i',u,h)
            valid=abs(det)>1e-12;inverse=np.zeros(len(det));inverse[valid]=1/det[valid]
            s=point-a;bary0=inverse*np.einsum('ij,ij->i',s,h);q=np.cross(s,u)
            bary1=inverse*(q@direction);distance=inverse*np.einsum('ij,ij->i',v,q)
            hit=valid&(bary0>=-1e-9)&(bary1>=-1e-9)&(bary0+bary1<=1+1e-9)&(distance>1e-6)
            if not hit.any():hidden=False;break
        removed[i]=hidden
    return removed

def audit_exterior_rays(triangles,removed,restore_visible=True):
    """Independent first-hit audit from exterior, including both end views."""
    lo=triangles.min(axis=(0,1));hi=triangles.max(axis=(0,1));origins=[];directions=[]
    for x in np.linspace(lo[0]+1e-5,hi[0]-1e-5,127):
        for angle in np.linspace(0,2*np.pi,96,endpoint=False):
            direction=np.array([0,np.cos(angle),np.sin(angle)])
            origins.append(np.array([x,0,(lo[2]+hi[2])/2])+direction*2)
            directions.append(-direction)
    for side in (-1,1):
        for y in np.linspace(lo[1],hi[1],33):
            for z in np.linspace(lo[2],hi[2],33):
                origins.append([(lo[0]-1) if side<0 else (hi[0]+1),y,z]);directions.append([-side,0,0])
    origins=np.asarray(origins);directions=np.asarray(directions)
    a=triangles[:,0];u=triangles[:,1]-a;v=triangles[:,2]-a
    changed=0;hits=0;maximum=0;restore=set()
    for start in range(0,len(origins),64):
        origin=origins[start:start+64,None,:];direction=directions[start:start+64,None,:]
        h=np.cross(direction,v);det=np.einsum('ij,kij->ki',u,h)
        valid=abs(det)>1e-12;inverse=np.zeros_like(det);inverse[valid]=1/det[valid]
        s=origin-a;b0=inverse*np.einsum('kij,kij->ki',s,h);q=np.cross(s,u)
        b1=inverse*np.sum(direction*q,axis=2);distance=inverse*np.einsum('ij,kij->ki',v,q)
        hit=valid&(b0>=-1e-9)&(b1>=-1e-9)&(b0+b1<=1+1e-9)&(distance>0)
        distance=np.where(hit,distance,np.inf);closest=distance.argmin(axis=1);old=distance.min(axis=1);distance[:,removed]=np.inf;new=distance.min(axis=1)
        finite=np.isfinite(old);hits+=int(finite.sum());error=abs(old[finite]-new[finite]);changed+=int((error>1e-7).sum())
        restore.update(closest[finite][error>1e-7].tolist())
        if len(error):maximum=max(maximum,float(error.max()))
    if changed and restore_visible:
        removed[list(restore)]=False
        verified=audit_exterior_rays(triangles,removed,False)
        verified.update(restored_exterior_triangles=sorted(restore),initial_candidate_changed_hits=changed,initial_candidate_max_difference_m=maximum)
        return verified
    assert changed==0,('exterior hull envelope changed',changed,maximum)
    return dict(rays=len(origins),hits=hits,changed_first_hits=changed,max_first_hit_difference_m=maximum,note='Dense independent ray audit; not a proof over every exterior ray.')

def main():
    OUT.mkdir(exist_ok=True)
    old=np.loadtxt(OLD,delimiter=',',skiprows=1)
    manifest=json.loads(OLD.with_suffix('.geometry.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(REPLAY/'model.mjb'))
    data=mujoco.MjData(model);data.qpos[:]=np.load(REPLAY/'initial_state.npz')['qpos'];mujoco.mj_forward(model,data)
    trajectory=np.loadtxt(REPLAY/'policy_trajectory.csv',delimiter=',',skiprows=1)
    assert trajectory[0,0]==0 and trajectory[-1,0]>=12-1e-9
    shell_v,shell_f,_=local_surface(model,data,'rowing/shell_white_visual',0)
    deck_v,deck_f,_=local_surface(model,data,'rowing/deck_blue_visual',0)
    shell=shell_v[shell_f];occluders=np.concatenate([shell,deck_v[deck_f]])
    inner_all=np.concatenate([interior_triangles(shell,shell,occluders),interior_triangles(deck_v[deck_f],shell,occluders)])
    hull_audit=audit_exterior_rays(occluders,inner_all)
    keep=np.ones(len(old),dtype=bool);tags=np.empty(len(old),dtype=np.int32)
    summary=[];cursor=0;max_error=0;hull_removed=[]
    for tag,surface in enumerate(manifest['surfaces']):
        name=surface['geometry'];part=surface['part']
        vertices,faces,orientation=local_surface(model,data,name,part)
        signature=hashlib.sha256(np.round(vertices,12).astype('<f8').tobytes()+faces.astype('<i8').tobytes()).hexdigest()
        assert signature==surface['sha256'],(name,'CAD identity mismatch')
        spacing=.0044 if name.endswith('_blade_visual') else .006
        inner=inner_all[:len(shell)] if name.endswith('shell_white_visual') else inner_all[len(shell):] if name.endswith('deck_blue_visual') else np.zeros(len(faces),dtype=bool)
        if name.endswith(('shell_white_visual','deck_blue_visual')):hull_removed.extend(inner.tolist())
        start=cursor
        for face_index,face in enumerate(faces):
            a,b,c=vertices[face];normal=np.cross(b-a,c-a);area=np.linalg.norm(normal)/2
            if area<=1e-14:continue
            normal/=2*area;n=max(1,int(np.ceil(np.sqrt(area/spacing**2))))
            uv=[]
            for i in range(n):
                for j in range(n-i):
                    uv.append(((i+1/3)/n,(j+1/3)/n))
                    if i+j<n-1:uv.append(((i+2/3)/n,(j+2/3)/n))
            uv=np.asarray(uv);count=len(uv);assert count==n*n
            expected=np.empty((count,8));expected[:,0]=part
            expected[:,1:4]=(1-uv.sum(1))[:,None]*a+uv[:,0,None]*b+uv[:,1,None]*c
            expected[:,4:7]=normal;expected[:,7]=area/count
            error=np.max(abs(expected-old[cursor:cursor+count]));max_error=max(max_error,float(error))
            assert error<2e-12,(name,cursor,error)
            if inner[face_index]:keep[cursor:cursor+count]=False
            cursor+=count
        stop=cursor;tags[start:stop]=tag;rows=old[start:stop]
        reason='retained exterior component; no resampling';clearance=None
        if inner.any():reason='omit inward-facing internal CAD layer only when all seven face probes are hidden from transverse exterior'
        if name.endswith(('cockpit_visual','rails_visual')):
            keep[start:stop]=False;reason='user-requested omission of boat interior/seat rail CAD'
        elif name.endswith(('_grip_visual','_shaft_visual')):
            candidate=np.ones(len(rows),dtype=bool) if '_grip_' in name else rows[:,2]*(1 if part==1 else -1)<0
            clearance=clearance_lower_bound(rows[candidate,1:4],part,trajectory)
            remove=candidate.copy();remove[candidate]=clearance>.10
            keep[start:stop][remove]=False
            reason='omit inboard handle-side points only when continuous replay clearance bound exceeds 0.10 m; retain entire outboard shaft'
        summary.append(dict(geometry=name,part=part,old_start=start,old_stop=stop,old_count=len(rows),kept_count=int(keep[start:stop].sum()),removed_count=int((~keep[start:stop]).sum()),removed_internal_triangles=np.flatnonzero(inner).tolist(),kept_area_m2=float(rows[keep[start:stop],7].sum()),reason=reason,minimum_candidate_clearance_lower_bound_m=None if clearance is None or not len(clearance) else float(clearance.min())))
    assert cursor==len(old)
    assert np.array_equal(inner_all,hull_removed)
    for s in summary:
        if s['geometry'].endswith('_blade_visual'):assert s['removed_count']==0
    result=old[keep];path=OUT/'policy_markers_exterior.csv'
    np.savetxt(path,result,delimiter=',',header='part,x,y,z,nx,ny,nz,area',comments='',fmt='%.17g')
    ids=np.flatnonzero(keep).astype('<u8');ids.tofile(OUT/'original_marker_ids.u64')
    np.savez_compressed(OUT/'selection.npz',keep=keep,geometry_tag=tags)
    report=dict(method='Verified subset of original quadrature; no change to kept locations, normals or areas',markers=len(result),original_markers=len(old),removed_markers=int((~keep).sum()),quadrature_max_abs_error=max_error,surfaces=summary,area_by_part_m2=[float(result[result[:,0]==part,7].sum()) for part in range(3)],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),render_geometry_unchanged=True,limitations=['The 0.10 m clearance is a static policy-envelope buffer, not a proven bound on waves or spray.','Omitted boat interior does not model water entering the cockpit.','Inner hull/underside deck classification uses seven transverse visibility probes per triangle, not a watertight Boolean CAD union; ambiguous faces remain.','This subset does not establish independence of remaining IB constraints or CFD stability.'])
    report['hull_exterior_ray_audit']=hull_audit
    path.with_suffix('.geometry.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)
    render(old,keep)

def render(old,keep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Microsoft YaHei'
    fig,axes=plt.subplots(3,2,figsize=(13,9),facecolor='white')
    names=['船体（局部坐标）','左桨（局部坐标）','右桨（局部坐标）']
    for part in range(3):
        selected=old[:,0]==part
        views=[(1,2,'俯视 x-y'),(2,3,'截面/侧视 y-z')] if part==0 else [(2,1,'俯视 y-x'),(2,3,'侧视 y-z')]
        for column,(a,b,label) in enumerate(views):
            ax=axes[part,column]
            for mask,color,name in [(selected&~keep,'#d8716b','删除'),(selected&keep,'#176a89','保留')]:
                points=old[mask][::max(1,int(mask.sum())//18000)]
                ax.scatter(points[:,a],points[:,b],s=.6,c=color,label=name,rasterized=True)
            ax.set_aspect('equal');ax.set_title(names[part]+' '+label);ax.set_xlabel('xyz'[a-1]+' / m');ax.set_ylabel('xyz'[b-1]+' / m');ax.legend(markerscale=6,loc='upper right',bbox_to_anchor=(1,1.65) if part else (1,1));ax.grid(alpha=.15)
            if part:ax.set_ylim(min(old[selected,b].min()-.08,-.22),max(old[selected,b].max()+.08,.22))
    fig.suptitle('IB 点精简：红色删除，蓝色保留；两片桨叶和外侧桨杆完整保留')
    fig.tight_layout();fig.savefig(OUT/'ib_selection.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
