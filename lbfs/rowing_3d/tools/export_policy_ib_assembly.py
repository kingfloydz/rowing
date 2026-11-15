"""Export policy replay geometry and rigid 6DOF trajectories for the LBFS IB solver.

The MuJoCo replay already contains the boat free pose and all oar joint poses.
This utility resolves them into three world rigid bodies: hull, port oar and
starboard oar.  Surface triangles are quadrature-sampled in each part's local
frame, while the trajectory CSV stores the corresponding world pose and
velocities.  The C++ solver then transforms the local markers on the device at
each RK stage and never integrates solid dynamics.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import mujoco

def quat_conj(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])
def qmul(a,b):
    w,x,y,z=a; W,X,Y,Z=b
    return np.array([w*W-x*X-y*Y-z*Z,w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W])
def body_ancestors(m,bid):
    out=[]
    while bid:
        out.append(bid); bid=int(m.body_parentid[bid])
    return out
def sample_triangle(a,b,c,spacing):
    area=.5*np.linalg.norm(np.cross(b-a,c-a))
    if area<=1e-14:return []
    n=np.cross(b-a,c-a);n/=np.linalg.norm(n)
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('spacing must be positive')
    q=max(1,int(np.ceil(np.sqrt(area/(spacing*spacing)))))
    pts=[]
    # q^2 equal-area subtriangles; their centroids integrate linear fields
    # exactly. Both orientations are needed to cover the parent triangle.
    for i in range(q):
        for j in range(q-i):
            u=(i+1/3)/q;v=(j+1/3)/q
            pts.append((1-u-v)*a+u*b+v*c)
            if i+j<q-1:
                u=(i+2/3)/q;v=(j+2/3)/q
                pts.append((1-u-v)*a+u*b+v*c)
    return [(p,n,area/(q*q)) for p in pts]

def consistent_winding(vertices, faces):
    """Orient manifold-adjacent faces consistently, preserving majority area.

    Open/nonmanifold CAD is not closed or given an invented solid interior.
    Its global orientation is retained. Closed meshes additionally have
    their orientation checked by signed volume.
    """
    faces=faces.copy();edges={};adj=[[] for _ in faces]
    for fi,tri in enumerate(faces):
        for a,b in zip(tri,np.roll(tri,-1)):
            key=(min(int(a),int(b)),max(int(a),int(b)))
            edges.setdefault(key,[]).append((fi,int(a)<int(b)))
    for owners in edges.values():
        if len(owners)==2:
            (a,da),(b,db)=owners;change=int(da==db)
            adj[a].append((b,change));adj[b].append((a,change))
    area=.5*np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)
    flip=np.full(len(faces),-1,dtype=np.int8)
    for start in range(len(faces)):
        if flip[start]>=0:continue
        flip[start]=0;queue=[start]
        for a in queue:
            for b,change in adj[a]:
                wanted=int(flip[a])^change
                if flip[b]<0:flip[b]=wanted;queue.append(b)
                elif flip[b]!=wanted:raise ValueError('CAD contains non-orientable manifold adjacency')
        ids=np.array(queue)
        if area[ids[flip[ids]==1]].sum()>area[ids[flip[ids]==0]].sum():flip[ids]^=1
    changed=np.flatnonzero(flip==1)
    faces[changed]=faces[changed][:,[0,2,1]]
    # Do not infer outward orientation from a volume integral on open meshes.
    closed=all(len(v)==2 for v in edges.values())
    signed_volume=float(np.einsum('ij,ij->i',vertices[faces[:,0]],np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()/6)
    if closed and signed_volume<0:
        faces=faces[:,[0,2,1]];flip^=1;signed_volume=-signed_volume
    changed=np.flatnonzero(flip==1)
    return faces,dict(flipped_triangles=changed.tolist(),flipped_area_m2=float(area[changed].sum()),
        closed=closed,boundary_edges=sum(len(v)==1 for v in edges.values()),
        nonmanifold_edges=sum(len(v)>2 for v in edges.values()),signed_volume_m3=signed_volume)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--rollout',type=Path,required=True);ap.add_argument('--markers',type=Path,required=True);ap.add_argument('--trajectory',type=Path);ap.add_argument('--spacing',type=float,default=.005)
    ap.add_argument('--blade-spacing',type=float,help='Separate area-equivalent spacing for labelled blade_visual meshes; leaves hull and shafts unchanged')
    ap.add_argument('--markers-only',action='store_true',help='Export geometry without rewriting an existing verified trajectory')
    ap.add_argument('--parts',nargs=3,default=['rowing/boat','rowing/port_oar','rowing/starboard_oar'])
    args=ap.parse_args()
    if not args.markers_only and args.trajectory is None:ap.error('--trajectory is required unless --markers-only')
    if not np.isfinite(args.spacing) or args.spacing<=0:ap.error('--spacing must be positive')
    if args.blade_spacing is not None and (not np.isfinite(args.blade_spacing) or args.blade_spacing<=0):ap.error('--blade-spacing must be positive')
    m=mujoco.MjModel.from_binary_path(str(args.model));z=np.load(args.rollout);qpos=z['qpos'];times=z['time']
    if qpos.ndim!=2 or qpos.shape[0]<2:raise ValueError('rollout must contain at least two qpos frames')
    # rollout.npz stores states after each policy step.  Add the MuJoCo reset
    # state at t=0 so the fluid and policy trajectories share the same origin.
    if float(times[0])>1e-12:
        initial=args.rollout.parent/'initial_state.npz'
        if not initial.exists():raise ValueError('Actual reset state initial_state.npz is required; qpos0 is not an observed state')
        qpos=np.vstack([np.load(initial)['qpos'],qpos]);times=np.concatenate([[0.0],times])
    if not np.all(np.isfinite(qpos)) or not np.all(np.diff(times)>0):raise ValueError('Invalid trajectory samples')
    ids=[m.body(name).id for name in args.parts]
    part_bodies=[]
    for root in ids:
        fixed={root}
        for bid in range(root+1,m.nbody):
            if int(m.body_parentid[bid]) in fixed and int(m.body_jntnum[bid])==0:fixed.add(bid)
        part_bodies.append(fixed)
    # Convert every visual mesh in the selected body subtree to body-local
    # surface markers.  Mesh geoms contain both hull and oar shaft/blade CAD.
    markers=[];d=mujoco.MjData(m);surfaces=[];duplicates=[];seen={}
    d.qpos[:]=qpos[0];mujoco.mj_forward(m,d)
    for part,bodies in enumerate(part_bodies):
        for g in range(m.ngeom):
            bid=int(m.geom_bodyid[g])
            if bid not in bodies or int(m.geom_type[g])!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mid=int(m.geom_dataid[g]);va=int(m.mesh_vertadr[mid]);vn=int(m.mesh_vertnum[mid]);fa=int(m.mesh_faceadr[mid]);fn=int(m.mesh_facenum[mid])
            # Faces index this mesh's vertex slice, not the global vertex pool.
            verts=m.mesh_vert[va:va+vn];R=d.geom_xmat[g].reshape(3,3);p=d.geom_xpos[g].copy();root_bid=ids[part];Rb=d.xmat[root_bid].reshape(3,3);pb=d.xpos[root_bid].copy()
            local=(verts@R.T+p-pb)@Rb
            faces,orientation=consistent_winding(local,m.mesh_face[fa:fa+fn].astype(np.int64))
            # MuJoCo may use the identical CAD surface twice: once for visual
            # rendering and once for collision. It is one physical IB surface.
            # Deduplicate in the rigid part's frame before quadrature; different
            # sampling spacings otherwise hide the duplication at point level.
            signature=hashlib.sha256(np.round(local,12).astype('<f8').tobytes()+faces.astype('<i8').tobytes()).hexdigest()
            key=(part,signature)
            area=float(.5*np.linalg.norm(np.cross(local[faces[:,1]]-local[faces[:,0]],local[faces[:,2]]-local[faces[:,0]]),axis=1).sum())
            if key in seen:
                duplicates.append(dict(part=part,geometry=m.geom(g).name,duplicate_of=seen[key],area_m2=area))
                continue
            seen[key]=m.geom(g).name
            surfaces.append(dict(part=part,geometry=m.geom(g).name,area_m2=area,triangles=int(fn),sha256=signature,orientation=orientation))
            spacing=args.blade_spacing if args.blade_spacing is not None and (m.geom(g).name or '').endswith('_blade_visual') else args.spacing
            for k in range(fn):
                tri=faces[k]
                if tri.min()<0 or tri.max()>=vn:raise ValueError('Mesh face index out of local bounds')
                for pt,n,a in sample_triangle(local[tri[0]],local[tri[1]],local[tri[2]],spacing):
                    markers.append([part,float(pt[0]),float(pt[1]),float(pt[2]),float(n[0]),float(n[1]),float(n[2]),float(a)])
    if set(int(row[0]) for row in markers)!={0,1,2}:raise ValueError('Missing hull or oar surface geometry')
    args.markers.parent.mkdir(parents=True,exist_ok=True)
    with args.markers.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['part','x','y','z','nx','ny','nz','area']);w.writerows(markers)
    args.markers.with_suffix('.geometry.json').write_text(json.dumps(dict(
        method='Exact body-local vertex/face identity rounded to 1e-12 m; remove duplicate CAD surfaces before quadrature',
        surfaces=surfaces,excluded_duplicate_surfaces=duplicates,markers=len(markers),
        area_by_part_m2=[sum(row[7] for row in markers if row[0]==part) for part in range(3)],
        sha256=hashlib.sha256(args.markers.read_bytes()).hexdigest()),indent=2))
    if args.markers_only:
        print(f'markers={len(markers)}; trajectory unchanged');return
    # World pose of each selected rigid body.  Body 41/43 already include the
    # policy oar joints, so their world transforms are the desired 6DOF state.
    if set(int(row[0]) for row in markers)!={0,1,2}:raise ValueError('Missing hull or oar surface geometry')
    rows=[]
    for k,t in enumerate(times):
        d.qpos[:]=qpos[k];mujoco.mj_forward(m,d); row=[float(t)]
        for bid in ids:
            row.extend(d.xpos[bid].tolist());row.extend(d.xquat[bid].tolist());row.extend([0.,0.,0.])
            row.extend([0.,0.,0.])
        rows.append(row)
    for i in range(len(rows)):
        if i==0:ip=0;inext=1
        elif i==len(rows)-1:ip=i-1;inext=i
        else:ip=i-1;inext=i+1
        dt=float(times[inext]-times[ip]);
        for part in range(3):
            off=1+part*13; qa=np.array(rows[ip][off+3:off+7]);qb=np.array(rows[inext][off+3:off+7]);
            if(float(np.dot(qa,qb))<0):qb=-qb
            rows[i][off+7:off+10]=((np.array(rows[inext][off:off+3])-np.array(rows[ip][off:off+3]))/dt).tolist()
            dq=qmul(qb,quat_conj(qa));sn=np.linalg.norm(dq[1:]);angle=2*np.arctan2(sn,dq[0]);omega=dq[1:]*(angle/(sn*dt) if sn>1e-12 else 2/dt)
            rows[i][off+10:off+13]=omega.tolist()
    with args.trajectory.open('w',newline='') as f:
        w=csv.writer(f);header=['time']
        for name in ('hull','port_oar','starboard_oar'):header += [name+s for s in ('_px','_py','_pz','_qw','_qx','_qy','_qz','_vx','_vy','_vz','_wx','_wy','_wz')]
        w.writerow(header);w.writerows(rows)
    print(f'markers={len(markers)} trajectory_frames={len(rows)}')
if __name__=='__main__':main()
