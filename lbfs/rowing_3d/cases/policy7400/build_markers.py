"""New-model quadrature with the previously reviewed exterior/dry selection rules."""
from pathlib import Path
import sys,os,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import mujoco
from export_policy_ib_assembly import body_ancestors,sample_triangle
from reduce_policy_ib_geometry import local_surface,interior_triangles,audit_exterior_rays
from trim_high_policy_ib import clearance

def main():
    replay=HERE/'replay';m=mujoco.MjModel.from_binary_path(str(replay/'model.mjb'));d=mujoco.MjData(m)
    d.qpos[:]=np.load(replay/'initial_state.npz')['qpos'];mujoco.mj_forward(m,d)
    traj=np.loadtxt(replay/'policy_trajectory.csv',delimiter=',',skiprows=1);water=json.loads((replay/'metadata.json').read_text())['water_level_m']
    roots=[m.body('rowing/'+n).id for n in ('boat','port_oar','starboard_oar')]
    sv,sf,_=local_surface(m,d,'rowing/shell_white_visual',0);dv,df,_=local_surface(m,d,'rowing/deck_blue_visual',0)
    shell=sv[sf];hull=np.concatenate([shell,dv[df]])
    inner=np.r_[interior_triangles(shell,shell,hull),interior_triangles(dv[df],shell,hull)]
    audit=audit_exterior_rays(hull,inner)
    records=[];surfaces=[];seen=set()
    for part,root in enumerate(roots):
        fixed={root}
        for bid in range(root+1,m.nbody):
            if int(m.body_parentid[bid]) in fixed and int(m.body_jntnum[bid])==0:fixed.add(bid)
        for g in range(m.ngeom):
            name=m.geom(g).name or ''
            if m.geom_bodyid[g] not in fixed or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            v,f,orientation=local_surface(m,d,name,part)
            key=(part,hashlib.sha256(np.round(v,12).astype('<f8').tobytes()+f.astype('<i8').tobytes()).hexdigest())
            if key in seen:continue
            seen.add(key)
            if name.endswith(('cockpit_visual','rails_visual')):
                surfaces.append(dict(name=name,part=part,kept=0,reason='boat interior/rails omitted'));continue
            spacing=.00175 if name.endswith('_blade_visual') else .0035 if name.endswith('_shaft_visual') else .006
            mask=inner[:len(sf)] if name.endswith('shell_white_visual') else inner[len(sf):] if name.endswith('deck_blue_visual') else np.zeros(len(f),dtype=bool)
            rows=[]
            for face,omit in zip(f,mask):
                if omit:continue
                for p,n,a in sample_triangle(*v[face],spacing):rows.append([part,*p,*n,a])
            a=np.asarray(rows);keep=np.ones(len(a),dtype=bool)
            if len(a) and not name.endswith('_blade_visual'):
                height=clearance(a[:,1:4],part,traj,water);keep &= height<=.4
                if name.endswith('_grip_visual'):keep &= height<=.1
                if name.endswith('_shaft_visual'):keep &= ~((a[:,2]*(1 if part==1 else -1)<0)&(height>.1))
            records.append(a[keep]);surfaces.append(dict(name=name,part=part,spacing_m=spacing,kept=int(keep.sum()),removed_high=int((~keep).sum()),removed_inner_triangles=int(mask.sum()),area_m2=float(a[keep,7].sum()),orientation=orientation))
            print(name,int(keep.sum()),flush=True)
    a=np.concatenate(records);assert np.isfinite(a).all() and np.all(a[:,7]>0)
    assert np.max(abs(np.linalg.norm(a[:,4:7],axis=1)-1))<1e-6
    path=HERE/'policy_markers_exterior.csv';np.savetxt(path,a,delimiter=',',header='part,x,y,z,nx,ny,nz,area',comments='',fmt='%.17g')
    report=dict(markers=len(a),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_sha256=hashlib.sha256((replay/'model.mjb').read_bytes()).hexdigest(),surfaces=surfaces,hull_ray_audit=audit,
                selection='new CAD + old audited selection rules; retain blades completely; exclude hidden interior, cockpit/rails, provably high handles; fixed 0.4m high omission',
                limitation='Initial-water clearance does not prove maximum wave height. CAD exterior ray classification is not watertight Boolean union.')
    path.with_suffix('.geometry.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('TOTAL',len(a),flush=True)

if __name__=='__main__':main()
