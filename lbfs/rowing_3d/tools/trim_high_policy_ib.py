"""Fixed high-point omission from the already reviewed exterior IB subset.

Uses the same conservative continuous nlerp/translation height bound as
reduce_policy_ib_geometry.py; retained CSV rows are copied byte-for-byte.
No runtime activation/deactivation is introduced.
"""
import argparse,hashlib,json,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def clearance(points,part,trajectory,water):
    off=1+13*part;q=trajectory[:,off+3:off+7].copy();q/=np.linalg.norm(q,axis=1)[:,None]
    r=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    theta=2*np.arccos(np.clip(abs(np.sum(q[:-1]*q[1:],axis=1)),0,1));z=trajectory[:,off+2]
    result=np.empty(len(points))
    for start in range(0,len(points),512):
        p=points[start:start+512];height=r[:,2,:]@p.T+z[:,None]
        bound=np.minimum(height[:-1],height[1:])-.5*abs(np.diff(z))[:,None]
        bound-=2*np.sin(theta[:,None]/4)*np.linalg.norm(p,axis=1)[None,:]
        result[start:start+len(p)]=bound.min(axis=0)-water
    return result

def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--clearance',type=float,default=.4);args=ap.parse_args()
    if not np.isfinite(args.clearance) or args.clearance<=0:raise ValueError('positive finite clearance required')
    src=root/'designs/ib_exterior_policy1100_20261106';path=src/'policy_markers_exterior.csv'
    manifest=json.loads(path.with_suffix('.geometry.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    trajectory_path=root/'renders/policy_replay/final_policy_1100/policy_trajectory.csv'
    trajectory=np.loadtxt(trajectory_path,delimiter=',',skiprows=1)
    assert trajectory[0,0]==0 and trajectory[-1,0]>=12 and np.all(np.diff(trajectory[:,0])>0)
    a=np.loadtxt(path,delimiter=',',skiprows=1);low=np.empty(len(a));water=-.12027640640735626
    for part in range(3):
        ids=np.flatnonzero(a[:,0]==part);low[ids]=clearance(a[ids,1:4],part,trajectory,water)
    keep=low<=args.clearance;ids=np.fromfile(src/'original_marker_ids.u64',dtype='<u8');summary=[]
    for s in manifest['surfaces']:
        mask=(ids>=s['old_start'])&(ids<s['old_stop'])
        if s['geometry'].endswith('_blade_visual') and np.any(mask&~keep):raise ValueError('threshold would remove blade points')
        summary.append(dict(geometry=s['geometry'],before=int(mask.sum()),after=int((mask&keep).sum()),removed=int((mask&~keep).sum())))
    args.output.mkdir(parents=True,exist_ok=False);out=args.output/'policy_markers_exterior_hightrim.csv'
    lines=path.read_bytes().splitlines(keepends=True);assert len(lines)==len(a)+1
    out.write_bytes(lines[0]+b''.join(line for line,retained in zip(lines[1:],keep) if retained))
    ids[keep].tofile(args.output/'original_marker_ids.u64');np.savez_compressed(args.output/'selection.npz',keep=keep,minimum_clearance_m=low)
    report=dict(markers=int(keep.sum()),source_markers=len(a),additionally_removed=int((~keep).sum()),threshold_above_initial_water_m=args.clearance,initial_water_level_m=water,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_sha256=manifest['sha256'],trajectory_sha256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),surfaces=summary,retained_rows_byte_identical=True,blades_complete=True,dynamic_reactivation=False,limitations=['The replay bound is relative to initial water, not a proof of the maximum wave/spray height.'])
    out.with_suffix('.geometry.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
