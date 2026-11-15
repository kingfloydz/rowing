"""Read v2/v3 surface streams; merge same-time ranks into world-coordinate PLY.

Requires numpy. Each frame has <4I63f header, followed by count*3 float32.
The vertex count is divisible by three; every consecutive triple is a face.
v2: last-stage IB reaction in ALE basis; v3: exterior stress in world basis.
"""
import argparse
import json
import struct
from pathlib import Path
import numpy as np

HEADER = struct.Struct('<4I63f')

def frames(path):
    with Path(path).open('rb') as f:
        last=-float('inf')
        while True:
            raw=f.read(HEADER.size)
            if not raw: return
            if len(raw)!=HEADER.size: raise ValueError(f'{path}: truncated header')
            h=HEADER.unpack(raw)
            magic,version,count,reserved=h[:4]
            if magic!=0x3153564c or version not in (2,3) or count%3: raise ValueError(f'{path}: invalid surface header')
            time,cx,cy,yaw,ax,ay=h[4:10]
            if not np.isfinite(h[4:]).all() or time<=last: raise ValueError(f'{path}: invalid frame time/metadata')
            last=time
            # Bound read by remaining file length, including malicious counts.
            if count*12>Path(path).stat().st_size-f.tell(): raise ValueError(f'{path}: truncated vertices')
            v=np.frombuffer(f.read(count*12),dtype='<f4').reshape(-1,3).copy()
            if not np.isfinite(v).all(): raise ValueError(f'{path}: invalid vertices')
            x=v[:,0].copy()-ax;y=v[:,1].copy()-ay
            c,s=np.cos(yaw),np.sin(yaw)
            v[:,0]=cx+c*x-s*y;v[:,1]=cy+s*x+c*y
            yield {'version':version,'time_s':time,'ale':[cx,cy,yaw,ax,ay],
                   'world_poses':np.array(h[10:49]).reshape(3,13).tolist(),
                   ('exterior_force_world' if version==3 else 'last_stage_ib_reaction_ale'):np.array(h[49:67]).reshape(3,6).tolist()},v

def ply(path,vertices):
    n=len(vertices)
    if n%3 or n>=2**31: raise ValueError('PLY face index capacity exceeded')
    header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\n'
            'property float x\nproperty float y\nproperty float z\n'
            f'element face {n//3}\nproperty list uchar int vertex_indices\nend_header\n')
    faces=np.empty(n//3,dtype=[('size','u1'),('indices','<i4',(3,))])
    faces['size']=3;faces['indices']=np.arange(n,dtype=np.int32).reshape(-1,3)
    with Path(path).open('xb') as f:
        f.write(header.encode('ascii'));f.write(vertices.astype('<f4',copy=False).tobytes());f.write(faces.tobytes())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('streams',type=Path,nargs='+');ap.add_argument('--output',type=Path)
    ap.add_argument('--validate-only',action='store_true',help='Check all frames without creating large duplicate PLY files')
    a=ap.parse_args()
    if not a.validate_only:
        if a.output is None:ap.error('--output is required unless --validate-only')
        a.output.mkdir(parents=True,exist_ok=False)
    sources=[iter(frames(p)) for p in a.streams];index=0
    while True:
        batch=[next(s,None) for s in sources]
        if all(v is None for v in batch):break
        if any(v is None for v in batch):raise ValueError('rank streams have different frame counts')
        ref=batch[0][0]
        for meta,_ in batch[1:]:
            if meta['version']!=ref['version'] or meta['world_poses']!=ref['world_poses'] or meta['time_s']!=ref['time_s'] or meta['ale']!=ref['ale']:raise ValueError('rank metadata mismatch')
        if a.validate_only:
            print(json.dumps({'frame':index,'time_s':ref['time_s'],'rank_count':len(batch),'vertices':sum(len(v) for _,v in batch),'finite_and_synchronized':True}))
            index+=1;continue
        ply(a.output/f'surface_{index:05d}.ply',np.concatenate([v for _,v in batch]))
        meta={'time_s':ref['time_s'],'ale':ref['ale'],'world_poses':ref['world_poses'],
              'version':ref['version'], 'force_method':'exterior_stress_world' if ref['version']==3 else 'last_stage_ib_reaction_ale', 'body_loads':np.sum([m['exterior_force_world' if ref['version']==3 else 'last_stage_ib_reaction_ale'] for m,_ in batch],axis=0).tolist()}
        (a.output/f'surface_{index:05d}.json').write_text(json.dumps(meta),encoding='utf-8');index+=1
    print(f'{index} frames converted')

if __name__=='__main__':main()
