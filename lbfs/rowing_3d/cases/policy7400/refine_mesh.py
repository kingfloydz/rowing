"""User-requested local density changes; keep the measured blade plateaus."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
HERE=Path(__file__).resolve().parent

def axis(domain,fine,bands,h,outer=None):
    snapped=sorted([(np.floor(a/h)*h,np.ceil(b/h)*h) for a,b in fine])
    merged=[]
    for a,b in snapped:
        if merged and a<=merged[-1][1]+h*1e-8:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    all_bands=[(a,b,h) for a,b in merged]+bands
    def distance_coordinate(x):
        # Continuous size field; accelerate growth only outside the two blades.
        out=np.log(1.06)*x
        if outer is not None:out+=(np.log(1.08)-np.log(1.06))*(np.minimum(x-outer[0],0)+np.maximum(x-outer[1],0))
        return out
    pieces=[];cursor=domain[0]
    for a,b in merged+[[domain[1],domain[1]]]:
        if a>cursor:
            xx=np.linspace(cursor,a,240001);sx=distance_coordinate(xx);hh=np.full_like(xx,.8)
            for lo,hi,target in all_bands:
                distance=np.maximum(np.maximum(distance_coordinate(lo)-sx,sx-distance_coordinate(hi)),0)
                hh=np.minimum(hh,target+distance)
            mon=np.r_[0,np.cumsum(np.diff(xx)*.5*(1/hh[1:]+1/hh[:-1]))]
            pieces.append(np.interp(np.linspace(0,mon[-1],int(np.ceil(mon[-1]*1.002))+1),mon,xx))
        if b>a:pieces.append(a+np.arange(round((b-a)/h)+1)*h)
        cursor=b
    edges=np.concatenate([p if i==0 else p[1:] for i,p in enumerate(pieces)])
    w=np.diff(edges);ratios=np.maximum(w[1:]/w[:-1],w[:-1]/w[1:]);assert np.all(w>0) and ratios.max()<=1.0801
    if outer is None:assert ratios.max()<=1.0601
    regions=[]
    for a,b in merged:
        k=np.flatnonzero((edges[:-1]>=a-1e-10)&(edges[1:]<=b+1e-10));assert np.max(abs(w[k]-h))<1e-11
        regions.append(dict(bounds_m=[a,b],cell_begin=int(k[0]),cell_end_exclusive=int(k[-1]+1)))
    return edges,regions,dict(n=len(w),min_m=float(w.min()),max_m=float(w.max()),max_adjacent_ratio=float(ratios.max()))

def main():
    archive=HERE/'before_force_mesh_update'
    if not archive.exists():
        archive.mkdir()
        for name in ('mesh_design.json','axis_x.txt','axis_y.txt','axis_z.txt','mesh_review.png'):
            shutil.copy2(HERE/name,archive/name)
    old=json.loads((archive/'mesh_design.json').read_text());r=dict(old);h=old['h_blade_m']
    hull=r['envelope']['shell_white_visual'];lo=np.array(hull['low_m']);hi=np.array(hull['high_m'])
    fine=[[old['uniform_blocks'][0]['x']['bounds_m']],[b['y']['bounds_m'] for b in old['uniform_blocks']],[old['uniform_blocks'][0]['z']['bounds_m']]]
    inside_left=fine[1][0][1];inside_right=fine[1][1][0]
    bands=[[(lo[0]-.15,hi[0]+.10,.012),(lo[0]-1.15,lo[0]-.15,.020),(lo[0]-3.55,lo[0]-1.15,.040),(-12.4,lo[0]-3.55,.080)],
           [(lo[1]-.15,hi[1]+.15,.012),(inside_left,inside_right,.016)],[]]
    for a in (0,1):
        for shaft in r['wet_shaft_envelope'].values():bands[a].append((shaft['low_m'][a]-.03,shaft['high_m'][a]+.03,.007))
    outer=(fine[1][0][0],fine[1][-1][1]);results=[axis(d,f,b,h,outer if a==1 else None) for a,(d,f,b) in enumerate(zip(old['bounds_m'],fine,bands))]
    dims=[len(v[0])-1 for v in results]
    for a,res in zip('xyz',results):np.savetxt(HERE/f'axis_{a}.txt',res[0],fmt='%.17g')
    r.update(cells=int(np.prod(dims)),dimensions=dims,bands_m=bands,axes=[v[2] for v in results],
             uniform_blocks=[dict(x=results[0][1][0],y=y,z=results[2][1][0],h_m=h) for y in results[1][1]],
             cells_with_internal_halo_4x2_h12=(dims[0]+72)*(dims[1]+24)*dims[2],
             refinement_revision=3,inflow_fine_end_m=float(hi[0]+.10),outer_y_growth_limit=1.08,
             validation='Same measured blade envelope/plateaus as revision1. Continuous size field, <=6% width growth except <=8% beyond outside blade edges. Positive widths and exact 3.5mm plateaus checked.',
             axis_sha256={a:hashlib.sha256((HERE/f'axis_{a}.txt').read_bytes()).hexdigest() for a in 'xyz'})
    (HERE/'mesh_design.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    print(json.dumps(dict(cells=r['cells'],dimensions=dims,halo_cells=r['cells_with_internal_halo_4x2_h12'],axes=r['axes'],inflow_end=r['inflow_fine_end_m']),indent=2))

if __name__=='__main__':main()
