"""Run one process controlling eight local GPUs with portable, verified inputs."""
from pathlib import Path
import argparse, gzip, hashlib, json, os, subprocess
ROOT=Path(__file__).resolve().parents[1]

def prepare(case):
    for name,digest in json.loads((case/'inputs.sha256.json').read_text()).items():
        if hashlib.sha256((case/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Input checksum mismatch: '+name)
    for name in ('policy_markers_exterior.csv','replay/model.mjb'):
        target=case/name
        data=gzip.decompress((case/(name+'.gz')).read_bytes())
        if target.exists():
            if target.read_bytes()!=data:raise ValueError('Existing input differs: '+name)
        else:target.write_bytes(data)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',type=Path,default=ROOT/'cases/policy7400')
    ap.add_argument('--solver',type=Path,default=ROOT/'build/lu2025')
    ap.add_argument('--output',type=Path,default=ROOT/'runs/policy7400')
    ap.add_argument('--end-s',type=float)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--x-cuts',default='303,625,952')
    ap.add_argument('--checkpoint-every',type=int,default=50000)
    a=ap.parse_args();case=a.case.resolve();prepare(case)
    if a.prepare_only:return
    solver=a.solver.resolve();assert solver.is_file(),solver
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if len(names)<8:raise RuntimeError(f'Eight local GPUs required; found {len(names)}')
    print('GPU models:',names,flush=True) # No GPU model restriction.
    cfg=dict(line.split(maxsplit=1) for line in (case/'case.cfg').read_text().splitlines() if line.strip() and not line.startswith('#'))
    for key in ('mesh_axis_x','mesh_axis_y','mesh_axis_z','policy_motion_csv','ale_motion_csv','policy_markers_csv'):
        cfg[key]=str((case/cfg[key]).resolve())
    if a.end_s is not None:
        if not 0<a.end_s<=11.6:raise ValueError('Duration must lie in (0,11.6]')
        cfg['end_s']=str(a.end_s)
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    config=output/'case.cfg';config.write_text('\n'.join(k+' '+v for k,v in cfg.items())+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('LU_')}
    env.update(LU_COLUMN_PHASE_REPAIR='1',LU_PENCIL_X_CUTS=a.x_cuts,LU_CHECKPOINT_EVERY=str(a.checkpoint_every))
    cmd=[str(solver),str(config),'--multi-gpu','8','--decomp','xy','--output',str(output/'output')]
    (output/'launch.json').write_text(json.dumps(dict(command=cmd,gpu_models=names,solver_sha256=hashlib.sha256(solver.read_bytes()).hexdigest(),environment={k:v for k,v in env.items() if k.startswith('LU_')}),indent=2))
    with (output/'solver.log').open('w') as log:
        r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    (output/'exit_code.txt').write_text(str(r.returncode)+'\n')
    raise SystemExit(r.returncode)
if __name__=='__main__':main()
