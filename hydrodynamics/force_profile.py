"""Small, versioned force-law profiles; no runtime trajectory dependency."""
import json
from pathlib import Path
import math
import hashlib

from .models import ROOT


def load_force_profile(profile=None, repository=ROOT):
    if profile == 'prior':
        return dict(schema_version=1,status='engineering_prior',blade_cd_max=2.,
                    blade_cl_max=.9,blade_skin_cd=.01)
    path=Path(profile) if profile is not None else Path(repository)/'data/cfd_force_calibration.json'
    if profile is None and not path.exists():
        return load_force_profile('prior',repository)
    p=json.loads(path.read_text(encoding='utf-8'))
    if profile is None and p.get('default_enabled',True) is False:
        prior=load_force_profile('prior',repository)
        prior['candidate_disabled_reason']=p.get('disabled_reason','Candidate is opt-in only')
        return prior
    if p.get('schema_version') != 1:
        raise ValueError('Unknown force profile schema')
    for k,lo,hi in (('blade_cd_max',.5,6.),('blade_cl_max',0.,1.8),('blade_skin_cd',0.,.1)):
        value=float(p[k])
        if not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Invalid calibrated coefficient: '+k)
    if p.get('left_right_shared',True) is not True:
        raise ValueError('CFD force profile must share coefficients across oars')
    for name,expected in p.get('required_blade_inertia',{}).items():
        actual=json.loads((Path(repository)/'data/blade_inertia.json').read_text())[name]
        if actual != expected:
            raise ValueError('Calibration inertia prior mismatch: '+name)
    for side,expected in p.get('source',{}).get('geometry_sha256',{}).items():
        if side not in ('port','starboard'):
            raise ValueError('Unknown blade side in calibration')
        mesh=Path(repository)/'boat_meshes'/f'{side}_blade_3mm.stl'
        if hashlib.sha256(mesh.read_bytes()).hexdigest() != expected:
            raise ValueError('Calibration blade geometry mismatch: '+side)
    return p
