import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np
import warp as wp

from hydrodynamics.models import ROOT
from hydrodynamics.cfd_calibration import (read_archive, interpolate_trajectory,
    align_forces, oar_basis)
from hydrodynamics.force_profile import load_force_profile
from hydrodynamics.training import TrainingParameters, blade_traction
from rowing_mjlab.hydro import WarpHydrodynamics, blade_force_coefficients


@wp.kernel
def force_samples(velocity:wp.array(dtype=wp.vec3),coeff:wp.vec3,result:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    result[i]=blade_force_coefficients(velocity[i],wp.vec3(1.,0.,0.),wp.vec3(0.,1.,0.),.1,1000.,coeff)


class CfdCalibrationTests(unittest.TestCase):
    def test_archive_integrity(self):
        text='kind,id,x_m,y_m,z_m,fx_N,fy_N,fz_N,mx_Nm,my_Nm,mz_Nm\n'
        text+='\n'.join(f'{name},{i},0,0,0,1,2,3,4,5,6' for i,name in enumerate(('hull','port_oar','starboard_oar')))
        manifest=dict(source_segments=[{}],records=[dict(path='frame.csv',sha256=hashlib.sha256(text.encode()).hexdigest(),time_output_reconstructed_s=.01,segment=0)])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.zip'
            def write(payload):
                with zipfile.ZipFile(path,'w') as z:
                    z.writestr('frame.csv',payload);z.writestr('manifest.json',json.dumps(manifest));z.writestr('raw/segment_0/case.cfg','water_level_m 0')
            write(text)
            self.assertEqual(read_archive(path)[2].shape,(1,3,9))
            write(text+'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):read_archive(path)

    def test_ale_rotation_and_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ale.csv'
            path.write_text('t_s,Cx_m,Cy_m,yaw_rad\n0,10,20,1.5707963267948966\n1,10,20,1.5707963267948966\n')
            raw=np.zeros((2,3,9));raw[:,:,:3]=[1,0,2];raw[:,:,3:6]=[3,0,0];raw[:,:,6:9]=[0,4,0]
            states=[dict(position=np.tile([10,21,2],(2,1))) for _ in range(3)]
            wrench,error=align_forces(raw,states,path,np.array([0.,1.]))
            np.testing.assert_allclose(wrench[0,0],[0,3,0,-4,0,0],atol=1e-12)
            self.assertLess(error.max(),1e-12)

    def test_force_law_passivity_and_cpu_warp_parity(self):
        p=load_force_profile(ROOT/'data/cfd_force_calibration.json');coeff=wp.vec3(p['blade_cd_max'],p['blade_cl_max'],p['blade_skin_cd'])
        rng=np.random.default_rng(210);v=rng.normal(size=(120,3)).astype(np.float32)*4;v[0]=0
        out=wp.zeros(len(v),dtype=wp.vec3,device='cpu')
        wp.launch(force_samples,len(v),inputs=[wp.array(v,dtype=wp.vec3,device='cpu'),coeff,out],device='cpu')
        f=out.numpy()
        self.assertTrue(np.all(np.sum(v*f,axis=1)<=1e-4))
        np.testing.assert_array_equal(f[0],np.zeros(3))
        prior=TrainingParameters(blade_cd_max=p['blade_cd_max'],blade_cl_max=p['blade_cl_max'],blade_skin_cd=p['blade_skin_cd'])
        expected=blade_traction(v,np.array([1.,0,0]),np.array([0.,1,0]),.1,p=prior)
        np.testing.assert_allclose(f,expected,rtol=2e-5,atol=2e-3)

    def test_offline_basis_matches_runtime_clipping_and_com_torque(self):
        trajectory=ROOT/'renders/policy_replay/final_policy_1100/policy_trajectory.csv'
        if not trajectory.exists():self.skipTest('Local recorded trajectory is not installed')
        t=np.array([.1,1.5,2.4,3.5,6.6,7.1])
        states=interpolate_trajectory(trajectory,t);level=-.12027640640735626
        basis,_=oar_basis(states,level)
        model=SimpleNamespace(body_rootid=np.arange(3),body=lambda name:SimpleNamespace(id={'boat':0,'port_oar':1,'starboard_oar':2}[name]))
        hydro=WarpHydrodynamics(model,len(t),'cpu',force_profile='prior')
        profile=load_force_profile(ROOT/'data/cfd_force_calibration.json');hydro.blade_coefficients=wp.vec3(profile['blade_cd_max'],profile['blade_cl_max'],profile['blade_skin_cd'])
        pos=np.stack([s['position'] for s in states],axis=1).astype(np.float32)
        R=np.stack([s['rotation'] for s in states],axis=1).astype(np.float32)
        com=pos+np.einsum('nbij,j->nbi',R,np.array([.07,-.03,.04])).astype(np.float32)
        # cvel translation evaluated at subtree_com = origin in this fixture.
        cv=np.stack([np.c_[s['omega'],s['velocity']] for s in states],axis=1).astype(np.float32)
        d=SimpleNamespace(xpos=wp.array(pos,dtype=wp.vec3,device='cpu'),xmat=wp.array(R,dtype=wp.mat33,device='cpu'),
            xipos=wp.array(com,dtype=wp.vec3,device='cpu'),subtree_com=wp.array(pos,dtype=wp.vec3,device='cpu'),
            cvel=wp.array(cv,dtype=wp.spatial_vector,device='cpu'),xfrc_applied=wp.zeros((len(t),3),dtype=wp.spatial_vector,device='cpu'))
        water=np.zeros((len(t),5),np.float32);water[:,0]=level;water[:,1]=1000;hydro.water.assign(water)
        hydro(None,d);moving=d.xfrc_applied.numpy().copy()
        d.cvel.zero_();hydro(None,d);dynamic=moving-d.xfrc_applied.numpy()
        dynamic[:,:,3:]+=np.cross(com-pos,dynamic[:,:,:3])
        expected=np.einsum('nsck,k->nsc',basis[:,:,:,:3],np.array([profile['blade_cd_max']/2,profile['blade_cl_max']/.9,1]))
        np.testing.assert_allclose(dynamic[:,1:],expected,atol=.04,rtol=3e-4)

    def test_profile_rejects_invalid_coefficients(self):
        self.assertEqual(load_force_profile('prior')['blade_cd_max'],2.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'profile.json'
            path.write_text(json.dumps(dict(schema_version=1,blade_cd_max=-1,blade_cl_max=.9,blade_skin_cd=.01)))
            with self.assertRaises(ValueError):load_force_profile(path)

    def test_audited_candidate_is_not_enabled_by_default(self):
        self.assertEqual(load_force_profile()['status'],'engineering_prior')
        candidate=load_force_profile(ROOT/'data/cfd_force_calibration.json')
        self.assertFalse(candidate['default_enabled'])
        self.assertGreater(candidate['blade_cd_max'],2.)


if __name__=='__main__':unittest.main()
