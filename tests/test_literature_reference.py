"""The ten targets must retain literature signals, regardless of auxiliary IK."""
import unittest,json,hashlib
import numpy as np
from hydrodynamics.models import ROOT


class LiteratureReferenceTests(unittest.TestCase):
    def test_preserved_source_shapes_and_separate_pose_fit(self):
        meta=json.loads((ROOT/'data/reference20/metadata.json').read_text())
        if meta.get('reference_kind')!='literature_soft':self.skipTest('not literature-derived')
        for record in meta['literature_lineage']['sources']:
            self.assertEqual(hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest(),record['sha256'])
        a=np.load(ROOT/'data/reference20/trajectory.npz');q=a['q'][:-1];names=list(a['names'])
        o=np.loadtxt(ROOT/'data/rowing_reference.csv',delimiter=',',skiprows=1)[:-1]
        h=np.loadtxt(ROOT/'data/reference9/aligned_priors.csv',delimiter=',',skiprows=1)[:-1]
        t=np.loadtxt(ROOT/'data/reference9/trunk_vertical_prior.csv',delimiter=',',skiprows=1)[:-1]
        self.assertEqual(len(q),len(o));n=len(q);p=meta['literature_lineage']['parameters']
        shift=lambda values,delay:np.roll(values,round(delay*n),axis=0)
        source=shift(o[:,1:],p['oar_delay'])
        for side,j,sign in [('port',0,1),('starboard',3,-1)]:
            original=sign*o[:,1+j];expected=sign*np.deg2rad(-12+77*(sign*source[:,j]-original.min())/np.ptp(original))
            np.testing.assert_allclose(q[:,names.index(side+'_sweep')],expected,atol=1e-8)
            original=o[:,j+3];expected=np.deg2rad(65)*(source[:,j+2]-original.min())/np.ptp(original)
            np.testing.assert_allclose(q[:,names.index(side+'_feather')],expected,atol=1e-8)
        np.testing.assert_allclose(q[:,names.index('seat_slide')],-.22+.44*shift(h[:,1],p['seat_delay']),atol=1e-8)
        for side,c in zip(('left','right'),meta['literature_lineage']['elbow_zero_calibration']):
            expected=c['zero_rad']-shift(h[:,2],-p['elbow_advance'])
            np.testing.assert_allclose(q[:,names.index(side+'_elbow_joint')],expected,atol=1e-8)
        np.testing.assert_allclose(a['trunk'][:-1],p['trunk_amplitude_scale']*shift(t[:,1],-p['trunk_advance']),atol=1e-8)
