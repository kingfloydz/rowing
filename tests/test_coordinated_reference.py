"""Validate the runtime interpolant under its declared hard/soft contract."""
import unittest,json
import mujoco,numpy as np,torch
from scipy.spatial.transform import Rotation
from hydrodynamics.models import ROOT
from rowing_mjlab.asset import build_spec
from rowing_mjlab.coordinated_reference import CoordinatedReference,available
from rowing_mjlab.reference import RowingReference,PERIOD,IMITATION_NAMES
from rowing_mjlab.trunk import TrunkReference


@unittest.skipUnless(available(),'coordinated reference is not installed')
class CoordinatedReferenceTests(unittest.TestCase):
    def test_runtime_uses_one_clock_and_declared_reference_contract(self):
        self.assertEqual(PERIOD,3.)
        m=build_spec().compile();d=mujoco.MjData(m);ref=CoordinatedReference('cpu')
        phase=torch.linspace(0,1,801,dtype=torch.float64)
        values=ref(phase).numpy();qi=[m.joint(n).qposadr[0] for n in ref.names]
        row=RowingReference('cpu',m);trunk=TrunkReference('cpu')
        np.testing.assert_allclose(row(phase).numpy(),values[:,[ref.names.index(n) for n in IMITATION_NAMES]],atol=1e-6)
        np.testing.assert_allclose(trunk(phase).numpy(),values[:,-1],atol=1e-6)
        np.testing.assert_allclose(values[0],values[-1],atol=1e-7)
        soft=ref.metadata.get('reference_kind') in ('soft_symmetric','literature_soft')
        if soft:
            from rowing_mjlab.symmetry import MIRROR_PAIRS
            self.assertTrue(np.isfinite(values).all())
            for part,sign in MIRROR_PAIRS:
                if ref.metadata.get('reference_kind')=='literature_soft' and 'wrist' in part:
                    # Crossing heights and opposite feather axes allow wrist differences;
                    # the ten-channel source tests cover bilateral targets.
                    continue
                error=values[:,ref.names.index('left_'+part+'_joint')]-sign*values[:,ref.names.index('right_'+part+'_joint')]
                self.assertLess(np.rad2deg(np.sqrt(np.mean(error**2))),8.)
        for q in values[:,:-1]:
            d.qpos[:]=m.qpos0;d.qpos[qi]=q;mujoco.mj_forward(m,d)
            if soft:
                for i,n in enumerate(ref.names):
                    lo,hi=m.jnt_range[m.joint(n).id]
                    self.assertGreaterEqual(q[i],lo-1e-6)
                    self.assertLessEqual(q[i],hi+1e-6)
                continue
            for side in ('left','right'):
                for part in ('hand','foot'):
                    e=m.equality(side+'_'+part+'_fixed').id;a,b=m.eq_obj1id[e],m.eq_obj2id[e]
                    ra=d.xmat[a].reshape(3,3);rb=d.xmat[b].reshape(3,3);rel=m.eq_data[e]
                    self.assertLess(np.linalg.norm(d.xpos[a]+ra@rel[3:6]-d.xpos[b]),.001)
                    rot=Rotation.from_quat(rel[6:10][[1,2,3,0]]).as_matrix()
                    self.assertLess(Rotation.from_matrix((ra@rot).T@rb).magnitude(),np.deg2rad(.2))
            self.assertFalse(any(c.dist<-.001 for c in d.contact))
        reset=ref.joints(torch.tensor(ref.metadata['reset_phase'],dtype=torch.float64),ref.names).numpy()
        if not soft:np.testing.assert_allclose(reset,m.qpos0[qi],atol=2e-6)
        else:
            # A soft reference need not pass through the physical reset. The
            # existing startup blend must still begin at the real reset target.
            actual=torch.tensor(m.qpos0[[m.joint(n).qposadr[0] for n in IMITATION_NAMES]])
            _,_,_,target=row.target(torch.tensor(0.),torch.tensor(ref.metadata['reset_phase']),actual)
            np.testing.assert_allclose(target.numpy(),actual.numpy(),atol=1e-7)

    def test_soft_reference_elbows_move_posterolaterally(self):
        ref=CoordinatedReference('cpu')
        kind=ref.metadata.get('reference_kind')
        if kind not in ('soft_symmetric','literature_soft'):self.skipTest('not a soft reference')
        m=build_spec().compile();d=mujoco.MjData(m);qi=[m.joint(n).qposadr[0] for n in ref.names]
        lo,hi=(.45,.55) if kind=='literature_soft' else (.36,.46)
        values=ref.joints(torch.linspace(lo,hi,41,dtype=torch.float64),ref.names).numpy()
        offsets=[]
        for q in values:
            d.qpos[:]=m.qpos0;d.qpos[qi]=q;mujoco.mj_kinematics(m,d)
            rb=d.xmat[m.body('boat').id].reshape(3,3)
            row=[]
            for s in ('left','right'):
                shoulder,elbow,wrist=d.xanchor[[m.joint(s+'_'+p+'_joint').id for p in ('shoulder_yaw','elbow','wrist_roll')]]
                v=wrist-shoulder
                origin=shoulder+v*((elbow-shoulder)@v)/(v@v) if kind=='literature_soft' else shoulder
                row.append((elbow-origin)@rb)
            offsets.append(row)
        mean=np.mean(offsets,axis=0)
        self.assertTrue(np.all(mean[:,0]>.03))
        self.assertGreater(-mean[0,1],.01)
        self.assertGreater(mean[1,1],.01)


if __name__=='__main__':unittest.main()
