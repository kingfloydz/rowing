"""Exercise versioned stream metadata, world transforms and rank load sums."""
import importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
SCRIPT=Path(__file__).resolve().parents[1]/'tools/read_rowing_video.py'
spec=importlib.util.spec_from_file_location('reader',SCRIPT)
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)

class SurfaceStream(unittest.TestCase):
    def make(self,path,version,force):
        h=[0x3153564c,version,3,0]+[.05,2,3,0,0,0]+[0.]*39+[force]*18
        path.write_bytes(reader.HEADER.pack(*h)+np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='<f4').tobytes())

    def test_versions_transform_and_sum(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=p/'a.bin';b=p/'b.bin'
            self.make(a,3,1.);self.make(b,3,2.)
            meta,vertices=next(reader.frames(a))
            np.testing.assert_allclose(vertices[0],[2,3,0])
            self.assertEqual(meta['version'],3)
            subprocess.run([sys.executable,str(SCRIPT),str(a),str(b),'--output',str(p/'ply')],check=True,capture_output=True)
            m=json.loads((p/'ply/surface_00000.json').read_text())
            np.testing.assert_allclose(m['body_loads'],np.full((3,6),3.))
            self.make(a,2,1.)
            self.assertIn('last_stage_ib_reaction_ale',next(reader.frames(a))[0])
            bad=subprocess.run([sys.executable,str(SCRIPT),str(a),str(b),'--validate-only'],capture_output=True)
            self.assertNotEqual(bad.returncode,0)

    def test_truncation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.bin';self.make(p,3,1.);p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):list(reader.frames(p))

if __name__=='__main__':unittest.main()
