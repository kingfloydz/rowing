"""Short numerical smoke check of optional polar + rotating added mass coupling."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hydrodynamics.mujoco_adapter import RowingHydrodynamics

m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));d=mujoco.MjData(m)
h=RowingHydrodynamics(m,blade_model='caplan_flat',added_mass_rotation=True)
j=m.joint('boat_free').id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j]
d.qpos[qa+2]=.1193943;d.qvel[va]=1.;d.qvel[va+5]=.15
start=time.perf_counter();residual=0.
for _ in range(50):
    r=h.step(d);residual=max(residual,r['added_mass_residual_N'])
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
result=dict(steps=50,sim_time_s=float(d.time),wall_time_s=time.perf_counter()-start,
            max_implicit_residual_N=residual,finite_state=True,
            blade_model='caplan_flat',added_mass_rotation=True,
            note='CPU reference; 0.1 s numerical check, not validation of sustained rowing or experimental force accuracy.')
(ROOT/'results/extended/scene_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
