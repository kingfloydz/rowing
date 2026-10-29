"""Render existing reference kinematically from the current training reset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from hydrodynamics.models import ROOT
from rowing_mjlab.reset_pose import apply_reset_joints
from rowing_mjlab.reference import RowingReference, IMITATION_NAMES, PERIOD


def main():
    from rowing_mjlab.coordinated_reference import available
    if available():
        from tools.render_reference20 import render
        from rowing_mjlab.coordinated_reference import CoordinatedReference
        kind=CoordinatedReference('cpu').metadata.get('reference_kind')
        folder={'soft_symmetric':'reference20_soft','literature_soft':'reference10_literature'}.get(kind,'reference20')
        render(ROOT/'renders'/folder)
        return
    output = ROOT/'renders/reference_reset'
    output.mkdir(parents=True, exist_ok=True)
    root = ET.parse(ROOT/'scene.xml').getroot()
    apply_reset_joints(root)
    root.find('compiler').set('meshdir', str(ROOT))
    boat = root.find("./worldbody/body[@name='boat']")
    floating = json.loads((ROOT/'data/reset_float_pose.json').read_text())
    boat.set('pos', ' '.join(map(str, floating['position_m'])))
    boat.set('quat', ' '.join(map(str, floating['quaternion_wxyz'])))
    root.find('visual/global').set('offwidth', '1440')
    root.find('visual/global').set('offheight', '720')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    reference = RowingReference('cpu', model)
    addresses = [model.joint(n).qposadr[0] for n in IMITATION_NAMES]
    reset = torch.tensor(model.qpos0[addresses], dtype=torch.float32)
    phase0 = reference.nearest(reset[:6])
    mujoco.mj_forward(model, data)
    center = data.xpos[model.body('torso_link').id].copy()
    center[2] -= .12
    renderer = mujoco.Renderer(model, height=480, width=480)
    views = [('TOP', 90, -89), ('REAR', 180, -12), ('SIDE', 90, -12)]
    frames, report = [], []
    # One initial hold, the actual 1 s startup blend, and first slow cycle.
    times = np.r_[np.zeros(8), np.linspace(0, 3*PERIOD, 100), np.repeat(3*PERIOD, 8)]
    for k, elapsed in enumerate(times):
        phase, rate, blend, target = reference.target(torch.tensor(float(elapsed)), phase0, reset)
        data.qpos[:] = model.qpos0
        data.qpos[addresses] = target.numpy()
        mujoco.mj_forward(model, data)
        canvas = Image.new('RGB', (1440, 540), '#18202b')
        draw = ImageDraw.Draw(canvas)
        for v, (label, azimuth, elevation) in enumerate(views):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = center
            camera.distance = 2.25
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(data, camera=camera)
            for side, color in [('left', [0, 1, 1, 1]), ('right', [1, .4, .05, 1])]:
                ids = [model.joint(side+'_'+part+'_joint').id
                       for part in ('shoulder_yaw', 'elbow', 'wrist_roll')]
                points = data.xanchor[ids]
                for a, b in zip(points[:-1], points[1:]):
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .009, a, b)
                    renderer.scene.ngeom += 1
            canvas.paste(Image.fromarray(renderer.render()), (v*480, 40))
            draw.text((v*480+15, 12), label, fill='white')
        draw.text((15, 522), f't={elapsed:.2f}s  phase={float(phase):.3f}  blend={float(blend):.2f} | LEFT cyan / RIGHT orange | Kinematic reference; other joints held at RESET', fill='white')
        frames.append(canvas)
        if k in (0, 33, 58, 83):
            canvas.save(output/f'frame_{k:03d}.png')
        report.append({'time': float(elapsed), 'phase': float(phase), 'elbows_deg': np.rad2deg(target.numpy()[7:]).tolist()})
    renderer.close()
    frames[0].save(output/'reference.gif', save_all=True, append_images=frames[1:], duration=90, loop=0)
    (output/'report.json').write_text(json.dumps({'phase0': float(phase0), 'calibration': reference.elbow_calibration, 'frames': report}, indent=2))
    print(output)


if __name__ == '__main__':
    main()
