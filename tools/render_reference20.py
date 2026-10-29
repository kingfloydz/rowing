"""Render whole-body reference targets and actual reset, with a water plane."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
import imageio.v2 as imageio
from PIL import Image
from hydrodynamics.models import ROOT


def render(folder):
    from rowing_mjlab.coordinated_reference import available,CoordinatedReference
    folder.mkdir(parents=True,exist_ok=True)
    reset_phase=.15
    if available():
        import torch
        from rowing_mjlab.asset import build_spec
        m=build_spec().compile();ref=CoordinatedReference('cpu')
        values=ref.joints(torch.linspace(0,1,401,dtype=torch.float64),ref.names).numpy()
        poses=np.tile(m.qpos0,(len(values),1));poses[:,[m.joint(n).qposadr[0] for n in ref.names]]=values
        reset_phase=ref.metadata['reset_phase']
    else:
        m=mujoco.MjModel.from_binary_path(str(folder/'candidate.mjb'))
        poses=np.load(folder/'dense.npz')['qpos']
    d=mujoco.MjData(m)
    water=json.loads((ROOT/'data/reset_float_pose.json').read_text())['water_level_m']
    m.vis.global_.offwidth=1600;m.vis.global_.offheight=1000
    head=m.body('head_pitch_link').id
    for g in range(m.ngeom):
        b=m.geom_bodyid[g]
        while b and b!=head:b=m.body_parentid[b]
        if b==head:m.geom_rgba[g,3]=0
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0;opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT]=False
    renderer=mujoco.Renderer(m,height=544,width=800)
    def frame(i):
        d.qpos[:]=m.qpos0 if i is None else poses[i];mujoco.mj_kinematics(m,d);images=[]
        for az,el,dist in [(90,-15,3.2),(180,-60,6.5)]:
            c=mujoco.MjvCamera();c.lookat[:]=[0,0,.3];c.azimuth=az;c.elevation=el;c.distance=dist
            renderer.update_scene(d,camera=c,scene_option=opt)
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array([8.,8.,.001]),np.array([0,0,water]),np.eye(3).ravel(),np.array([.15,.55,.75,.18]))
            renderer.scene.ngeom+=1
            images.append(renderer.render().copy())
        return np.concatenate(images,axis=1)
    try:
        imageio.imwrite(folder/'reset.png',frame(None) if available() else frame(round(reset_phase*(len(poses)-1))))
        for phase in (.1,.25,.45,.7,.9):imageio.imwrite(folder/f'phase_{round(phase*100):02d}.png',frame(round(phase*(len(poses)-1))))
        frames=[frame(round(i/90*(len(poses)-1))) for i in range(90)]
        with imageio.get_writer(folder/'reference20.mp4',fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
            for f in frames:writer.append_data(f)
        preview=[Image.fromarray(f).resize((960,324)) for f in frames[::3]]
        preview[0].save(folder/'reference20.gif',save_all=True,append_images=preview[1:],duration=100,loop=0)
    finally:renderer.close()


if __name__=='__main__':render(ROOT/'renders/reference20')
