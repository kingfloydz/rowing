"""Render actual boat CAD and the current level-heave waterline."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hydrodynamics.models import wet_triangles
OUT=ROOT/'results/rowimu'
info=json.loads((OUT/'summary.json').read_text());z=info['level_heave_m'];water=info['water_z_m']
meshes={}
for name in ['shell_white','deck_blue','cockpit','rigger','rails','fin']:
 mesh=trimesh.load_mesh(ROOT/f'boat_meshes/{name}.stl');mesh.apply_translation([0,0,z]);meshes[name]=mesh
hull=meshes['shell_white'];draft=water-hull.bounds[0,2];gatez=.2074977344+z
colors={'shell_white':'#c9ced4','deck_blue':'#235da5','cockpit':'#7a8590','rigger':'#d7a246','rails':'#525b63','fin':'#6b7880'}
fig=plt.figure(figsize=(14,7),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[1.05,1])
ax=fig.add_subplot(grid[0,:],projection='3d')
for name,mesh in meshes.items():
 ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolors=colors[name],edgecolors='none',alpha=1))
waterpoly=np.array([[-3.55,-1.05,0],[3.8,-1.05,0],[3.8,1.05,0],[-3.55,1.05,0]])
ax.add_collection3d(Poly3DCollection([waterpoly],facecolors='#58bada',alpha=.30,edgecolors='#3195bc',linewidths=.7))
section=hull.section(plane_origin=[0,0,water],plane_normal=[0,0,1])
if section is not None:
 for line in section.discrete:ax.plot(line[:,0],line[:,1],line[:,2]+.001,c='#007d9b',lw=2)
ax.scatter([0,0],[.79,-.79],[gatez,gatez],c='#c45e16',s=25,depthshade=False)
ax.set(xlim=(-3.55,3.8),ylim=(-1.05,1.05),zlim=(-.24,.45),xlabel='Boat x (m)',ylabel='y (m)',zlabel='Height (m)')
ax.set_box_aspect((7.35,2.1,.95),zoom=1.65);ax.view_init(elev=22,azim=-65);ax.set_axis_off()
ax.set_title('Actual boat CAD | translucent water plane at world z = 0',pad=1)
side=fig.add_subplot(grid[1,0])
side.axhspan(-.3,0,color='#dceff6');side.axhline(0,c='#007d9b',lw=1.4)
for name in ['shell_white','deck_blue','cockpit','fin']:
 mesh=meshes[name];side.add_collection(PolyCollection(mesh.triangles[:,:,[0,2]],facecolors=colors[name],edgecolors='none'))
wet=wet_triangles(hull.triangles,water);side.add_collection(PolyCollection(wet[:,:,[0,2]],facecolors='#388fb1',edgecolors='none',alpha=.5))
side.scatter([0],[gatez],s=30,c='#c45e16',zorder=5);side.annotate(f'Oarlock: {gatez:.3f} m',xy=(0,gatez),xytext=(.7,.33),arrowprops={'arrowstyle':'->'},fontsize=9)
side.annotate('',xy=(-1.0,0),xytext=(-1.0,-draft),arrowprops={'arrowstyle':'<->','color':'#164b60'})
side.text(-.9,-draft/2,f'Hull draft {draft:.3f} m',va='center',fontsize=9)
side.set(xlim=(-3.4,3.6),ylim=(-.25,.43),xlabel='Boat x (m)',ylabel='Height above water (m)',title='Side projection (vertical scale enlarged)');side.grid(alpha=.15)
front=fig.add_subplot(grid[1,1]);front.axhspan(-.25,0,color='#dceff6');front.axhline(0,c='#007d9b',lw=1.4)
cut=hull.section(plane_origin=[0,0,0],plane_normal=[1,0,0])
if cut is not None:
 for line in cut.discrete:front.plot(line[:,1],line[:,2],c='#52616e',lw=2)
# Rigger projection shows relation between hull and oarlock supports.
rig=meshes['rigger'];front.add_collection(PolyCollection(rig.triangles[:,:,[1,2]],facecolors='#d7a246',edgecolors='none',alpha=.6))
front.scatter([.79,-.79],[gatez,gatez],s=35,c='#c45e16',zorder=5)
front.annotate('',xy=(.9,gatez),xytext=(.9,0),arrowprops={'arrowstyle':'<->','color':'#a14f13'})
front.text(.92,gatez/2,f'{gatez:.3f} m',rotation=90,va='center',fontsize=9)
front.set(xlim=(-1.05,1.05),ylim=(-.25,.43),xlabel='Transverse y (m)',ylabel='Height above water (m)',title='Hull section at x = 0 + projected rigger');front.set_aspect('equal',adjustable='box');front.grid(alpha=.15)
ax.remove();fig.set_layout_engine(None);fig.set_size_inches(14,4.8)
side.set_position([.06,.16,.55,.65]);front.set_position([.69,.16,.28,.65])
fig.suptitle(f'Boat and waterline | total loaded mass {92.3764016049845:.2f} kg | level-heave balance',fontsize=15)
fig.savefig(OUT/'boat_waterline.png',dpi=175);fig.savefig(OUT/'boat_waterline.svg');plt.close(fig)
print(json.dumps({'water_z_m':water,'boat_heave_m':z,'hull_draft_m':draft,'oarlock_height_above_water_m':gatez,'note':'H2 mass included; H2 and oars hidden for boat visibility. Level attitude, not free trim equilibrium.'},indent=2))
