"""Generate the compact hand: STEP in mm, simulation STL/GLB in m.
Requires cadquery, trimesh, matplotlib, numpy. Optional local CAD runtime:
  python -m pip install --target logs/cad_runtime cadquery
"""
from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent
runtime=OUT.parents[1]/'logs/cad_runtime'
if runtime.is_dir(): sys.path.append(str(runtime))
import cadquery as cq
import numpy as np
import trimesh
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BORE,OUTER,LENGTH=34.,50.,70.
NECK_D,NECK_EXPOSED,NECK_BLANK_LENGTH=50.,4.,20.
ROOT_FILLET,END_FILLET,LIP_FILLET=5.,2.,1.
CENTER=NECK_EXPOSED+OUTER/2

def build():
    sleeve=cq.Workplane('XY').center(CENTER,0).circle(OUTER/2).extrude(LENGTH).translate((0,0,-LENGTH/2))
    # Round the distal connector rim before union to avoid tangent equal-radius
    # cylinder intersections and to provide a smooth exposed shoulder.
    neck=cq.Workplane('YZ').circle(NECK_D/2).extrude(NECK_BLANK_LENGTH).edges('>X').fillet(ROOT_FILLET)
    solid=sleeve.union(neck)
    seam=[e for e in solid.val().Edges() if e.geomType() in ('ELLIPSE','BSPLINE')]
    solid=solid.newObject(seam).fillet(ROOT_FILLET)
    ends=[e for e in solid.val().Edges() if e.geomType()=='CIRCLE' and abs(abs(e.Center().z)-LENGTH/2)<1e-5]
    solid=solid.newObject(ends).fillet(END_FILLET)
    bore=cq.Workplane('XY').center(CENTER,0).circle(BORE/2).extrude(LENGTH+20).translate((0,0,-LENGTH/2-10))
    solid=solid.cut(bore)
    lips=[e for e in solid.val().Edges() if e.geomType()=='CIRCLE' and abs(e.radius()-BORE/2)<1e-5 and abs(abs(e.Center().z)-LENGTH/2)<1e-5]
    return solid.newObject(lips).fillet(LIP_FILLET)

def main():
    solid=build()
    assert solid.val().isValid() and len(solid.solids().vals())==1
    # A slightly undersized gauge must pass through the full bore.
    gauge=cq.Workplane('XY').center(CENTER,0).circle(BORE/2-.01).extrude(LENGTH+20).translate((0,0,-LENGTH/2-10))
    assert solid.intersect(gauge).val().Volume()<1e-6
    cq.exporters.export(solid,str(OUT/'compact_hand.step'))
    verts,faces=solid.val().tessellate(.08,.12)
    mesh=trimesh.Trimesh(vertices=[v.toTuple() for v in verts],faces=faces,process=True)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    exported=mesh.copy();exported.apply_scale(.001);exported.export(OUT/'compact_hand.stl')
    exported.visual.face_colors=[179,179,179,255]
    trimesh.Scene(exported).export(OUT/'assembly.glb')
    info=dict(units='mm',bore_diameter=BORE,outer_diameter=OUTER,sleeve_length=LENGTH,
              connector_diameter=NECK_D,connector_exposed_length=NECK_EXPOSED,
              connector_blank_length=NECK_BLANK_LENGTH,mount_to_handle_axis=CENTER,
              junction_fillet=ROOT_FILLET,outer_end_fillet=END_FILLET,bore_lip_fillet=LIP_FILLET,
              cad_volume_mm3=solid.val().Volume(),valid_single_solid=True,watertight_mesh=True,
              bore_gauge_clear=True,step_units='mm',stl_and_glb_units='m')
    (OUT/'dimensions.json').write_text(json.dumps(info,indent=2))
    fig=plt.figure(figsize=(13,8),facecolor='#f5f7fa')
    ax=fig.add_subplot(121,projection='3d',facecolor='#f5f7fa')
    normals=mesh.face_normals;light=np.array([-1,-2,3.]);light/=np.linalg.norm(light)
    shades=.55+.45*np.maximum(normals@light,0)
    colors=shades[:,None]*np.array([.7,.7,.7])
    ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolor=colors,edgecolor='none'))
    ax.set(xlim=(-5,70),ylim=(-35,35),zlim=(-40,40),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)')
    ax.set_box_aspect((75,70,80));ax.view_init(elev=32,azim=-60)
    ax.set_title('Single solid | rounded junction',pad=18)
    ax2=fig.add_subplot(122,facecolor='#f5f7fa')
    sec=mesh.section(plane_origin=[0,0,0],plane_normal=[0,1,0])
    for points in sec.discrete:
        ax2.fill(points[:,0],points[:,2],color='#aaaaaa',alpha=.8)
        ax2.plot(points[:,0],points[:,2],color='#16546a',linewidth=1)
    ax2.plot([CENTER,CENTER],[-40,40],'--',color='#c39445',linewidth=1)
    def dim(a,b,text,offset=(0,0)):
        ax2.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='<->',color='#263447'))
        mid=(np.array(a)+b)/2+offset;ax2.text(*mid,text,ha='center',va='center',fontsize=10)
    dim((0,-44),(CENTER,-44),f'{CENTER:g} mm',(0,-5))
    dim((70,-35),(70,35),'70 mm',(8,0))
    dim((CENTER-OUTER/2,44),(CENTER+OUTER/2,44),'OD 50 mm',(0,5))
    dim((CENTER-BORE/2,29),(CENTER+BORE/2,29),'ID 34 mm',(0,-5))
    ax2.annotate('R5 junction',xy=(CENTER-OUTER/2-2,29),xytext=(-5,57),fontsize=10,arrowprops=dict(arrowstyle='->'))
    ax2.text(-5,-58,'Connector: diameter 50 mm\nExposed length: 4 mm; blank length: 20 mm\nSleeve end edges R2; bore lips R1\nX=0: wrist mount; Z: handle axis',va='top',fontsize=10,linespacing=1.7)
    ax2.set(xlim=(-10,90),ylim=(-90,65));ax2.set_aspect('equal');ax2.axis('off')
    ax2.set_title('Axial section | dimensions')
    fig.suptitle('ROWING / COMPACT SLEEVE HAND',fontsize=17,fontweight='bold',x=.05,ha='left')
    fig.subplots_adjust(left=.04,right=.98,bottom=.10,top=.89,wspace=.15)
    fig.savefig(OUT/'preview.png',dpi=160);plt.close(fig)
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
