"""Convert documented task envelopes to CAD joint limits; not hardware certification."""
from pathlib import Path
import sys,json,re,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tools.render_rowimu import oar_rotation


def main():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));d=mujoco.MjData(m)
    envelope={'sweep':[-80.,50.],'elevation':[-25.,10.],'feather':[-10.,110.]}
    audit={'kind':'Engineering simulation limits; not measured hardware stops',
        'physical_angles_deg':envelope,'joint_margin_deg':2.,'rounding_deg':5.,
        'basis':{'sweep':'World Rowing 2020 practical rigging: experienced sculling catch 55-72+ deg, finish 40-45 deg. -80..50 is a chosen margin, not a quoted hard stop.',
          'elevation':'RowIMU Fig6 trajectory plus user-selected -6deg offset, expanded to -25..10. Engineering envelope, not manufacturer data.',
          'feather':'RowIMU Fig7 covers feathered (~0) to squared (~90+) blade; -10..110 is chosen clearance, not a manufacturer stop.'},
        'sources':['https://worldrowing.com/wp-content/uploads/2020/10/Practical-Boat-Rigging-Gianni-Postilione-Conny-Draper.pdf','https://ojs.ub.uni-konstanz.de/cpa/article/view/6500/5867'],
        'caveat':'Box bounds enclose sampled physical orientations. Independent joint limits do not enforce the exact physical-angle envelope or guarantee robot reachability/collision clearance.',
        'mapping':'MuJoCo forward kinematics includes body quaternions, axes and ref; initial assembly pose is retained.', 'joints':{}}
    for side,sgn in [('port',1),('starboard',-1)]:
        names=[side+'_'+k for k in envelope];jids=[m.joint(n).id for n in names];qa=m.jnt_qposadr[jids];b=m.body(side+'_oar').id
        solutions=[];maxerror=0.
        for angles in itertools.product(np.linspace(-80,50,3),np.linspace(-25,10,8),np.linspace(-10,110,9)):
            R=oar_rotation(*np.deg2rad(angles),sgn)
            # Initialize on the anatomical branch, preserving signed left/right rotations.
            x0=np.deg2rad([sgn*(angles[0]+44),sgn*(angles[1]+18),90-angles[2]])
            def error(q):
                d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
                return Rotation.from_matrix(d.xmat[b].reshape(3,3).T@R).as_rotvec()
            sol=least_squares(error,x0,xtol=1e-11,ftol=1e-11,gtol=1e-11)
            maxerror=max(maxerror,np.linalg.norm(sol.fun));solutions.append(np.rad2deg(sol.x))
        assert maxerror<1e-7
        q=np.array(solutions);lo=5*np.floor((q.min(0)-2)/5);hi=5*np.ceil((q.max(0)+2)/5)
        for i,name in enumerate(names):
            audit['joints'][name]={'range_deg':[float(lo[i]),float(hi[i])],'range_rad':np.deg2rad([lo[i],hi[i]]).tolist(),
             'sampled_min_max_deg':[float(q[:,i].min()),float(q[:,i].max())],'orientation_error_max_rad':maxerror}
    for filename in ['scene.xml','preview.xml']:
        path=ROOT/filename;text=path.read_text(encoding='utf-8')
        for name,info in audit['joints'].items():
            value=' '.join(f'{v:.12f}' for v in info['range_rad'])
            pattern=r'(<joint name="'+name+r'"[^>]*range=")[^"]+("[^>]*>)'
            text,n=re.subn(pattern,lambda match:match[1]+value+match[2],text)
            assert n==1
        path.write_text(text,encoding='utf-8')
        check=mujoco.MjModel.from_xml_path(str(path))
        for name in audit['joints']:
            j=check.joint(name);assert j.range[0]<=check.qpos0[j.qposadr[0]]<=j.range[1]
    (ROOT/'data/oar_motion_limits.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v['range_deg'] for k,v in audit['joints'].items()},indent=2))

if __name__=='__main__':main()
