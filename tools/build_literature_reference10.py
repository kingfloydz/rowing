"""Literature-derived ten channels; whole-body IK is only an auxiliary preview."""
import sys,json,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np,trimesh
from scipy.optimize import least_squares,brentq
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from hydrodynamics.models import ROOT
from rowing_mjlab.asset import build_spec
from rowing_mjlab.symmetry import MIRROR_PAIRS
from tools.literature_reference10 import build as build_ten, NAMES


def smooth(x):
    x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-diagnostic',action='store_true');args=parser.parse_args()
    dest=ROOT/'data/reference20';out=ROOT/'renders/reference10_literature'
    if args.baseline_diagnostic:out=out/'baseline_diagnostic'
    out.mkdir(parents=True,exist_ok=True);output_dest=out if args.baseline_diagnostic else dest
    backup=out/'previous_reference'
    if not backup.exists():shutil.copytree(dest,backup)
    previous=json.loads((dest/'metadata.json').read_text())
    m=build_spec().compile();d=mujoco.MjData(m)
    names=list(np.load(dest/'trajectory.npz')['names']);qi=np.array([m.joint(n).qposadr[0] for n in names]);jids=[m.joint(n).id for n in names]
    idx=lambda n:names.index(n)
    oars=np.array([idx(s+'_'+j) for s in ('port','starboard') for j in ('sweep','elevation','feather')])
    seat=idx('seat_slide');human=np.array([i for i in range(len(names)) if i not in oars and i!=seat and names[i] not in ('left_elbow_joint','right_elbow_joint')])
    bounds=m.jnt_range[jids].copy();bounds[:,0]+=.04;bounds[:,1]-=.04
    sides=('left','right');eq=[m.equality(s+'_'+p+'_fixed').id for p in ('hand','foot') for s in sides]
    anchors=[[m.joint(s+'_'+p+'_joint').id for p in ('shoulder_yaw','elbow','wrist_roll')] for s in sides]
    ob=[m.body(s+'_oar').id for s in ('port','starboard')]
    verts=[trimesh.load_mesh(ROOT/f'boat_meshes/{s}_blade_3mm.stl').vertices for s in ('port','starboard')]
    water=json.loads((ROOT/'data/reset_float_pose.json').read_text())['water_level_m']
    boat=m.body('boat').id;torso=m.body('torso_link').id
    pairs=[(idx('left_'+p+'_joint'),idx('right_'+p+'_joint'),sign) for p,sign in MIRROR_PAIRS]
    n=400;phase=np.arange(n)/n;targets=[];poses=[]
    ten,baseline,affine,lineage=build_ten(m,phase)
    if args.baseline_diagnostic:ten=baseline.copy()
    neutral=m.qpos0[qi].copy()
    for l,r,sign in pairs:
        neutral[l]=(neutral[l]+sign*neutral[r])/2;neutral[r]=sign*neutral[l]
    neutral[idx('waist_yaw_joint')]=neutral[idx('waist_roll_joint')]=0
    last=neutral.copy()
    for k,p in enumerate(phase):
        q=neutral.copy()
        for i,name in enumerate(NAMES[:9]):q[idx(name)]=ten[k,i]
        trunk=ten[k,9]
        bend=np.clip((1.336754268-q[idx('left_elbow_joint')])/np.deg2rad(101.4),0,1)
        def residual(x):
            full=q.copy();full[human]=x;d.qpos[:]=m.qpos0;d.qpos[qi]=full;mujoco.mj_kinematics(m,d)
            rr=d.xmat.reshape(-1,3,3);rb=rr[boat];result=[]
            hand_targets=[]
            for j,e in enumerate(eq):
                a,b=m.eq_obj1id[e],m.eq_obj2id[e];rel=m.eq_data[e]
                # Foot closure stays strong; hand position and orientation are soft.
                rot=Rotation.from_quat(rel[6:10][[1,2,3,0]]).as_matrix()
                if j<2:
                    point=rot.T@(np.array([.029,0,0])-rel[3:6]);target=d.xpos[b]+rr[b]@point
                    result.extend(12*(d.xpos[a]+rr[a]@np.array([.029,0,0])-target))
                    hand_targets.append(target)
                else:
                    result.extend(45*(d.xpos[a]+rr[a]@rel[3:6]-d.xpos[b]))
                result.extend(Rotation.from_matrix((rr[a]@rot).T@rr[b]).as_rotvec()*(.6 if j<2 else 4))
            for j,ids in enumerate(anchors):
                shoulder,elbow,wrist=d.xanchor[ids]
                h=(hand_targets[j]-shoulder)@rb
                # Boat +X is behind the rower. User requests posterolateral
                # flexion: backward pull plus moderate, mirrored abduction.
                outward=-1 if j==0 else 1
                target=np.array([.5*h[0]*(1-bend)+.14*bend,outward*(.012+.060*bend),-.16-.04*bend])
                local=(elbow-shoulder)@rb
                result.extend((local-target)*np.array([9,14,5]))
            result.extend([15*(np.arctan2(rr[torso,0,2],rr[torso,2,2])-trunk)])
            result.extend([5*full[idx('waist_yaw_joint')],5*full[idx('waist_roll_joint')]])
            # Similar bilateral style, not exact mirroring: arms may adapt to
            # the left-over-right crossing and unequal hand heights.
            result.extend([(.4 if 'wrist' in names[l] else 1.2 if 'shoulder' in names[l] else 2 if 'elbow' in names[l] else 3)*(full[l]-sign*full[r]) for l,r,sign in pairs])
            result.extend(.12*(x-neutral[human]))
            return np.array(result)
        starts=[last[human]]
        if k==0:starts.append(neutral[human])
        fits=[least_squares(residual,np.clip(x,bounds[human,0]+1e-6,bounds[human,1]-1e-6),bounds=(bounds[human,0],bounds[human,1]),max_nfev=90,ftol=1e-7,xtol=1e-8,gtol=1e-7) for x in starts]
        fit=min(fits,key=lambda r:r.cost);q[human]=fit.x;last=q.copy();poses.append(q)
        if k%40==0:print('Auxiliary IK',k,n,'cost',fit.cost,flush=True)
    poses=np.array(poses)
    poses[:,human]=gaussian_filter1d(poses[:,human],2,axis=0,mode='wrap')
    # Periodic smoothing only changes the visualization posture, not water targets.
    phase=np.r_[phase,1.];poses=np.vstack([poses,poses[0]])
    trunk=np.r_[ten[:,9],ten[0,9]]
    np.savez_compressed(output_dest/'trajectory.npz',phase=phase,q=poses,trunk=trunk,names=names)
    meta={k:v for k,v in previous.items() if k in ('original_gripper_mount','physical_reset_calibration','grip_adjustment_rotation_vector_deg','grip_adjustment_type','grip_point_shift_toward_oarlock_m','technique_url')}
    meta.update(strokes_per_minute=20,period_s=3,reset_phase=previous['reset_phase'],reference_kind='literature_soft',
        reset_on_reference=False,drive_interior=[.11,.41],recovery_interior=[.56,.96],flat_recovery_interior=[.60,.80],
        transitions={'approach_and_entry':[[.96,1],[0,.11]],'extraction':[[.41,.56]]},
        source=lineage['provenance'],literature_lineage=lineage,elbow_direction='posterolateral',
        original_literature_preserved=True,independent_phase_offsets_removed=True,
        limitation='Ten-channel soft reference. Auxiliary full-body poses do not redefine elbow or trunk targets. Closure and collision errors are measured, not hard gates. No dynamic stability claim.')
    for label,values in [('baseline',baseline),('affine',affine),('adjusted',ten)]:
        table=np.column_stack([phase,np.vstack([values,values[0]])])
        np.savetxt(out/(label+'_ten_channels.csv'),table,delimiter=',',header=','.join(('phase',*NAMES)),comments='')
    (output_dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(output_dest,flush=True)


if __name__=='__main__':main()
