"""Ten reference channels derived explicitly from retained literature curves."""
import json,hashlib
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq,minimize
import trimesh
from hydrodynamics.models import ROOT
from rowing_mjlab.retarget import elbow_calibration
from rowing_mjlab.reference import IMITATION_NAMES

NAMES=(*IMITATION_NAMES,'trunk_vertical')
PARAMETERS=dict(period_s=3.,oar_delay=.06,seat_delay=.02,elbow_advance=.02,trunk_advance=.02,
    sweep_range_deg=[-12.,65.],feather_range_deg=[0.,65.],seat_range_m=[-.22,.22],
    elbow_amplitude_scale=1.,trunk_amplitude_scale=.85,
    drive_source_phase=[.05,.35],recovery_source_phase=[.50,.90],crossing_gap_m=.035)


def smooth(x):
    x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))


def build(model,phase):
    files=['data/rowing_reference.csv','data/reference9/aligned_priors.csv','data/reference9/trunk_vertical_prior.csv']
    o,s,t=[np.loadtxt(ROOT/f,delimiter=',',skiprows=1) for f in files]
    curve=lambda table,p:CubicSpline(table[:,0],table[:,1:],bc_type='periodic')(np.asarray(p)%1)
    p=PARAMETERS;phase=np.asarray(phase);op=(phase-p['oar_delay'])%1
    os=curve(o,op);ss=curve(s,phase-p['seat_delay']);es=curve(s,phase+p['elbow_advance']);ts=curve(t,phase+p['trunk_advance'])[:,0]
    calibration=elbow_calibration(model)
    base=np.column_stack([curve(o,phase),-.32423946769661277+.57757947871773*curve(s,phase)[:,0],
        *[c['zero_rad']-curve(s,phase)[:,1] for c in calibration],curve(t,phase)[:,0]])
    result=np.zeros((len(phase),10));result[:,:6]=os
    for i,sign in [(0,1),(3,-1)]:
        original=sign*o[:,i+1];v=sign*os[:,i]
        result[:,i]=sign*np.deg2rad(-12+77*(v-original.min())/np.ptp(original))
    for i in (2,5):
        original=o[:,i+1]
        result[:,i]=np.deg2rad(65)*(os[:,i]-original.min())/np.ptp(original)
    result[:,6]=-.22+.44*ss[:,0]
    for i,c in enumerate(calibration):result[:,7+i]=c['zero_rad']-p['elbow_amplitude_scale']*es[:,1]
    result[:,9]=p['trunk_amplitude_scale']*ts
    affine=result.copy();d=mujoco.MjData(model)
    qi=[model.joint(n).qposadr[0] for n in NAMES[:6]]
    bodies=[model.body(s+'_oar').id for s in ('port','starboard')]
    vertices=[trimesh.load_mesh(ROOT/f'boat_meshes/{s}_blade_3mm.stl').vertices for s in ('port','starboard')]
    water=json.loads((ROOT/'data/reset_float_pose.json').read_text())['water_level_m']
    def pose(angles):
        d.qpos[:]=model.qpos0;d.qpos[qi]=angles;mujoco.mj_kinematics(model,d)
    for k,source_phase in enumerate(op):
        angles=result[k,:6].copy()
        wet=smooth(source_phase/.05)-smooth((source_phase-.35)/.15)
        dry=smooth((source_phase-.35)/.15)-smooth((source_phase-.90)/.10)
        bottom=max(-.43,.050-(1-dry)*.8);top=min(.32,-.030+(1-wet)*.8)
        intervals=[]
        for side,(v,b) in enumerate(zip(vertices,bodies)):
            def extrema(e):
                aa=angles.copy();aa[3*side+1]=e;pose(aa)
                z=(v@d.xmat[b].reshape(3,3).T+d.xpos[b])[:,2]-water
                return z.min(),z.max()
            lower=brentq(lambda e:extrema(e)[0]-bottom,-.45,.45)
            upper=brentq(lambda e:extrema(e)[1]-top,-.45,.45)
            lo,hi=sorted([lower,upper]);intervals.append((lo,hi))
            angles[3*side+1]=np.clip(angles[3*side+1],lo,hi)
        pose(angles)
        sites=[model.site(s+'_grasp').id for s in ('starboard','port')]
        points=d.site_xpos[sites].copy();weight=smooth((.30-np.linalg.norm(points[0,:2]-points[1,:2]))/.20)
        gap=.012+.023*weight
        initial=angles[[1,4]].copy()
        def clearance(x):
            aa=angles.copy();aa[[1,4]]=x;pose(aa)
            return d.site_xpos[sites[0],2]-d.site_xpos[sites[1],2]-gap
        if clearance(initial)<0:
            fit=minimize(lambda x:np.sum(np.array([1.,4.])*(x-initial)**2),initial,method='SLSQP',bounds=intervals,
                constraints=[dict(type='ineq',fun=clearance)],options=dict(ftol=1e-12,maxiter=60))
            if clearance(fit.x)<-1e-6:raise RuntimeError('Cannot clear handle crossing while retaining water bounds')
            angles[[1,4]]=fit.x
        result[k,:6]=angles
    source_records=[dict(path=f,sha256=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()) for f in files]
    return result,base,affine,dict(parameters=p,sources=source_records,elbow_zero_calibration=calibration,
        provenance='Literature curves retained; explicit amplitude/offset and phase shifts, with minimum elevation corrections for water and crossing. No IK replacement of the ten channels.')
