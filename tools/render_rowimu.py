"""Reconstruct RowIMU plots for current CAD; not a measured dual-oar dataset."""
from pathlib import Path
import sys,json,hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from shapely import Polygon, unary_union
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares,brentq
from scipy.spatial.transform import Rotation
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader
from hydrodynamics.training_adapter import TrainingRowingHydrodynamics
from hydrodynamics.training import blade_traction
from hydrodynamics.models import triangle_geometry,wet_triangles,hydrostatic_wrench,G
OUT=ROOT/'results/rowimu'
URL='https://ojs.ub.uni-konstanz.de/cpa/article/download/6500/5867'


def oar_rotation(theta,beta,gamma,side_sign):
    """Physical sweep, shaft elevation and blade feather -> world rotation."""
    chord=np.array([-np.sin(theta)*np.cos(beta),side_sign*np.cos(theta)*np.cos(beta),np.sin(beta)])
    flatnormal=np.array([np.cos(theta),side_sign*np.sin(theta),0.])
    normal=np.sin(gamma)*flatnormal+np.cos(gamma)*np.cross(chord,flatnormal)
    yaxis=side_sign*chord
    return np.column_stack([normal,yaxis,np.cross(normal,yaxis)])


def digitize():
    path=OUT/'rowimu2015.pdf'
    if not path.exists():path.write_bytes(urllib.request.urlopen(URL,timeout=60).read())
    images=list(PdfReader(path).pages[2].images)
    # Pixel coordinates inspected against tick labels in original embedded plots.
    specs={'sweep_deg':(4,134,518,138,2.06),'elevation_deg':(2,124,498,143,12.36),
           'feather_deg':(1,128,514,312,2.44),'handle_force_N':(0,126,507,285,1.335)}
    phase=np.linspace(0,1,401);result={'phase':phase};audit={}
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(name,(idx,x0,x1,y0,scale)) in zip(axes.flat,specs.items()):
        im=np.asarray(images[idx].image.convert('RGB')).astype(float)
        mask=(im.max(2)-im.min(2)>35)&(im.min(2)<225)
        cols=np.arange(x0,x1+1);values=[]
        for x in cols:
            ys=np.where(mask[:,max(0,x-1):x+2].any(axis=1))[0]
            values.append(np.percentile((y0-ys)/scale,[10,50,90]) if len(ys) else [np.nan]*3)
        vals=np.array(values)
        for k in range(3):
            good=np.isfinite(vals[:,k]);vals[:,k]=np.interp(cols,cols[good],vals[good,k])
        curves=np.array([np.interp(phase,(cols-x0)/(x1-x0),vals[:,k]) for k in range(3)])
        y=gaussian_filter1d(curves[1,:-1],5 if name!='handle_force_N' else 2,mode='wrap');y=np.r_[y,y[0]]
        result[name]=y;result[name+'_p10_pixels']=curves[0];result[name+'_p90_pixels']=curves[2]
        ax.imshow(im.astype(np.uint8));ax.plot(x0+phase*(x1-x0),y0-y*scale,'k--',lw=1.3)
        ax.set_title(name+' | dashed: extracted representative');ax.axis('off')
        audit[name]={'embedded_image_index':idx,'x_at_phase_0':x0,'x_at_phase_1':x1,'y_at_zero':y0,'pixels_per_unit':scale}
    fig.suptitle('Digitization QA: medians of colored curve pixels, not means of 21 raw strokes')
    fig.tight_layout();fig.savefig(OUT/'digitization_check.png',dpi=150);plt.close(fig)
    df=pd.DataFrame(result);df.to_csv(OUT/'digitized_reference.csv',index=False)
    return df,audit


def main():
    OUT.mkdir(exist_ok=True);ref,calibration=digitize();phase=ref.phase.to_numpy();T=60/26
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));d=mujoco.MjData(m);h=TrainingRowingHydrodynamics(m)
    boatq=int(m.joint('boat_free').qposadr[0]);boatv=int(m.joint('boat_free').dofadr[0])
    # Recompute vertical balance with CURRENT profile at original joint pose, level hull.
    def balance(z):
        d.qpos[boatq+2]=z;mujoco.mj_forward(m,d)
        return sum(a['force_N'][2] for a in h.force_terms(d)[1])-m.body_mass.sum()*G
    z=brentq(balance,-.3,.4,xtol=1e-11);d.qpos[boatq+2]=z;mujoco.mj_forward(m,d)
    initial_q=d.qpos.copy();summary={'source_url':URL,'source_sha256':hashlib.sha256((OUT/'rowimu2015.pdf').read_bytes()).hexdigest(),
        'data_status':'One-oar printed curves digitized; second oar mirrored; no synchronized bilateral measurements',
        'reference_statistic':'Median colored pixels per column; bands are pixel spread, not statistical confidence intervals',
        'smoothing':'Periodic Gaussian: sigma 1.25% cycle for angles, 0.5% for displayed measured force; then periodic cubic spline',
        'pixel_calibration':calibration,'stroke_rate_spm':26,'period_s':T,'water_z_m':0,'level_heave_m':z,
        'static_condition':'Vertical force balance at original pose, roll/pitch held level; not free trim equilibrium',
        'motion_condition':'Prescribed oar motion, hull attitude/heave fixed; H2 NOT executing these trajectories',
        'force_condition':'Current harmonic quasi-steady training closure plus geometric buoyancy; no slamming/history/added mass',
        'speed_condition':'U=0 is stationary hull; U=1..4 are prescribed forward speeds, not simulated speed response',
        'handle_force_definition':'Hydrodynamic moment about signed vertical oarlock axis divided by horizontal handle lever, opposing water load; excludes oar inertia, gravity, hand torque and H2 dynamics',
        'source_angle_mapping':'Physical elevation = printed elevation minus 6 deg, user-selected; feather 90deg=square',
        'plot_coordinates':'World boat x unchanged; display axis reversed so bow is left and catch at left, finish at right',
        'tip_geometry':'Single distal end edge between its upper and lower endpoints; excludes top/bottom perimeter',
        'waterline_status':'Physical vertical-balance waterline unchanged; user-selected -6deg offset; force prediction remains unvalidated',
        'sides':{}}
    allrows=[];motions={};target=np.deg2rad(ref[['sweep_deg','elevation_deg','feather_deg']].to_numpy())
    target[:,1]-=np.deg2rad(6.)
    summary['elevation_offset_deg']=-6.
    summary['offset_provenance']='Explicit user-selected geometric retargeting, not measured RowIMU calibration'
    for side,entry,sgn in zip(['port','starboard'],h.blades,[1,-1]):
        body,surf,volume,meta=entry;jids=[m.joint(side+'_'+k).id for k in ['sweep','elevation','feather']]
        qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];limits=m.jnt_range[jids];prev=d.qpos[qa].copy();qs=[];errors=[]
        # Desired actual rotation, converted to repository joint axes and offsets.
        for theta,beta,gamma in target:
            R=oar_rotation(theta,beta,gamma,sgn)
            def err(q):
                d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
                actual=d.xmat[body].reshape(3,3)
                return Rotation.from_matrix(actual.T@R).as_rotvec()
            fit=least_squares(err,np.clip(prev,limits[:,0]+1e-10,limits[:,1]-1e-10),bounds=(limits[:,0],limits[:,1]),xtol=1e-11,ftol=1e-11,gtol=1e-11)
            prev=fit.x;qs.append(prev.copy());errors.append(np.linalg.norm(fit.fun))
        qs=np.array(qs);qs[-1]=qs[0];spl=CubicSpline(phase*T,qs,bc_type='periodic');qdot=spl(phase*T,1)
        # Distal end edge only: exclude upper/lower perimeter edges.
        outline=unary_union([Polygon(t[:,1:]) for t in surf])
        boundary=np.array(outline.exterior.coords);aa=boundary[:-1];bb=boundary[1:]
        delta=bb-aa;far=max(sgn*boundary[:,0])
        keep=(np.minimum(sgn*aa[:,0],sgn*bb[:,0])>far-.03)&(abs(delta[:,1])>4*abs(delta[:,0]))
        edge_points=np.concatenate([aa[keep],bb[keep]])
        yz=edge_points[[np.argmin(edge_points[:,1]),np.argmax(edge_points[:,1])]]
        tip=np.column_stack([np.full(2,surf[:,:,0].mean()),yz])
        positions=[];sections=[];wetfraction=[];forces={u:[] for u in [0,1,2,3,4]};powers=[];levers=[];handles=[]
        for i,t in enumerate(phase):
            d.qpos[:]=initial_q;d.qpos[qa]=qs[i];d.qvel[:]=0;d.qvel[va]=qdot[i]
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            origin=d.xpos[body].copy();rot=d.xmat[body].reshape(3,3).copy();world=h.world(d,body,surf)
            positions.append(world.copy())
            sections.append(tip@rot.T+origin)
            handle=d.site(side+'_grasp').xpos.copy();handles.append(handle)
            lever=np.linalg.norm((handle-origin)[:2]);levers.append(lever)
            wet=wet_triangles(world);area,n,centres=triangle_geometry(wet)
            wetfraction.append(area.sum()/meta['area_m2'])
            _,buoyancy_moment,_=hydrostatic_wrench(h.world(d,body,volume),origin)
            v,omega=h.motion(d,body)
            for u in forces:
                vel=v+np.cross(omega,centres-origin)+np.array([u,0,0])
                ff=blade_traction(vel,rot[:,0],sgn*rot[:,1],area)
                moment=np.cross(centres-origin,ff).sum(axis=0)+buoyancy_moment
                equivalent=-sgn*moment[2]/lever
                forces[u].append(equivalent)
                power=float(np.sum(ff*vel));powers.append(power)
                if i in (40,80,160,240) and u in (0,4):
                    d.qvel[boatv]=u;mujoco.mj_forward(m,d)
                    _,loads,_=h.force_terms(d)
                    adapter_moment=sum((a['torque_Nm']+np.cross(a['point_m']-origin,a['force_N']) for a in loads if a['body']==body),np.zeros(3))
                    np.testing.assert_allclose(adapter_moment,moment,rtol=1e-10,atol=1e-9)
                    d.qvel[boatv]=0
                if not len(wet):assert abs(equivalent)<1e-9
                assert np.isfinite(equivalent)
                allrows.append(dict(side=side,phase=t,time_s=t*T,speed_m_s=u,handle_water_equivalent_N=equivalent,
                    water_force_x_N=ff[:,0].sum(),water_force_y_N=ff[:,1].sum(),wet_fraction=wetfraction[-1],relative_power_W=power,
                    sweep_q=qs[i,0],elevation_q=qs[i,1],feather_q=qs[i,2]))
        motions[side]={'positions':np.array(positions),'sections':np.array(sections),'wet':np.array(wetfraction),'forces':forces,'handles':np.array(handles)}
        summary['sides'][side]={'pivot_height_m':float(origin[2]),'orientation_retarget_max_error_deg':float(np.rad2deg(max(errors))),
            'joint_min_deg':np.rad2deg(qs.min(0)).tolist(),'joint_max_deg':np.rad2deg(qs.max(0)).tolist(),
            'handle_horizontal_lever_range_m':[float(min(levers)),float(max(levers))],
            'wet_fraction_range':[float(min(wetfraction)),float(max(wetfraction))],
            'blade_z_range_m':[float(np.min(positions,axis=(0,1,2))[2]),float(np.max(positions,axis=(0,1,2))[2])],
            'max_relative_water_power_W':max(powers),'wet_phase_range':[float(phase[np.array(wetfraction)>1e-4].min()),float(phase[np.array(wetfraction)>1e-4].max())],
            'force_peaks_N':{str(u):float(max(f)) for u,f in forces.items()}}
        assert max(powers)<1e-7
    pd.DataFrame(allrows).to_csv(OUT/'predictions.csv',index=False)
    # Figure 9-style side projections: actual blade outlines at uniform time intervals.
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,sharey=True)
    for ax,(side,motion) in zip(axes,motions.items()):
        ax.axhspan(-.5,0,color='#d8edf5');ax.axhline(0,color='#257393',lw=1)
        for i in range(0,len(phase)-1,8):
            ends=motion['sections'][i];ax.plot(ends[:,0],ends[:,2],color=plt.cm.turbo(phase[i]),lw=3,solid_capstyle='round')
        vertices=motion['sections'].reshape(-1,3);ax.set_xlim(vertices[:,0].max()+.12,vertices[:,0].min()-.12)
        ax.set_ylim(vertices[:,2].min()-.06,vertices[:,2].max()+.07);ax.set_aspect('equal',adjustable='box')
        ax.set_title(side.capitalize()+' | single outermost blade end edge');ax.set_ylabel('Height above water (m)');ax.grid(alpha=.12)
    axes[-1].set_xlabel('Boat-relative x (m); bow / forward to the LEFT; catch at left, finish at right')
    fig.suptitle('Figure 9-style tip path | shaft elevation offset -6 deg; fixed physical waterline',fontsize=14)
    fig.text(.5,.015,'Single-oar curves mirrored; user-selected -6 deg offset, not measured bilateral motion or H2 playback.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,.94,.96])
    cax=fig.add_axes([.95,.20,.012,.55]);fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,100),cmap='turbo'),cax=cax,label='Stroke cycle (%)')
    fig.savefig(OUT/'figure9_blade_paths.png',dpi=190);fig.savefig(OUT/'figure9_blade_paths.svg');plt.close(fig)
    # Figure 8-style force: stationary-hull result plus speed sensitivity.
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for side,motion in motions.items():
        axes[0].plot(phase*100,motion['forces'][0],label=side,ls='-' if side=='port' else '--')
    axes[0].plot(phase*100,ref.handle_force_N,color='k',ls=':',label='RowIMU printed curve (different boat / moving trial)')
    axes[0].set_title('Stationary hull, U = 0 m/s | hydrodynamic-equivalent normal handle force')
    for u in [1,2,3,4]:
        pair=np.mean([np.array(motions[s]['forces'][u]) for s in motions],axis=0)
        axes[1].plot(phase*100,pair,label=f'U = {u} m/s (mean per oar)')
    axes[1].plot(phase*100,ref.handle_force_N,'k:',label='RowIMU reference (not matched-condition validation)')
    axes[1].set_title('Sensitivity to prescribed boat speed | same fixed heave and oar cycle')
    for ax in axes:
        ax.axvline(40,color='#999',ls='--',lw=.8);ax.axhline(0,color='#aaa',lw=.7);ax.set_ylabel('Equivalent handle force (N)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Stroke cycle (%)')
    fig.suptitle('Figure 8-style prediction | shaft elevation offset -6 deg',fontsize=14)
    fig.text(.5,.01,'User-selected -6 deg elevation offset; water-load equivalent only, not validated handle force.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.96]);fig.savefig(OUT/'figure8_handle_force.png',dpi=180);fig.savefig(OUT/'figure8_handle_force.svg');plt.close(fig)
    summary['verification']={'adapter_moment_checks':16,'passivity':'all sampled wetted tractions non-positive power','dry_force':'zero','finite_forces':True}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['sides'],indent=2));print('level heave',z)

if __name__=='__main__':main()
