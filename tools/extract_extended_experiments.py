"""Reproducible published-figure digitization, never raw experimental records.

Uses pdfplumber (bundled Python) and Pillow. Source PDFs remain outside the repo.
Curve identities, colours, symbols and axis calibrations were visually checked.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pdfplumber
from pypdf import PdfReader
from extract_experiments import label, find_objects

ROOT=Path(__file__).resolve().parents[1]


def write(name,rows):
    with (ROOT/'data'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(name,len(rows))


def main():
    ap=argparse.ArgumentParser()
    for name in ['caplan','barre','reijtenbagh']: ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    # Caplan Fig.6: isolated filled squares in embedded image, exact native pixels.
    im=np.asarray(PdfReader(a.caplan).pages[4].images[0].image.convert('L'))
    dark=im<100
    eroded=np.ones_like(dark)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]: eroded &= np.roll(dark,(dy,dx),(0,1))
    labels,_=label(eroded);points=[]
    for y,x in find_objects(labels):
        h,w=y.stop-y.start,x.stop-x.start
        if 2<=h<=9 and 2<=w<=9 and 98<x.start<925 and 40<y.start<410:
            points.append(((x.start+x.stop-1)/2,(y.start+y.stop-1)/2))
    rows=[]
    for deg in range(0,181,5):
        group=sorted([(x,y) for x,y in points if round((x-104.5)/(920-104.5)*180/5)*5==deg],key=lambda p:p[1])
        if deg==45: continue  # Two series overlap; do not invent separate ordinates.
        if deg in (0,180): group=group*2
        if len(group)!=2: raise ValueError((deg,group))
        upper,lower=group
        cd,cl=(lower,upper) if deg<45 else (upper,lower)
        rows.append(dict(angle_deg=deg,cd=(269-cd[1])/107.5,cl=(269-cl[1])/107.5,
                         cd_pixel_x=cd[0],cd_pixel_y=cd[1],cl_pixel_x=cl[0],cl_pixel_y=cl[1],
                         coefficient_digitization_bound=2.5/107.5,
                         overlap_boundary=int(deg in (0,180)),pdf_page=5,figure=6))
    write('caplan2007_flat_polar.csv',rows)
    p=pdfplumber.open(a.barre)
    # Fig.8 right-hand panels only: left panels repeat the same force data vs theta.
    maps={'cd':[(784,),(1024,1025),(1289,1290),(1291,)],
          'cl':[(2457,),(2754,2755),(3089,),(3090,)]}
    conditions=[(3,.77),(3,.71),(4,.77),(4,.71)]
    rows=[]
    for field,series in maps.items():
        for (k,eta),ids in zip(conditions,series):
            pts=np.concatenate([np.asarray(p.pages[8].curves[i]['pts']) for i in ids])
            for x,y in pts:
                angle=(x-310.707)/(458.281-310.707)*180
                value=(464.750-y)/(464.750-308.487)*2.5 if field=='cd' else (648.721-y)/(648.721-538.515)*1.5
                rows.append(dict(rotation_rate_rad_s=k,advance_number=eta,coefficient=field,
                                 angle_deg=angle,value=value,pdf_x=x,pdf_y_top=y,pdf_page=9,figure=8))
    write('barre2010_dynamic_polars.csv',rows)
    # Fig.11: squares=K3, triangles=K3.5, diamonds=K3.75, circles=K4.
    # Illustrator represents circles as six-point Bezier paths, not as ellipses.
    rows=[];q=p.pages[11]
    for shape,objs in [('rect',q.rects),('curve',q.curves)]:
        for c in objs:
            x=(c['x0']+c['x1'])/2;y=(c['top']+c['bottom'])/2
            if not (190<x<480 and 3.7<c['width']<4.1 and 3.7<c['height']<4.1): continue
            if shape=='curve' and c['fill']:continue
            k=3. if shape=='rect' else {4:3.5,5:3.75,6:4.}.get(len(c['pts']))
            if k is None:continue
            fr=.2+(x-124.075)/(486.654-124.075)*1.8
            value=(291.531-y)/(291.531-84.305)*4
            rows.append(dict(rotation_rate_rad_s=k,reduced_frequency=fr,
                             coefficient='cd_max' if value>1.25 else 'cl_max',value=value,
                             pdf_x=x,pdf_y_top=y,pdf_page=12,figure=11))
    # The two largest CL maxima exceed 1.25; distinguish by their lower y-series.
    for r in rows:
        if r['rotation_rate_rad_s']==3 and r['pdf_y_top']>205:r['coefficient']='cl_max'
    write('barre2010_maxima.csv',rows)
    # Fig.5 flat blade zero-torque point, right panel only, four conditions.
    rows=[]
    for (k,eta),idx in zip(conditions,[158,207,280,315]):
        for x,y in p.pages[6].curves[idx]['pts']:
            rows.append(dict(rotation_rate_rad_s=k,advance_number=eta,
                             angle_deg=(x-301.747)/(468.716-301.747)*180,
                             dip=(149.151-y)/(149.151-113.459),
                             clipped=int(y<=95.855),pdf_x=x,pdf_y_top=y,pdf_page=7,figure=5))
    write('barre2010_zero_torque_point.csv',rows)
    # 2026 Fig.12a and 14a: select only solid experimental paths, never dashed models.
    p=pdfplumber.open(a.reijtenbagh);rows=[]
    for page,kind,params,ids,x0,x1,y0,dy in [
        (20,'constant_jerk',[.1,.2,.3,.4,.6,.8],[(0,1,2,3,4),(6,7),(9,10),(12,13),(15,),(17,)],76.793,227.213,165.550,24.007),
        (22,'sqrt_velocity',[.16,.25,.36,.49,.64],[(11,),(13,),(15,),(17,),(19,)],77.400,226.386,166.336,23.778)]:
        tmax=3. if kind=='constant_jerk' else 2.
        for param,segments in zip(params,ids):
            for idx in segments:
                c=p.pages[page].curves[idx]
                assert c['dash'][0]==[]
                for x,y in c['pts']:
                    rows.append(dict(motion=kind,parameter=param,segment=idx,time_s=(x-x0)/(x1-x0)*tmax,
                                     measured_excess_force_N=(y0-y)/dy,final_speed_m_s=.6,
                                     pdf_page=page+1,figure=12 if page==20 else 14,pdf_x=x,pdf_y_top=y))
    write('reijtenbagh2026_excess_force.csv',rows)
    meta={
        'warning':'Published plotted data; not original sensor time series. Figure interpolation is not independent experimental validation.',
        'caplan2007':{'doi':'10.1080/02640410600809985','geometry':'quarter-scale flat Big-Blade outline; no curvature',
                      'area_m2':.007742,'speed_m_s':.75,'rho_kg_m3':999,'top_edge_depth_m':0,
                      'extraction':'native embedded raster filled-square centres; axes x=104.5..920 for 0..180deg; y=269,107.5 pixels/unit',
                      'exclusions':'45deg overlapping markers; 0/180 overlap used only as training boundaries',
                      'uncertainty':'+/-2.5 pixels is a digitization bound, not experimental uncertainty'},
        'barre2010':{'doi':'10.1243/17543371JSET43','geometry':'0.300m chord x 0.140m height rectangular flat blade',
                     'area_m2':.042,'top_edge_depth_m':0,'kinematics':'theta_dot=K*sin(theta), Vboat=eta*K*Le',
                     'extraction':'PDF vector paths; Fig8 and5 right panels only to avoid duplicate data; Fig11 marker shapes',
                     'uncertainty':'coordinates rounded in publication; assume +/-0.15 PDF point for sensitivity, not sensor accuracy',
                     'limitations':'Fig11 K groups have different fr ranges; leave-K-out within overlap only. Actual lever not specified in accessible table; no inferred dimensional Fx test.'},
        'reijtenbagh2026':{'doi':'10.1017/jfm.2025.11107','geometry_m':[.2,.1,.004],
                          'rho_kg_m3':998,'nu_m2_s':.978e-6,'top_edge_depth_m':.15,'cd':1.3,
                          'quantity':'Fa=FD-FQS; dry plate/strut inertia already removed',
                          'extraction':'solid coloured PDF paths; published theoretical dashed paths excluded',
                          'limitations':'Nominal motion differs from actual robot motion (Appendix A). Some force paths clip at Fa=0; source curves are filtered. Paper model is itself calibrated on related experiments.'},
        'sha256':{key:hashlib.sha256(getattr(a,key).read_bytes()).hexdigest() for key in ['caplan','barre','reijtenbagh']}}
    (ROOT/'data/provenance_extended.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


if __name__=='__main__':main()
