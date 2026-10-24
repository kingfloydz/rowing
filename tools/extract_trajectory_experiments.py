"""Digitize published vector paths, retaining source locations and curve identities.

Run with pdfplumber's Python. These are figure traces, NOT raw sensor records.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]


def write(name, rows):
    with (ROOT/'data'/name).open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(name, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grift', type=Path, required=True)
    ap.add_argument('--day', type=Path, required=True)
    ap.add_argument('--caplan', type=Path)
    a = ap.parse_args()
    rows = []
    with pdfplumber.open(a.grift) as p:
        q = p.pages[12]
        for beta, ids in [(0, (4,5,7,10,11,13)), (15, (2,3,6,8,9,12))]:
            # Dashed velocity follows dX/dt, solid follows dY/dt. The plotted
            # Fig.8(d) legend appears interchanged; retain that audit explicitly.
            for name, idx in zip(('x_m','y_m','theta_deg','vx_m_s','vy_m_s','alpha_deg'), ids):
                if name=='theta_deg': x0,x1,y0,dy = 86.19,227.34,246.88,1.0085
                elif name=='alpha_deg':x0,x1,y0,dy=86.192,227.343,441.296,(441.296-356.361)/200
                elif name in ('x_m','y_m'): x0,x1,y0,dy = 277.92,418.45,117.24,125.8
                else: x0,x1,y0,dy = 277.92,418.45,256.965,50.425
                for x,y in q.curves[idx]['pts']:
                    rows.append(dict(beta_deg=beta, variable=name, phase=(x-x0)/(x1-x0),
                                     value=(y0-y)/dy, pdf_page=13, figure=8,
                                     curve_index=idx, pdf_x=x, pdf_y_top=y))
        write('grift2021_kinematics.csv', rows)
        rows = []
        # Fig.9b: raw Fn,Ft; Fig.9d is a rescaling of these same data.
        for k, ids in [(.5,(0,1)),(.75,(2,3)),(1.,(4,5))]:
            for name, idx in zip(('fn_N','ft_N'), ids):
                for x,y in p.pages[13].curves[idx]['pts']:
                    rows.append(dict(beta_deg=0,kappa=k,variable=name,phase=(x-283.28)/(428.19-283.28)*2-.5,
                                     value=(108.62-y)/1.371,pdf_page=14,figure=9,curve_index=idx,pdf_x=x,pdf_y_top=y))
        # Fig.22: beta15 is a different blade mounting and a genuinely different
        # tip trajectory. beta0 is a duplicate control, never a new test case.
        for beta, offset in [(0,0),(15,2)]:
            for name,idx in [('fn_N',offset),('ft_N',offset+1),('fx_N',offset+4),('fy_N',offset+5)]:
                x0,x1=(84.052,231.789) if idx<4 else (268.214,415.950)
                for x,y in p.pages[28].curves[idx]['pts']:
                    rows.append(dict(beta_deg=beta,kappa=1.,variable=name,phase=(x-x0)/(x1-x0)*2-.5,
                                     value=(132.602-y)/1.97746,pdf_page=29,figure=22,curve_index=idx,pdf_x=x,pdf_y_top=y))
        # Fig.10 Fx,Fy are a separate coordinate check of the Fig.9 kappa1 case.
        for name,idx in [('fx_N',2),('fy_N',3)]:
            for x,y in p.pages[14].curves[idx]['pts']:
                rows.append(dict(beta_deg=0,kappa=1.,variable=name,phase=(x-91.23)/(233.48-91.23)*2-.5,
                                 value=(125.27-y)/1.9365,pdf_page=15,figure=10,curve_index=idx,pdf_x=x,pdf_y_top=y))
        write('grift2021_force_trajectories.csv',rows)
    # Day Fig.3: digitize the excursion; periodic smooth differentiation avoids
    # interpolation over the gaps in the dashed velocity and acceleration paths.
    rows=[]
    with pdfplumber.open(a.day) as p:
        q=p.pages[5]
        for x,y in q.curves[0]['pts']:
            rows.append(dict(phase=(x-88.396)/(263.170-88.396),excursion_m=(138.335-y)/155.688,
                             pdf_page=6,figure=3,curve_index=0,pdf_x=x,pdf_y_top=y))
    write('day2011_rowing_excursion.csv',rows)
    rows=[]
    for motion,surface,increase in [('sinusoidal','free_transition',1.8),('rowing','free_transition',3.),
                                   ('sinusoidal','studs_400mm',1.9),('rowing','studs_400mm',2.4)]:
        rows.append(dict(motion=motion,surface=surface,mean_speed_m_s=4.,period_s=1.84,
                         resistance_increase_percent=increase,pdf_page=13,table=4))
    write('day2011_mean_unsteady_resistance.csv',rows)
    if a.caplan:
        # Manually digitized hollow markers in native embedded Fig8 bitmap,
        # PDF page6. Keep coordinates and +/-3 pixel reading uncertainty.
        # Endpoints overlap other symbols; treat these readings as low confidence.
        cd_y=[280,278,265,245,166,122,100,98,69,58,60,67,90,120,158,205,244,265,276]
        cl_y=[278,257,213,161,142,145,174,207,241,277,312,348,383,405,423,403,368,334,298]
        rows=[]
        for deg,yd,yl in zip(range(0,181,10),cd_y,cl_y):
            rows.append(dict(angle_deg=deg,cd=max(0,(279.5-yd)/107.6),cl=(279.5-yl)/107.6,
                             pixel_x=114.5+deg*(920-114.5)/180,cd_pixel_y=yd,cl_pixel_y=yl,
                             coefficient_reading_bound=3/107.6,pdf_page=6,figure=8,
                             low_confidence_endpoint=int(deg in (0,180))))
        write('caplan2007_curved_polar.csv',rows)
    provenance={
        'grift':{'doi':'10.1017/jfm.2021.318','sha256':hashlib.sha256(a.grift.read_bytes()).hexdigest(),
                 'kind':'published vector paths, not raw measurements',
                 'velocity_legend_audit':'Fig8d dashed matches dX/dt; solid matches dY/dt. Duration is inferred using positions and velocities only.',
                 'force_basis':'Measured Fn,Ft use the shaft/transducer basis n=(-cos(theta),-sin(theta)), t=(sin(theta),-cos(theta)), also for beta15. Blade chord rotates by theta+beta. Independently checked against Fx,Fy.',
                 'notes':['Do not count Fig22 beta0 or Fig10 as independent validation.',
                          'Measured water forces already have dry blade/strut inertia subtracted.',
                          'Plot sampling density does not equal experimental sample count.',
                          'Only 0<=phase<=1 kinematics available. No verification of recovery or the full release tail.']},
        'day':{'doi':'10.1177/1475090211406775','sha256':hashlib.sha256(a.day.read_bytes()).hexdigest(),
               'kind':'Fig3 excursion vector path and Table4 printed values',
               'notes':['Table4 is total mean resistance relative to steady at the same mean speed.',
                        'Not an additive 3 percent correction to instantaneous R(U).',
                        'Free-transition and tripped cases are distinct conditions.']}}
    if a.caplan:
        provenance['caplan_curved']={'doi':'10.1080/02640410600809985',
            'sha256':hashlib.sha256(a.caplan.read_bytes()).hexdigest(),
            'source':'PDF page6 Fig8 native embedded image0; manually read hollow square/circle centres at 10 degree intervals',
            'bounds':'3 pixels = 0.0279 coefficient reading bound; overlapped endpoints lower confidence',
            'scope':'Longitudinal curvature only, no transverse curvature, no dynamic data; reverse-side reflection unvalidated.'}
    (ROOT/'data/provenance_trajectories.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')


if __name__=='__main__': main()
