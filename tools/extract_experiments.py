"""Extract published figures, not raw sensor data. Requires pdfplumber and Pillow.

python tools/extract_experiments.py --sources /path/to/downloaded/pdfs
Grift 2019 Fig. 4/10: vector coordinates in publisher PDF.
Day 2011 Fig. 11b: orange marker pixel centres after Poppler rendering at 150 dpi.
Axis calibration and colour-to-condition assignments were visually checked.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image

def label(mask):
    labels=np.zeros(mask.shape,dtype=int); n=0
    for y,x in zip(*np.where(mask)):
        if labels[y,x]: continue
        n+=1; labels[y,x]=n; stack=[(y,x)]
        while stack:
            yy,xx=stack.pop()
            for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)):
                yn,xn=yy+dy,xx+dx
                if 0<=yn<mask.shape[0] and 0<=xn<mask.shape[1] and mask[yn,xn] and not labels[yn,xn]:
                    labels[yn,xn]=n; stack.append((yn,xn))
    return labels,n

def find_objects(labels):
    result=[]
    for i in range(1,labels.max()+1):
        y,x=np.where(labels==i)
        result.append((slice(y.min(),y.max()+1),slice(x.min(),x.max()+1)))
    return result

ROOT = Path(__file__).resolve().parents[1]


def write_csv(name, fields, rows):
    target = ROOT / 'data' / name
    target.parent.mkdir(exist_ok=True)
    with target.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    args = p.parse_args()
    source = args.sources
    pdf = pdfplumber.open(source / 'grift2019.pdf')
    # Fig. 4 PDF page 10: h=0 at x188.912; 50 mm per 50.178 points.
    rows = []
    seen = set()
    for c in pdf.pages[9].curves:
        width, height = c['x1']-c['x0'], c['bottom']-c['top']
        if c['fill'] or not (3 < width < 3.6 and 3 < height < 3.6):
            continue
        x, y = (c['x0']+c['x1'])/2, (c['top']+c['bottom'])/2
        if not (138 < x < 391 and 65 < y < 135):
            continue
        key = (round(x, 5), round(y, 5))
        if key in seen:
            continue
        seen.add(key)
        h_mm = (x-188.912)/50.178*50
        cd = (211.961-y)/43.2355*0.5
        rows.append(dict(h_m=round(h_mm/1000, 7), cd=round(cd, 6),
                         pdf_x=x, pdf_y_top=y, figure='4', pdf_page=10,
                         extraction='vector_marker_not_raw_sensor'))
    rows.sort(key=lambda r:(r['h_m'],r['cd']))
    write_csv('grift2019_steady_markers.csv', list(rows[0]), rows)
    # Fig. 10a: colours in curve order start at a=.82, then wrap in legend.
    curve_accels = [.82, 1.02, 1.23, 1.43, 1.64, .41, .61]
    rows = []
    for i, c in enumerate(pdf.pages[16].curves[:12]):
        a, velocity = (curve_accels[i], .30) if i < 7 else (.82, [.20,.25,.30,.35,.40][i-7])
        xzero = 88.786 if i < 7 else 268.555
        scale = (228.544-88.786)/3 if i < 7 else (408.312-268.555)/3
        for x, y in c['pts']:
            t, force = (x-xzero)/scale, (148.449-y)/((148.449-70.657)/5)
            if -.002 <= t <= 3.002:
                rows.append(dict(trial=f'fig10{"a" if i<7 else "b"}_{i if i<7 else i-7}',
                                 time_s=t, measured_total_force_N=force,
                                 acceleration_m_s2=a, target_speed_m_s=velocity,
                                 depth_m=.1, plate_mass_kg=.4, pdf_page=17,
                                 extraction='published_filtered_vector_curve'))
    write_csv('grift2019_acceleration_curves.csv', list(rows[0]), rows)
    # Day's plot is embedded as a raster; identify the orange hollow diamonds.
    im=np.asarray(Image.open(source/'day_fig11.png').convert('RGB')).astype(int)
    r,g,b=im[:,:,0],im[:,:,1],im[:,:,2]
    mask=(r>180)&(g>50)&(g<200)&(b<130)&(r>g+35)
    roi=np.zeros_like(mask);roi[625:695,390:925]=True
    components,n=label(mask&roi)
    blobs=[]
    for idx,sl in enumerate(find_objects(components),1):
        if sl is None: continue
        yy,xx=np.where(components[sl]==idx)
        if len(xx)<5:continue
        x=(sl[1].start+sl[1].stop-1)/2
        y=(sl[0].start+sl[0].stop-1)/2
        if sl[1].stop-sl[1].start>18:continue
        blobs.append([sl[1].start,sl[1].stop-1,sl[0].start,sl[0].stop-1])
    # Blue circles overlap some orange diamonds and split their coloured outline.
    # Merge fragments at the same x; use the combined bounding box, not each fragment.
    groups=[]
    for box in sorted(blobs):
        centre=(box[0]+box[1])/2
        if groups and abs(centre-(groups[-1][0]+groups[-1][1])/2)<5:
            q=groups[-1]; groups[-1]=[min(q[0],box[0]),max(q[1],box[1]),min(q[2],box[2]),max(q[3],box[3])]
        else: groups.append(box)
    rows=[]
    for box in groups:
        x,y=(box[0]+box[1])/2,(box[2]+box[3])/2
        u=1.75+(x-380)/(975-380)*2.5
        ct=.0035-(y-547)/(770-547)*.001
        rows.append(dict(speed_m_s=u,ct=ct,pixel_x=x,pixel_y=y,
                         ct_digitization_bound=2/(770-547)*.001,
                         source='Day2011_fig11b_free_transition',pdf_page=12))
    rows.sort(key=lambda r:r['speed_m_s'])
    write_csv('day2011_hull_resistance.csv',list(rows[0]),rows)
    meta={
      'notice':'Published figure digitization; these are NOT raw experimental records.',
      'grift2019':{'doi':'10.1017/jfm.2019.102','license':'CC BY 4.0',
        'url':'https://www.cambridge.org/core/services/aop-cambridge-core/content/view/EB6F8190B82EF385B98BFA5D70C3EB10/S0022112019001022a.pdf/drag_force_on_an_accelerating_submerged_plate.pdf',
        'geometry_m':[.2,.1,.005],'rho_kg_m3':1000,'mass_kg':.4,
        'time_axis_point_bound':.05/((228.544-88.786)/3),
        'force_axis_point_bound_N':.05/((148.449-70.657)/5),
        'steady_depth_assignment':'Marker/grid systematic offset ~0.24 mm. Model groups nearest integer-mm nominal depths, tolerance 0.5 mm. Extracted CSV retains original vector coordinates.',
        'limitations':'Vector curve is already filtered/decimated in publication. Quantization bounds are not experimental uncertainty. Digitization cannot recover sensor sampling or covariance.'},
      'day2011':{'doi':'10.1177/1475090211406775',
        'url':'https://strathprints.strath.ac.uk/35941/1/IMechE_An_experimental_study_of_unsteady_hydro.pdf',
        'waterline_length_m':7.937,'waterline_beam_m':.280,'draft_m':.105,
        'displacement_kg':118.4,'wetted_area_m2':2.340,
        'limitations':'Different hull from repository. Orange free-transition diamonds only. Pixel calibration at 150 dpi; approximate +/-2 pixels, not original measurement error.'},
      'hashes':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['grift2019.pdf','day2011.pdf','day_fig11.png']}}
    (ROOT/'data'/'provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Grift markers',len(seen),'Day markers',len(rows),'written to',ROOT/'data')


if __name__=='__main__':main()
