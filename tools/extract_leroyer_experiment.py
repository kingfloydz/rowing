"""Manual seed positions on native raster figures, snapped to nearby dark ink.

These are 31 paired readings of ONE simplified stroke, not 31 experiments.
The source was visually inspected at native and enlarged resolution. Source
images remain outside this checkout. Inspection overlays are scratch artifacts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
THETA=np.arange(15,166,5)
SEEDS={
    'fx': [40,60,69,95,116,119,137,170,198,214,228,239,243,240,229,224,
           229,241,250,251,242,228,217,209,197,176,147,114,82,59,52],
    'fy': [116,94,94,64,51,68,77,69,62,74,88,106,118,132,142,153,
           165,182,201,219,234,245,259,276,291,290,273,254,241,233,211],
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('pdf',type=Path)
    ap.add_argument('--earlier-pdf',type=Path)
    ap.add_argument('--inspection-dir',type=Path,required=True);a=ap.parse_args()
    a.inspection_dir.mkdir(parents=True,exist_ok=True)
    reader=PdfReader(a.pdf);records=[];axes={}
    for component,page,index,zero,scale in [('fx',10,0,41.,4.1),('fy',9,0,154.,3.08)]:
        im=reader.pages[page-1].images[index].image.convert('RGB')
        pixels=np.asarray(im.convert('L'));draw=ImageDraw.Draw(im)
        axes[component]=dict(pdf_page=page,image_index=index,x_zero=59.,x_180=458.,
                             y_zero=zero,pixels_per_newton=scale)
        for theta,seed in zip(THETA,SEEDS[component]):
            x=59+(458-59)*theta/180
            # Snap within three pixels only; no inference across neighboring
            # curves. Overlapping curves carry the full reading-error bound.
            ys=np.arange(max(0,seed-3),min(pixels.shape[0],seed+4))
            ink=np.maximum(0.,120-pixels[ys,int(round(x))].astype(float))
            y=float(np.sum(ys*ink)/ink.sum()) if ink.sum()>0 else float(seed)
            records.append(dict(theta_deg=int(theta),component=component,force_N=(zero-y)/scale,
                                pixel_x=float(x),pixel_y=y,manual_seed_y=seed,
                                reading_bound_N=3./scale,pdf_page=page,
                                figure=10 if component=='fx' else 9))
            draw.ellipse((x-2,y-2,x+2,y+2),outline='red',width=1)
        im.resize((im.width*3,im.height*3)).save(a.inspection_dir/f'leroyer_{component}_digitization.png')
    target=ROOT/'data/leroyer2010_force_trajectory.csv'
    with target.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    provenance=dict(doi='10.1080/02640414.2010.485646',source_path=str(a.pdf),
        sha256=hashlib.sha256(a.pdf.read_bytes()).hexdigest(),method='manual native raster seeds with local ink snap',
        axes=axes,reading_uncertainty='Conservative +/-3 native pixels per ordinate; not a statistical confidence interval.',
        experiment_accuracy='Paper reports force accuracy better than 0.5 N and about +/-3% measurement uncertainty.',
        data_family='Barre1998 Nantes towing-tank database, shared with Barre/Kobus2010. Independent campaign NOT established.',
        condition=dict(rotation_rate_rad_s=3.,boat_speed_m_s=2.91,advance_number_printed=.742,
                       lever_m=1.308,chord_m=.3,height_m=.14,area_m2=.042,top_edge_depth_m=0.),
        lever_note='Le=1.308 m is explicit in Leroyer2008 Fig2, and gives Leroyer2010 Table I centre normal speed 1.014 m/s. Printed eta=.742 is rounded; eta=2.91/(3*1.308) used.',
        scope='One simplified permanently immersed stroke; not measured human rowing or catch/release validation.',
        sample_count='31 paired angle samples, 62 force ordinates, one condition. CFD curves are not experimental observations.')
    (ROOT/'data/provenance_leroyer2010.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    print(target, len(records))
    if a.earlier_pdf: extract_earlier(a.earlier_pdf)


def extract_earlier(path):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        curves=pdf.pages[13].curves
        maps={
            'minus_fx':([0,1,2],199.009,349.226,190.001,(190.001-69.542)/120),
            'fy_plot':([368,369,370],387.910,538.872,132.556,(200.011-65.103)/120),
            'minus_mz':([921,922,923],389.986,538.745,379.100,(379.100-254.566)/160),
            'dip_over_half_chord':([1290],330.963924,556.163866,497.585680,(569.858812-497.585680)/.8),
        }
        rows=[]
        for name,(ids,x0,x180,y0,scale) in maps.items():
            for ci in ids:
                for x,y in curves[ci]['pts']:
                    rows.append(dict(variable=name,theta_deg=180*(x-x0)/(x180-x0),value=(y0-y)/scale,
                                     pdf_x=x,pdf_y=y,pdf_curve=ci,pdf_page=14,
                                     figure={'minus_fx':19,'fy_plot':19,'minus_mz':20,'dip_over_half_chord':21}[name]))
    target=ROOT/'data/leroyer2008_force_moment.csv'
    with target.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    provenance=dict(doi='10.1007/s00773-007-0256-7',source_url='https://hal.science/hal-00699499/document',
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pdf_page=14,vector_maps=maps,
        condition=dict(rotation_rate_rad_s=3.,advance_number_printed=.5887,boat_speed_m_s=2.31,lever_m=1.308),
        family='Same Nantes towing-tank database as Leroyer2010 and Barre2010, different advance condition.',
        sign_note='Fig19 ordinate is printed -Fy, inconsistent with Fig4/Eq3. Preserve fy_plot; evaluate both signs against measured moment and Fig21 before choosing physical Fy.',
        units_note='Fig20 axis prints N, but moment Mz has units N m by its definition and Eq3.',
        reading_bound='Assumed +/-0.2 PDF points for vector coordinates; not sensor uncertainty.',
        sample_note=f'{len(rows)} drawing vertices, NOT independent sensor samples or experiments. Fig18 and Fig20a duplicates excluded.')
    (ROOT/'data/provenance_leroyer2008.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(target,len(rows))


if __name__=='__main__':main()
