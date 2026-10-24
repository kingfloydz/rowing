"""Dode2022 Fig14 experimental marker centres from connecting-path vertices."""
import argparse,csv,hashlib,json
from pathlib import Path
import pdfplumber
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('pdf',type=Path);a=ap.parse_args()
    rows=[];theory=[]
    with pdfplumber.open(a.pdf) as p:
        q=p.pages[15]
        for i,fn,hull,epsilon in [(0,.3,'A',.1),(1,.3,'A',.2),(2,.3,'A',.25),(3,.3,'A',.3),
                                   (4,.3,'B',.3),(19,.57,'C',.1),(20,.57,'C',.2),(21,.57,'C',.3)]:
            x0,dx,y0,dy=(112.55,26.174,170.312,22.63275) if fn==.3 else (289.616,32.7185,68.459,16.202)
            for x,y in q.curves[i]['pts']:
                rows.append(dict(hull=hull,fn=fn,epsilon=epsilon,omega_nd=(x-x0)/dx,a0_normalized=(y0-y)/dy,
                                 plot_clipped=int(y<68.459),pdf_page=16,figure=14,curve_index=i,pdf_x=x,pdf_y_top=y))
        for i,fn in [(5,.3),(22,.57)]:
            x0,dx,y0,dy=(112.55,26.174,170.312,22.63275) if fn==.3 else (289.616,32.7185,68.459,16.202)
            for x,y in q.curves[i]['pts']:
                theory.append(dict(fn=fn,omega_nd=(x-x0)/dx,a0_normalized=(y0-y)/dy,pdf_x=x,pdf_y_top=y))
    for name,data in [('dode2022_wave_experiments.csv',rows),('dode2022_published_theory.csv',theory)]:
        with (ROOT/'data'/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        print(name,len(data))
    (ROOT/'data/provenance_waves.json').write_text(json.dumps(dict(doi='10.1017/jfm.2022.592',sha256=hashlib.sha256(a.pdf.read_bytes()).hexdigest(),
        source='Fig14, PDF page16: experimental marker centres at vector polyline vertices, checked visually',
        omitted='Hull A epsilon15% is rasterized/overlapped; not digitized. One C epsilon20% vertex above plot boundary excluded from metrics.',
        normalization='A0=(mean Cw-Cw_steady)/(max Cw_steady * epsilon²), not total force',
        independence='Multiple sizes, frequencies, amplitudes; published averages of repeated measurements, not raw repeats.',
        uncertainty='Fig14 has no individual error bars; uncertainty in A0 is amplified by division by epsilon².'),indent=2),encoding='utf8')


if __name__=='__main__':main()
