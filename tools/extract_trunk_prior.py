"""Digitize Figure 2's T6 mean curve from Pitto et al. (2024).

Usage: python tools/extract_trunk_prior.py paper.pdf --qa-dir output-directory
Requires pypdf, Pillow, numpy, scipy; no PDF is copied into the repository.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--qa-dir', type=Path, required=True)
    args = parser.parse_args()
    im = list(PdfReader(args.pdf).pages[5].images)[0].image.convert('RGB')
    if im.size != (3296, 1460):
        raise ValueError('Unexpected Figure 2 resolution; recalibrate pixel coordinates')
    gray = np.asarray(im).mean(axis=2)
    xs = np.arange(1245, 2218)
    ys = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        if 1753 <= x <= 1760:  # printed Finish event dashed line
            continue
        dark = np.flatnonzero(gray[150:385, x] < 90) + 150
        runs = np.split(dark, np.flatnonzero(np.diff(dark) > 1) + 1)
        runs = [run for run in runs if 5 <= len(run) <= 35]
        if runs:
            ys[i] = np.median(max(runs, key=len))
    valid = np.isfinite(ys)
    if valid.mean() < .97:
        raise ValueError('Insufficient mean-curve coverage; inspect the PDF')
    ys = np.interp(xs, xs[valid], ys[valid])
    deg = (304.0-ys)/((304.0-181.5)/40)
    out = ROOT/'data/reference9'
    np.savetxt(out/'trunk_vertical_raw.csv', np.c_[(xs-1245)/(2217-1245), deg],
               delimiter=',', header='figure_fraction,trunk_t6_vertical_deg', comments='', fmt='%.12g')
    phase = np.linspace(0, 1, 401)
    x = 1245+(2217-1245)*phase
    # Preserve the original printed Catch-to-Catch fraction, not elapsed time.
    values = np.interp(x, xs, deg)
    values[0] = values[-1] = (values[0]+values[-1])/2
    np.savetxt(out/'trunk_vertical_prior.csv', np.c_[phase, np.deg2rad(values)],
               delimiter=',', header='phase,trunk_t6_vertical_rad_prior', comments='', fmt='%.12g')
    metadata = dict(
        source='Pitto et al., Influence of Neuromuscular Activity and Technical Determinants on Scull Rowing Performance',
        doi='10.3390/app14199055', page=6, figure=2, panel='Trunk vertical (T6)',
        statistic='printed mean curve across 12 male rowers; not raw synchronized data',
        angle='relative to vertical; positive backward, negative forward',
        digitization=dict(image_size=list(im.size), catch_x=1245, finish_x=1756,
                          next_catch_x=2217, zero_y=304, forty_deg_y=181.5,
                          valid_column_fraction=float(valid.mean()), pixel_angle_deg=40/122.5),
        phase='original printed Catch-to-Catch fraction; no Finish remapping',
        smoothing='resampling and periodic endpoint reconciliation only',
        runtime_phase='identity; only Catch is aligned; not measured elapsed time',
        robot_proxy='torso_link local +Z projected into the vertical plane along boat heading; world vertical zero',
        range_deg=[float(values.min()),float(values.max())],
        limitation='rigid robot torso proxy for human T6 sensor; no lumbar or pelvis-angle imitation')
    (out/'trunk_vertical_provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    args.qa_dir.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(im)
    draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(240,40,40), width=3)
    im.crop((1190,60,2250,430)).save(args.qa_dir/'trunk_digitization_overlay.png')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
