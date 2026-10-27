"""Rebuild priors with catch translation only, without within-cycle time warps."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/reference9'


def periodic_source(x, y):
    # Reconcile only the periodic seam. Digitized elbow endpoints miss ~0.5%.
    endpoint = (y[0]+y[-1])/2
    interior = (x > 1e-8) & (x < 1-1e-8)
    x = np.r_[0, x[interior], 1]
    y = np.r_[endpoint, y[interior], endpoint]
    return PchipInterpolator(np.r_[x[:-1]-1, x, x[1:]+1],
                             np.r_[y[:-1], y, y[1:]])


def main():
    seat = np.loadtxt(DATA/'wirms_stage4_raw.csv', delimiter=',', skiprows=1)
    elbow = np.loadtxt(DATA/'pitto_left_elbow_raw.csv', delimiter=',', skiprows=1)
    trunk = np.loadtxt(DATA/'trunk_vertical_raw.csv', delimiter=',', skiprows=1)
    phase = np.linspace(0,1,401)
    seat_catch = float(seat[seat[:,1].argmin(),0])
    s = periodic_source(seat[:,0],seat[:,1])((phase+seat_catch)%1)
    s = (s-seat[:,1].min())/np.ptp(seat[:,1])
    e = np.deg2rad(periodic_source(elbow[:,0],elbow[:,1])(phase))
    t = np.deg2rad(periodic_source(trunk[:,0],trunk[:,1])(phase))
    s[-1],e[-1],t[-1] = s[0],e[0],t[0]
    np.savetxt(DATA/'aligned_priors.csv', np.c_[phase,s,e], delimiter=',', comments='',
               header='phase,seat_normalized_prior,human_left_elbow_flexion_rad_prior',fmt='%.12g')
    np.savetxt(DATA/'trunk_vertical_prior.csv', np.c_[phase,t], delimiter=',', comments='',
               header='phase,trunk_t6_vertical_rad_prior',fmt='%.12g')
    report = dict(method='catch-only cyclic translation; no finish alignment or nonlinear phase warp',
        seat_catch_source_phase=seat_catch,
        seat_catch_definition='minimum seat displacement (turnaround), proxy within printed Catch interval',
        seat_source_peak_phase=float(seat[seat[:,1].argmax(),0]),
        seat_aligned_peak_phase=float((seat[seat[:,1].argmax(),0]-seat_catch)%1),
        elbow_catch_source_phase=0, trunk_catch_source_phase=0, oar_catch_source_phase=0,
        elbow_finish_source_fraction=(653-142)/(1122-142),
        trunk_finish_source_fraction=(1756-1245)/(2217-1245),
        oar_finish_source_phase=.4,
        seam='average first/last extracted values; interpolate missing endpoint margins; no other smoothing',
        limitation='Pitto figure fractions are not measured elapsed time; sources are not synchronized recordings',
        amplitude='original paper shape retained; existing robot seat amplitude and elbow geometric conversion retained')
    (DATA/'catch_alignment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    p=DATA/'source_provenance.json';meta=json.loads(p.read_text())
    meta['target_events']={'catch_phase':0,'release_phase':'source-specific; no forced alignment'}
    meta['seat_phase_method']='cyclic shift by raw seat minimum; no drive/recovery rescaling'
    meta['elbow_phase_method']='original printed Catch-to-Catch figure fraction; no Finish remapping'
    meta['smoothing']='PCHIP resampling and endpoint reconciliation only; raw curves retained'
    meta['runtime_alignment']='catch_alignment.json; supersedes historical model_seat_mapping below'
    p.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    p=DATA/'trunk_vertical_provenance.json';meta=json.loads(p.read_text())
    meta['phase']='original printed Catch-to-Catch figure fraction; Finish is not remapped to 0.4'
    meta['runtime_phase']='identity; Catch aligned at 0; no nonlinear time warp'
    meta['smoothing']='PCHIP resampling and endpoint reconciliation only'
    meta['range_deg']=[float(np.rad2deg(t.min())),float(np.rad2deg(t.max()))]
    p.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
