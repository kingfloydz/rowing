"""Extract published seat/left-elbow curves, retaining source and phase provenance.

No training references or scene files are overwritten. These are cross-study
priors, NOT nine-channel synchronized experimental measurements.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
from PIL import Image
from pypdf import PdfReader
from pypdf.generic import ContentStream
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/reference9'
QA = ROOT / 'results/reference9'


def save_csv(path, names, values):
    np.savetxt(path, values, delimiter=',', header=','.join(names), comments='', fmt='%.12g')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def purple_paths(page):
    """Figure 4 is a PDF Form XObject with MATLAB vector polylines."""
    stream = ContentStream(page['/Resources']['/XObject']['/Im9'].get_object(), page.pdf)
    color, points, paths = None, [], []
    for args, op in stream.operations:
        if op == b'RG':
            color = np.array(args, dtype=float)
        elif op == b'G':
            color = None
        elif op in (b'm', b'l'):
            points.append(np.array(args, dtype=float))
        elif op == b'S':
            if color is not None and np.allclose(color, [.4941, .1843, .5569]) and len(points) > 100:
                paths.append(np.array(points))
            points = []
        elif op in (b'f', b'f*', b'n'):
            points = []
    assert len(paths) == 3, 'Unexpected Figure 4 vector structure'
    assert all(len(p) == 1001 for p in paths)
    return paths


def periodic(x, y):
    """Small smoothing and endpoint averaging; preserve unmodified raw export."""
    p = np.linspace(0, 1, 1001)
    a = np.interp(p, x, y)
    a[0] = a[-1] = (a[0]+a[-1])/2
    a = gaussian_filter1d(a[:-1], 2, mode='wrap')
    return CubicSpline(p, np.r_[a, a[0]], bc_type='periodic')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--papers', type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    QA.mkdir(exist_ok=True, parents=True)
    seat_pdf = args.papers/'sensors-23-01060.pdf'
    elbow_pdf = args.papers/'Influence_of_Neuromuscular_Activity_and_Technical_.pdf'
    paths = purple_paths(PdfReader(seat_pdf).pages[7])
    # Raw MATLAB path coordinates. Tick positions verified against the PDF.
    # Seat: y=573 -> 0 mm; y=205.285714 -> 600 mm.
    # Force: y=1105.25 -> 0 N; each 100 N -> 71.5 coordinate units.
    source_phase = (paths[0][:, 0]-400)/3043
    seat = (573-paths[0][:, 1])/(429/700)/1000
    force = (1105.25-paths[1][:, 1])/.715
    save_csv(OUT/'wirms_stage4_raw.csv', ['source_phase','seat_m','handle_force_N'],
             np.column_stack([source_phase, seat, force]))

    embedded = list(PdfReader(elbow_pdf).pages[5].images)
    assert len(embedded) == 1
    im = embedded[0].image.convert('RGB')
    assert im.size == (3296, 1460), 'Figure image changed; recalibrate'
    im.save(QA/'pitto_figure2.png')
    grey = np.asarray(im).mean(axis=2)
    xs = np.arange(147, 1118)
    ys = np.full(len(xs), np.nan)
    for j, x in enumerate(xs):
        if abs(x-653) <= 5:  # dashed Finish line, not the mean curve
            continue
        dark = np.flatnonzero(grey[585:827, x] < 85)+585
        if len(dark):
            # The bold mean is much darker than the SD band/grid.
            ys[j] = np.median(dark)
    good = np.isfinite(ys)
    assert good.mean() > .97
    ys = np.interp(xs, xs[good], ys[good])
    elbow_deg = (830-ys)/(232/120)
    elbow_phase = (xs-142)/(1122-142)
    save_csv(OUT/'pitto_left_elbow_raw.csv', ['figure_fraction','left_elbow_flexion_deg'],
             np.column_stack([elbow_phase, elbow_deg]))

    # Force crossings are operational event proxies, NOT measured blade entry.
    # Record sensitivity to threshold; exact cross-study synchronization is absent.
    events = {}
    peak = int(force.argmax())
    for frac in (.02,.05,.10):
        threshold = frac*force.max()
        before = np.flatnonzero(force[:peak] < threshold)[-1]
        after = peak+np.flatnonzero(force[peak:] < threshold)[0]
        catch = np.interp(threshold, force[before:before+2], source_phase[before:before+2])
        release = np.interp(threshold, force[after-1:after+1][::-1], source_phase[after-1:after+1][::-1])
        events[str(frac)] = dict(catch=float(catch), release=float(release))
    catch, release = (events['0.05'][k] for k in ('catch','release'))
    phi = np.linspace(0, 1, 401)
    # Piecewise event mapping, then periodic smoothing removes derivative kinks.
    src = np.where(phi <= .4, catch+(release-catch)*phi/.4,
                   release+(1+catch-release)*(phi-.4)/.6) % 1
    seat_curve = periodic(source_phase, seat)
    sp = seat_curve(src)
    sp = (sp-sp.min())/np.ptp(sp)
    sp = periodic(phi, sp)(phi)
    ef = (653-142)/(1122-142)
    esrc = np.where(phi <= .4, ef*phi/.4, ef+(1-ef)*(phi-.4)/.6)
    ep = np.interp(esrc, elbow_phase, elbow_deg)
    ep = periodic(phi, ep)(phi)
    save_csv(OUT/'aligned_priors.csv',
             ['phase','seat_normalized_prior','human_left_elbow_flexion_rad_prior'],
             np.column_stack([phi,sp,np.deg2rad(ep)]))

    import mujoco
    from scipy.optimize import minimize_scalar
    model=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
    data=mujoco.MjData(model)
    start_phase=json.loads((ROOT/'designs/compact_hand/reference_start.json').read_text())['shared_phase']
    reset_seat=float(model.qpos0[model.joint('seat_slide').qposadr[0]])
    # Experimental sensor origin, CAD joint zero, reset pose and catch position
    # are different quantities. Use a fixed boat-frame anatomical datum.
    mujoco.mj_kinematics(model,data)
    foot_x=float(np.mean([data.xanchor[model.joint(s+'_ankle_pitch_joint').id,0] for s in ('left','right')]))
    seat_x_at_qzero=float(data.xpos[model.body('seat').id,0]-reset_seat)
    datum_distance=seat_x_at_qzero-foot_x
    # Conservative seed from lower-body scans (120 deg -> about 34 deg knees).
    # Offset and span remain free parameters in the full-cycle fit.
    seat_offset,seat_span=-.20,.43
    model_seat=seat_offset+seat_span*sp
    elbow_extension=[]
    for side in ('left','right'):
        j=model.joint(side+'_elbow_joint')
        ids=[model.joint(side+'_'+part+'_joint').id for part in ('shoulder_yaw','elbow','wrist_roll')]
        def bending(q):
            data.qpos[:]=model.qpos0;data.qpos[j.qposadr[0]]=q
            mujoco.mj_kinematics(model,data)
            u=data.xanchor[ids[1]]-data.xanchor[ids[0]]
            v=data.xanchor[ids[2]]-data.xanchor[ids[1]]
            return np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))
        fit=minimize_scalar(bending,bounds=tuple(model.jnt_range[j.id]),method='bounded')
        elbow_extension.append(float(fit.x))
    elbows=np.array(elbow_extension)+np.deg2rad(ep[:,None])
    # Keep un-clipped priors, exposing any limit conflict instead of hiding it.
    oar_table=np.loadtxt(ROOT/'data/rowing_reference.csv',delimiter=',',skiprows=1)
    names=['port_sweep','port_elevation','port_feather','starboard_sweep','starboard_elevation',
           'starboard_feather','seat_slide','left_elbow_joint','right_elbow_joint']
    save_csv(OUT/'rowing_reference9_priors.csv',['phase']+names,
             np.column_stack([phi,oar_table[:,1:],model_seat,elbows]))
    seat_mapping={'model_joint':'seat_slide','reset_coordinate_m_information_only':reset_seat,
                  'reference_start_phase_information_only':start_phase,
                  'reference_datum':'boat-frame mean left/right ankle pitch joint center (fixed by foot welds)',
                  'seat_point':'seat body frame origin; not anatomical hip or mesh center',
                  'mean_ankle_center_x_m':foot_x,'seat_x_at_joint_zero_m':seat_x_at_qzero,
                  'distance_at_joint_zero_m':datum_distance,
                  'physical_coordinate_formula':'seat_minus_ankle_longitudinal_distance = distance_at_joint_zero + q_seat',
                  'direction_sign':1,'offset_m':seat_offset,'span_m':seat_span,
                  'calibration_status':'geometry_informed_seed_not_fitted',
                  'formula':'q_seat(phi) = fitted_offset + fitted_span * normalized_shape(phi)',
                  'range_m':[float(model_seat.min()),float(model_seat.max())],
                  'experimental_origin_not_transferred':True,
                  'reset_is_not_a_reference_anchor':True,
                  'position_at_selected_start_m':float(CubicSpline(phi,model_seat,bc_type='periodic')(start_phase)),
                  'range_is_not_full_chain_reachability':True}
    (OUT/'seat_mapping.json').write_text(json.dumps(seat_mapping,indent=2),encoding='utf-8')

    # Re-render the source page for a literal vector-to-page overlay.
    subprocess.run(['pdftoppm','-f','8','-singlefile','-scale-to','2200','-png',
                    str(seat_pdf),str(QA/'wirms_page8')],check=True,capture_output=True)
    page = Image.open(QA/'wirms_page8.png')
    # Exact PDF graphics transforms: Form path -> page points -> raster pixels.
    # Form /Im9: y flip at 419, scale 1/8, then scale 2 and translation;
    # page placement: scale .63943 and translation (35.716,440.932).
    px = page.width/595.276
    py = page.height/841.89
    x_image = (35.716+.63943*(2*paths[0][:,0]-563.349976)/8)*px
    y_image = (841.89-(440.932+.63943*(419-(2*paths[0][:,1]-285.999176)/8)))*py
    fig, ax = plt.subplots(figsize=(12,5))
    ax.imshow(page)
    ax.plot(x_image,y_image,
            'k--',lw=1.1,label='Stage 4: extracted PDF vector')
    ax.set_xlim(x_image.min()-60,x_image.max()+20)
    ax.set_ylim(y_image.max()+45,y_image.min()-35)
    ax.legend();ax.set_title('WiRMS Figure 4 seat curve: vector extraction overlay')
    fig.tight_layout();fig.savefig(QA/'seat_digitization_overlay.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4))
    ax.imshow(im);ax.plot(xs,ys,'r--',lw=1,label='Extracted mean (not SD)')
    ax.set_xlim(75,1150);ax.set_ylim(860,505);ax.legend()
    ax.set_title('Pitto Figure 2 left elbow: digitization overlay')
    fig.tight_layout();fig.savefig(QA/'elbow_digitization_overlay.png',dpi=150);plt.close(fig)

    meta = {
        'status':'Cross-study kinematic priors, NOT synchronized human recordings; not yet H2-feasible',
        'sources':{
            'seat':{'doi':'10.3390/s23031060','pdf_sha256':digest(seat_pdf),'page':8,'figure':4,
                    'series':'Stage 4, participant 3 example','method':'1001 PDF vector vertices',
                    'protocol_spm':26,'pooled_observed_spm':28.97,
                    'pooled_rate_not_individual_rate':True,'raw_seat_range_m':[float(seat.min()),float(seat.max())]},
            'elbow':{'doi':'10.3390/app14199055','pdf_sha256':digest(elbow_pdf),'page':6,'figure':2,
                     'series':'left elbow mean across 12 rowers; right not measured',
                     'method':'black mean-line pixels from embedded 3296x1460 image',
                     'calibration':{'x_catch':142,'x_finish':653,'x_next_catch':1122,'y_zero':830,
                                    'pixels_per_deg':232/120},
                     'raw_range_deg':[float(elbow_deg.min()),float(elbow_deg.max())]},
            'oars':json.loads((ROOT/'data/rowing_reference_provenance.json').read_text()),
        },
        'period_s':60/26,'samples_including_periodic_endpoint':401,
        'target_events':{'catch_phase':0,'release_phase':.4},
        'seat_phase_method':'5% positive handle-force peak crossings; approximate events, not blade tracking',
        'seat_event_threshold_sensitivity':events,
        'elbow_phase_method':'map printed Catch-Finish-Catch separately to 0,0.4,1; figure fraction is NOT time',
        'smoothing':'periodic Gaussian sigma=0.2% cycle and periodic cubic interpolation; raw curves retained',
        'seat_units':'raw metres; aligned shape normalized; robot amplitude/offset not yet identified',
        'elbow_units':'raw anatomical degrees, aligned anatomical radians; not H2 encoder coordinates',
        'uncertainty':'PDF plotting precision is not experimental sensor accuracy; elbow pixel scale ~0.52 deg/pixel',
        'model_seat_mapping':seat_mapping,
        'elbow_extension_coordinates_rad':elbow_extension,
        'nine_channel_priors':'rowing_reference9_priors.csv; NOT feasible trajectory. Elbow limits deliberately not clipped.',
        'scene_sha256':digest(ROOT/'scene.xml'),
    }
    (OUT/'source_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'raw_seat_range_m':meta['sources']['seat']['raw_seat_range_m'],
                      'raw_elbow_range_deg':meta['sources']['elbow']['raw_range_deg'],
                      'seat_events':events},indent=2))


if __name__ == '__main__':
    main()
