# 172-degree seat/knee calibration

Target: sagittal hip–knee–ankle included angle 172 degrees (flexion 8 degrees), represented by H2 hip_pitch, knee, and ankle_pitch joint anchors. These are robot proxies, not anatomically registered human landmarks. The corresponding motor coordinate is independently calculated; do not command 172 degrees to the knee motor.

`tools/calibrate_seat_knee.py` calibrates the finish with both feet welded to the boat and the existing pelvis-seat hinge. Seat limits are relaxed in the optimizer only. Other joint limits are retained. The finish solve prefers the original pelvis hinge posture; the seat sweep then prefers the calibrated finish hinge posture, with tiny continuity regularization. This selection rule is needed because knee angle and seat position do not form a unique mapping when pelvis orientation is unconstrained.

The source is WiRMS participant 3 Stage 4, as previously extracted in `data/reference9/wirms_stage4_raw.csv`. No amplitude scaling is applied: `q = q_finish + source_seat - max(source_seat)`. Absolute source sensor coordinates are not transferred. Model coordinates refer to the seat body origin; render coordinates are in the boat frame at its saved pose. The blue foot marker is the mean ankle-axis center, not a sole-contact point.

Outputs:
- `calibration.json`: target, model hash, selected posture rule, coordinates, validity thresholds.
- `seat_to_knee_map.csv`: seat coordinate to geometric included angles and motor angles, plus foot closure errors and validity. Failed rows are diagnostic best fits only.
- `wirms_seat_to_knee.csv`: original source phase and seat displacement, mapped robot seat coordinate, geometric and motor angles. Out-of-domain angles are NaN, not clipped or extrapolated.
- `mapping_poses.npz`: lower-body coordinates used for each sample.
- `seat_travel_on_boat.png`: actual compiled CAD, finish, contiguous accepted boundary, and unscaled source-stroke endpoint; diagram and curves.

Python API: `from tools.calibrate_seat_knee import seat_positions_to_knee`. Pass one or many seat **joint coordinates in metres**. The returned `angles_rad` columns are left/right geometric included angles, then left/right knee motor coordinates. `valid` is false outside the connected calibrated interval. Outputs are radians; CSV angles are degrees. Regenerate the map when scene geometry, foot attachments, or the posture selection rule changes.

Validation: foot translation error < 0.1 mm, rotation error < 0.1 degree, retained leg/hinge bounds. Seat grid resolution is approximately 1.93 mm; the accepted boundary is not an exact global reachability limit. Alternative poses may exist. The finish and lookup endpoint checks pass, and out-of-range inputs return NaN. No hand/oar closure, collision, dynamic, or interpolation-between-samples pose validation has been completed. The calibration is not a training-ready full-body reference. No scene/reset/training files were changed.
