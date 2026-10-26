# Nine-channel imitation

Active targets are port sweep/elevation/feather, starboard sweep/elevation/feather,
seat_slide, left_elbow_joint, right_elbow_joint. All share phase, half-speed first
cycle, and the 0.3 s reset blend. The command observation now contains 13 scalars
(previously 10); old policy checkpoints are not directly shape-compatible.

The imitation reward is 0.75 * blade_area + 0.60 * oars + 0.25 * seat + 0.15 * elbows.
Oars retain their existing 0.40/0.35/0.25 sweep/elevation/feather score and
10/4/15 degree scales. Seat uses exp(-(error/0.05 m)^2). Elbows use
exp(-mean((error/15 degrees)^2)). Torque, action-rate and collision costs remain.
Weights are in config.py; scales are in reference.py. Seat error in metres and
elbow RMSE in degrees are logged separately from oar RMSE.

The aligned seat waveform is normalized to the calibrated 0.57757947871773 m
stroke with finish at 0.2533400110211172 m. Calibration is guarded by the scene
hash. Oar source curves are unchanged. Human elbow flexion from aligned_priors.csv
(in radians) is used directly as left_elbow_joint and right_elbow_joint qpos.
No extension offset, geometric inversion, rescaling or clipping is applied.
Only left elbow measurements are available, so the right target mirrors the left.
The common 0.3 s reset blend still applies before tracking the reference directly.
Targets use shape-preserving interpolation to avoid seat/elbow overshoot.

These are cross-study soft priors, not a synchronized or verified full-body
trajectory. Scalar joint-limit checks do not certify simultaneous closed-chain
reachability, collision freedom, or dynamic feasibility. No training is launched
by this change.

## Projected wetted blade area

The extra target is the underwater blade planform projected along the boat's
longitudinal axis: A_wet * abs(normal dot boat_forward), separately per blade.
It is a geometric proxy, NOT projection along instantaneous relative water flow.
The same projected blade outline as the hydrodynamics model is clipped against
the water plane. Runtime triangle submerged fractions are computed on device.
A periodic reference table is generated from reference oar angles with the boat
at its reset pose and nominal waterline. Actual area uses the current boat/blade
poses and waterline. Both share reference phase and the 0.3 s initial-area blend.
This tracks the full-cycle area waveform rather than maximizing area throughout
recovery. The reward is exp(-mean(((A-A_ref)/(0.15*A_full))^2)), weight 0.75.
The total imitation maximum is now 1.75; penalty weights are unchanged.
Area RMSE is logged in square metres. Actor observation dimensions are unchanged.
