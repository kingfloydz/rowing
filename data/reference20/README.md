# Installed reference (2026-11-11)

The selected legacy-policy cycle is installed for training: 3.2 s periodic motion,
with the existing 5.2 s first cycle retained. Both passive wrist connector hinges
remain enabled. The raised gates and tilted rigger mesh are exported as portable
XML/STL; the reset arms and waist were fitted to close both hand welds.

The source joint trajectory is preserved exactly. Recorded floating-base travel
is omitted from the targets. The trunk target is evaluated at the reset hull
attitude so it shares the periodic joint clock. Source hashes and measured soft
reference residuals are in metadata.json; export equivalence is in installation.json.

This is a soft kinematic reference, not a validated trained policy. Inherited
leg/waist limit excursions reach about 0.177 rad, foot-weld residuals about 3 mm
and 2.24 degrees, and collision penetration about 2.4 mm. Core drive is submerged
and core recovery dry; entry/extraction transitions are not included in those
claims. The float reset is inherited and has not been recalibrated after the
small gate-mass relocation.

Penalty curriculum starts after 1000 completed PPO updates, including when
resuming older checkpoints: action_rate changes from -0.05 to -2 and angular
velocity from -1 to -5. Environment count and episode resets do not affect it.
