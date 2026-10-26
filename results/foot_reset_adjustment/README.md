# Foot reset adjustment

Applied to scene.xml and preview.xml:
- Reset ankle roll: left +12.5 to +9.5 degrees, right -12.5 to -9.5 degrees. Both the baked ankle-roll body orientation and joint reference were updated, preserving the physical meaning of joint coordinates.
- Recomputed both foot weld relative poses from the new reset; hand welds and all other reset joints were preserved.
- Foot weld torquescale: 1 to 0.5, reducing the relative weighting of orientation errors. This is not a statement that angular stiffness is exactly halved. Position-constraint solref `.004 1` and solimp `.99 .999 .001 .5 2` remain identical. Hand torquescale is unchanged.
- Seat travel limits: [-0.34, 0.27] m, covering the 172-degree-calibrated source stroke [-0.3242395, 0.2533400] m with margin. Seat reset and damping unchanged; robot joint limits unchanged.

Checks: scene and preview compile, reset equality residual below 1e-8; mjlab build_spec inherits the reset and torque scaling; no reset collision penetrations deeper than 1 mm in the training model. The saved lower-body poses were rechecked after the seat-limit change: all 301 samples of the full 0.5775795 m stroke satisfy foot position/orientation and joint limits. Exact closure was possible, so stroke reachability does not rely on accepting soft weld errors.

Current calibration and seat-to-knee conversion are in `seat_knee_172/`. Previous results under `results/seat_knee_172/` describe the old foot configuration. The conversion helper checks the scene hash and rejects stale maps; supply the current `seat_knee_172/seat_to_knee_map.csv` explicitly. Use the same posture selection rule described in the calibration. This remains a lower-body calibration, not a full-body validated policy reference: hands/oars, full-cycle collisions, and dynamics have not been validated.

The `.before` files are pre-change snapshots preserving pre-existing local changes. `reset_change_report.json` records parameters and hashes. The adjustment is not pushed or committed.
