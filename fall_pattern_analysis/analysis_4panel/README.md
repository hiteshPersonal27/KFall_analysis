# Phase-Aligned Signal Shape — 4-Panel (+ Gyroscope Magnitude)

Same event-locked phase-alignment method as `../analysis_3panel/`, extended with a
4th signal: **gyroscope magnitude** (`sqrt(GyrX²+GyrY²+GyrZ²)`, 5 Hz low-pass
filtered). Tilt (from Euler angles) is a fused/integrated orientation estimate,
not the same thing as raw angular velocity — and the dataset's earlier baseline
exploration had already suggested gyro shows a sharp, clean signature that
tilt's noisy behavior doesn't capture on its own.

## Run

```bash
python3 visualize_pattern_4panel.py
```

Same 4 plots as `analysis_3panel/` (grand average, inter-subject overlay,
intra-subject small multiples, trial heatmap), now with gyro magnitude as a 4th
signal in the grand-average and inter-subject views. (Small multiples and the
trial heatmap stay ACC_M-only in both versions — repeating a 32-panel grid or a
2,319-row raster per signal would clutter rather than clarify.)

## Finding — the notable one

Gyroscope magnitude turned out to be **as clean as ACC_M/VV**: smooth,
monotonic rise from a ~30–50°/s baseline near onset, sharp peak (~200°/s) right
at impact, then drops. This revised an earlier (threshold-rule-derived)
impression that gyro was a weak signal — the weakness was in that ad hoc
threshold test, not the raw signal itself. Net result: **three of the four raw
IMU channels — ACC_M, VV, gyro magnitude — show a consistent, generalizable
fall signature; tilt alone does not.** This is the finding that later motivated
excluding tilt from the rolling-regression (`../rolling_regression/`) work.

Full narrative: `../docs/pattern_analysis.md`.
