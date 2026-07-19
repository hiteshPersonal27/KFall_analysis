# Paper-Threshold Validation (Rule A/B/C)

Re-implements the KFall paper's (Yu, Jang & Xiong, 2021) own threshold-based benchmark
algorithm exactly, and tests the question this whole project started from: **is
acceleration magnitude alone enough to detect a fall, or is that a myth?**

This folder also hosts `analyze_pattern.py`, the **shared pipeline** every other method
in `fall_pattern_analysis/` imports (`compute_signals`, `lowpass_filter`,
`discover_subjects`, `load_sensor_data`, `load_labels`, `get_fall_label_info`,
`FALL_TASK_IDS`) — so it plays double duty as both its own implementation and the
common signal-processing core.

## Run

```bash
python3 analyze_pattern.py
```

Prints a VV sanity check, then sensitivity/specificity/lead-time for each rule, then
saves 4 comparison plots to `plots/` and per-trial results to `pattern_results.csv`.

## The three rules

| Rule | Definition | Sensitivity | Specificity |
|---|---|---|---|
| **A** | `ACC_M < 0.8g` at any point (accel-only) | 100.0% | 17.4% |
| **B** | Paper's full algorithm: `ACC_M<0.8g AND VV>0.3` + tilt confirmation within 10 frames | 95.6% | 34.9% |
| **C** | Gyro-based (`max(\|GyrX\|,\|GyrZ\|)` vs. standing baseline), fainting-sitting subset only | 100.0% (subset) | not tested |
| *Paper's own Threshold algorithm (reference)* | — | 95.50% | 83.43% |

Rule B's sensitivity (95.6%) matching the paper's reported 95.50% is the key
validation signal that the vertical-velocity/rotation math is implemented correctly.

## Signals computed here

- **ACC_M** — accel magnitude, 5 Hz low-pass filtered.
- **VV (vertical velocity)** — body-frame accel rotated into the world frame via
  the fused Euler-angle orientation, projected onto vertical, integrated with a
  1-second windowed integral (bounds drift).
- **Tilt** — `max(|Pitch|, |Roll|)`.

## Key finding

Accel magnitude alone (Rule A) has no real discriminating power — it fires on
100% of falls but also 82.6% of ordinary daily activities. Adding vertical
velocity + tilt (Rule B) roughly doubles specificity for almost no sensitivity
cost, confirming the paper's 3-signal design was necessary, not arbitrary.

Full narrative: `../docs/pattern_analysis.md`.
