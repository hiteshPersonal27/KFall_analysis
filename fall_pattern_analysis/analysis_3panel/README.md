# Phase-Aligned Signal Shape — 3-Panel (ACC_M, VV, Tilt)

Answers a different question than `paper_threshold_validation/`: not "does a
threshold rule fire correctly," but **"what does the fall signal actually look
like, shape-wise, across the whole dataset?"**

Every fall trial is time-normalized onto a common **phase axis** — 0% = labeled
onset frame, 100% = labeled impact frame, extended −50%…+150% for context — the
standard event-locked averaging technique (same idea as ERP averaging), which
lets trials of very different absolute duration be overlaid and averaged
meaningfully.

## Run

```bash
python3 visualize_pattern_3panel.py
```

Imports the validated pipeline from `../paper_threshold_validation/analyze_pattern.py`
(no signal math duplicated). Saves 4 plots to `plots/`:

1. **`pattern_grand_average.png`** — mean ± 1 std per signal across all 2,319 fall
   trials on the phase axis. "Is there one consistent global shape?"
2. **`pattern_intersubject_overlay.png`** — one line per subject (32) + global mean.
3. **`pattern_intrasubject_small_multiples.png`** — a panel per subject, individual
   trials (thin) + subject mean (bold), ACC_M only.
4. **`pattern_trial_heatmap.png`** — every trial as one row, phase on x-axis, color
   = ACC_M — the single-trial data underlying the averages.

## Finding

ACC_M and VV both show a sharp, consistent, well-timed pattern (dip then spike
precisely at phase 100). Tilt is comparatively weak and noisy — wide spread even
before onset, no sharp localized feature. This motivated `analysis_4panel/`
(adds gyroscope magnitude, which turned out to be as clean as ACC_M/VV).

Full narrative: `../docs/pattern_analysis.md`.
