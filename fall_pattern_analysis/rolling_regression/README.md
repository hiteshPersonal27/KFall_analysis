# Rolling-Regression β Signals (Phase 1: visualization)

An explainable, real-time-deployable pre-impact fall detector based on a **causal
rolling regression** — watching the *local shape* of the signal instead of raw
thresholds. This folder is **Phase 1 (visualization only)**: it confirms the
regression-derived signals separate falls from ADLs before any detector is built.

## Method

At every frame `t`, fit the last `WINDOW = 25` frames (0.25 s, past-only/causal):
- linear fit → **β₁** = local slope (trend)
- quadratic fit → **β₂** = local curvature (bump)

β₁/β₂ become new per-frame signals — flat during ordinary motion, spiking when a
fall's dynamics begin. Computed on all three validated strong channels: **ACC_M, VV,
GYR_M** (reusing `../analyze_pattern.py`; tilt excluded as it's the noisy one).

Vectorized: the window x-values are fixed (`0..W-1`), so each fit is one matrix
multiply against a precomputed pseudo-inverse over sliding windows — fast.

**Order matters**: β is computed in raw frame-time (causal), *then* phase-aligned for
plotting — never the reverse (that would be non-causal). Falls align onset→impact
(0→100%); ADLs align over full trial duration (0→100%).

## Run

```bash
python3 visualize_beta_signals.py
```

Prints sanity checks + trial counts, saves `plots/beta_ACC_M.png`, `beta_VV.png`,
`beta_GYR_M.png`. Each figure overlays the fall grand-average (mean ± 1 std) against
the false-positive-prone ADLs (D10 stumble, D13 sit-down, D14 quick-sit, D04 jump)
and the all-ADL mean.

## Phase 1 finding (summary)

The β signals **do separate falls from ADLs** — risky ADLs stay near flat while falls
show large β excursions. Crucially, **VV β₁ separates earliest** (positive slope
building steadily from onset through the descent), making it the strongest *pre-impact*
discriminator; ACC_M β separates strongly but mostly *at* impact (less lead time);
GYR_M β₁ separates in the mean but with a wide per-trial spread.

## Not in this phase

No thresholds, alarm rule, or sensitivity/specificity/lead-time table — that's
**Phase 2**, now justified by this visualization.
