# Signal-Quality Visualization

Interactive Plotly tools to visually confirm the KFall dataset is reliable by
inspecting the signal per sensor across trials. Signal-quality checks — not
classification.

Two tools live here:

| File | What it is |
|---|---|
| `build_signal_dashboard.py` → `signal_dashboard.html` | **Interactive dashboard** with subject/task/mode selection (see below). The main tool. |
| `build_signal_quality.py` → `signal_quality.html` | Simpler grand-average-only view (mean ± 1 std across all fall trials), no selectors. |

---

## Interactive dashboard (`build_signal_dashboard.py`)

```bash
python3 build_signal_dashboard.py
```

Regenerates and opens `signal_dashboard.html` (self-contained; Plotly embedded).
Prints trials loaded per mode (expected 2,319 fall + ~2,717 ADL).

### View modes (radio buttons)

1. **Subject × Task** — pick a subject and a task (any of the 15 fall tasks F01–F15
   **or** 21 ADL tasks D01–D21). Shows that subject's individual trials as thin
   lines + their mean as a bold line. No std band — kept sharp. Use this to compare
   a fall against a normal activity for the same person.
2. **Subject, all tasks** — pick a subject and a **task group (Falls / ADLs)**;
   overlays each task's mean in that group as one line, colored by task, with a
   legend.
3. **Task, all subjects** — pick any task (fall **or** ADL); overlays all 32
   subjects' means, one line each.

The overlays stay **single-family** (all falls *or* all ADLs, never mixed) so the
x-axis keeps one consistent meaning:
- **Fall tasks**: x = *% of onset→impact* (0% = labeled onset, 100% = labeled impact).
- **ADL tasks**: x = *% of trial duration* (0% = trial start, 100% = trial end),
  since ADLs have no fall onset/impact labels. A red note appears above the plot and
  the x-axis label switches automatically whenever ADLs are shown.

The per-sensor dropdown (Accelerometer / Gyroscope / Orientation) applies in all
modes. Mode 1 shows a magnitude panel + a raw-3-axes panel; overlay modes show one
clean line per entity on the sensor's primary channel (ACC_M / gyro magnitude /
tilt).

### Phase axis

- **Fall tasks**: 0% = labeled onset frame, 100% = labeled impact frame, extended
  −50%…+150% (real pre/post-event data).
- **ADL tasks** (no fall onset/impact labels exist): each trial is normalized over
  its full duration (0% = first frame, 100% = last frame); values outside 0–100%
  are NaN, so lines break instead of drawing misleading flats. A red note appears
  above the plot when an ADL task is selected, since the axis meaning differs.

### Aesthetics

Sharp thin lines (raw axes width 1, means 2–2.4), light-gray gridlines on a white
background, fixed per-sensor y-ranges so scales stay comparable across selections,
dashed reference lines at phase 0 and 100, and proper axis units (g, m/s, deg/s,
deg). No std bands in overlay modes.

> The generated `signal_dashboard.html` is ~26 MB (it embeds every trial for
> client-side switching). It is regenerable from the script, so consider keeping it
> out of version control (see the repo `.gitignore`).

---

## Grand-average view (`build_signal_quality.py`)

Interactive Plotly tool to visually confirm the KFall dataset is reliable by
inspecting the **event-locked grand-average signal per sensor** across all fall
trials. This is a signal-quality check — not classification.

## Run

```bash
python3 build_signal_quality.py
```

Regenerates and opens `signal_quality.html` (a single self-contained file with
Plotly embedded — no internet needed to view). Prints the number of aggregated fall
trials (expected ~2,319).

## What it shows

Every fall trial (tasks 20–34) is phase-aligned onto a common axis where 0% = fall
onset frame and 100% = fall impact frame (extended −50% to +150%), then averaged.
The plot shows **mean ± 1 std** across all trials, with dashed reference lines at
phase 0 and 100.

- **Dropdown** switches sensor: Accelerometer / Gyroscope / Orientation.
- **Top panel**: the derived/magnitude channel(s) — ACC_M & VV (accelerometer),
  gyro magnitude (gyroscope), or tilt (orientation).
- **Bottom panel**: the three raw axes for that sensor.
- **Interaction**: hover for exact values, legend-click to toggle an axis (its band
  hides with it), zoom/pan.

## Consistency with the validated pipeline

All signal definitions are imported directly from `../analyze_pattern.py` (see
`../docs/pattern_analysis.md`) — the same 5 Hz low-pass filter, ACC_M / gyro-
magnitude Euclidean norms, and world-frame vertical-velocity computation used in the
paper-faithful analysis whose Rule B reproduces the paper's 95.50% sensitivity.
Accelerometer and gyroscope axes are shown 5 Hz low-pass filtered (the same filtered
axes whose norms give ACC_M and gyro magnitude, so the raw-axis and magnitude panels
reconcile); Euler angles are shown raw, matching how the pipeline derives tilt.

## Interpretation

If the dataset is sound, each channel should show a consistent, sharply-timed shape
with a narrow std band when averaged across all trials — the signal-quality evidence
that the pre-impact fall pattern is a genuine, global property of the data.
