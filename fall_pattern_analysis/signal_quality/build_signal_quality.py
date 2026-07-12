"""
Interactive signal-quality visualization for the KFall dataset.

Purpose: visually confirm the dataset is reliable by inspecting the event-locked
grand-average signal per sensor across all fall trials -- a clean signal-quality
check, NOT classification. If the dataset is sound, every sensor channel should show
a consistent, low-variance shape when all ~2,319 fall trials are phase-aligned and
averaged.

Reuses the validated preprocessing pipeline from ../analyze_pattern.py exactly
(see ../docs/pattern_analysis.md) so results stay consistent with the paper-faithful
analysis (Rule B matches the paper's 95.50% sensitivity):
  - 5 Hz low-pass Butterworth filter (lowpass_filter)
  - ACC_M = Euclidean norm of filtered accel axes (compute_signals)
  - VV    = body-frame accel rotated to world frame via scipy Rotation (intrinsic
            Z-Y-X), 1 s windowed integral to bound drift, sign so VV>0 = falling
            downward (compute_signals)
  - gyro magnitude = Euclidean norm of filtered gyro axes
  - tilt = max(|Pitch|, |Roll|) from raw Euler angles

Phase axis (event-locked averaging): 0% = fall onset frame, 100% = fall impact
frame, extended -50%..+150% for context, on a fixed 201-point grid -- identical to
the analysis_3panel / analysis_4panel convention.

Output: a single self-contained interactive Plotly HTML (signal_quality.html) with a
per-sensor dropdown (Accelerometer / Gyroscope / Orientation). For each sensor:
  - top panel: derived/magnitude channel(s)  (ACC_M & VV | gyro magnitude | tilt)
  - bottom panel: the three raw axes
Both panels show mean ± 1 std across all fall trials, with dashed reference lines at
phase 0 and 100. Hover for values, legend-click to toggle axes, zoom/pan.

Run from anywhere:  python3 fall_pattern_analysis/signal_quality/build_signal_quality.py
"""

import os
import sys
import webbrowser

import numpy as np

# Reuse the validated pipeline: signal_quality/ -> fall_pattern_analysis/ holds
# analyze_pattern.py.
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_DIR)
from analyze_pattern import (  # noqa: E402
    discover_subjects, load_sensor_data, load_labels, get_fall_label_info,
    compute_signals, lowpass_filter, FALL_TASK_IDS,
)

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(SCRIPT_DIR, "signal_quality.html")

PHASE_MIN, PHASE_MAX, N_PHASE = -50, 150, 201
PHASE_GRID = np.linspace(PHASE_MIN, PHASE_MAX, N_PHASE)


# ----------------------------------------------------------------------------- #
# Data assembly
# ----------------------------------------------------------------------------- #

def resample_to_phase(frames, values, onset, impact):
    """Phase-align one channel onto the common grid (identical to analysis_Npanel)."""
    duration = impact - onset
    if duration <= 0:
        return None
    target_frames = onset + PHASE_GRID / 100.0 * duration
    return np.interp(target_frames, frames, values)


def build_phase_aligned_dataset():
    """
    Collect every fall trial, phase-aligned, for all channels.

    Per-channel filtering follows the pipeline exactly:
      - Acc X/Y/Z and Gyr X/Y/Z: 5 Hz low-pass (same filtered axes whose norms give
        ACC_M and gyro magnitude, so the raw-axis panel reconciles with the
        magnitude panel).
      - Euler X/Y/Z: raw (the pipeline derives tilt from raw Euler angles).
    """
    subjects = discover_subjects()
    channels = ["acc_m", "vv", "gyro_m", "tilt",
                "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ",
                "EulerX", "EulerY", "EulerZ"]
    collected = {c: [] for c in channels}
    n_trials = 0

    for subject in subjects:
        df_label = load_labels(subject)
        if df_label is None:
            continue
        for task in FALL_TASK_IDS:
            for trial in range(1, 6):
                df = load_sensor_data(subject, task, trial)
                if df is None:
                    continue
                label_info = get_fall_label_info(df_label, task, trial)
                if label_info is None:
                    continue
                onset, impact = label_info["onset"], label_info["impact"]
                frames = df["FrameCounter"].values
                if onset < frames.min() or impact > frames.max():
                    continue

                acc_m, vv = compute_signals(df)  # filtered ACC_M, world-frame VV
                gyro_m = np.sqrt(
                    lowpass_filter(df["GyrX"].values) ** 2
                    + lowpass_filter(df["GyrY"].values) ** 2
                    + lowpass_filter(df["GyrZ"].values) ** 2
                )
                tilt = np.maximum(df["EulerY"].abs().values, df["EulerX"].abs().values)

                raw = {
                    "AccX": lowpass_filter(df["AccX"].values),
                    "AccY": lowpass_filter(df["AccY"].values),
                    "AccZ": lowpass_filter(df["AccZ"].values),
                    "GyrX": lowpass_filter(df["GyrX"].values),
                    "GyrY": lowpass_filter(df["GyrY"].values),
                    "GyrZ": lowpass_filter(df["GyrZ"].values),
                    "EulerX": df["EulerX"].values,
                    "EulerY": df["EulerY"].values,
                    "EulerZ": df["EulerZ"].values,
                }
                derived = {"acc_m": acc_m, "vv": vv, "gyro_m": gyro_m, "tilt": tilt}

                trial_aligned = {}
                ok = True
                for name, series in {**derived, **raw}.items():
                    aligned = resample_to_phase(frames, series, onset, impact)
                    if aligned is None:
                        ok = False
                        break
                    trial_aligned[name] = aligned
                if not ok:
                    continue

                for name, aligned in trial_aligned.items():
                    collected[name].append(aligned)
                n_trials += 1

    stats = {}
    for name, arrs in collected.items():
        stacked = np.array(arrs)
        stats[name] = (stacked.mean(axis=0), stacked.std(axis=0))
    return stats, n_trials


# ----------------------------------------------------------------------------- #
# Plot construction
# ----------------------------------------------------------------------------- #

# Sensor -> (magnitude panel channels, raw-axis channels), with display metadata.
# Each channel: (key, display label, color).
SENSORS = {
    "Accelerometer": {
        "magnitude": [
            ("acc_m", "ACC_M (accel magnitude)", "#1abc9c"),
            ("vv", "VV (vertical velocity)", "#3498db"),
        ],
        "magnitude_axis_title": "ACC_M (g)  /  VV (m/s)",
        "raw": [
            ("AccX", "Acc X", "#e74c3c"),
            ("AccY", "Acc Y", "#2ecc71"),
            ("AccZ", "Acc Z", "#3498db"),
        ],
        "raw_axis_title": "Acceleration (g)",
    },
    "Gyroscope": {
        "magnitude": [
            ("gyro_m", "Gyro magnitude", "#9b59b6"),
        ],
        "magnitude_axis_title": "Angular velocity magnitude (deg/s)",
        "raw": [
            ("GyrX", "Gyr X (roll rate)", "#e74c3c"),
            ("GyrY", "Gyr Y (pitch rate)", "#2ecc71"),
            ("GyrZ", "Gyr Z (yaw rate)", "#3498db"),
        ],
        "raw_axis_title": "Angular velocity (deg/s)",
    },
    "Orientation": {
        "magnitude": [
            ("tilt", "Tilt = max(|Pitch|,|Roll|)", "#e67e22"),
        ],
        "magnitude_axis_title": "Tilt (deg)",
        "raw": [
            ("EulerX", "Roll (Euler X)", "#e74c3c"),
            ("EulerY", "Pitch (Euler Y)", "#2ecc71"),
            ("EulerZ", "Yaw (Euler Z)", "#3498db"),
        ],
        "raw_axis_title": "Angle (deg)",
    },
}


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def add_channel_traces(fig, stats, key, label, color, row, sensor_name):
    """
    Add a mean line + a ±1 std shaded band for one channel.

    Returns the number of traces added (3: band-upper, band-lower, mean). All share
    a legendgroup so a single legend click toggles the mean and its band together
    (layout.legend.groupclick='togglegroup').
    """
    mean, std = stats[key]
    upper, lower = mean + std, mean - std
    group = f"{sensor_name}::{key}"

    fig.add_trace(go.Scatter(
        x=PHASE_GRID, y=upper, mode="lines",
        line=dict(width=0), hoverinfo="skip",
        legendgroup=group, showlegend=False, name=f"{label} +1std",
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=PHASE_GRID, y=lower, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor=_hex_to_rgba(color, 0.18),
        hoverinfo="skip", legendgroup=group, showlegend=False, name=f"{label} -1std",
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=PHASE_GRID, y=mean, mode="lines",
        line=dict(color=color, width=2.5), legendgroup=group, showlegend=True,
        name=label,
        hovertemplate=(f"<b>{label}</b><br>phase %{{x:.0f}}%<br>"
                       "mean %{y:.3f}<br>±1std %{customdata:.3f}<extra></extra>"),
        customdata=std,
    ), row=row, col=1)
    return 3


def build_figure(stats, n_trials):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("Derived / magnitude channel", "Raw sensor axes"),
    )

    # Track which traces belong to which sensor, for the dropdown visibility masks.
    trace_sensor = []

    for sensor_name, cfg in SENSORS.items():
        for key, label, color in cfg["magnitude"]:
            n = add_channel_traces(fig, stats, key, label, color, 1, sensor_name)
            trace_sensor.extend([sensor_name] * n)
        for key, label, color in cfg["raw"]:
            n = add_channel_traces(fig, stats, key, label, color, 2, sensor_name)
            trace_sensor.extend([sensor_name] * n)

    sensor_names = list(SENSORS.keys())
    default_sensor = sensor_names[0]

    # Initial visibility: only the default sensor's traces shown.
    for tr, s in zip(fig.data, trace_sensor):
        tr.visible = (s == default_sensor)

    # Dropdown buttons: each sets a full visibility mask + relabels the y-axes.
    buttons = []
    for sensor_name in sensor_names:
        cfg = SENSORS[sensor_name]
        vis = [s == sensor_name for s in trace_sensor]
        buttons.append(dict(
            label=sensor_name,
            method="update",
            args=[
                {"visible": vis},
                {"yaxis.title.text": cfg["magnitude_axis_title"],
                 "yaxis2.title.text": cfg["raw_axis_title"]},
            ],
        ))

    # Dashed reference lines at onset (phase 0) and impact (phase 100), both panels.
    shapes = []
    for phase_x in (0, 100):
        for yref in ("y domain", "y2 domain"):
            shapes.append(dict(
                type="line", x0=phase_x, x1=phase_x, y0=0, y1=1,
                xref="x", yref=yref,
                line=dict(color="gray", width=1.2, dash="dash"),
            ))
    annotations = list(fig.layout.annotations) + [
        dict(x=0, y=1.02, xref="x", yref="y domain", text="onset (0%)",
             showarrow=False, font=dict(size=10, color="gray"), xanchor="center"),
        dict(x=100, y=1.02, xref="x", yref="y domain", text="impact (100%)",
             showarrow=False, font=dict(size=10, color="gray"), xanchor="center"),
    ]

    fig.update_layout(
        title=dict(
            text=(f"KFall Signal-Quality Check &mdash; Event-Locked Grand Average "
                  f"(mean &plusmn; 1 std, n = {n_trials} fall trials)<br>"
                  "<sup>Phase-aligned: 0% = fall onset, 100% = fall impact. "
                  "Use the dropdown to switch sensor; click legend to toggle axes.</sup>"),
            x=0.5, xanchor="center",
        ),
        updatemenus=[dict(
            buttons=buttons, direction="down", showactive=True,
            x=0.0, xanchor="left", y=1.16, yanchor="top",
            pad=dict(t=2, b=2, l=4, r=4), bgcolor="#f7f7f7",
        )],
        legend=dict(groupclick="togglegroup", orientation="v",
                    x=1.01, xanchor="left", y=1.0),
        hovermode="x unified",
        shapes=shapes,
        annotations=annotations,
        template="plotly_white",
        height=780, width=1150,
        margin=dict(t=110, r=230),
    )
    fig.update_xaxes(title_text="Phase (% of onset&rarr;impact duration)", row=2, col=1)
    fig.update_yaxes(title_text=SENSORS[default_sensor]["magnitude_axis_title"], row=1, col=1)
    fig.update_yaxes(title_text=SENSORS[default_sensor]["raw_axis_title"], row=2, col=1)

    # Dropdown context label.
    fig.add_annotation(
        x=0.0, y=1.205, xref="paper", yref="paper", xanchor="left",
        text="<b>Sensor:</b>", showarrow=False, font=dict(size=12),
    )
    return fig


# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Building phase-aligned grand average across all fall trials "
          "(reusing validated pipeline from ../analyze_pattern.py)...")
    stats, n_trials = build_phase_aligned_dataset()
    print(f"Aggregated {n_trials} fall trials onto the phase axis "
          f"({PHASE_MIN}% to {PHASE_MAX}%, 0=onset, 100=impact).")
    if n_trials < 2300:
        print(f"WARNING: expected ~2,300+ fall trials, got {n_trials}. "
              "Check the data layout / label columns.")
    else:
        print("Trial count is in the expected range (~2,300+). Dataset looks complete.")

    fig = build_figure(stats, n_trials)
    fig.write_html(OUT_HTML, include_plotlyjs=True, full_html=True)
    print(f"\nSaved interactive visualization to {OUT_HTML}")

    try:
        webbrowser.open(f"file://{OUT_HTML}")
        print("Opened in your default browser.")
    except Exception as exc:  # pragma: no cover
        print(f"(Could not auto-open a browser: {exc}. Open the file manually.)")
