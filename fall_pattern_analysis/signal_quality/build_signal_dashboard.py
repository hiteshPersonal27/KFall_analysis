"""
Interactive KFall signal dashboard with subject/task selection.

Signal-quality inspection tool (not classification). Reuses the validated
preprocessing pipeline from ../analyze_pattern.py exactly (see ../docs/
pattern_analysis.md): 5 Hz low-pass filter, ACC_M / gyro-magnitude Euclidean norms,
world-frame vertical velocity (VV), tilt = max(|Pitch|,|Roll|), and phase alignment.

Phase axis:
  - Fall tasks (F01-F15, ids 20-34): 0% = labeled onset frame, 100% = labeled
    impact frame, extended -50%..+150% (real pre/post-event data exists there).
  - ADL tasks (D01-D21, ids 1-19 & 35-36): no fall onset/impact labels exist, so
    each trial is normalized over its full duration (0% = first frame, 100% = last
    frame); values outside 0-100% are set to NaN (no data there) so lines break
    cleanly rather than drawing misleading flats.

Three view modes:
  1. One subject, one task  -- individual trials (thin) + mean (bold), no std band.
                               Supports BOTH ADL and fall tasks (fall-vs-ADL compare).
  2. One subject, all tasks -- overlay each FALL task's mean, colored by task.
  3. One task, all subjects -- overlay each subject's mean for one FALL task.
  (Modes 2/3 are falls-only so the onset->impact x-axis has a single meaning.)

Per-sensor dropdown (Accelerometer / Gyroscope / Orientation). Mode 1 shows a
magnitude panel + a raw-3-axes panel; overlay modes show one clean line per entity
on the sensor's primary magnitude channel (ACC_M / gyro magnitude / tilt).

Output: a single self-contained interactive signal_dashboard.html (Plotly embedded).

Run:  python3 fall_pattern_analysis/signal_quality/build_signal_dashboard.py
"""

import os
import sys
import json
import webbrowser

import numpy as np

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "paper_threshold_validation")
sys.path.insert(0, PIPELINE_DIR)
from analyze_pattern import (  # noqa: E402
    discover_subjects, load_sensor_data, load_labels, get_fall_label_info,
    compute_signals, lowpass_filter, FALL_TASK_IDS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(SCRIPT_DIR, "signal_dashboard.html")

# 81 points over -50..150 = 2.5% steps. This is downsampling, not smoothing (no
# extra filtering beyond the pipeline's 5 Hz), and keeps the impact spike (~10-20%
# wide) crisply sampled while keeping the embedded JSON to a reasonable size.
PHASE_MIN, PHASE_MAX, N_PHASE = -50, 150, 81
PHASE_GRID = np.linspace(PHASE_MIN, PHASE_MAX, N_PHASE)

# ADL task descriptions (paper Table 2). ADL ids: D01-D19 = 1-19, D20-D21 = 35-36.
ADL_DESCRIPTIONS = {
    1: "Stand for 30 s", 2: "Bend to tie shoelace and get up", 3: "Pick up object from floor",
    4: "Gently jump to reach an object", 5: "Sit to the ground and get up (normal)",
    6: "Walk normally with turn (4 m)", 7: "Walk quickly with turn (4 m)",
    8: "Jog normally with turn (4 m)", 9: "Jog quickly with turn (4 m)",
    10: "Stumble while walking", 11: "Sit on a chair for 30 s", 12: "Sit on the sofa for 30 s",
    13: "Sit down / get up from chair (normal)", 14: "Sit down / get up from chair (quick)",
    15: "Try to get up and collapse into chair", 16: "Sit on sofa and get up (normal)",
    17: "Lie on the bed for 30 s", 18: "Lie down to bed and get up (normal)",
    19: "Lie down to bed and get up (quick)", 35: "Walk up/down stairs (normal)",
    36: "Walk up/down stairs (quick)",
}
ADL_TASK_IDS = list(range(1, 20)) + [35, 36]

# Rounding precision per channel to keep the embedded JSON compact. Wide-range
# channels (gyro deg/s, Euler deg, tilt deg, gyro magnitude) are rounded to whole
# units -- visually indistinguishable at their scale but ~1 char/value smaller.
CHANNEL_ROUND = {
    "acc_m": 2, "vv": 2, "gyro_m": 0, "tilt": 0,
    "AccX": 2, "AccY": 2, "AccZ": 2,
    "GyrX": 0, "GyrY": 0, "GyrZ": 0,
    "EulerX": 0, "EulerY": 0, "EulerZ": 0,
}
CHANNELS = list(CHANNEL_ROUND.keys())


def resample_fall(frames, values, onset, impact):
    duration = impact - onset
    if duration <= 0:
        return None
    target = onset + PHASE_GRID / 100.0 * duration
    return np.interp(target, frames, values)


def resample_adl(frames, values):
    start, end = frames[0], frames[-1]
    duration = end - start
    if duration <= 0:
        return None
    target = start + PHASE_GRID / 100.0 * duration
    out = np.interp(target, frames, values)
    # No data outside the trial itself -> NaN so the line breaks instead of clamping.
    out[(PHASE_GRID < 0) | (PHASE_GRID > 100)] = np.nan
    return out


def compute_all_channels(df):
    acc_m, vv = compute_signals(df)
    gyro_m = np.sqrt(
        lowpass_filter(df["GyrX"].values) ** 2
        + lowpass_filter(df["GyrY"].values) ** 2
        + lowpass_filter(df["GyrZ"].values) ** 2
    )
    tilt = np.maximum(df["EulerY"].abs().values, df["EulerX"].abs().values)
    return {
        "acc_m": acc_m, "vv": vv, "gyro_m": gyro_m, "tilt": tilt,
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


def build_dataset():
    subjects = discover_subjects()
    data = {}          # data[subj][task][channel] = [ [trial arrays] ]
    task_labels = {}   # task_id -> {code, type, desc}
    counts = {"mode1_trials": 0, "fall_trials": 0, "adl_trials": 0}
    range_accum = {c: [] for c in CHANNELS}

    # Fall task descriptions come from the label file (canonical: first subject).
    canon_label = load_labels(subjects[0])
    for tid in FALL_TASK_IDS:
        info_desc = None
        pattern = f"({tid})"
        rows = canon_label[canon_label["Task Code (Task ID)"].str.contains(pattern, regex=False, na=False)]
        code = f"F{tid - 19:02d}"
        if not rows.empty:
            info_desc = str(rows.iloc[0]["Description"]).strip()
        task_labels[tid] = {"code": code, "type": "fall", "desc": info_desc or code}
    for tid in ADL_TASK_IDS:
        did = tid if tid <= 19 else (tid - 15)  # 35->20, 36->21
        task_labels[tid] = {"code": f"D{did:02d}", "type": "adl",
                            "desc": ADL_DESCRIPTIONS.get(tid, f"D{did:02d}")}

    all_task_ids = sorted(FALL_TASK_IDS + ADL_TASK_IDS)

    for subject in subjects:
        df_label = load_labels(subject)
        data[subject] = {}
        for task in all_task_ids:
            is_fall = task in FALL_TASK_IDS
            per_channel = {c: [] for c in CHANNELS}
            n_here = 0
            for trial in range(1, 6):
                df = load_sensor_data(subject, task, trial)
                if df is None:
                    continue
                frames = df["FrameCounter"].values
                if is_fall:
                    info = get_fall_label_info(df_label, task, trial) if df_label is not None else None
                    if info is None:
                        continue
                    onset, impact = info["onset"], info["impact"]
                    if onset < frames.min() or impact > frames.max():
                        continue

                chans = compute_all_channels(df)
                aligned = {}
                ok = True
                for c in CHANNELS:
                    a = resample_fall(frames, chans[c], onset, impact) if is_fall \
                        else resample_adl(frames, chans[c])
                    if a is None:
                        ok = False
                        break
                    aligned[c] = a
                if not ok:
                    continue

                for c in CHANNELS:
                    per_channel[c].append(np.round(aligned[c], CHANNEL_ROUND[c]))
                    range_accum[c].append(aligned[c])
                n_here += 1

            if n_here > 0:
                enc = {}
                for c in CHANNELS:
                    as_int = CHANNEL_ROUND[c] == 0
                    enc[c] = [
                        [None if (v != v) else (int(v) if as_int else float(v)) for v in arr]
                        for arr in per_channel[c]
                    ]
                data[subject][task] = enc
                counts["mode1_trials"] += n_here
                counts["fall_trials" if is_fall else "adl_trials"] += n_here

    # Fixed, robust y-ranges per channel (1st/99th percentile, padded 8%).
    yrange = {}
    for c in CHANNELS:
        stacked = np.concatenate([a for a in range_accum[c]]) if range_accum[c] else np.array([0, 1])
        stacked = stacked[~np.isnan(stacked)]
        lo, hi = np.percentile(stacked, 1), np.percentile(stacked, 99)
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        yrange[c] = [float(lo - pad), float(hi + pad)]

    return subjects, all_task_ids, task_labels, data, yrange, counts


# ----------------------------------------------------------------------------- #
# Sensor / panel configuration (mirrors the pipeline's channel groupings)
# ----------------------------------------------------------------------------- #
SENSORS = {
    "Accelerometer": {
        "primary": "acc_m",
        "mag": [["acc_m", "ACC_M", "#1abc9c"], ["vv", "VV", "#3498db"]],
        "raw": [["AccX", "Acc X", "#e74c3c"], ["AccY", "Acc Y", "#2ecc71"], ["AccZ", "Acc Z", "#3498db"]],
        "mag_unit": "ACC_M (g) / VV (m/s)", "raw_unit": "Acceleration (g)",
        "primary_unit": "ACC_M (g)",
    },
    "Gyroscope": {
        "primary": "gyro_m",
        "mag": [["gyro_m", "Gyro magnitude", "#9b59b6"]],
        "raw": [["GyrX", "Gyr X", "#e74c3c"], ["GyrY", "Gyr Y", "#2ecc71"], ["GyrZ", "Gyr Z", "#3498db"]],
        "mag_unit": "Angular velocity (deg/s)", "raw_unit": "Angular velocity (deg/s)",
        "primary_unit": "Gyro magnitude (deg/s)",
    },
    "Orientation": {
        "primary": "tilt",
        "mag": [["tilt", "Tilt", "#e67e22"]],
        "raw": [["EulerX", "Roll (X)", "#e74c3c"], ["EulerY", "Pitch (Y)", "#2ecc71"], ["EulerZ", "Yaw (Z)", "#3498db"]],
        "mag_unit": "Tilt (deg)", "raw_unit": "Angle (deg)",
        "primary_unit": "Tilt (deg)",
    },
}


def combined_range(yrange, keys):
    los = [yrange[k][0] for k in keys]
    his = [yrange[k][1] for k in keys]
    return [min(los), max(his)]


def build_html(subjects, all_task_ids, task_labels, data, yrange, counts):
    # Precompute per-sensor fixed ranges for the magnitude and raw panels.
    sensor_meta = {}
    for name, cfg in SENSORS.items():
        sensor_meta[name] = {
            "primary": cfg["primary"],
            "mag": cfg["mag"], "raw": cfg["raw"],
            "mag_unit": cfg["mag_unit"], "raw_unit": cfg["raw_unit"],
            "primary_unit": cfg["primary_unit"],
            "mag_range": combined_range(yrange, [m[0] for m in cfg["mag"]]),
            "raw_range": combined_range(yrange, [r[0] for r in cfg["raw"]]),
            "primary_range": yrange[cfg["primary"]],
        }

    tasks_meta = [{"id": t, **task_labels[t]} for t in all_task_ids]
    fall_ids = [t for t in all_task_ids if task_labels[t]["type"] == "fall"]
    adl_ids = [t for t in all_task_ids if task_labels[t]["type"] == "adl"]

    payload = {
        "phase": [float(x) for x in PHASE_GRID],
        "subjects": subjects,
        "tasks": tasks_meta,
        "fall_ids": fall_ids,
        "adl_ids": adl_ids,
        "sensors": sensor_meta,
        "data": data,
        "counts": counts,
    }
    blob = json.dumps(payload, separators=(",", ":"))

    import plotly
    plotly_js_path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    with open(plotly_js_path, encoding="utf-8") as f:
        plotly_js = f.read()

    html = _HTML_TEMPLATE.replace("/*PLOTLY_JS*/", plotly_js).replace('"/*DATA*/"', blob)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)


_HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>KFall Signal Dashboard</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #ffffff; color: #222; }
  #controls { display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
              padding: 12px 18px; border-bottom: 1px solid #e2e2e2; background: #fafafa; }
  .grp { display: flex; flex-direction: column; gap: 3px; }
  .grp label { font-size: 11px; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: .04em; }
  select { padding: 5px 8px; font-size: 13px; border: 1px solid #ccc; border-radius: 5px; background: #fff; }
  .modes { display: flex; gap: 10px; }
  .modes label { font-size: 13px; font-weight: 500; text-transform: none; letter-spacing: 0; color: #222; cursor: pointer; }
  #chart { width: 100%; height: calc(100vh - 66px); }
  .hidden { display: none !important; }
</style></head>
<body>
<div id="controls">
  <div class="grp"><label>View mode</label>
    <div class="modes">
      <label><input type="radio" name="mode" value="1" checked> Subject &times; Task</label>
      <label><input type="radio" name="mode" value="2"> Subject, all tasks</label>
      <label><input type="radio" name="mode" value="3"> Task, all subjects</label>
    </div>
  </div>
  <div class="grp"><label>Sensor</label>
    <select id="sensor"></select></div>
  <div class="grp" id="groupWrap"><label>Task group</label>
    <select id="group"><option value="fall">Falls (F01–F15)</option>
      <option value="adl">ADLs (D01–D21)</option></select></div>
  <div class="grp" id="subjWrap"><label>Subject</label>
    <select id="subject"></select></div>
  <div class="grp" id="taskWrap"><label>Task</label>
    <select id="task"></select></div>
</div>
<div id="chart"></div>

<script>/*PLOTLY_JS*/</script>
<script>
const DB = "/*DATA*/";
const PHASE = DB.phase, SUBJECTS = DB.subjects, TASKS = DB.tasks, FALL_IDS = DB.fall_ids,
      ADL_IDS = DB.adl_ids, SENSORS = DB.sensors, DATA = DB.data;
const TASK_BY_ID = {}; TASKS.forEach(t => TASK_BY_ID[t.id] = t);

const GRID = "#dcdcdc", REF = "#888";
const AXIS_FONT = { family: "Helvetica, Arial, sans-serif", size: 13, color: "#333" };
// X-axis meaning differs by task type: falls anchor on onset->impact; ADLs have no
// such event, so they anchor on their own full trial duration.
const X_FALL = "Phase (% of onset→impact)";
const X_ADL  = "Phase (% of trial duration: start→end)";
const ADL_NOTE = { x: 0.5, y: 1.045, xref: "paper", yref: "paper", showarrow: false,
  font: { size: 11, color: "#c0392b" },
  text: "ADL task: phase = 0% trial start → 100% trial end (no fall onset/impact labels)" };

// ---- controls ----
const $ = id => document.getElementById(id);
function fill(sel, items, val, text) {
  sel.innerHTML = "";
  items.forEach(it => { const o = document.createElement("option");
    o.value = val(it); o.textContent = text(it); sel.appendChild(o); });
}
fill($("sensor"), Object.keys(SENSORS), s => s, s => s);
fill($("subject"), SUBJECTS, s => s, s => "SA" + String(s).padStart(2, "0"));
fill($("task"), TASKS, t => t.id, t => t.code + " — " + t.desc);

function currentMode() { return document.querySelector('input[name=mode]:checked').value; }

function syncControls() {
  const m = currentMode();
  // Subject: hidden in mode 3. Task selector: hidden in mode 2.
  // Group toggle (Falls/ADLs): only meaningful in mode 2 (which task family to overlay).
  $("subjWrap").classList.toggle("hidden", m === "3");
  $("taskWrap").classList.toggle("hidden", m === "2");
  $("groupWrap").classList.toggle("hidden", m !== "2");
  // Mode 1 and mode 3 both list all tasks (falls + ADLs); each overlay/view stays
  // axis-consistent because only one task is shown at a time.
  if (m === "1" || m === "3")
    fill($("task"), TASKS, t => t.id,
         t => t.code + " — " + t.desc + (t.type === "adl" ? "  [ADL]" : ""));
}

// ---- helpers ----
function nanMean(arrs) {
  const n = PHASE.length, out = new Array(n).fill(0), cnt = new Array(n).fill(0);
  arrs.forEach(a => { for (let i = 0; i < n; i++) { const v = a[i];
    if (v !== null && v !== undefined) { out[i] += v; cnt[i]++; } } });
  for (let i = 0; i < n; i++) out[i] = cnt[i] ? out[i] / cnt[i] : null;
  return out;
}
function palette(n) {
  const c = []; for (let i = 0; i < n; i++) c.push(`hsl(${Math.round(360 * i / n)},68%,48%)`);
  return c;
}
const REF_LINES = (yref) => [0, 100].map(x => ({ type: "line", x0: x, x1: x, xref: "x",
  yref: yref, y0: 0, y1: 1, line: { color: REF, width: 1.2, dash: "dash" } }));

function baseLayout(panels, sensorCfg, modeLabel, xTitle) {
  xTitle = xTitle || X_FALL;
  const L = { template: "plotly_white", paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    margin: { t: 54, r: 210, b: 54, l: 66 }, hovermode: "x unified",
    font: { family: "Helvetica, Arial, sans-serif", size: 12 },
    legend: { orientation: "v", x: 1.01, xanchor: "left", y: 1, font: { size: 11 },
              groupclick: "togglegroup" },
    title: { text: modeLabel, x: 0.5, xanchor: "center", font: { size: 15 } },
    shapes: [] };
  const axCommon = { showgrid: true, gridcolor: GRID, gridwidth: 1, zeroline: false,
    showline: true, linecolor: "#999", ticks: "outside", tickfont: { size: 11 },
    mirror: false };
  if (panels === 2) {
    L.xaxis = Object.assign({}, axCommon, { anchor: "y", domain: [0, 1],
      title: { text: "", font: AXIS_FONT }, matches: "x2", showticklabels: false });
    L.xaxis2 = Object.assign({}, axCommon, { anchor: "y2", domain: [0, 1], showticklabels: true,
      title: { text: xTitle, font: AXIS_FONT } });
    L.yaxis = Object.assign({}, axCommon, { domain: [0.56, 1],
      title: { text: sensorCfg.mag_unit, font: AXIS_FONT }, range: sensorCfg.mag_range.slice(), fixedrange: false });
    L.yaxis2 = Object.assign({}, axCommon, { domain: [0, 0.44],
      title: { text: sensorCfg.raw_unit, font: AXIS_FONT }, range: sensorCfg.raw_range.slice() });
    L.shapes = REF_LINES("y domain").concat(REF_LINES("y2 domain"));
  } else {
    L.xaxis = Object.assign({}, axCommon, { anchor: "y", domain: [0, 1],
      title: { text: xTitle, font: AXIS_FONT } });
    L.yaxis = Object.assign({}, axCommon, { domain: [0, 1],
      title: { text: sensorCfg.primary_unit, font: AXIS_FONT }, range: sensorCfg.primary_range.slice() });
    L.shapes = REF_LINES("y domain");
  }
  return L;
}

function line(x, y, name, color, width, opts) {
  return Object.assign({ x, y, name, mode: "lines", type: "scatter",
    line: { color, width, shape: "linear" }, connectgaps: false }, opts || {});
}

// ---- renderers ----
function renderMode1(sensorName) {
  const cfg = SENSORS[sensorName], subj = +$("subject").value, task = +$("task").value;
  const rec = (DATA[subj] || {})[task];
  const tinfo = TASK_BY_ID[task];
  const isAdl = tinfo.type === "adl";
  const title = `Mode 1 &nbsp; SA${String(subj).padStart(2,"0")} &nbsp;&middot;&nbsp; `
    + `${tinfo.code} ${tinfo.desc} &nbsp;<span style="color:#888">(${tinfo.type})</span>`;
  const L = baseLayout(2, cfg, title, isAdl ? X_ADL : X_FALL);
  if (isAdl) L.annotations = [ADL_NOTE];  // clarify axis meaning for ADL
  const traces = [];
  if (!rec) {
    Plotly.react("chart", [], Object.assign(L, { title: { text: title + " &mdash; no data", x: .5 } }), {responsive:true});
    return 0;
  }
  // magnitude panel (xy)
  cfg.mag.forEach(([key, lab, color]) => {
    const arrs = rec[key] || [];
    arrs.forEach((a, i) => traces.push(line(PHASE, a, lab + " trial", color, 1,
      { xaxis: "x", yaxis: "y", opacity: 0.28, legendgroup: key, showlegend: false,
        hoverinfo: "skip" })));
    traces.push(line(PHASE, nanMean(arrs), lab + " (mean)", color, 2.4,
      { xaxis: "x", yaxis: "y", legendgroup: key,
        hovertemplate: `${lab} %{y:.3f}<extra></extra>` }));
  });
  // raw axes panel (x y2)
  cfg.raw.forEach(([key, lab, color]) => {
    const arrs = rec[key] || [];
    arrs.forEach((a, i) => traces.push(line(PHASE, a, lab + " trial", color, 1,
      { xaxis: "x2", yaxis: "y2", opacity: 0.25, legendgroup: key, showlegend: false,
        hoverinfo: "skip" })));
    traces.push(line(PHASE, nanMean(arrs), lab + " (mean)", color, 2.0,
      { xaxis: "x2", yaxis: "y2", legendgroup: key,
        hovertemplate: `${lab} %{y:.2f}<extra></extra>` }));
  });
  Plotly.react("chart", traces, L, { responsive: true });
  return (rec[cfg.primary] || []).length;
}

function renderOverlay(sensorName, mode) {
  const cfg = SENSORS[sensorName], primary = cfg.primary;
  const traces = []; let count = 0;
  let isAdl, L;
  if (mode === "2") {
    // One subject, overlay every task of the chosen family (falls OR ADLs) --
    // kept single-family so the x-axis has one consistent meaning.
    const subj = +$("subject").value;
    const group = $("group").value;                 // "fall" | "adl"
    isAdl = group === "adl";
    const ids = isAdl ? ADL_IDS : FALL_IDS;
    const famLabel = isAdl ? "all ADL tasks" : "all fall tasks";
    L = baseLayout(1, cfg, "", isAdl ? X_ADL : X_FALL);
    L.title.text = `Mode 2 &nbsp; SA${String(subj).padStart(2,"0")} &mdash; ${famLabel} `
      + `(mean per task, ${sensorName})`;
    const colors = palette(ids.length);
    ids.forEach((tid, i) => {
      const rec = (DATA[subj] || {})[tid]; if (!rec || !rec[primary]) return;
      const t = TASK_BY_ID[tid];
      traces.push(line(PHASE, nanMean(rec[primary]), t.code, colors[i], 1.4,
        { hovertemplate: `${t.code} ${t.desc}<br>%{y:.3f}<extra></extra>` }));
      count++;
    });
  } else {
    // One task (fall OR ADL), overlay every subject's mean.
    const task = +$("task").value, t = TASK_BY_ID[task];
    isAdl = t.type === "adl";
    L = baseLayout(1, cfg, "", isAdl ? X_ADL : X_FALL);
    L.title.text = `Mode 3 &nbsp; ${t.code} ${t.desc} &mdash; all subjects `
      + `(mean per subject, ${sensorName})`;
    const colors = palette(SUBJECTS.length);
    SUBJECTS.forEach((subj, i) => {
      const rec = (DATA[subj] || {})[task]; if (!rec || !rec[primary]) return;
      traces.push(line(PHASE, nanMean(rec[primary]), "SA" + String(subj).padStart(2,"0"),
        colors[i], 1.2, { hovertemplate: `SA${String(subj).padStart(2,"0")}<br>%{y:.3f}<extra></extra>` }));
      count++;
    });
  }
  if (isAdl) L.annotations = [ADL_NOTE];
  Plotly.react("chart", traces, L, { responsive: true });
  return count;
}

function render() {
  const m = currentMode(), sensor = $("sensor").value;
  if (m === "1") renderMode1(sensor);
  else renderOverlay(sensor, m);
}

document.querySelectorAll('input[name=mode]').forEach(r =>
  r.addEventListener("change", () => { syncControls(); render(); }));
["sensor", "subject", "task", "group"].forEach(id => $(id).addEventListener("change", render));

syncControls();
render();
console.log("KFall dashboard ready. Trials embedded:", DB.counts);
</script>
</body></html>
"""


if __name__ == "__main__":
    print("Building phase-aligned dataset (reusing ../analyze_pattern.py pipeline)...")
    subjects, all_task_ids, task_labels, data, yrange, counts = build_dataset()

    n_fall_tasks = sum(1 for t in all_task_ids if task_labels[t]["type"] == "fall")
    n_adl_tasks = sum(1 for t in all_task_ids if task_labels[t]["type"] == "adl")
    print(f"\nSubjects: {len(subjects)}  |  tasks: {len(all_task_ids)} "
          f"({n_fall_tasks} fall, {n_adl_tasks} ADL)")
    print("Trials loaded per mode:")
    print(f"  Mode 1 (subject x task): {counts['mode1_trials']} total trials "
          f"({counts['fall_trials']} fall + {counts['adl_trials']} ADL), any of {len(all_task_ids)} tasks")
    print(f"  Mode 2 (subject, all tasks): Falls/ADLs toggle -> overlay "
          f"{n_fall_tasks} fall OR {n_adl_tasks} ADL task means for one subject")
    print(f"  Mode 3 (task, all subjects): any of {len(all_task_ids)} tasks "
          f"(fall or ADL) overlaid across {len(subjects)} subjects")
    print("  (Overlays stay single-family so the x-axis has one meaning: falls = "
          "onset->impact, ADLs = trial start->end.)")

    if counts["fall_trials"] < 2300:
        print(f"  WARNING: expected ~2,300+ fall trials, got {counts['fall_trials']}.")
    else:
        print("  Fall-trial count in expected range (~2,300+). Consistent with pipeline.")

    build_html(subjects, all_task_ids, task_labels, data, yrange, counts)
    size_mb = os.path.getsize(OUT_HTML) / 1024 / 1024
    print(f"\nSaved {OUT_HTML} ({size_mb:.1f} MB, self-contained)")

    try:
        webbrowser.open(f"file://{OUT_HTML}")
        print("Opened in your default browser.")
    except Exception as exc:  # pragma: no cover
        print(f"(Could not auto-open a browser: {exc})")
