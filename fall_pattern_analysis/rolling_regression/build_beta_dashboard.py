"""
Interactive Savitzky-Golay beta-signal dashboard for the rolling-regression
fall-detection method.

The rolling regression IS a causal Savitzky-Golay derivative filter: a sliding
window is fit with a low-order polynomial, giving beta1 = smoothed slope (1st
derivative) and beta2 = smoothed curvature (2nd derivative), stamped at the
window's LAST sample (causal -- uses only past data, real-time deployable). Runs
on the three validated strong channels: ACC_M, VV, GYR_M.

This dashboard fixes three critiques of the earlier static aggregate plots
(rolling_regression/plots/beta_*.png):
  1. Blending all 15 fall types (and all ADLs) into one mean hid task-to-task
     shape differences -> Mode 2 shows each task's OWN mean, no blending.
  2. The +/-1 std band conflated within-task and between-task variance -> Mode 2's
     per-task lines make heterogeneity visible instead of hiding it in one band.
  3. Falls use an onset->impact x-axis, ADLs use start->end -- overlaying them
     mixes axis meanings -> Mode 3 (separability) compares scalar beta
     DISTRIBUTIONS at a fixed pre-impact lead, needing no shared time axis at all.

beta is always computed in raw frame-time first, THEN phase-aligned for display --
never the reverse (that would break causality).

Reuses the validated pipeline from ../analyze_pattern.py (5 Hz filter, ACC_M, VV,
gyro norm) and the SG kernel construction from visualize_beta_signals.py.

Output: a single self-contained interactive beta_dashboard.html (Plotly embedded).

Run:  python3 fall_pattern_analysis/rolling_regression/build_beta_dashboard.py
"""

import os
import sys
import json
import warnings
import webbrowser

import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_DIR)
from analyze_pattern import (  # noqa: E402
    discover_subjects, load_sensor_data, load_labels, get_fall_label_info,
    compute_signals, lowpass_filter, FALL_TASK_IDS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(SCRIPT_DIR, "beta_dashboard.html")

FS = 100.0
WINDOWS = [15, 25, 35]                 # SG window sizes (frames), user-selectable
LEADS_MS = [0, 50, 100, 150, 200, 300]  # pre-impact leads to sample falls at
LEAD_FRAMES = sorted({round(ms / 1000.0 * FS) for ms in LEADS_MS})

PHASE_MIN, PHASE_MAX, N_PHASE = -50, 150, 41
PHASE_GRID = np.linspace(PHASE_MIN, PHASE_MAX, N_PHASE)

CHANNELS = ["ACC_M", "VV", "GYR_M"]
CHANNEL_UNITS = {"ACC_M": "g", "VV": "m/s", "GYR_M": "deg/s"}
BETA_UNITS = {
    "beta1": {"ACC_M": "g/frame", "VV": "(m/s)/frame", "GYR_M": "(deg/s)/frame"},
    "beta2": {"ACC_M": "g/frame^2", "VV": "(m/s)/frame^2", "GYR_M": "(deg/s)/frame^2"},
}
RAW_ROUND = {"ACC_M": 2, "VV": 3, "GYR_M": 1}
BETA_ROUND = {
    "ACC_M": {"beta1": 4, "beta2": 5},
    "VV": {"beta1": 4, "beta2": 5},
    "GYR_M": {"beta1": 2, "beta2": 3},
}

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
RISKY_ADLS = {10: "D10 stumble", 13: "D13 sit-down", 14: "D14 quick-sit", 4: "D04 jump"}


# ----------------------------------------------------------------------------- #
# Savitzky-Golay kernels (fixed per window size; see visualize_beta_signals.py)
# ----------------------------------------------------------------------------- #
def make_kernels(w):
    x = np.arange(w, dtype=float)
    p1 = np.linalg.pinv(np.vstack([np.ones(w), x]).T)          # linear fit
    p2 = np.linalg.pinv(np.vstack([np.ones(w), x, x**2]).T)    # quadratic fit
    return p1[1], p2[2]  # slope-extraction vector, curvature-extraction vector


KERNELS = {w: make_kernels(w) for w in WINDOWS}


def rolling_beta(signal, w):
    n = len(signal)
    b1 = np.full(n, np.nan)
    b2 = np.full(n, np.nan)
    if n < w:
        return b1, b2
    win = np.lib.stride_tricks.sliding_window_view(signal, w)
    k1, k2 = KERNELS[w]
    b1[w - 1:] = win @ k1
    b2[w - 1:] = win @ k2
    return b1, b2


def channels_of(df):
    acc_m, vv = compute_signals(df)
    gyr_m = np.sqrt(
        lowpass_filter(df["GyrX"].values) ** 2
        + lowpass_filter(df["GyrY"].values) ** 2
        + lowpass_filter(df["GyrZ"].values) ** 2
    )
    return {"ACC_M": acc_m, "VV": vv, "GYR_M": gyr_m}


def resample(frames, values, start, end, adl):
    duration = end - start
    if duration <= 0:
        return None
    target = start + PHASE_GRID / 100.0 * duration
    out = np.interp(target, frames, values)
    if adl:
        out[(PHASE_GRID < 0) | (PHASE_GRID > 100)] = np.nan
    return out


def round_or_none(v, decimals):
    if v != v:  # NaN
        return None
    return round(float(v), decimals)


# ----------------------------------------------------------------------------- #
# Main data build
# ----------------------------------------------------------------------------- #
def build_dataset():
    subjects = discover_subjects()

    canon_label = load_labels(subjects[0])
    task_labels = {}
    for tid in FALL_TASK_IDS:
        pattern = f"({tid})"
        rows = canon_label[canon_label["Task Code (Task ID)"].str.contains(pattern, regex=False, na=False)]
        code = f"F{tid - 19:02d}"
        desc = str(rows.iloc[0]["Description"]).strip() if not rows.empty else code
        task_labels[tid] = {"code": code, "type": "fall", "desc": desc}
    for tid in ADL_TASK_IDS:
        did = tid if tid <= 19 else (tid - 15)
        task_labels[tid] = {"code": f"D{did:02d}", "type": "adl", "desc": ADL_DESCRIPTIONS.get(tid, f"D{did:02d}")}
    all_task_ids = sorted(FALL_TASK_IDS + ADL_TASK_IDS)

    trial_data = {}  # trial_data[subj][task] = {"raw": {ch:[arr]}, "beta": {W:{ch:{"beta1":[arr],"beta2":[arr]}}}}
    mean_accum = {t: {w: {ch: {"beta1": [], "beta2": []} for ch in CHANNELS} for w in WINDOWS} for t in all_task_ids}
    fall_lead_vals = {w: {ch: {"beta1": {L: [] for L in LEAD_FRAMES}, "beta2": {L: [] for L in LEAD_FRAMES}}
                          for ch in CHANNELS} for w in WINDOWS}
    adl_max_vals = {t: {w: {ch: {"beta1": [], "beta2": []} for ch in CHANNELS} for w in WINDOWS}
                     for t in ADL_TASK_IDS}
    raw_range_accum = {ch: [] for ch in CHANNELS}
    beta_range_accum = {w: {ch: {"beta1": [], "beta2": []} for ch in CHANNELS} for w in WINDOWS}
    counts = {"fall_trials": 0, "adl_trials": 0}

    for subject in subjects:
        df_label = load_labels(subject)
        trial_data[subject] = {}
        for task in all_task_ids:
            is_fall = task in FALL_TASK_IDS
            raw_trials = {ch: [] for ch in CHANNELS}
            beta_trials = {w: {ch: {"beta1": [], "beta2": []} for ch in CHANNELS} for w in WINDOWS}
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
                    start, end = info["onset"], info["impact"]
                    if start < frames.min() or end > frames.max():
                        continue
                else:
                    start, end = frames[0], frames[-1]

                chans = channels_of(df)
                raw_aligned = {}
                ok = True
                for ch in CHANNELS:
                    a = resample(frames, chans[ch], start, end, adl=not is_fall)
                    if a is None:
                        ok = False
                        break
                    raw_aligned[ch] = a
                if not ok:
                    continue

                beta_aligned = {}
                for w in WINDOWS:
                    beta_aligned[w] = {}
                    for ch in CHANNELS:
                        b1, b2 = rolling_beta(chans[ch], w)
                        a1 = resample(frames, b1, start, end, adl=not is_fall)
                        a2 = resample(frames, b2, start, end, adl=not is_fall)
                        beta_aligned[w][ch] = (a1, a2)

                        if is_fall:
                            impact_frame = end
                            for lead in LEAD_FRAMES:
                                target = impact_frame - lead
                                if target < frames.min():
                                    continue
                                v1 = np.interp(target, frames, b1)
                                v2 = np.interp(target, frames, b2)
                                if v1 == v1:
                                    fall_lead_vals[w][ch]["beta1"][lead].append(float(v1))
                                if v2 == v2:
                                    fall_lead_vals[w][ch]["beta2"][lead].append(float(v2))
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                m1 = np.nanmax(np.abs(b1))
                                m2 = np.nanmax(np.abs(b2))
                            if m1 == m1:
                                adl_max_vals[task][w][ch]["beta1"].append(float(m1))
                            if m2 == m2:
                                adl_max_vals[task][w][ch]["beta2"].append(float(m2))

                for ch in CHANNELS:
                    raw_trials[ch].append(raw_aligned[ch])
                    raw_range_accum[ch].append(raw_aligned[ch])
                for w in WINDOWS:
                    for ch in CHANNELS:
                        a1, a2 = beta_aligned[w][ch]
                        beta_trials[w][ch]["beta1"].append(a1)
                        beta_trials[w][ch]["beta2"].append(a2)
                        beta_range_accum[w][ch]["beta1"].append(a1)
                        beta_range_accum[w][ch]["beta2"].append(a2)
                        mean_accum[task][w][ch]["beta1"].append(a1)
                        mean_accum[task][w][ch]["beta2"].append(a2)
                n_here += 1

            if n_here > 0:
                enc_raw = {ch: [[round_or_none(v, RAW_ROUND[ch]) for v in arr] for arr in raw_trials[ch]]
                           for ch in CHANNELS}
                enc_beta = {}
                for w in WINDOWS:
                    enc_beta[w] = {}
                    for ch in CHANNELS:
                        enc_beta[w][ch] = {
                            bkey: [[round_or_none(v, BETA_ROUND[ch][bkey]) for v in arr]
                                   for arr in beta_trials[w][ch][bkey]]
                            for bkey in ("beta1", "beta2")
                        }
                trial_data[subject][task] = {"raw": enc_raw, "beta": enc_beta}
                counts["fall_trials" if is_fall else "adl_trials"] += n_here

    def nanmean_round(arrs, decimals):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(np.array(arrs), axis=0) if arrs else np.full(N_PHASE, np.nan)
        return [round_or_none(v, decimals) for v in m]

    task_means = {}
    for t in all_task_ids:
        task_means[t] = {}
        for w in WINDOWS:
            task_means[t][w] = {}
            for ch in CHANNELS:
                task_means[t][w][ch] = {
                    bkey: nanmean_round(mean_accum[t][w][ch][bkey], BETA_ROUND[ch][bkey])
                    for bkey in ("beta1", "beta2")
                }

    def fixed_range(values, pad_frac=0.08):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return [-1.0, 1.0]
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return [-1.0, 1.0]
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        pad = pad_frac * (hi - lo if hi > lo else max(abs(hi), 1.0))
        return [float(lo - pad), float(hi + pad)]

    raw_range = {ch: fixed_range(np.concatenate(raw_range_accum[ch]) if raw_range_accum[ch] else [])
                 for ch in CHANNELS}
    beta_range = {}
    for w in WINDOWS:
        beta_range[w] = {}
        for ch in CHANNELS:
            beta_range[w][ch] = {}
            for bkey in ("beta1", "beta2"):
                vals = beta_range_accum[w][ch][bkey]
                flat = np.concatenate(vals) if vals else np.array([])
                beta_range[w][ch][bkey] = fixed_range(flat)

    print(f"Fall trials: {counts['fall_trials']}  |  ADL trials: {counts['adl_trials']}")

    return {
        "subjects": subjects,
        "all_task_ids": all_task_ids,
        "task_labels": task_labels,
        "trial_data": trial_data,
        "task_means": task_means,
        "fall_lead_vals": fall_lead_vals,
        "adl_max_vals": adl_max_vals,
        "raw_range": raw_range,
        "beta_range": beta_range,
        "counts": counts,
    }


# ----------------------------------------------------------------------------- #
# HTML assembly
# ----------------------------------------------------------------------------- #
def build_html(d):
    tasks_meta = [{"id": t, **d["task_labels"][t]} for t in d["all_task_ids"]]
    fall_ids = [t for t in d["all_task_ids"] if d["task_labels"][t]["type"] == "fall"]
    adl_ids = [t for t in d["all_task_ids"] if d["task_labels"][t]["type"] == "adl"]
    risky_meta = [{"id": tid, "label": lbl} for tid, lbl in RISKY_ADLS.items()]

    payload = {
        "phase": [float(x) for x in PHASE_GRID],
        "windows": WINDOWS,
        "channels": CHANNELS,
        "leadsMs": LEADS_MS,
        "leadFrames": LEAD_FRAMES,
        "channelUnits": CHANNEL_UNITS,
        "betaUnits": BETA_UNITS,
        "subjects": d["subjects"],
        "tasks": tasks_meta,
        "fallIds": fall_ids,
        "adlIds": adl_ids,
        "riskyAdls": risky_meta,
        "trialData": d["trial_data"],
        "taskMeans": d["task_means"],
        "fallLeadVals": d["fall_lead_vals"],
        "adlMaxVals": d["adl_max_vals"],
        "rawRange": d["raw_range"],
        "betaRange": d["beta_range"],
        "counts": d["counts"],
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
<html><head><meta charset="utf-8"><title>KFall Beta Dashboard (Savitzky-Golay)</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #ffffff; color: #222; }
  #controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
              padding: 10px 18px; border-bottom: 1px solid #e2e2e2; background: #fafafa; }
  .grp { display: flex; flex-direction: column; gap: 3px; }
  .grp label { font-size: 11px; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: .04em; }
  select { padding: 5px 8px; font-size: 13px; border: 1px solid #ccc; border-radius: 5px; background: #fff; }
  .modes { display: flex; gap: 10px; }
  .modes label { font-size: 13px; font-weight: 500; text-transform: none; letter-spacing: 0; color: #222; cursor: pointer; }
  #note { padding: 4px 18px; font-size: 12px; color: #555; background: #fffbe6; border-bottom: 1px solid #f0e6b0; }
  #chart { width: 100%; height: calc(100vh - 96px); }
  .hidden { display: none !important; }
</style></head>
<body>
<div id="controls">
  <div class="grp"><label>View mode</label>
    <div class="modes">
      <label><input type="radio" name="mode" value="1" checked> Trial explorer</label>
      <label><input type="radio" name="mode" value="2"> Per-task means</label>
      <label><input type="radio" name="mode" value="3"> Separability</label>
    </div>
  </div>
  <div class="grp"><label>Channel</label><select id="channel"></select></div>
  <div class="grp"><label>Derivative</label>
    <select id="beta"><option value="beta1">&beta;1 (slope)</option><option value="beta2">&beta;2 (curvature)</option></select></div>
  <div class="grp"><label>SG window</label><select id="window"></select></div>
  <div class="grp" id="subjWrap"><label>Subject</label><select id="subject"></select></div>
  <div class="grp" id="taskWrap"><label>Task</label><select id="task"></select></div>
  <div class="grp" id="groupWrap"><label>Task group</label>
    <select id="group"><option value="fall">Falls (F01-F15)</option><option value="adl">ADLs (D01-D21)</option></select></div>
  <div class="grp" id="leadWrap"><label>Pre-impact lead</label><select id="lead"></select></div>
</div>
<div id="note"></div>
<div id="chart"></div>

<script>/*PLOTLY_JS*/</script>
<script>
const DB = "/*DATA*/";
const PHASE = DB.phase, WINDOWS = DB.windows, CHANNELS = DB.channels,
      LEADS_MS = DB.leadsMs, LEAD_FRAMES = DB.leadFrames,
      CH_UNITS = DB.channelUnits, BETA_UNITS = DB.betaUnits,
      SUBJECTS = DB.subjects, TASKS = DB.tasks, FALL_IDS = DB.fallIds, ADL_IDS = DB.adlIds,
      RISKY = DB.riskyAdls, TRIAL = DB.trialData, MEANS = DB.taskMeans,
      FALL_LEAD = DB.fallLeadVals, ADL_MAX = DB.adlMaxVals,
      RAW_RANGE = DB.rawRange, BETA_RANGE = DB.betaRange;
const TASK_BY_ID = {}; TASKS.forEach(t => TASK_BY_ID[t.id] = t);

const GRID = "#dcdcdc", REF = "#888";
const AXIS_FONT = { family: "Helvetica, Arial, sans-serif", size: 13, color: "#333" };
const X_FALL = "Phase (% of onset→impact)";
const X_ADL  = "Phase (% of trial duration: start→end)";
const ADL_NOTE = { x: 0.5, y: 1.05, xref: "paper", yref: "paper", showarrow: false,
  font: { size: 11, color: "#c0392b" },
  text: "ADL task: phase = 0% trial start → 100% trial end (no fall onset/impact labels)" };

const $ = id => document.getElementById(id);
function fill(sel, items, val, text) {
  sel.innerHTML = "";
  items.forEach(it => { const o = document.createElement("option");
    o.value = val(it); o.textContent = text(it); sel.appendChild(o); });
}
fill($("channel"), CHANNELS, c => c, c => c);
fill($("window"), WINDOWS, w => w, w => w + " frames (" + (w*10) + " ms)");
fill($("subject"), SUBJECTS, s => s, s => "SA" + String(s).padStart(2, "0"));
fill($("task"), TASKS, t => t.id, t => t.code + " — " + t.desc + (t.type==="adl" ? "  [ADL]" : ""));
fill($("lead"), LEADS_MS, ms => ms, ms => ms + " ms before impact");

function currentMode() { return document.querySelector('input[name=mode]:checked').value; }

function syncControls() {
  const m = currentMode();
  $("subjWrap").classList.toggle("hidden", m === "3");
  $("taskWrap").classList.toggle("hidden", m === "2" || m === "3");
  $("groupWrap").classList.toggle("hidden", m !== "2");
  $("leadWrap").classList.toggle("hidden", m !== "3");
  if (m === "1") fill($("task"), TASKS, t => t.id, t => t.code + " — " + t.desc + (t.type==="adl" ? "  [ADL]" : ""));
}

function nanMean(arrs) {
  const n = PHASE.length, out = new Array(n).fill(0), cnt = new Array(n).fill(0);
  arrs.forEach(a => { for (let i = 0; i < n; i++) { const v = a[i];
    if (v !== null && v !== undefined) { out[i] += v; cnt[i]++; } } });
  for (let i = 0; i < n; i++) out[i] = cnt[i] ? out[i] / cnt[i] : null;
  return out;
}
function palette(n) { const c = []; for (let i = 0; i < n; i++) c.push(`hsl(${Math.round(360*i/n)},68%,48%)`); return c; }
const REF_LINES = (yref) => [0, 100].map(x => ({ type: "line", x0: x, x1: x, xref: "x",
  yref: yref, y0: 0, y1: 1, line: { color: REF, width: 1.2, dash: "dash" } }));
function line(x, y, name, color, width, opts) {
  return Object.assign({ x, y, name, mode: "lines", type: "scatter",
    line: { color, width, shape: "linear" }, connectgaps: false }, opts || {});
}

function betaLabel(bkey) { return bkey === "beta1" ? "β1 (slope)" : "β2 (curvature)"; }
function betaUnit(ch, bkey) { return BETA_UNITS[bkey][ch]; }

function baseLayout(panels, yTitleTop, yTitleBottom, yRangeTop, yRangeBottom, modeLabel, xTitle) {
  xTitle = xTitle || X_FALL;
  const L = { template: "plotly_white", paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    margin: { t: 54, r: 210, b: 54, l: 70 }, hovermode: "x unified",
    font: { family: "Helvetica, Arial, sans-serif", size: 12 },
    legend: { orientation: "v", x: 1.01, xanchor: "left", y: 1, font: { size: 10 }, groupclick: "togglegroup" },
    title: { text: modeLabel, x: 0.5, xanchor: "center", font: { size: 14 } }, shapes: [] };
  const ax = { showgrid: true, gridcolor: GRID, gridwidth: 1, zeroline: false,
    showline: true, linecolor: "#999", ticks: "outside", tickfont: { size: 11 } };
  if (panels === 2) {
    L.xaxis = Object.assign({}, ax, { anchor: "y", domain: [0,1], title: {text:""}, matches: "x2", showticklabels: false });
    L.xaxis2 = Object.assign({}, ax, { anchor: "y2", domain: [0,1], title: { text: xTitle, font: AXIS_FONT } });
    L.yaxis = Object.assign({}, ax, { domain: [0.56, 1], title: { text: yTitleTop, font: AXIS_FONT }, range: yRangeTop });
    L.yaxis2 = Object.assign({}, ax, { domain: [0, 0.44], title: { text: yTitleBottom, font: AXIS_FONT }, range: yRangeBottom });
    L.shapes = REF_LINES("y domain").concat(REF_LINES("y2 domain"));
  } else {
    L.xaxis = Object.assign({}, ax, { anchor: "y", domain: [0,1], title: { text: xTitle, font: AXIS_FONT } });
    L.yaxis = Object.assign({}, ax, { domain: [0,1], title: { text: yTitleTop, font: AXIS_FONT }, range: yRangeTop });
    L.shapes = REF_LINES("y domain");
  }
  return L;
}

// ---- Mode 1: Trial explorer ----
function renderMode1() {
  const ch = $("channel").value, bkey = $("beta").value, w = +$("window").value;
  const subj = +$("subject").value, task = +$("task").value;
  const tinfo = TASK_BY_ID[task];
  const isAdl = tinfo.type === "adl";
  const rec = (TRIAL[subj] || {})[task];
  $("note").textContent = isAdl
    ? "ADL task selected: x-axis = 0% trial start → 100% trial end (no fall onset/impact labels)."
    : "Fall task selected: x-axis = 0% labeled onset → 100% labeled impact.";

  const title = `Mode 1  SA${String(subj).padStart(2,"0")} · ${tinfo.code} ${tinfo.desc} `
    + `(${ch}, ${betaLabel(bkey)}, SG window=${w})`;
  const L = baseLayout(2, `${ch} (${CH_UNITS[ch]})`, `${betaLabel(bkey)} (${betaUnit(ch,bkey)})`,
    RAW_RANGE[ch].slice(), BETA_RANGE[w][ch][bkey].slice(), title, isAdl ? X_ADL : X_FALL);
  if (isAdl) L.annotations = [ADL_NOTE];

  const traces = [];
  if (!rec) { Plotly.react("chart", [], L, {responsive:true}); return; }

  const rawArrs = rec.raw[ch] || [];
  rawArrs.forEach(a => traces.push(line(PHASE, a, ch+" trial", "#1abc9c", 1,
    { xaxis:"x", yaxis:"y", opacity:0.28, legendgroup:"raw", showlegend:false, hoverinfo:"skip" })));
  traces.push(line(PHASE, nanMean(rawArrs), ch+" (mean)", "#1abc9c", 2.4,
    { xaxis:"x", yaxis:"y", legendgroup:"raw", hovertemplate:`${ch} %{y:.3f}<extra></extra>` }));

  const betaArrs = (rec.beta[w] && rec.beta[w][ch]) ? rec.beta[w][ch][bkey] : [];
  betaArrs.forEach(a => traces.push(line(PHASE, a, betaLabel(bkey)+" trial", "#c0392b", 1,
    { xaxis:"x2", yaxis:"y2", opacity:0.25, legendgroup:"beta", showlegend:false, hoverinfo:"skip" })));
  traces.push(line(PHASE, nanMean(betaArrs), betaLabel(bkey)+" (mean)", "#c0392b", 2.2,
    { xaxis:"x2", yaxis:"y2", legendgroup:"beta", hovertemplate:`${betaLabel(bkey)} %{y:.5f}<extra></extra>` }));

  Plotly.react("chart", traces, L, { responsive: true });
}

// ---- Mode 2: Per-task means (no blended band -- every task its own line) ----
function renderMode2() {
  const ch = $("channel").value, bkey = $("beta").value, w = +$("window").value;
  const subj = +$("subject").value, group = $("group").value;
  const isAdl = group === "adl";
  const ids = isAdl ? ADL_IDS : FALL_IDS;
  $("note").textContent = "Each line is that task's OWN mean β curve — no cross-task blending, "
    + "so heterogeneity between fall/ADL types stays visible.";

  const title = `Mode 2  SA${String(subj).padStart(2,"0")} — all ${isAdl?"ADL":"fall"} tasks `
    + `(mean per task, ${ch}, ${betaLabel(bkey)}, SG window=${w})`;
  const L = baseLayout(1, `${betaLabel(bkey)} (${betaUnit(ch,bkey)})`, null,
    BETA_RANGE[w][ch][bkey].slice(), null, title, isAdl ? X_ADL : X_FALL);
  if (isAdl) L.annotations = [ADL_NOTE];

  const traces = []; const colors = palette(ids.length);
  ids.forEach((tid, i) => {
    const rec = (TRIAL[subj] || {})[tid];
    const m = (MEANS[tid] && MEANS[tid][w] && MEANS[tid][w][ch]) ? MEANS[tid][w][ch][bkey] : null;
    if (!m) return;
    const t = TASK_BY_ID[tid];
    traces.push(line(PHASE, m, t.code, colors[i], 1.4,
      { hovertemplate: `${t.code} ${t.desc}<br>%{y:.5f}<extra></extra>` }));
  });
  Plotly.react("chart", traces, L, { responsive: true });
}

// ---- Mode 3: Separability (box distributions, no shared time axis needed) ----
function renderMode3() {
  const ch = $("channel").value, bkey = $("beta").value, w = +$("window").value, leadMs = +$("lead").value;
  const leadFrame = LEAD_FRAMES[LEADS_MS.indexOf(leadMs)];
  $("note").textContent = "Falls: β sampled at the chosen lead before impact. ADLs: worst-case max|β| "
    + "over the WHOLE trial (the false-positive-relevant statistic — a threshold detector fires if ANY "
    + "moment crosses it). Distribution overlap = expected false-alarm rate.";

  const title = `Mode 3  Separability at ${leadMs} ms pre-impact (${ch}, ${betaLabel(bkey)}, SG window=${w})`;
  const L = { template: "plotly_white", paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    margin: { t: 54, r: 40, b: 90, l: 70 }, showlegend: false,
    title: { text: title, x: 0.5, xanchor: "center", font: { size: 14 } },
    xaxis: { showgrid: false, tickangle: -20 },
    yaxis: { showgrid: true, gridcolor: GRID, zeroline: true, zerolinecolor: "#999",
      title: { text: `${betaLabel(bkey)} (${betaUnit(ch,bkey)})`, font: AXIS_FONT } } };

  const traces = [];
  const fv = (FALL_LEAD[w] && FALL_LEAD[w][ch]) ? FALL_LEAD[w][ch][bkey][leadFrame] : [];
  traces.push({ type: "box", y: fv || [], name: "All falls", marker: { color: "#c0392b" }, boxpoints: false });

  RISKY.forEach(r => {
    const av = (ADL_MAX[r.id] && ADL_MAX[r.id][w] && ADL_MAX[r.id][w][ch]) ? ADL_MAX[r.id][w][ch][bkey] : [];
    traces.push({ type: "box", y: av || [], name: r.label, marker: { color: "#2980b9" }, boxpoints: false });
  });

  const allAdlVals = [];
  ADL_IDS.forEach(tid => {
    const av = (ADL_MAX[tid] && ADL_MAX[tid][w] && ADL_MAX[tid][w][ch]) ? ADL_MAX[tid][w][ch][bkey] : [];
    allAdlVals.push(...av);
  });
  traces.push({ type: "box", y: allAdlVals, name: "All ADLs", marker: { color: "#7f8c8d" }, boxpoints: false });

  Plotly.react("chart", traces, L, { responsive: true });
}

function render() {
  const m = currentMode();
  if (m === "1") renderMode1();
  else if (m === "2") renderMode2();
  else renderMode3();
}

document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener("change", () => { syncControls(); render(); }));
["channel","beta","window","subject","task","group","lead"].forEach(id => $(id).addEventListener("change", render));

syncControls();
render();
console.log("KFall beta dashboard ready. Trials embedded:", DB.counts);
</script>
</body></html>
"""


if __name__ == "__main__":
    print("Building Savitzky-Golay beta dataset (reusing ../analyze_pattern.py pipeline)...")
    d = build_dataset()
    build_html(d)
    size_mb = os.path.getsize(OUT_HTML) / 1024 / 1024
    print(f"\nSaved {OUT_HTML} ({size_mb:.1f} MB, self-contained)")

    try:
        webbrowser.open(f"file://{OUT_HTML}")
        print("Opened in your default browser.")
    except Exception as exc:  # pragma: no cover
        print(f"(Could not auto-open a browser: {exc})")
