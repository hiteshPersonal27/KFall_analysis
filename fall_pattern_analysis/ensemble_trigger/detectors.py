"""
Three independent, causal, per-frame pre-impact fall detectors, per
ensemble_trigger_plan.md. Each takes a 1-D signal array (in raw frame-time, computed
from data up to and including the current frame only) and returns the first frame
index at which it fires, or None if it never fires.

Signal choice per detector (see ensemble_trigger_plan.md for the rationale):
  - Threshold : ACC_M (raw accelerometer magnitude)
  - CUSUM     : VV's beta1 (Savitzky-Golay slope of vertical velocity, from
                rolling_regression/) -- the project's earliest pre-impact discriminator
  - Shapelet  : ACC_M (raw), matching the plan's bump/dip shape narrative

No signal math is duplicated here -- ACC_M/VV come from
paper_threshold_validation/analyze_pattern.py's compute_signals(), and the rolling
regression construction matches rolling_regression/visualize_beta_signals.py's
rolling_beta() (window=25 frames, same fixed pseudo-inverse trick).
"""

import numpy as np

FS = 100.0
SG_WINDOW = 25  # frames (~0.25s), matches rolling_regression/'s validated choice

# ----------------------------------------------------------------------------- #
# Savitzky-Golay beta1 (slope) -- same construction as rolling_regression/
# ----------------------------------------------------------------------------- #
_X = np.arange(SG_WINDOW, dtype=float)
_P1 = np.linalg.pinv(np.vstack([np.ones(SG_WINDOW), _X]).T)  # rows = [b0, b1]


def rolling_beta1(signal):
    """Causal SG slope. NaN for the first SG_WINDOW-1 frames (window not full)."""
    n = len(signal)
    beta1 = np.full(n, np.nan)
    if n < SG_WINDOW:
        return beta1
    win = np.lib.stride_tricks.sliding_window_view(signal, SG_WINDOW)
    beta1[SG_WINDOW - 1:] = win @ _P1[1]
    return beta1


# ----------------------------------------------------------------------------- #
# Detector 1: Threshold (on ACC_M)
# ----------------------------------------------------------------------------- #
def threshold_fire(acc_m, c):
    """fire when ACC_M(t) < c (the dip that precedes/accompanies a fall)."""
    idx = np.where(acc_m < c)[0]
    return int(idx[0]) if len(idx) else None


# ----------------------------------------------------------------------------- #
# Detector 2: CUSUM (on VV's beta1)
# ----------------------------------------------------------------------------- #
def _cusum_accumulate(delta):
    """
    Vectorized-ish two-sided CUSUM recursion: S(t) = max(0, S(t-1) + delta(t)).
    Uses numpy's ufunc.accumulate machinery (faster than a raw Python loop, avoids
    a manual per-frame for-loop) since this recursion has no closed-form vector op.
    """
    acc = np.frompyfunc(lambda a, b: a + b if a + b > 0 else 0.0, 2, 1)
    return acc.accumulate(delta, dtype=object).astype(float)


def cusum_fire(beta1_vv, k, h, baseline_frames=100):
    """
    Two-sided CUSUM on VV's beta1. mu = mean of the trial's own first
    `baseline_frames` valid (non-NaN) samples -- "the calm early part of the trial,"
    per the plan. Returns (first_fire_idx_or_None, side) where side is 'plus' if the
    upward total fired first, 'minus' if the downward total fired first (rough
    fall-direction indicator, per the plan's "useful bonus").
    """
    valid = beta1_vv[~np.isnan(beta1_vv)]
    if valid.size < baseline_frames:
        baseline_frames = max(valid.size // 2, 1)
    mu = valid[:baseline_frames].mean() if valid.size else 0.0

    x = np.nan_to_num(beta1_vv, nan=mu)  # NaN frames (pre-window) treated as baseline
    dev = x - mu
    s_plus = _cusum_accumulate(dev - k)
    s_minus = _cusum_accumulate(-dev - k)

    fire_plus = np.where(s_plus > h)[0]
    fire_minus = np.where(s_minus > h)[0]
    t_plus = int(fire_plus[0]) if len(fire_plus) else None
    t_minus = int(fire_minus[0]) if len(fire_minus) else None

    if t_plus is None and t_minus is None:
        return None, None
    if t_minus is None or (t_plus is not None and t_plus <= t_minus):
        return t_plus, "plus"
    return t_minus, "minus"


# ----------------------------------------------------------------------------- #
# Detector 3: Shapelet (on ACC_M)
# ----------------------------------------------------------------------------- #
SHAPELET_LEN = 50  # frames (~0.5s) -- long enough to capture the dip/bump+impact shape


def _zscore(arr, eps=1e-8):
    """Normalize along the last axis so matching is shape-based, not
    offset/scale-based -- without this, distance is dominated by which windows
    happen to share ACC_M's ~1g baseline level rather than the dip/bump shape
    itself (the standard fix in the shapelet literature, e.g. Ye & Keogh)."""
    mean = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, keepdims=True)
    return (arr - mean) / (std + eps)


def window_distance(shapelet, windows):
    """Z-normalized sum of squared differences, shapelet (L,) vs. windows (N, L) or
    (L,) -> (N,) or scalar. The shapelet itself is stored/plotted in raw units
    (interpretable g); normalization happens only here, at comparison time."""
    shp_z = _zscore(shapelet)
    win_z = _zscore(windows)
    return np.sum((win_z - shp_z) ** 2, axis=-1)


def min_dist_to_trials(candidate, trial_signals):
    """
    For each whole-trial signal, the minimum z-normalized distance from the
    candidate to ANY position within it (sliding window, vectorized per trial).
    This is the standard shapelet scoring statistic -- NOT a mean distance to a
    fixed sample of pre-cut windows, which is fragile: most of a trial (even a
    "fall" trial) is calm padding around the brief event, so a heterogeneous
    sample of random windows doesn't actually represent "this trial's best
    match," and z-normalization can make that calm padding look deceptively
    similar to almost anything. Taking the minimum finds wherever in the trial
    the candidate actually fits best, which is what a runtime detector would
    effectively do too (it fires at the first position that matches well).
    """
    out = np.empty(len(trial_signals))
    for i, sig in enumerate(trial_signals):
        if len(sig) < len(candidate):
            out[i] = np.inf
            continue
        windows = np.lib.stride_tricks.sliding_window_view(sig, len(candidate))
        out[i] = window_distance(candidate, windows).min()
    return out


def discover_shapelet(candidate_source_signals, adl_trial_sample, fall_trial_sample,
                       n_candidates=150, rng=None):
    """
    Offline shapelet discovery (train data only), accelerated via random sampling
    (per the plan -- exhaustive subsequence search is too heavy).

    candidate_source_signals: signals to draw raw SHAPELET_LEN candidate snippets
        from (biased to each fall trial's onset region, where the discriminative
        shape actually lives).
    adl_trial_sample, fall_trial_sample: lists of WHOLE-TRIAL signal arrays (not
        pre-cut windows) used to score each candidate via min_dist_to_trials.

    Score = mean(ADL trials' min-distance) - mean(fall trials' min-distance);
    higher = candidate sits close to real falls, far from ADLs. Returns the
    best-scoring candidate (raw units) and its score.
    """
    rng = rng or np.random.default_rng(42)
    best_score, best_shapelet = -np.inf, None
    pool = [s for s in candidate_source_signals if len(s) >= SHAPELET_LEN]
    for _ in range(n_candidates):
        sig = pool[rng.integers(len(pool))]
        start = rng.integers(0, len(sig) - SHAPELET_LEN + 1)
        cand = sig[start:start + SHAPELET_LEN]
        d_adl = min_dist_to_trials(cand, adl_trial_sample).mean()
        d_fall = min_dist_to_trials(cand, fall_trial_sample).mean()
        score = d_adl - d_fall
        if score > best_score:
            best_score, best_shapelet = score, cand.copy()
    return best_shapelet, best_score


def shapelet_fire(acc_m, shapelet, c_match):
    """fire when dist(shapelet, window ending at t) < c_match."""
    n = len(acc_m)
    if n < SHAPELET_LEN:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(acc_m, SHAPELET_LEN)
    dist = window_distance(shapelet, windows)  # dist[i] = distance for window ending at i+L-1
    idx = np.where(dist < c_match)[0]
    return int(idx[0] + SHAPELET_LEN - 1) if len(idx) else None
