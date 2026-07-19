# Ensemble Pre-Impact Fall Trigger

A **voting ensemble** that pools three independent detectors into one pre-impact fall
trigger. At every frame the system decides *fire* or *don't fire* using **only past frames**
(causal → real-time deployable). The goal: fire *after* a fall begins but *before* the body
hits the ground, so a protective device has time to act.

## Status

**Implemented and evaluated.** Three detectors + majority-vote pooling, tuned on a
held-out by-subject train/test split (26/6 subjects, matching the source paper's own
ratio). All numbers below are on the **test subjects only** — never used for tuning.

## Run

```bash
python3 build_ensemble.py
```

Prints the train/test subject split, tuned parameters, sanity checks, and final
test-set metrics; saves `ensemble_results.csv` and 4 plots to `plots/`.

## The three ensemble members

| Detector | Signal | Fires when | Test sensitivity | Test specificity |
|---|---|---|---|---|
| **Threshold** | ACC_M (raw) | crosses a fixed cutoff at the current frame | 95.0% | 65.3% |
| **CUSUM** | VV's β1 (slope) | accumulated departure from baseline passes a limit | 91.8% | 57.8% |
| **Shapelet** | ACC_M (raw) | recent window matches a learned fall shape | 51.9% | 63.7% |

Votes are pooled per frame: `V(t) = v_threshold + v_cusum + v_shapelet` (0–3).

| Rule | Sensitivity | Specificity | Lead time |
|---|---|---|---|
| ANY (≥1) | 99.8% | 32.1% | 793 ms |
| **MAJORITY (≥2, default)** | **94.5%** | **62.2%** | **302 ms** |
| UNANIMOUS (=3) | 44.4% | 92.5% | 172 ms |
| *Paper's Threshold (reference)* | 95.50% | 83.43% | 333 ms |
| *Paper's SVM (reference)* | 99.77% | 94.87% | 385 ms |
| *Paper's ConvLSTM (reference)* | 99.32% | 99.01% | 403 ms |

## The key finding: partial, not full, complementarity

The per-activity false-alarm breakdown (`plots/false_alarm_breakdown.png`) shows:
- **Threshold and CUSUM are strongly correlated** — both false-alarm at ~100% on
  jumping/jogging (D04, D08, D09) and stumbling (D10). Unsurprising: ACC_M and
  VV are physically related, so both react to genuinely fast, dynamic movement.
- **Shapelet fails on a different set of activities** — mostly static/lying
  ones (D01, D02, D05, D12, D17, D18) — genuine complementarity with the other two.
- One activity, D17 (lie down to bed and get up quickly), fools **all three**.

So voting helps *some* (MAJORITY edges past Threshold alone on a few activities where
only one detector misfires) but not as much as it could, because two of the three
members share a real, physical source of correlated error rather than being fully
independent — exactly the failure mode `ensemble_trigger_plan.md` warned about.
**MAJORITY does not beat the paper's own three benchmarks** on this evaluation
protocol; it sits between the paper's Threshold and its trained classifiers,
while remaining fully explainable (every firing traces to one of three simple,
interpretable rules — unlike the paper's ConvLSTM).

## Two real implementation bugs found and fixed during this build

1. **Un-normalized shapelet distance.** Plain sum-of-squared-differences distance
   is dominated by ACC_M's shared ~1g baseline, not the dip/bump shape itself — the
   shapelet never fired on anything, including a known clean fall. Fixed with
   z-normalization (standard practice in the shapelet literature).
2. **Naive "fall trial" sampling for shapelet scoring.** Most of a fall trial is
   calm walking/standing before the event — a uniform random sample of windows
   from "fall trials" is mostly non-event, indistinguishable from ADL windows after
   z-normalization. Confirmed empirically: the *true* dip+spike shape scored
   *worse* against this naive sample than against ADL windows. Fixed by scoring
   each candidate via its **minimum distance to any position within a whole
   sampled trial** (the standard shapelet-discovery statistic), not a mean
   distance to pre-cut window snippets.

Both are documented in `detectors.py`'s and `build_ensemble.py`'s comments at the
exact functions involved.

## Files

- **`ensemble_trigger_plan.md`** — the methodology and build plan (formulas, every
  detector's rationale, voting logic). **Read this first.**
- `detectors.py` — the three detector implementations (Threshold, CUSUM, Shapelet
  discovery + matching).
- `build_ensemble.py` — orchestration: data loading, train/test split, tuning,
  evaluation, plots.
- `ensemble_results.csv` — per-trial results (every detector's fire frame + lead
  time, ensemble decision for all 3 voting rules, train/test split membership).
- `plots/` — sensitivity/specificity bars, per-activity false-alarm heatmap, the
  learned shapelet curve, lead-time comparison.

## Reuses from elsewhere in this project

- `../paper_threshold_validation/analyze_pattern.py` — validated pipeline
  (`compute_signals`, `discover_subjects`, `load_sensor_data`, `load_labels`,
  `get_fall_label_info`, `FALL_TASK_IDS`, `ADL_TASK_IDS`).
- The causal Savitzky–Golay β1 construction from `../rolling_regression/`
  (same fixed pseudo-inverse trick, window=25 frames) — reimplemented locally in
  `detectors.py`'s `rolling_beta1` to keep this folder self-contained.
