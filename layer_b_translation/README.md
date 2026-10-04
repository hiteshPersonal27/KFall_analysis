# Layer B — Translation Layer

Converts a confirmed fall event (Layer A output: fall window + `T_onset`/`T_impact`)
into concrete actuator commands: direction, urgency, strategy, and a torque-vs-time
profile per joint. See `docs/Fall_Prevention_Actuator_Input_Pipeline.md` and
`docs/LayerB_Implementation_Plan.md` for the full design context, and
**`docs/KNOWN_ISSUES_AND_FIXES.md` for the authoritative record of what's been
fixed, how it was validated, and what's still genuinely open** — read that
before trusting any specific number out of this pipeline.

Every function is pure and takes already-computed KFall signals (`acc_m`,
`gyr_m`, `vv`, `pitch`/`roll`/`yaw`, `onset`, `impact`) as produced by
`paper_implementation/data.py::load_trial`.

- **Phase 1** (direction, CoM, time-to-impact, capture point, torque, torque
  profile, recovery strategy) — implemented, validated on a train/test
  subject split (`split_utils.py`), full results in `results/`.
- **Phase 2** (stance-leg classifier, cause/context classifier) —
  implemented. Cause/context has ground-truth validation; stance-leg does
  not (KFall has no ground-truth label for it) and is flagged accordingly.

## Validation methodology

Every calibrated constant below is fit on **train subjects only** and scored
on **held-out test subjects only**, reusing the existing seeded 80/20
subject-level split (`paper_implementation/data.py::make_subject_split`, 26
train / 6 test subjects). See `docs/KNOWN_ISSUES_AND_FIXES.md` for why this
matters and what changed when it was adopted.

## Explicit stubs / frozen placeholders (to be replaced once available)

| Stub | Placeholder value | Replaced by |
|---|---|---|
| Patient body mass | 70 kg | Layer C calibration entry |
| Leg length / lever arm | 0.9 m (de Leva adult average) | Layer C calibration entry / anthropometric table |
| Base-of-support half-width | 0.20m forward / 0.12m backward,lateral (direction-conditioned, illustrative) | Phase 2 stance-leg identification (2.1), or real horizontal CoM sensing |
| Actuator response latency | 0.10 s | Real bench-measured value once actuator hardware is selected |
| Hip/knee torque split | 70% hip / 30% knee | Literature-backed biomechanical value |
| Safety torque ceiling | 40 Nm per joint | Real Layer D value (physician/PT + biomech literature) |

## Pipeline (dependency order)

```
direction_classifier.py      (1.1)  -> direction                         [signed roll/pitch @ peak GYR_M]
com_estimation.py            (1.2)  -> com_position, com_velocity        [single-segment proxy, hardware-limited]
time_to_impact.py            (1.3)  -> time_to_impact_ms, reaction_time_ms  [causal v0+gyr_early regression]
capture_point.py             (1.4)  -> xcom_deficit, first_crossing_lead_time_ms  [direction-aware BoS + causal scan]
torque_magnitude.py          (1.5)  -> torque_hip_Nm, torque_knee_Nm     [impulse-momentum kinematic formula]
torque_profile.py            (1.6)  -> torque-vs-time array              [sized to reaction_time_ms]
recovery_strategy.py         (1.7)  -> strategy                          [derived from torque vs. ceiling]
phase2_models/cause_context_classifier.py  (2.2) -> cause_context        [ground-truth validated]
phase2_models/stance_leg_classifier.py     (2.1) -> stance_leg           [unvalidated, no ground truth]
```

`pipeline.py` chains all of these into one `run_layer_b(trial)` call.

## Run

```bash
python3 layer_b_translation/pipeline.py              # runs on a handful of sample KFall fall trials
python3 -m pytest layer_b_translation/tests/test_layer_b.py   # validates against KFall trials + sanity bounds
python3 layer_b_translation/tests/run_full_dataset.py          # full train/test revalidation, regenerates results/
```

## Headline results (see `results/summary.txt` and per-parameter files for full detail)

| Component | Held-out result |
|---|---|
| 1.1 Direction accuracy | 90.7% (gate: ≥80%, cleared) |
| 1.3 Time-to-impact correlation | r=0.534 (gate: ≥0.1 improvement over constant baseline, cleared) |
| 1.5/1.7 Torque/strategy | 100% of trials exceed the 40Nm ceiling → always "stepping" (reported as a finding, not tuned away — see `docs/KNOWN_ISSUES_AND_FIXES.md`) |
