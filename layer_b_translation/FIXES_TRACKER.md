# Layer B Phase 1 Fix Tracker

Working doc for implementing `docs/LayerB_Implementation_Plan.md`-adjacent fix
plan (saved at `/home/hitesh_mr_fedora/.claude/plans/ok-so-what-is-hidden-nautilus.md`).
Tracks progress step by step so nothing gets missed across a long session.
Update the status + notes as each step lands.

| Step | What | Status | Notes |
|---|---|---|---|
| 0 | Ground-truth direction mapping | ✅ Done | `phase1_formulas/fall_direction_labels.py` — TASK_TO_TRUE_DIRECTION dict, sourced from docs Table 2 |
| 1 | Adopt train/test subject split | ✅ Done | `split_utils.py` — reuses `paper_implementation/data.py::make_subject_split` (seed 42). 26 train subjects (1880 fall trials), 6 test subjects (439 fall trials) |
| 2 | Direction classifier redesign (1.1) | ✅ Done | Rewrote using signed roll-dominance rule at peak-GYR_M frame (roll separates fwd/bwd, pitch separates lateral -- empirical finding, opposite of naive anatomical guess, see calibration note in code). ROLL_DOMINANCE_RATIO=1.9 frozen on train (88.0% train acc). **Held-out test: 90.7%** (398/439) -- gate cleared, no ML fallback needed |
| 3 | Time-to-impact regression (1.3) | ✅ Done | 3-param causal regression `t ~ a + b*v0 + c*gyr_early` (gyr_early = max GYR_M over first 150ms post-onset, causal — rejected a whole-window gyr_peak feature that scored higher (r=0.473) but wasn't causally available). Frozen on train. **Held-out: r=0.534, MAE=101.8ms** vs constant baseline r≈0, MAE=118.1ms — gate cleared. Added `usable_reaction_time_ms()` to account for the 150ms decision delay downstream (torque profile now sized to this, not full predicted duration) |
| 4 | Direction-aware BoS + causal re-eval (1.2/1.4) | ✅ Done | CoM (1.2) unchanged (hardware-constrained, not a bug -- confirmed no fake fix applied). BoS half-width now direction-CONDITIONED (0.20m forward / 0.12m backward,lateral -- illustrative, not fit to data, honestly can't be spatially asymmetric since this CoM proxy is vertical-only, not horizontal, documented in code). Added `find_first_crossing_frame` -- causal per-frame deficit was already causal, only the `.max()` aggregation peeked at the future; new `first_crossing_lead_time_ms` reports real streaming-equivalent lead time, sanity-checked (always ≤ onset→impact duration) |
| 5 | Torque from impulse-momentum (1.5/1.6/1.7) | ✅ Done | First formula attempt (`mass*deficit/time/latency`) rejected pre-dataset via toy sanity check -- gave implausible hundreds-of-Nm torques regardless of constant choice (double time-divisor). Replaced with kinematic `F=2*mass*deficit/effective_T²` (same template as time_to_impact.py's own h=0.5at² model), `ACTUATOR_LATENCY_S=0.10s` frozen via toy check only. `recovery_strategy.py` "step" decision now derived from `required_torque_total_nm > safety_ceiling_nm` directly, config's `xcom_deficit_high_m` percentile threshold removed entirely. **Finding (not adjusted)**: given this dataset's deficits (~0.8-1.2m) and the 40Nm ceiling, nearly all sample trials now select "stepping" -- reported honestly as a consequence of the crude single-IMU CoM proxy producing large deficits, not tuned away |
| 6 | Revalidate + regenerate results | ✅ Done | `tests/run_full_dataset.py` rewritten: runs on train/test split separately, reports both side by side with subject counts, in all 9 `results/*.txt` files + `summary.txt` + full CSV. 0 pipeline errors across all 2,319 trials. Both pre-registered gates (1.1 ≥80% held-out, 1.3 r≥0.1 improvement) CLEARED |

## Final gate outcomes

| Gate | Target | Result | Outcome |
|---|---|---|---|
| 1.1 direction accuracy | ≥80% held-out | 90.7% (398/439) | ✅ CLEARED (no ML fallback needed) |
| 1.3 time-to-impact correlation | held-out r ≥ 0.1 improvement over constant baseline (r≈0) | r=0.534 | ✅ CLEARED |
| 1.5/1.7 torque/strategy | no accuracy gate — physically-derived formula, saturation rate reported as finding | 100% of trials exceed 40Nm ceiling → always "stepping" | Reported honestly, not tuned away (see 1.5 result file for why) |

All 6 pytest tests pass. All 7 originally-identified problems addressed per the plan; see `docs/LayerB_Implementation_Plan.md` and `results/summary.txt` for full detail.

## Key discipline (do not violate)

- **Calibrate on train subjects only. Score on test subjects only. Never both on the same data.**
- Pre-register gates (stated above) before running on test subjects — no retroactively lowering the bar.
- If a gate fails: iterate on train subjects (including trying an ML fallback per user's instruction), evaluate held-out test **once** when finalized — not repeatedly peeked at and adjusted against.
- New constants (e.g. ACTUATOR_LATENCY_S) get frozen before looking at their downstream effect (e.g. saturation rate), not tuned after the fact.

## Files touched so far

- `layer_b_translation/phase1_formulas/fall_direction_labels.py` (new)
- `layer_b_translation/split_utils.py` (new)
- `layer_b_translation/tests/calibrate_direction.py` (new, calibration-only, not part of pytest suite)
