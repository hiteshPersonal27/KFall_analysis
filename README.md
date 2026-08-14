# KFall Pre-Impact Fall Pattern Analysis

Signal-processing investigation of whether the pre-impact fall signature in the
[KFall dataset](https://doi.org/10.3389/fnagi.2021.692865) (Yu, Jang & Xiong, 2021)
generalizes across subjects and fall types, and which sensor signals actually carry
that pattern. Two related but separate investigations live here:

- `fall_pattern_analysis/` — the original signal-pattern investigation (below).
- `paper_implementation/` — a from-scratch reproduction of the source paper's own
  three benchmark algorithms (threshold, SVM, ConvLSTM), evaluated against the
  paper's published Table 3 numbers. See `paper_implementation/README.md`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires the KFall sensor/label data to be extracted into `sensor_data/` and
`label_data/` at the project root (from `sensor_data_new.zip` / `label_data_new.zip`).

## Project layout

```
KFall/
├── fall_pattern_analysis/            # Global-pattern validation (this investigation)
│   ├── paper_threshold_validation/   # Paper-faithful Rule A/B/C: sensitivity & specificity
│   │   ├── analyze_pattern.py        #   shared pipeline (ACC_M/VV/GYR_M) + this method's own
│   │   ├── pattern_results.csv       #   per-trial results (2,319 fall + 2,717 ADL trials)
│   │   └── plots/
│   ├── analysis_3panel/              # Phase-aligned signal shape: ACC_M, VV, tilt
│   ├── analysis_4panel/              #   + gyroscope magnitude
│   ├── signal_quality/               # Interactive Plotly dashboards (subject/task explorer)
│   ├── rolling_regression/           # Causal Savitzky-Golay β1/β2 (slope/curvature) signals
│   ├── ensemble_trigger/             # Voting ensemble: Threshold + CUSUM + Shapelet
│   └── docs/
│       ├── pattern_analysis.md                 # technical walkthrough (full project history)
│       ├── KFall_Pattern_Analysis_Report.docx  # formal report
│       ├── build_report.py                     # regenerates the .docx
│       └── paper.pdf                           # source publication
│
├── paper_implementation/              # Reproduction of the source paper's own benchmark
│   ├── data.py                        #   shared data layer (windowing, subject split)
│   ├── threshold.py                   #   Algorithm A: threshold-based (paper's Fig. 6 rule)
│   ├── svm_model.py                   #   Algorithm B: SVM on 40 hand-crafted features
│   ├── convlstm_model.py              #   Algorithm C: ConvLSTM on raw sensor windows (GPU)
│   ├── evaluate.py                    #   unified comparison vs. the paper's Table 3
│   ├── results/                       #   per-trial predictions + comparison.md
│   └── README.md                      #   run commands + full methodology notes
│
├── sensor_data/, label_data/, *.zip   # raw dataset (gitignored)
├── requirements.txt
└── .gitignore
```

Every folder under `fall_pattern_analysis/` has its own `README.md` with more
detail. Every script resolves its paths relative to its own file location (not
the current working directory), and `analyze_pattern.py` — the shared pipeline
all other methods import — lives in `paper_threshold_validation/`. Run any
script directly, e.g.:

```bash
python3 fall_pattern_analysis/paper_threshold_validation/analyze_pattern.py
python3 fall_pattern_analysis/analysis_4panel/visualize_pattern_4panel.py
python3 fall_pattern_analysis/signal_quality/build_signal_dashboard.py
python3 fall_pattern_analysis/rolling_regression/build_beta_dashboard.py
```

## Summary of findings

- **Acceleration magnitude alone is not a reliable fall detector.** It catches
  every fall (100% sensitivity) but also fires on 82.6% of ordinary daily
  activities (17.4% specificity).
- **Adding vertical velocity and orientation (the source paper's full algorithm)**
  roughly doubles specificity (34.9%) for a negligible sensitivity cost (95.6%,
  closely matching the paper's own reported 95.50%).
- **Phase-aligned visualization** (event-locked averaging across all 2,319 fall
  trials) confirms three of the four available signals — acceleration magnitude,
  vertical velocity, and gyroscope magnitude — show a sharp, consistent, globally
  reproducible pattern, verified down to individual trials and across all 32
  subjects. Orientation tilt is comparatively weak and noisy.
- **Conclusion**: the dataset does contain a genuine, generalizable fall
  signature, but fixed-threshold rules on it are not sufficient as a standalone
  detector — consistent with the source paper's own finding that trained
  classifiers (SVM, ConvLSTM) substantially outperform its threshold algorithm.
- **A 3-detector voting ensemble** (Threshold + CUSUM + Shapelet, evaluated on a
  held-out test set) reaches 94.5% sensitivity / 62.2% specificity — better than
  any single member's balance, but still short of the paper's own benchmarks,
  because two of the three detectors (Threshold, CUSUM) share correlated errors
  on fast/dynamic ADLs rather than being fully independent. Fully explainable —
  every detection traces to one of three simple, interpretable rules.

Full detail, methodology, and figures: see `fall_pattern_analysis/docs/pattern_analysis.md`
(technical) or `fall_pattern_analysis/docs/KFall_Pattern_Analysis_Report.docx` (formal report).

## Model comparison (post validation/bug-fix pass)

A deep validation pass across `paper_implementation/` and every `experiments/`
subproject found and fixed three real bugs: a window-labeling bug that
silently dropped fall trials with a short onset-to-impact duration from every
window-based evaluation; a transformer classification-head bug (two
architectures copied the LSTM baseline's "last timestep" convention despite
using non-causal self-attention, fixed to global average pooling); and an SVM
cross-validation subject-leakage bug plus an incorrectly-computed zero-crossing-rate
feature. A separate, dedicated leakage sweep checked every experiment for
normalization-stat leakage, subject overlap between splits, and other data
hygiene issues — everything else in the repo was clean. All models below were
retrained/re-evaluated after the fixes, on the identical corrected test set
(439 fall trials, 522 ADL trials):

| Model | TP | FN | TN | FP | Sensitivity | Specificity | Lead time (ms) |
|---|--:|--:|--:|--:|--:|--:|--:|
| Threshold (paper repro) | 384 | 55 | 436 | 86 | 87.47% | 83.52% | 347 ± 135 |
| SVM (paper repro) | 367 | 72 | 402 | 120 | 83.60% | 77.01% | 226 ± 123 |
| **ConvLSTM (paper repro)** | **415** | **24** | **489** | **33** | **94.53%** | **93.68%** | 224 ± 136 |
| CWT + 2D-CNN (`experiments/cwt_lstm`) | 368 | 71 | 385 | 137 | 83.83% | 73.75% | 228 ± 126 |
| ViT / PreFallKD-style repro (`experiments/vit_prefallkd`) | 415 | 24 | 457 | 65 | 94.53% | 87.55% | 225 ± 136 |
| Conv+Transformer, pooled/6-token (`experiments/conv_transformer`) | 409 | 30 | 423 | 99 | 93.17% | 81.03% | 227 ± 136 |
| Conv+Transformer, no-pool/50-token (`experiments/conv_transformer_nopool`) | 415 | 24 | 455 | 67 | 94.53% | 87.16% | 224 ± 137 |
| GAF, per-window normalization (`experiments/gaf_mtf`) | 354 | 85 | 48 | 474 | 80.64% | 9.20% | 219 ± 118 |
| GAF, global normalization (`experiments/gaf_global`) | 415 | 24 | 347 | 175 | 94.53% | 66.48% | 223 ± 131 |

**The ConvLSTM baseline from the original paper reproduction remains the
best-performing model overall** — no newer architecture tried in this repo
(attention-based or image-encoding) beat its specificity. Four independent
architectures (ConvLSTM, ViT/PreFallKD, Conv+Transformer no-pool, GAF global
norm) all converge on the exact same 24 false negatives, strong evidence
these particular fall trials are genuinely hard cases (very short
onset-to-impact duration) rather than a modeling artifact of any one
approach.

## Reference

Yu, X., Jang, J., & Xiong, S. (2021). A Large-Scale Open Motion Dataset (KFall) and
Benchmark Algorithms for Detecting Pre-impact Fall of the Elderly Using Wearable
Inertial Sensors. *Frontiers in Aging Neuroscience*, 13:692865.
