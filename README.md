# KFall Pre-Impact Fall Pattern Analysis

Signal-processing investigation of whether the pre-impact fall signature in the
[KFall dataset](https://doi.org/10.3389/fnagi.2021.692865) (Yu, Jang & Xiong, 2021)
generalizes across subjects and fall types, and which sensor signals actually carry
that pattern.

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

## Reference

Yu, X., Jang, J., & Xiong, S. (2021). A Large-Scale Open Motion Dataset (KFall) and
Benchmark Algorithms for Detecting Pre-impact Fall of the Elderly Using Wearable
Inertial Sensors. *Frontiers in Aging Neuroscience*, 13:692865.
