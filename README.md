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
├── legacy_baseline/          # Original exploratory work
│   ├── generate_plots.py     #   5 single-trial example plots (freefall SVM, gyro
│   ├── kfall_analysis.ipynb  #   whiplash, roll/pitch trajectory, CWT scalogram,
│   ├── explanation.md        #   label window isolation) + rationale
│   └── plots/
│
├── fall_pattern_analysis/    # Global-pattern validation (this investigation)
│   ├── analyze_pattern.py    #   Rule A/B/C sensitivity & specificity testing
│   ├── pattern_results.csv   #   per-trial results (2,319 fall + 2,744 ADL trials)
│   ├── plots/                #   rule-comparison plots
│   ├── analysis_3panel/      #   phase-aligned signal shape: ACC_M, VV, tilt
│   ├── analysis_4panel/      #   + gyroscope magnitude
│   └── docs/
│       ├── pattern_analysis.md              # technical walkthrough
│       ├── KFall_Pattern_Analysis_Report.docx  # formal report
│       ├── build_report.py                  # regenerates the .docx
│       └── paper.pdf                        # source publication
│
├── sensor_data/, label_data/, *.zip   # raw dataset (gitignored)
├── requirements.txt
└── .gitignore
```

Each script resolves its own paths relative to its own file location, so any of
them can be run directly regardless of the current working directory, e.g.:

```bash
python3 fall_pattern_analysis/analyze_pattern.py
python3 fall_pattern_analysis/analysis_4panel/visualize_pattern_4panel.py
python3 legacy_baseline/generate_plots.py
```

## Summary of findings

- **Acceleration magnitude alone is not a reliable fall detector.** It catches
  every fall (100% sensitivity) but also fires on 82.7% of ordinary daily
  activities (17.3% specificity).
- **Adding vertical velocity and orientation (the source paper's full algorithm)**
  roughly doubles specificity (35.8%) for a negligible sensitivity cost (95.6%,
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

Full detail, methodology, and figures: see `fall_pattern_analysis/docs/pattern_analysis.md`
(technical) or `fall_pattern_analysis/docs/KFall_Pattern_Analysis_Report.docx` (formal report).

## Reference

Yu, X., Jang, J., & Xiong, S. (2021). A Large-Scale Open Motion Dataset (KFall) and
Benchmark Algorithms for Detecting Pre-impact Fall of the Elderly Using Wearable
Inertial Sensors. *Frontiers in Aging Neuroscience*, 13:692865.
