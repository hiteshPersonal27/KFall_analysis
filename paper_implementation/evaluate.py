"""
Unified evaluation: reads the three per-trial result CSVs produced by
threshold.py, svm_model.py and convlstm_model.py, computes sensitivity /
specificity / lead time for each, and writes a Table-3-style comparison
against the paper's reference numbers.

Run (after all three algorithms have produced results/*.csv):
  python3 paper_implementation/evaluate.py
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
COMPARISON_MD = os.path.join(RESULTS_DIR, "comparison.md")

RESULT_FILES = {
    "Threshold": "threshold.csv",
    "SVM": "svm.csv",
    "ConvLSTM": "convlstm.csv",
}

PAPER_REFERENCE = {
    "Threshold": {"sensitivity": 0.9550, "specificity": 0.8343, "lead_mean": 333, "lead_std": 160},
    "SVM": {"sensitivity": 0.9977, "specificity": 0.9487, "lead_mean": 385, "lead_std": 159},
    "ConvLSTM": {"sensitivity": 0.9932, "specificity": 0.9901, "lead_mean": 403, "lead_std": 163},
}


def summarize(df):
    fall = df[df["is_fall"]]
    adl = df[~df["is_fall"]]
    tp = int(fall["detected"].sum())
    fn = len(fall) - tp
    tn = int((~adl["detected"]).sum())
    fp = len(adl) - tn
    sensitivity = tp / len(fall) if len(fall) else float("nan")
    specificity = tn / len(adl) if len(adl) else float("nan")
    lead = fall.loc[fall["detected"], "lead_time_ms"]
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": sensitivity, "specificity": specificity,
            "lead_mean": lead.mean(), "lead_std": lead.std()}


def build_comparison():
    rows = []
    missing = []
    for algo, fname in RESULT_FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        df = pd.read_csv(path)
        s = summarize(df)
        ref = PAPER_REFERENCE[algo]
        rows.append({
            "Algorithm": algo,
            "FN": s["fn"], "FP": s["fp"],
            "Sensitivity": s["sensitivity"], "Sensitivity (paper)": ref["sensitivity"],
            "Specificity": s["specificity"], "Specificity (paper)": ref["specificity"],
            "Lead time (ms)": f"{s['lead_mean']:.0f} +/- {s['lead_std']:.0f}",
            "Lead time (paper, ms)": f"{ref['lead_mean']} +/- {ref['lead_std']}",
        })
    return pd.DataFrame(rows), missing


def check_success_criteria(df):
    checks = []
    by_algo = {r["Algorithm"]: r for _, r in df.iterrows()}
    if "SVM" in by_algo and "Threshold" in by_algo:
        checks.append(("SVM outperforms Threshold (sensitivity)",
                        by_algo["SVM"]["Sensitivity"] > by_algo["Threshold"]["Sensitivity"]))
    if "ConvLSTM" in by_algo and "Threshold" in by_algo:
        checks.append(("ConvLSTM outperforms Threshold (sensitivity)",
                        by_algo["ConvLSTM"]["Sensitivity"] > by_algo["Threshold"]["Sensitivity"]))
    if "ConvLSTM" in by_algo:
        r = by_algo["ConvLSTM"]
        checks.append(("ConvLSTM most balanced sensitivity/specificity",
                        abs(r["Sensitivity"] - r["Specificity"]) < 0.15))
    if all(a in by_algo for a in ("Threshold", "SVM", "ConvLSTM")):
        checks.append(("Specificity ordering Threshold < SVM < ConvLSTM",
                        by_algo["Threshold"]["Specificity"] < by_algo["SVM"]["Specificity"]
                        < by_algo["ConvLSTM"]["Specificity"]))
    return checks


def to_markdown(df, checks, missing):
    lines = ["# KFall Baseline Reproduction -- Comparison\n"]
    if missing:
        lines.append(f"**Missing results (not yet run):** {', '.join(missing)}\n")
    if not df.empty:
        cols = ["Algorithm", "FN", "FP", "Sensitivity", "Sensitivity (paper)",
                "Specificity", "Specificity (paper)", "Lead time (ms)", "Lead time (paper, ms)"]
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        lines += [header, sep]
        for _, r in df.iterrows():
            cells = [str(r["Algorithm"]), str(r["FN"]), str(r["FP"]),
                     f"{r['Sensitivity']:.2%}", f"{r['Sensitivity (paper)']:.2%}",
                     f"{r['Specificity']:.2%}", f"{r['Specificity (paper)']:.2%}",
                     r["Lead time (ms)"], r["Lead time (paper, ms)"]]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("\n## Reproduction success criteria\n")
        for name, passed in checks:
            lines.append(f"- [{'x' if passed else ' '}] {name}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    df, missing = build_comparison()
    if missing:
        print(f"Warning: missing results for {missing} -- run the corresponding script(s) first.")
    if df.empty:
        print("No results found. Run threshold.py / svm_model.py / convlstm_model.py first.")
    else:
        pd.set_option("display.width", 140)
        print(df.to_string(index=False))
        checks = check_success_criteria(df)
        print("\nReproduction success criteria:")
        for name, passed in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

        md = to_markdown(df, checks, missing)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(COMPARISON_MD, "w") as f:
            f.write(md)
        print(f"\nSaved comparison table to {COMPARISON_MD}")
