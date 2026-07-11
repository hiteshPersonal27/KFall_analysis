# KFall Sensor Data Analysis: Implementation and Rationale

This document provides a detailed explanation of the exploratory data analysis (EDA) pipeline implemented for the KFall dataset, detailing the signal processing rationale, mathematical modeling, and clinical motivation for each visualization.

---

## 1. Context and Objectives
The KFall dataset is designed for **pre-impact fall detection** research. Unlike post-impact detection (which simply registers that a user has already fallen), pre-impact detection aims to identify the fall *during the descent phase* (typically within a 100–200 ms window prior to impact). This window is crucial for deploying active protective systems, such as wearable body airbags, to prevent hip and head injuries in the elderly.

The sensor data is collected at **100 Hz** from a wearable IMU placed on the lower back (L5 vertebra) containing:
1. **Triaxial Accelerometer (`AccX`, `AccY`, `AccZ`)** in units of $g$ (standard gravity).
2. **Triaxial Gyroscope (`GyrX`, `GyrY`, `GyrZ`)** in units of $\text{deg/s}$.
3. **Triaxial Euler Angles (`EulerX`, `EulerY`, `EulerZ`)** representing Roll, Pitch, and Yaw in degrees.

---

## 2. Explanation of the 5 Plots and Their Rationale

### Plot 1: The Pre-Impact Freefall Signal (SVM)
* **What it plots:** Time (X-axis) vs. Signal Vector Magnitude (Y-axis) of acceleration.
* **Mathematical Formula:** 
  $$SVM = \sqrt{\text{AccX}^2 + \text{AccY}^2 + \text{AccZ}^2}$$
* **Plot Details:** Highlights the lowest acceleration magnitude (weightlessness) and the highest spike (floor impact) with arrows, and shades the freefall region where $SVM < 0.6g$.
* **Rationale & Clinical Motivation:** 
  During steady standing or controlled walking, the body is subject only to normal gravity, so the $SVM$ value hovers around $1.0g$. When a fall begins, the body goes into a state of partial freefall, causing the acceleration magnitude to drop significantly towards $0.0g$ (weightlessness). Immediately after this freefall dip, a massive deceleration spike (often exceeding $3.0g$ to $8.0g$) occurs as the body hits the floor. 
  By tracking the $SVM$, we can identify the weightlessness phase as a primary, low-latency trigger for pre-impact fall detection.

---

### Plot 2: Rotational Whiplash (Gyroscope Comparison)
* **What it plots:** Time vs. Angular Velocity ($\text{deg/s}$) overlaying `GyrX` (Roll), `GyrY` (Pitch), and `GyrZ` (Yaw) for a normal walking trial (T06) compared side-by-side with a slip-fall trial (T32).
* **Rationale & Clinical Motivation:**
  Activities of Daily Living (ADLs) such as walking, sitting, or standing up display highly rhythmic, low-frequency, and low-amplitude angular velocity patterns. Conversely, a fall is characterized by a rapid, chaotic loss of control, resulting in high-frequency, high-amplitude angular velocity spikes (whiplash). 
  Comparing these signals side-by-side demonstrates the stark contrast in rotational dynamics: walking peaks generally stay below $50\text{ deg/s}$ to $100\text{ deg/s}$, whereas a slip-fall produces whiplash spikes exceeding $300\text{ deg/s}$ to $600\text{ deg/s}$. Gyroscope signals serve as excellent secondary features to prevent false positives from rapid ADL movements (e.g., sitting down quickly).

---

### Plot 3: Center of Mass Orientation Change (Roll vs. Pitch Phase Space)
* **What it plots:** Roll Angle (`EulerX`) on the X-axis vs. Pitch Angle (`EulerY`) on the Y-axis, colored by a time gradient (colormap). It marks the start point, end point, loss-of-balance onset, and floor impact.
* **Rationale & Clinical Motivation:**
  This 2D trajectory maps the displacement of the user's lower back (the body's center of mass) in 3D space. Under normal circumstances, the body stays within a narrow cylinder of stability. During a fall, the phase space trajectory spirals outwards as the body tilts.
  Plotting Roll vs. Pitch enables researchers to define a **"boundary of stability"** (or a point of no return). If the orientation angles cross this boundary, a fall is physically inevitable, meaning active protective devices can be safely deployed without fear of false triggers.

---

### Plot 4: Continuous Wavelet Transform (CWT) Scalogram
* **What it plots:** A two-panel chart showing the raw vertical acceleration (`AccZ`) on top and its 2D Wavelet Scalogram (Time-Frequency power spectrum) on the bottom using a Morlet wavelet.
* **Rationale & Clinical Motivation:**
  Standard Fourier Transforms (FFT) lose time information, which is unacceptable for analyzing transient events like falls. While Short-Time Fourier Transforms (STFT) keep time details, they suffer from a fixed resolution window.
  The **Continuous Wavelet Transform (CWT)** provides variable resolution: high frequency resolution at low frequencies, and high time resolution at high frequencies. This is perfect for fall detection because the onset of a fall is often preceded by a rapid high-frequency transient (such as the high-frequency slip of a foot or a trip against an obstacle) before the low-frequency descent begins. The scalogram uncovers these micro-instabilities in the frequency domain before they are clearly visible in the raw time-series.

---

### Plot 5: Label Window Isolation
* **What it plots:** Frame Index (X-axis) vs. SVM (Y-axis), with a translucent red shaded window highlighting the frames between the clinical labels `Fall_onset_frame` and `Fall_impact_frame`.
* **Rationale & Clinical Motivation:**
  In machine learning datasets, data must be correctly windowed to train predictive models (like LSTMs, GRUs, or Mamba architectures). 
  - **Fall Onset:** The exact frame where the participant loses balance.
  - **Fall Impact:** The frame where the impact occurs.
  The interval between these two frames is the **pre-impact detection window**. The goal of a pre-impact detection model is to trigger an alarm *after* the onset frame but *before* the impact frame. Visualizing this window against the sensor readings verifies the label ground truth, ensuring that training features are extracted precisely from the pre-impact phase, and not corrupted by post-impact sensor noise.

---

## 3. Technology Rationale
- **Pandas & NumPy:** Standard for fast, vectorized calculations (such as the Euclidean norm for SVM) and file handling.
- **Matplotlib & Seaborn:** Lightweight, highly customizable, and easy to export as high-quality PNGs (300 DPI) for clinical papers, reports, or slides without running a web server.
- **PyWavelets (`pywt`):** The industry standard Python library for wavelet transforms, providing highly optimized C-implemented algorithms for CWT computation.
