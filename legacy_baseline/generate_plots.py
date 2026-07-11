import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

# Cwd-independent paths: this script lives in legacy_baseline/, data lives at the
# KFall project root one level up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SENSOR_DIR = os.path.join(PROJECT_ROOT, "sensor_data")
LABEL_DIR = os.path.join(PROJECT_ROOT, "label_data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

# Create directory for saving plots
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------- Helper Functions -----------------

def load_sensor_data(subject_id, task_id, trial_id):
    file_path = f"{SENSOR_DIR}/SA{subject_id:02d}/S{subject_id:02d}T{task_id:02d}R{trial_id:02d}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sensor file not found: {file_path}")
    df = pd.read_csv(file_path)
    # Compute Signal Vector Magnitude (SVM)
    df['SVM'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
    return df

def load_labels(subject_id):
    label_path = f"{LABEL_DIR}/SA{subject_id:02d}_label.xlsx"
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    df = pd.read_excel(label_path)
    df['Task Code (Task ID)'] = df['Task Code (Task ID)'].ffill()
    df['Description'] = df['Description'].ffill()
    return df

def get_fall_label_info(df_label, task_id, trial_id):
    pattern = f"({task_id})"
    match_rows = df_label[
        (df_label['Task Code (Task ID)'].str.contains(pattern, regex=False, na=False)) &
        (df_label['Trial ID'] == trial_id)
    ]
    if not match_rows.empty:
        row = match_rows.iloc[0]
        return {
            'onset': int(row['Fall_onset_frame']),
            'impact': int(row['Fall_impact_frame']),
            'desc': row['Description']
        }
    return None

# ----------------- Plot 1: Pre-Impact Freefall Signal -----------------

def generate_plot1_freefall_signal(df, subject, task, trial):
    plt.figure(figsize=(10, 5))
    
    # Calculate key metrics
    max_idx = df['SVM'].idxmax()
    min_idx = df['SVM'].idxmin()
    max_val = df.loc[max_idx, 'SVM']
    min_val = df.loc[min_idx, 'SVM']
    t_max = df.loc[max_idx, 'TimeStamp(s)']
    t_min = df.loc[min_idx, 'TimeStamp(s)']
    
    # Plot SVM
    plt.plot(df['TimeStamp(s)'], df['SVM'], color='#1abc9c', linewidth=2, label='SVM (Acc Magnitude)')
    
    # Plot 1.0g Reference
    plt.axhline(1.0, color='#7f8c8d', linestyle='--', linewidth=1, label='1.0g Baseline (Steady Standing)')
    
    # Highlight weightlessness and impact peaks
    plt.scatter(t_min, min_val, color='#e67e22', s=100, zorder=5, label=f'Min SVM (Weightlessness): {min_val:.2f}g')
    plt.annotate(f"Weightlessness ({min_val:.2f}g)", xy=(t_min, min_val), xytext=(t_min - 0.5, min_val - 0.2),
                 arrowprops=dict(facecolor='#e67e22', shrink=0.08, width=1.5, headwidth=6))
    
    plt.scatter(t_max, max_val, color='#e74c3c', marker='x', s=100, linewidth=3, zorder=5, label=f'Max SVM (Impact Peak): {max_val:.2f}g')
    plt.annotate(f"Floor Impact ({max_val:.2f}g)", xy=(t_max, max_val), xytext=(t_max + 0.3, max_val - 1.0),
                 arrowprops=dict(facecolor='#e74c3c', shrink=0.08, width=1.5, headwidth=6))
    
    # Highlight freefall region (SVM < 0.6g)
    freefall_df = df[df['SVM'] < 0.6]
    if not freefall_df.empty:
        t_ff_start = freefall_df['TimeStamp(s)'].iloc[0]
        t_ff_end = freefall_df['TimeStamp(s)'].iloc[-1]
        plt.axvspan(t_ff_start, t_ff_end, color='#f1c40f', alpha=0.2, label='Freefall Phase (<0.6g)')
    
    plt.title(f"1. Pre-Impact Freefall Signal (Subject SA{subject:02d}, Task T{task:02d}, Trial R{trial:02d})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration Magnitude (g)")
    plt.xlim(df['TimeStamp(s)'].min(), df['TimeStamp(s)'].max())
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "plot1_freefall_signal.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Plot 1 to {output_path}")

# ----------------- Plot 2: Rotational Whiplash -----------------

def generate_plot2_rotational_whiplash(df_fall, df_walk, subject, task_fall, trial_fall):
    # Setup subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True)
    
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    labels = ['GyrX (Roll Rate)', 'GyrY (Pitch Rate)', 'GyrZ (Yaw Rate)']
    
    # Plot Walking Trial (T06)
    for col, color, label in zip(['GyrX', 'GyrY', 'GyrZ'], colors, labels):
        ax1.plot(df_walk['TimeStamp(s)'], df_walk[col], color=color, linewidth=1.5, label=label)
    ax1.set_title("Normal Walking Trial (T06) - Rhythmic, Controlled Rotation", fontsize=12)
    ax1.set_ylabel("Angular Velocity (°/s)")
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.set_xlim(df_walk['TimeStamp(s)'].min(), df_walk['TimeStamp(s)'].max())
    
    # Plot Fall Trial (T32)
    for col, color, label in zip(['GyrX', 'GyrY', 'GyrZ'], colors, labels):
        ax2.plot(df_fall['TimeStamp(s)'], df_fall[col], color=color, linewidth=2, label=label)
    ax2.set_title(f"Dynamic Fall Trial (T{task_fall:02d}) - Chaotic, Massive Whiplash Spikes", fontsize=12)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Angular Velocity (°/s)")
    ax2.set_xlim(df_fall['TimeStamp(s)'].min(), df_fall['TimeStamp(s)'].max())
    
    # Compute peak rotation values
    peak_walk = df_walk[['GyrX', 'GyrY', 'GyrZ']].abs().max().max()
    peak_fall = df_fall[['GyrX', 'GyrY', 'GyrZ']].abs().max().max()
    
    fig.suptitle(f"2. Rotational Whiplash Comparison (Subject SA{subject:02d})\nWalk Peak: {peak_walk:.1f}°/s | Fall Peak: {peak_fall:.1f}°/s (Whiplash Ratio: {peak_fall/peak_walk:.1f}x)")
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "plot2_rotational_whiplash.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Plot 2 to {output_path}")

# ----------------- Plot 3: Center of Mass Orientation Change -----------------

def generate_plot3_center_of_mass(df, subject, task, trial, label_info):
    plt.figure(figsize=(8, 7))
    
    # Plot the winding trajectory line
    # We will draw it as a continuous line, and scatter dots colored by time index
    sc = plt.scatter(df['EulerX'], df['EulerY'], c=df['TimeStamp(s)'], cmap='plasma', 
                     s=15, alpha=0.8, edgecolors='none', label='Body Trajectory')
    
    # Draw simple thin line connecting them
    plt.plot(df['EulerX'], df['EulerY'], color='#a0aec0', linewidth=0.5, alpha=0.5)
    
    # Add start and end points
    plt.scatter(df['EulerX'].iloc[0], df['EulerY'].iloc[0], color='#2ecc71', marker='o', s=120, edgecolors='black', zorder=10, label='Start Point')
    plt.scatter(df['EulerX'].iloc[-1], df['EulerY'].iloc[-1], color='#e74c3c', marker='s', s=120, edgecolors='black', zorder=10, label='End Point')
    
    # Add onset and impact markers if available
    if label_info:
        onset_frame = label_info['onset']
        impact_frame = label_info['impact']
        
        try:
            onset_row = df[df['FrameCounter'] == onset_frame].iloc[0]
            impact_row = df[df['FrameCounter'] == impact_frame].iloc[0]
            
            plt.scatter(onset_row['EulerX'], onset_row['EulerY'], color='#e67e22', marker='*', s=200, edgecolors='black', zorder=11, label='Fall Onset (Loss of Balance)')
            plt.scatter(impact_row['EulerX'], impact_row['EulerY'], color='#c0392b', marker='D', s=150, edgecolors='black', zorder=11, label='Floor Impact')
        except IndexError:
            pass
            
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (seconds)')
    
    plt.title(f"3. Center of Mass Orientation (Subject SA{subject:02d}, Task T{task:02d}, Trial R{trial:02d})")
    plt.xlabel("Roll Angle (EulerX, °)")
    plt.ylabel("Pitch Angle (EulerY, °)")
    plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "plot3_center_of_mass.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Plot 3 to {output_path}")

# ----------------- Plot 4: Continuous Wavelet Transform -----------------

def generate_plot4_cwt_scalogram(df, subject, task, trial):
    # Retrieve Z-axis acceleration
    acc_z = df['AccZ'].values
    time = df['TimeStamp(s)'].values
    fs = 100.0 # 100 Hz
    
    # Perform Continuous Wavelet Transform
    # We use a Morlet wavelet
    scales = np.arange(1, 128)
    coefs, freqs = pywt.cwt(acc_z, scales, 'morl', sampling_period=1.0/fs)
    power = np.abs(coefs) ** 2
    
    # Setup subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 2]})
    
    # Raw vertical acceleration on top
    ax1.plot(time, acc_z, color='#2c3e50', linewidth=1.5)
    ax1.set_title("Vertical Acceleration (AccZ) Signal", fontsize=11)
    ax1.set_ylabel("Acc (g)")
    ax1.set_xlim(time.min(), time.max())
    
    # Wavelet scalogram on bottom
    im = ax2.pcolormesh(time, freqs, power, shading='auto', cmap='inferno')
    ax2.set_title("Continuous Wavelet Transform (CWT) Scalogram (Morlet Wavelet)", fontsize=11)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.18)
    cbar.set_label('Wavelet Coefficient Power (Energy Intensity)')
    
    fig.suptitle(f"4. CWT Time-Frequency Scalogram (Subject SA{subject:02d}, Task T{task:02d}, Trial R{trial:02d})")
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "plot4_cwt_scalogram.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Plot 4 to {output_path}")

# ----------------- Plot 5: Label Window Isolation -----------------

def generate_plot5_label_isolation(df, subject, task, trial, label_info):
    plt.figure(figsize=(10, 5))
    
    # Plot SVM vs Frame Index
    plt.plot(df['FrameCounter'], df['SVM'], color='#34495e', linewidth=1.5, label='SVM (Acc Magnitude)')
    
    if label_info:
        onset = label_info['onset']
        impact = label_info['impact']
        
        # Draw shaded translucent red region between onset and impact
        plt.axvspan(onset, impact, color='#e74c3c', alpha=0.3, label=f"Fall Phase ({impact - onset} frames / {(impact - onset)*10} ms)")
        
        # Mark onset and impact frames
        try:
            val_onset = df[df['FrameCounter'] == onset]['SVM'].values[0]
            val_impact = df[df['FrameCounter'] == impact]['SVM'].values[0]
            
            plt.scatter(onset, val_onset, color='#d35400', s=100, zorder=5, label=f'Onset Frame: {onset}')
            plt.scatter(impact, val_impact, color='#c0392b', marker='x', s=100, linewidth=3, zorder=5, label=f'Impact Frame: {impact}')
        except IndexError:
            pass
            
    plt.title(f"5. Fall Label Window Isolation (Subject SA{subject:02d}, Task T{task:02d}, Trial R{trial:02d})\nFall Description: {label_info['desc'] if label_info else 'N/A'}")
    plt.xlabel("Frame Index (100 Hz sampling)")
    plt.ylabel("Acceleration Magnitude (g)")
    plt.xlim(df['FrameCounter'].min(), df['FrameCounter'].max())
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "plot5_label_isolation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Plot 5 to {output_path}")

# ----------------- Main Execution -----------------

if __name__ == "__main__":
    # Choose default parameters for generating KFall analysis:
    # Subject: SA06
    # Fall trial: T32 R01 (Forward fall while walking caused by a slip)
    # Walking trial: T06 R01 (Normal walking)
    sub = 6
    task_fall = 32
    trial_fall = 1
    task_walk = 6
    trial_walk = 1
    
    print(f"Loading data for Subject SA{sub:02d}...")
    df_fall = load_sensor_data(sub, task_fall, trial_fall)
    df_walk = load_sensor_data(sub, task_walk, trial_walk)
    df_label = load_labels(sub)
    
    label_info = get_fall_label_info(df_label, task_fall, trial_fall)
    print(f"Loaded label details for Fall Task {task_fall}, Trial {trial_fall}: {label_info}")
    
    print("\nGenerating plots...")
    generate_plot1_freefall_signal(df_fall, sub, task_fall, trial_fall)
    generate_plot2_rotational_whiplash(df_fall, df_walk, sub, task_fall, trial_fall)
    generate_plot3_center_of_mass(df_fall, sub, task_fall, trial_fall, label_info)
    generate_plot4_cwt_scalogram(df_fall, sub, task_fall, trial_fall)
    generate_plot5_label_isolation(df_fall, sub, task_fall, trial_fall, label_info)
    
    print("\nAll 5 plots successfully generated and saved to the 'plots/' directory!")
