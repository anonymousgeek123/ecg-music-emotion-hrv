# ECG Signal Preprocessing for Music-Induced Emotion Recognition

Course project — ECG signal processing pipeline for HRV-based 
emotion and stress analysis in response to music stimuli.

## Overview

Implements a multi-stage ECG preprocessing pipeline to extract 
clean heart rate variability (HRV) features and classify 
emotional responses to different music genres (Classical, 
Favorite, Rock).

## Pipeline

1. **8th-order Butterworth bandpass filter** (0.5–40 Hz) — 
   removes baseline wander and high-frequency noise
2. **2nd-order Notch filter** (50 Hz) — removes powerline 
   interference
3. **Adaptive LMS filter** — suppresses residual motion artifacts 
   and baseline drift

## Features Extracted

**HRV (Time Domain):** MeanNN, SDNN, RMSSD, pNN50, Sample Entropy  
**Statistical:** Mean, Standard Deviation, Z-score normalization  
**R-peak detection** via NeuroKit2

## Models

- Voting Classifier — music emotion detection (Classical / 
  Favorite / Rock)
- Voting Regressor — stress level prediction
- Evaluation: Accuracy, MSE, R², Confusion Matrix

## Music Type → Expected Physiological Response

| Music Type | Mean HR (BPM) | Expected Stress |
|------------|---------------|-----------------|
| Classical  | 62 ± 3        | Low             |
| Favorite   | 82 ± 4        | Medium          |
| Rock       | 105 ± 5       | High            |

## Results

![ECG Filtering](outputs/ecg_filtering.png)
![Music Classification](outputs/music_confusion_matrix.png)
![Stress Prediction](outputs/stress_confusion_matrix.png)
![Error Distribution](outputs/error_distribution.png)

## Error Analysis

- Mean Absolute Error and percentage error computed per prediction
- Error distribution histogram shows spread of stress prediction errors

## Tools & Libraries

Python · NeuroKit2 · SciPy · NumPy · scikit-learn · XGBoost · 
Matplotlib · Seaborn

## Files

- `ecg_emotion_pipeline.py` — full preprocessing + ML pipeline
- `report.pdf` — detailed methodology and results
- `outputs/` — generated plots and confusion matrices
