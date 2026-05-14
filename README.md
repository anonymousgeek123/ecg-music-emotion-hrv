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

| Music Type | Mean HR (BPM) | Expected Stre
