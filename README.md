# Deep Learning-Based Wearable IoT Health Monitoring System for Real-Time ECG and SpO2 Abnormality Detection

This repository contains the implementation accompanying the paper **"Deep Learning-Based Wearable IoT Health Monitoring System for Real-Time ECG and SpO2 Abnormality Detection."**

## Authors
- Vinnakota Jagadeesh

Department of Computer Science and Engineering, Vignan's Foundation for Science, Technology and Research, Guntur, India

## Abstract

Cardiovascular diseases remain the single largest contributor to global mortality. This project introduces a deep learning framework that performs real-time abnormality detection on two complementary physiological streams — **ECG** and **SpO2** — acquired from wearable IoT sensors. Four neural architectures are designed, trained, and compared:

1. **CNN-LSTM Baseline**
2. **Attention-Augmented Bidirectional LSTM (Attention-BiLSTM)** — the proposed primary contribution
3. **Multimodal ECG-SpO2 Fusion Network**
4. **Edge-Optimised Lightweight Model** for microcontroller deployment

The Attention-BiLSTM achieves **96.7% accuracy** and **AUC-ROC of 0.991** at **12.4 ms inference latency**, while the edge-optimised variant fits under **50 KB** with float16 quantisation — suitable for ARM Cortex-M4 deployment.

## Key Contributions

1. **Attention-BiLSTM architecture** — couples a self-attention gate with a Bidirectional LSTM to capture long-range QRS-T dependencies for arrhythmia classification.
2. **Temporally-aligned multimodal fusion** — dual-branch network combining ECG and SpO2 modality-specific encoders.
3. **Systematic model compression** — float32, float16, and INT8 quantisation evaluated across all four architectures.
4. **Production IoT pipeline** — end-to-end system from sensor acquisition through edge inference to a Flask dashboard with MQTT-driven real-time visualization and clinical alerting.

## Datasets

| Dataset | Total | Normal | Abnormal | Split |
|---|---|---|---|---|
| MIT-BIH ECG | 3,000 | 1,827 (61%) | 1,173 (39%) | 80/20 |
| Simulated SpO2 | 3,000 | 2,100 (70%) | 900 (30%) | 80/20 |

- **ECG**: MIT-BIH Arrhythmia Database (PhysioNet) — 48 half-hour ambulatory ECG recordings from 47 subjects, sampled at 360 Hz.
- **SpO2**: A physiologically-informed signal simulator (no large-scale public SpO2 dataset exists at the required temporal resolution), modeling cardiac modulation, respiratory variation, and clinical-zone Gaussian noise across three severity zones (Normal, Warning, Critical).

## Model Architectures

### 1. CNN-LSTM Baseline
```
Conv1D(64, k=5) → MaxPool → Conv1D(128, k=3) → MaxPool → LSTM(64) → Dense(64) → Sigmoid
```

### 2. Attention-BiLSTM (Proposed)
- Two convolutional blocks with batch norm + ReLU
- Self-attention gate (1×1 conv, sigmoid) modulating encoder output
- Bidirectional LSTM (64 units/direction, 0.2 recurrent dropout)
- Dense(128) → Dense(64, dropout=0.4) → Sigmoid

### 3. Multimodal ECG-SpO2 Fusion Model
- Independent ECG and SpO2 encoder branches
- Feature-level concatenation → Dense(128, dropout=0.3) → Sigmoid

### 4. Edge-Optimised Lightweight Model
```
Conv1D(16, k=5) → MaxPool(4) → Conv1D(32, k=3) → GlobalAvgPool → Dense(16) → Sigmoid
```
Occupies only 48 KB (Float32).

## Results

### Model Performance (MIT-BIH Test Set, n=600, 5-run average)

| Model | Accuracy | Precision | Recall | AUC | Size (KB) |
|---|---|---|---|---|---|
| CNN-LSTM | 0.945 | 0.932 | 0.918 | 0.978 | 245 |
| **Attention-BiLSTM** | **0.967** | **0.958** | **0.949** | **0.991** | 312 |
| Multimodal Fusion | 0.961 | 0.952 | 0.941 | 0.987 | 428 |
| Edge-Optimised | 0.918 | 0.901 | 0.887 | 0.952 | 48 |

### Model Size Across Quantisation Strategies (KB)

| Model | Float32 | Float16 | INT8 |
|---|---|---|---|
| CNN-LSTM | 245 | 123 | 62 |
| Attention-BiLSTM | 312 | 156 | 78 |
| Multimodal Fusion | 428 | 214 | 107 |
| Edge-Optimised | 48 | 24 | 12 |

### SpO2 Classifier Performance (Macro F1 = 0.924)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.982 | 0.991 | 0.986 | 420 |
| Warning | 0.887 | 0.854 | 0.870 | 118 |
| Critical | 0.912 | 0.923 | 0.917 | 62 |

## System Pipeline

```
IoT Sensors (ECG + SpO2) → Signal Preprocessing → [Attention-BiLSTM / Multimodal Fusion / Edge-Optimised] → Real-Time Inference Engine → IoT Dashboard + Alert System
```

### Signal Preprocessing
1. **Segmentation** — 2-second non-overlapping windows (720 samples) centered on R-peaks
2. **Quality control** — rejection of flat-line and saturated segments
3. **Normalisation** — per-window Z-score
4. **Augmentation** — additive Gaussian noise and temporal stretching for minority-class oversampling

### Heart Rate Estimation
Three-stage algorithm: Butterworth bandpass filter (5–15 Hz) → R-peak detection (`scipy.find_peaks`) → HR = 60/RR, clamped to 30–250 BPM. Mean absolute error: 2.3 BPM.

### IoT Dashboard
- `RealTimeMonitor` class with sliding-window inference (720-sample FIFO for ECG, 100-sample for SpO2)
- Alerts triggered at prediction confidence > 0.7
- MQTT integration with automatic fallback to simulation mode
- Flask server with live Plotly waveform rendering, metric cards, confidence trend, and event log (refreshed every 400 ms)

## Key Challenges Addressed

- **Compute and power constraints** — addressed via float16/INT8 quantisation and lightweight architecture design
- **Signal quality and artefact rejection** — robust preprocessing and quality-gating
- **Class imbalance** — balanced class-weight strategies and minority-class augmentation
- **Data privacy** — on-device inference and MQTT with TLS encryption
- **Model generalisation** — acknowledged limitation due to MIT-BIH's demographic scope

## Limitations

1. MIT-BIH contains limited demographic diversity (primarily middle-aged adults)
2. The SpO2 simulator does not reproduce real-world pulse oximeter noise under dynamic motion
3. Multimodal Fusion model at INT8 (107 KB) exceeds the sub-100 KB MCU target
4. All evaluation is retrospective; prospective clinical trial evidence is required for regulatory submission

## Future Work

1. Federated learning across wearable fleets for privacy-preserving model updates
2. Transformer-based pretraining (ECG-BERT) on 24-hour Holter recordings
3. Multi-task learning for simultaneous ECG classification, heart rate, and respiratory rate estimation
4. Uncertainty quantification via Monte Carlo dropout or deep ensembles
5. Prospective clinical validation targeting FDA 510(k) and EU MDR regulatory pathways

## Tech Stack

- **Deep Learning**: TensorFlow/Keras
- **Signal Processing**: SciPy, WFDB
- **Dashboard**: Flask, Plotly
- **IoT Communication**: MQTT
- **Deployment**: TensorFlow Lite (float16/INT8 quantisation) for ARM Cortex-M4 microcontrollers

## Acknowledgments

The authors thank the PhysioNet team for maintaining the MIT-BIH Arrhythmia Database, and the open-source communities behind TensorFlow, WFDB, SciPy, and Flask.

## Citation

If you use this work, please cite:

```
Vinnakota Jagadeesh, Ankamma Bollimuntha, P. Venkata Rajulu, Dega Balu Kotaiah,
"Deep Learning-Based Wearable IoT Health Monitoring System for Real-Time ECG and SpO2 Abnormality Detection"
```
