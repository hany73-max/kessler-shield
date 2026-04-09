# Orbital Collision Early-Warning System

An end-to-end Machine Learning pipeline designed to predict satellite collisions using high-dimensional orbital telemetry data. The system is engineered to provide an early-warning alert at least 48 hours prior to the Time of Closest Approach (TCA).

## Overview

Predicting orbital hazards is a problem defined by extreme class imbalance (a 1:1300 hazard-to-safe ratio). This project explores multiple architectural approaches—including Synthetic Minority Oversampling (SMOTE) and Unsupervised Anomaly Detection—before establishing a highly constrained classification engine optimized for 100% Recall.

## Pipeline Architecture

The workflow is broken down into two primary phases: Data Engineering and Model Optimization.

### 1. Data Engineering & Preprocessing
* **Temporal Filtering:** Data is strictly filtered to `time_to_tca >= 2.0` to ensure predictions are operationally viable for avoidance maneuvers.
* **Feature Engineering:** Cyclical orbital angles (azimuth, elevation) are encoded into sine/cosine pairs to preserve geometric continuity.
* **Data Leakage Prevention:** Post-event predictive estimates (e.g., `risk`, `max_risk_estimate`) are explicitly dropped.

### 2. Modeling & Optimization
Due to the extreme rarity of true collisions, standard classification paradigms fail. The pipeline documents the mathematical evaluation of three distinct approaches:
* **Synthetic Data Generation (SMOTE):** Discarded. Caused severe geometric overfitting on tree-based algorithms resulting in 0% validation recall.
* **Unsupervised Anomaly Detection:** Discarded. Evaluated via a custom ParameterGrid engine using `IsolationForest` and `OneClassSVM`. Proved that physical collisions are not mathematically "geometric outliers" in the context of standard orbit telemetry.
* **The Champion Engine (XGBoost + Threshold Tuning):** The final architecture utilizes aggressive Random Under-Sampling to balance the feature space, paired with a heavily constrained XGBoost classifier (shallow trees, low learning rate). 

## Results

By extracting the raw probabilities from the XGBoost engine and manually tuning the decision threshold via a Precision-Recall curve, the system achieved the mission-critical operational requirement:
* **Validation Recall:** 1.0 (100% of real hazards successfully flagged).
* **False Positive Reduction:** Maintained perfect recall while suppressing the false-alarm rate to operational levels.

## Repository Structure

```text
├── data/
│   ├── raw_data/                 # Original telemetry dumps
│   └── clean_data/               # Scaled, engineered, and sampled CSVs
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb # Temporal filtering and feature engineering
│   └── 02_modeling.ipynb         # Model training, grid search, and threshold tuning
├── requirements.txt
└── README.md
```

## Quick Start

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/orbital-collision-predictor.git](https://github.com/yourusername/orbital-collision-predictor.git)
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Launch the pipeline:
```bash
jupyter notebook
```

*** ### Note on Deployment
This repository represents the research and modeling phase. To deploy this engine into a real-time data stream, the scaling parameters and the champion XGBoost model (with the custom probability threshold) must be exported via `joblib` or `pickle` and wrapped in a production API.
