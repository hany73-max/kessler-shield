# 🛡️ Kessler-Shield: Orbital Collision Early-Warning System

**Kessler-Shield** is an end-to-end Machine Learning pipeline designed to predict high-risk orbital collisions between satellites and space debris using raw radar telemetry. 

Built entirely in Python, this system translates mathematical theory into a hardened, object-oriented production engine capable of processing live radar feeds and issuing collision warnings in milliseconds.

## 🧠 The Engineering Challenge
Predicting satellite collisions is fundamentally an anomaly detection problem. The orbital telemetry dataset features a massive **1:1300 class imbalance** (safe objects vastly outnumber collision hazards). 

Instead of relying on synthetic data sampling (which can distort real-world physics), this engine was built to learn directly from the imbalanced noise. By utilizing a custom **XGBClassifier** and mathematically adjusting the decision threshold using Precision-Recall trade-off analysis, Kessler-Shield successfully identifies hazards while maintaining a strictly controlled false-positive rate.

## 🏗️ System Architecture
The project is decoupled into isolated, production-ready modules:
```text
kessler-shield/
│
├── data/                   # Raw, training, and test telemetry (ignored in Git)
├── models/                 # Serialized .pkl artifacts (Model + Preprocessor Memory)
├── notebook/               # Initial EDA, Math proofs, and Model prototyping
└── src/                    
    ├── config.py           # Centralized configuration and model hyperparameters
    ├── process.py          # OOP Data Engine: Handles cyclical angle math & state memory
    ├── training.py         # Factory Script: Trains the model and serializes artifacts
    ├── predict.py          # Live Inference Engine: Scans new telemetry for hazards
    └── evaluate.py         # Diagnostics: Generates PR Curves and Feature Importance
```

How to Run the Diagnostics
To prove the model's effectiveness on unseen test data, you can run the evaluation dashboard. This will output the final classification metrics, generate a Confusion Matrix, and plot the system's Precision-Recall Curve.

Clone the repository.

Install the required dependencies:

```Bash
pip install -r requirements.txt
```
Run the evaluation script from the root directory:

```Bash
python src/evaluate.py
```
Built With

- Python 3

- XGBoost (Gradient Boosting Engine)

- Scikit-Learn (Preprocessing, Metrics, Thresholding)

- Pandas & NumPy (Data Manipulation & Math)

- Matplotlib & Seaborn (Diagnostic Visualization)