# 🛰️ kessler-shield

## 🚀 Overview
Low Earth Orbit (LEO) is facing a critical traffic crisis. With the exponential launch of mega-constellations and the accumulation of space debris, tracking radars issue hundreds of Conjunction Data Messages (CDMs) weekly. Currently, human operators must manually evaluate these warnings to determine if a satellite needs to execute a costly avoidance maneuver, or if the alert is a false alarm due to radar uncertainty (Probability Dilution).

This project builds an automated, machine-learning-driven **Orbital Collision Avoidance Engine** designed to process raw astrophysical telemetry and autonomously classify collision hazards days in advance.

## 📊 The Dataset
This engine is trained on real-world, anonymized Conjunction Data Messages from the **European Space Agency (ESA) Kelvins Collision Avoidance Challenge**. 

**The Challenge: Extreme Class Imbalance**
Space is vast. In the raw orbital data, roughly 99.9% of alerts resolve as safe passes as the uncertainty bubble shrinks. In the initial training matrix, there were **11 verified high-risk hazards against 11,931 safe passes**. A standard machine learning model would achieve 99.9% accuracy simply by predicting "Safe" every time—which would be catastrophic in orbit. 

## ⚙️ Architecture & Pipeline

### Phase 1: Physics Feature Engineering (Complete)
To prevent data leakage and train the engine for real-world deployment, the dataset was restructured using a "Time-Machine" architecture:
* **Target Extraction:** Isolated the final `risk` assessment at the Time of Closest Approach (TCA) as the absolute ground truth (1 = Hazard, 0 = Safe).
* **Feature Shift:** Extracted the radar features (Miss Distance, Relative Speed, Covariance Matrices) from CDMs issued **2 days prior** to TCA. The model must learn to predict the final truth using this noisy, 48-hour-out data.
* **Cyclical Encoding:** Converted circular spatial angles (Azimuth, Elevation) into trigonometric Sine/Cosine components to preserve physical geometry.
* **Arena Balancing:** Deployed **SMOTE (Synthetic Minority Over-sampling Technique)** to mathematically synthesize the physics of the 11 minority hazards, perfectly balancing the training matrix to prevent the Zero-Variance Trap.

### Phase 2: The Classification Arena (In Progress)
The pre-processed physics matrix will be fed into a comparative evaluation arena using Scikit-Learn to benchmark the performance of:
* Logistic Regression (Baseline Probability)
* Random Forest Classifier (Ensemble/Non-linear)
* Support Vector Machines (SVM)
* Extreme Gradient Boosting (XGBoost)

### Phase 3: Evaluation Metrics (Upcoming)
Due to the extreme class imbalance, standard Accuracy is discarded. Models will be evaluated purely on their ability to prevent orbital collisions and save fuel, utilizing:
* Confusion Matrices (False Positives vs. False Negatives)
* Precision-Recall Area Under Curve (PR-AUC)
* F1-Score

## 🛠️ Tech Stack
* **Language:** Python
* **Data Engineering:** Pandas, NumPy, Imbalanced-Learn (SMOTE)
* **Machine Learning:** Scikit-Learn, XGBoost
* **Domain:** Space Situational Awareness (SSA), Orbital Mechanics

---
*Built by a Machine Learning Engineer & Space Tech Specialist to solve real-world aerospace challenges.*