# 🛡️ Kessler-Shield: Orbital Collision Early-Warning System

**Kessler-Shield** is an end-to-end machine learning system that predicts high-risk orbital conjunctions (close approaches between satellites and debris) from raw Conjunction Data Message (CDM) telemetry — from raw data, to a trained model, to a live, deployed prediction service with its own interactive dashboard.

### 🚀 [Live Demo](https://kessler-shield-akmjrrdw9dsqfay8zvz33h.streamlit.app/)

The dashboard runs live against a separately deployed prediction API — upload a batch of conjunction events, or enter a single event manually and get an instant risk assessment.

---

## 🧠 The Engineering Challenge

Predicting satellite collisions is fundamentally an extreme anomaly-detection problem. The training data has a **~1:1300 class imbalance** — true collision hazards are vanishingly rare compared to safe conjunctions.

Instead of masking that imbalance with synthetic oversampling (which can distort the real physics), Kessler-Shield learns directly from the imbalanced data and uses **precision-recall trade-off analysis** to hand-tune the decision threshold, rather than defaulting to the naive 0.5 cutoff most classifiers ship with.

A second, less obvious challenge: several raw fields in this dataset (observation counts, residuals, orbit-determination arc span) are a proxy for *analyst attention* rather than physics — operators track events they already suspect are risky more closely. Those fields are deliberately excluded from training to avoid leaking human judgment into the model.

---

## 🏗️ System Architecture

The project is split into four independent layers — a training/evaluation pipeline, a reusable preprocessing engine, a prediction API, and a UI — each with a single job:

```text
kessler-shield/
│
├── app.py                  # Streamlit dashboard — batch analysis, manual input, model transparency
├── api.py                  # FastAPI service — the actual "brain," serves predictions over HTTP
├── requirements.txt        # Shared dependencies for both the API and the dashboard
│
├── data/                   # Raw, training, and test telemetry (gitignored)
├── models/                 # Serialized .pkl artifacts (trained model + fitted preprocessor)
├── notebook/               # Initial EDA, math derivations, and model prototyping
│
└── src/
    ├── config.py           # Single source of truth for hyperparameters & decision threshold
    ├── process.py          # OrbitalPreprocessor — cyclical angle encoding, imbalance-aware cleaning
    ├── training.py          # Fits the preprocessor + model, serializes both to models/
    ├── predict.py           # Batch inference over a CSV, using the shared preprocessor
    ├── evaluation.py        # Precision-recall curves, confusion matrix, threshold diagnostics
    └── main.py              # CLI entry point: `python src/main.py [train|predict|evaluate|all]`
```

**Why the API and the UI are separate services, not one script:** `api.py` owns the model and does the actual prediction work; `app.py` never touches `joblib`, the model, or the preprocessor directly — it only sends raw event data to the API over HTTP and displays whatever comes back. That split means the dashboard could be swapped for a different frontend entirely without touching the model-serving code at all.

---

## ✨ Features

- **Batch analysis** — upload a CDM-format CSV, get every conjunction event scored, risk-banded, and downloadable as a results file
- **Manual single-event input** — guided form for the raw fields the model actually uses, with inline explanations of what each one means and why it matters (e.g. why angles get cyclically encoded)
- **Risk gauge** — a visual, threshold-relative risk indicator rather than a bare probability number, since real risk probabilities here are tiny (threshold ≈ 0.0055) and a flat 0–1 scale would be meaningless
- **Model transparency tab** — feature importances and model configuration, so predictions aren't a black box
- **Adjustable decision threshold** — live-updates all views, demonstrating the precision/recall trade-off directly rather than just describing it

---

## 🛠️ Running It Locally

```bash
git clone https://github.com/hany73-max/kessler-shield.git
cd kessler-shield
pip install -r requirements.txt
```

**Train and evaluate the model:**
```bash
python src/main.py train      # fits the preprocessor + model, saves both to models/
python src/main.py evaluate   # precision-recall curve, confusion matrix, threshold diagnostics
python src/main.py all        # both, in order
```

**Run the API and the dashboard** (in two separate terminals):
```bash
uvicorn api:app --reload
streamlit run app.py
```

---

## 🚢 Deployment

- **API** (`api.py`) — deployed on Railway
- **Dashboard** (`app.py`) — deployed on Streamlit Community Cloud, configured via an `API_URL` secret pointing at the live API

Splitting these across two hosts mirrors how this would actually be deployed in production — a model-serving backend and a frontend as genuinely separate, independently scalable services.

---

## 📌 Known Limitations & Honest Notes

- The `t_cd_area_over_mass` / `t_cr_area_over_mass` / `c_cd_area_over_mass` / `c_cr_area_over_mass` fields (drag and reflectivity area-to-mass ratios) are currently dropped by the same filter that removes covariance cross-terms, due to a naming collision (`process.py`'s cross-term filter matches on a `t_c`/`c_c` prefix, which these fields also happen to start with). The current model was trained without them. Fixing the filter and retraining is a known next step.
- `scikit-learn` is pinned to an exact version (`==1.7.2`) rather than a range — this isn't a style preference, it's a hard requirement, since the pickled preprocessor breaks under different minor versions of `SimpleImputer`/`StandardScaler`.
- Railway's free tier may spin down after inactivity — the first prediction after idle time can take 20–30 seconds while the service wakes up.

---

## Built With

- **Python 3**
- **XGBoost** — gradient boosting classifier
- **scikit-learn** — preprocessing, imputation, scaling
- **FastAPI + Uvicorn** — model-serving API
- **Streamlit + Plotly** — interactive dashboard
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — diagnostic visualization
