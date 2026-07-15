"""
Kessler Shield — Satellite Collision Risk Dashboard (v2)
-----------------------------------------------------------
Streamlit front-end for the existing kessler-shield pipeline
(src/process.py + src/predict.py + models/champion_xgboost.pkl).

Drop this file in the REPO ROOT (same level as `src/` and `models/`) and run:

    pip install streamlit pandas numpy joblib xgboost scikit-learn plotly

    streamlit run app.py

This version adds:
  - A manual single-event input tab (guided fields for what the model
    actually uses, plus a free-form section for any other raw CDM columns)
  - A risk gauge for single predictions
  - Relative risk banding for batch results (Low / Watch / High / Critical) —
    fixed 0-1 bands don't make sense here since real risk probabilities are
    tiny (the tuned threshold is ~0.0055), so bands are threshold-relative
  - A model transparency tab showing feature importances and config
"""

import sys
from pathlib import Path

import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Paths & imports ──────────────────────────────────────────────────
# training.py / predict.py both do `from process import OrbitalPreprocessor`
# (a flat import, run with src/ on the path) — replicated here so joblib
# can find the class to unpickle.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from process import OrbitalPreprocessor  # noqa: F401  (needed for unpickling)
except ImportError as e:
    st.error(
        "Couldn't import `OrbitalPreprocessor` from src/process.py. "
        "Make sure this app.py sits in the repo root, next to `src/`.\n\n"
        f"Details: {e}"
    )
    st.stop()

try:
    from config import decision_threshold as DEFAULT_THRESHOLD
except ImportError:
    DEFAULT_THRESHOLD = 0.0055  # fallback, matches src/config.py


# ── Page setup & light styling ────────────────────────────────────────
st.set_page_config(
    page_title="Kessler Shield — Collision Risk Predictor",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .risk-badge {
        display:inline-block; padding: 4px 12px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem;
    }
    .risk-low      { background:#e6f4ea; color:#1e7e34; }
    .risk-watch    { background:#fff8e1; color:#9a6700; }
    .risk-high     { background:#fde8e8; color:#c0392b; }
    .risk-critical { background:#c0392b; color:white; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛰️ Kessler Shield")
st.caption(
    "Predicts high-risk satellite conjunction events from Conjunction Data "
    "Message (CDM) data, using a threshold-tuned XGBoost classifier trained "
    "on a heavily imbalanced (~1-in-1300) collision-risk dataset."
)


# ── Load model + preprocessor once ───────────────────────────────────
@st.cache_resource(show_spinner="Loading model and preprocessor…")
def load_artifacts():
    model = joblib.load(MODELS_DIR / "champion_xgboost.pkl")
    preprocessor = joblib.load(MODELS_DIR / "orbital_preprocessor.pkl")
    return model, preprocessor


try:
    model, preprocessor = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Couldn't find the trained model files. Expected:\n\n"
        f"- `{MODELS_DIR / 'champion_xgboost.pkl'}`\n"
        f"- `{MODELS_DIR / 'orbital_preprocessor.pkl'}`\n\n"
        f"Details: {e}"
    )
    st.stop()


# ── Shared helpers ────────────────────────────────────────────────────
def risk_band(probability: float, threshold: float) -> tuple[str, str]:
    """Bands are threshold-relative, not fixed 0-1 splits — real probabilities
    here are tiny, so a fixed 0.3/0.6/0.9 split would never fire."""
    if probability >= threshold * 5:
        return "Critical", "risk-critical"
    if probability >= threshold:
        return "High", "risk-high"
    if probability >= threshold * 0.5:
        return "Watch", "risk-watch"
    return "Low", "risk-low"


def make_gauge(probability: float, threshold: float) -> go.Figure:
    max_range = max(threshold * 8, probability * 1.5, 0.01)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"valueformat": ".5f"},
            title={"text": "Predicted risk probability"},
            gauge={
                "axis": {"range": [0, max_range]},
                "bar": {"color": "#2c3e50"},
                "steps": [
                    {"range": [0, threshold * 0.5], "color": "#e6f4ea"},
                    {"range": [threshold * 0.5, threshold], "color": "#fff8e1"},
                    {"range": [threshold, threshold * 5], "color": "#fde8e8"},
                    {"range": [threshold * 5, max_range], "color": "#c0392b"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
    return fig


def run_pipeline(df_raw: pd.DataFrame):
    """Batch path — stays local, doesn't go through the API.
    NOTE: transform_new_data() predicts once per unique event_id (the
    pipeline deduplicates multiple CDM messages per event down to the
    latest one ≥2 days out) — so the number of predictions almost never
    matches the number of raw uploaded rows. event_ids tells us which
    events actually got scored, so results can be built correctly instead
    of assuming a 1:1 row match with the upload."""
    X_scaled, y_true, event_ids = preprocessor.transform_new_data(df_raw)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    return probabilities, y_true, event_ids


import os

API_URL = os.environ.get("API_URL", "kessler-shield-production.up.railway.app")


def predict_via_api(row: dict) -> float:
    """Single-event path — this is the one that actually calls api.py."""
    response = requests.post(f"{API_URL}/predict", json=row)
    response.raise_for_status()
    result = response.json()
    return result["risk_probability"]


# ── Sidebar controls (shared across tabs) ────────────────────────────
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0005,
    max_value=0.05,
    value=float(DEFAULT_THRESHOLD),
    step=0.0005,
    format="%.4f",
    help=(
        "Probability above which an event is flagged high-risk. Lower = "
        "catches more true collisions but more false alarms. The project's "
        f"tuned default is {DEFAULT_THRESHOLD:.4f}, chosen from "
        "precision-recall analysis on the imbalanced training data."
    ),
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** XGBoost (max_depth=3, lr=0.01, 200 estimators)\n\n"
    "Predictions are only as good as the input CDM data — this tool doesn't "
    "fetch live tracking data itself."
)

tab_batch, tab_manual, tab_model = st.tabs(
    ["📁 Batch Analysis", "🎯 Single Event (Manual Input)", "📊 Model Info"]
)

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Batch analysis (CSV upload)
# ══════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Upload conjunction data")
    uploaded_file = st.file_uploader(
        "CSV in the same raw format as data/test_data/test_data.csv",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is None:
        st.info("👆 Upload a CDM-format CSV to analyze a batch of events.")
    else:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that file as a CSV: {e}")
            df_raw = None

        if df_raw is not None:
            st.write(f"Loaded **{len(df_raw):,}** rows from `{uploaded_file.name}`.")
            if st.button("🚀 Run risk analysis", type="primary"):
                try:
                    with st.spinner("Cleaning data, engineering features, scoring…"):
                        probabilities, y_true, event_ids = run_pipeline(df_raw)
                except Exception as e:
                    st.error(
                        "Preprocessing failed — the file is likely missing "
                        "expected columns or has a different schema than the "
                        f"training data.\n\nDetails: {e}"
                    )
                    st.stop()

                predictions = (probabilities >= threshold).astype(int)
                bands = [risk_band(p, threshold)[0] for p in probabilities]

                # One row per unique event actually scored — NOT one row per
                # raw uploaded row. The pipeline deduplicates multiple CDM
                # messages per event down to one before scoring, so this
                # will usually be shorter than the uploaded file, and that's
                # expected, not a bug.
                st.caption(
                    f"Scored {len(probabilities):,} unique events out of "
                    f"{len(df_raw):,} raw rows uploaded (multiple CDM messages "
                    "per event get collapsed to the latest one ≥2 days out)."
                )
                results = pd.DataFrame({"event_id": event_ids.values})
                results["risk_probability"] = probabilities
                results["risk_band"] = bands
                results["flagged_high_risk"] = predictions.astype(bool)
                if y_true is not None:
                    results["actual_risk_label"] = y_true.values
                results_sorted = results.sort_values("risk_probability", ascending=False)

                n_total = len(results)
                n_flagged = int(predictions.sum())
                band_counts = pd.Series(bands).value_counts()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Events assessed", f"{n_total:,}")
                c2.metric("Flagged high-risk", f"{n_flagged:,}")
                c3.metric("Critical", f"{band_counts.get('Critical', 0):,}")
                c4.metric("Flag rate", f"{n_flagged / n_total * 100:.2f}%")

                if y_true is not None and (y_true == 1).sum() > 0:
                    tp = int(((predictions == 1) & (y_true == 1)).sum())
                    actual_pos = int((y_true == 1).sum())
                    st.caption(
                        f"Ground-truth labels were present: caught {tp}/{actual_pos} "
                        f"actual high-risk events at this threshold."
                    )

                st.markdown("---")
                col_a, col_b = st.columns([2, 1])

                with col_a:
                    st.subheader("Results")
                    display_cols = ["risk_band", "risk_probability"] + [
                        c for c in results_sorted.columns
                        if c not in ("risk_band", "risk_probability")
                    ]

                    def _highlight(row):
                        colors = {
                            "Low": "#e6f4ea", "Watch": "#fff8e1",
                            "High": "#fde8e8", "Critical": "#f5b7b1",
                        }
                        bg = colors.get(row["risk_band"], "")
                        style = f"background-color: {bg}; color: #1a1a1a;" if bg else ""
                        return [style] * len(row)

                    st.dataframe(
                        results_sorted[display_cols]
                        .style.apply(_highlight, axis=1)
                        .format({"risk_probability": "{:.5f}"}),
                        use_container_width=True,
                        height=420,
                    )

                with col_b:
                    st.subheader("Risk breakdown")
                    fig_pie = px.pie(
                        names=band_counts.index, values=band_counts.values,
                        color=band_counts.index,
                        color_discrete_map={
                            "Low": "#a8d5ba", "Watch": "#ffe08a",
                            "High": "#f5a3a3", "Critical": "#c0392b",
                        },
                        hole=0.5,
                    )
                    fig_pie.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("Probability distribution")
                fig_hist = px.histogram(x=probabilities, nbins=50)
                fig_hist.add_vline(
                    x=threshold, line_dash="dash", line_color="#c0392b",
                    annotation_text="threshold",
                )
                fig_hist.update_layout(
                    xaxis_title="Predicted risk probability",
                    yaxis_title="Number of events",
                    height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.download_button(
                    "⬇️ Download full results as CSV",
                    data=results_sorted.to_csv(index=False).encode("utf-8"),
                    file_name="kessler_shield_results.csv",
                    mime="text/csv",
                )

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Manual single-event input
# ══════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Assess a single conjunction event")
    st.caption(
        "Fields below match the real raw CDM schema this pipeline was "
        "trained on — verified by actually running `_data_clean()` against "
        "the real column list, not guessed. `t_` = target object, "
        "`c_` = chaser object."
    )

    with st.form("manual_event_form"):
        st.markdown("**Event identity & timing**")
        id_col1, id_col2, id_col3 = st.columns(3)
        with id_col1:
            event_id = st.text_input("Event ID", value="manual-event-001")
        with id_col2:
            time_to_tca = st.number_input(
                "Time to closest approach (days)",
                min_value=2.0, max_value=30.0, value=5.0, step=0.5,
                help="⚠️ Must be ≥ 2.0 — events closer than this get filtered out entirely, matching training.",
            )
        with id_col3:
            c_object_type = st.selectbox(
                "Chaser object type",
                ["DEBRIS", "PAYLOAD", "ROCKET BODY", "TBA", "UNKNOWN"],
                help="Verified against the model's real trained categories (preprocessor.expected_columns).",
            )

        st.markdown("**Miss geometry**")
        g1, g2, g3 = st.columns(3)
        with g1:
            miss_distance = st.number_input("Miss distance (m)", min_value=0.0, value=500.0)
        with g2:
            relative_speed = st.number_input("Relative speed (m/s)", min_value=0.0, value=10000.0)
        with g3:
            mahalanobis_distance = st.number_input("Mahalanobis distance", min_value=0.0, value=3.0)

        st.markdown("**Orbital geometry (degrees)** — cyclically encoded, so wraparound is handled correctly")
        a1, a2, a3, a4, a5 = st.columns(5)
        with a1:
            geocentric_latitude = st.number_input("Geocentric latitude", -90.0, 90.0, 0.0)
        with a2:
            azimuth = st.number_input("Azimuth", 0.0, 360.0, 180.0)
        with a3:
            elevation = st.number_input("Elevation", -90.0, 90.0, 45.0)
        with a4:
            t_j2k_inc = st.number_input("Target inclination", 0.0, 180.0, 51.6)
        with a5:
            c_j2k_inc = st.number_input("Chaser inclination", 0.0, 180.0, 51.6)

        with st.expander("🛰️ Target object detail", expanded=False):
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                t_rcs_estimate = st.number_input("Target RCS estimate (m²)", min_value=0.0, value=1.0)
                t_j2k_sma = st.number_input("Target semi-major axis (km)", min_value=0.0, value=7000.0)
            with t2:
                t_sedr = st.number_input("Target SEDR", value=0.0)
                t_j2k_ecc = st.number_input("Target eccentricity", 0.0, 1.0, 0.001)
            with t3:
                t_span = st.number_input("Target OD arc span", min_value=0.0, value=5.0)
                t_h_apo = st.number_input("Target apogee height (km)", min_value=0.0, value=420.0)
            with t4:
                t_h_per = st.number_input("Target perigee height (km)", min_value=0.0, value=410.0)
                t_position_covariance_det = st.number_input("Target position covariance det.", min_value=0.0, value=1.0, format="%.6f")

        with st.expander("🛰️ Chaser object detail", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                c_rcs_estimate = st.number_input("Chaser RCS estimate (m²)", min_value=0.0, value=0.1)
                c_j2k_sma = st.number_input("Chaser semi-major axis (km)", min_value=0.0, value=7000.0)
            with c2:
                c_sedr = st.number_input("Chaser SEDR", value=0.0)
                c_j2k_ecc = st.number_input("Chaser eccentricity", 0.0, 1.0, 0.001)
            with c3:
                c_span = st.number_input("Chaser OD arc span", min_value=0.0, value=5.0)
                c_h_apo = st.number_input("Chaser apogee height (km)", min_value=0.0, value=420.0)
            with c4:
                c_h_per = st.number_input("Chaser perigee height (km)", min_value=0.0, value=410.0)
                c_position_covariance_det = st.number_input("Chaser position covariance det.", min_value=0.0, value=1.0, format="%.6f")

        with st.expander("📐 Uncertainty (sigma) terms", expanded=False):
            st.caption("Diagonal covariance terms, radial/transverse/normal position and velocity.")
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("*Target*")
                t_sigma_r = st.number_input("t_sigma_r", min_value=0.0, value=1.0)
                t_sigma_t = st.number_input("t_sigma_t", min_value=0.0, value=1.0)
                t_sigma_n = st.number_input("t_sigma_n", min_value=0.0, value=1.0)
                t_sigma_rdot = st.number_input("t_sigma_rdot", min_value=0.0, value=0.01)
                t_sigma_tdot = st.number_input("t_sigma_tdot", min_value=0.0, value=0.01)
                t_sigma_ndot = st.number_input("t_sigma_ndot", min_value=0.0, value=0.01)
            with s2:
                st.markdown("*Chaser*")
                c_sigma_r = st.number_input("c_sigma_r", min_value=0.0, value=1.0)
                c_sigma_t = st.number_input("c_sigma_t", min_value=0.0, value=1.0)
                c_sigma_n = st.number_input("c_sigma_n", min_value=0.0, value=1.0)
                c_sigma_rdot = st.number_input("c_sigma_rdot", min_value=0.0, value=0.01)
                c_sigma_tdot = st.number_input("c_sigma_tdot", min_value=0.0, value=0.01)
                c_sigma_ndot = st.number_input("c_sigma_ndot", min_value=0.0, value=0.01)

        with st.expander("☀️ Space weather", expanded=False):
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                F10 = st.number_input("F10 (solar flux)", min_value=0.0, value=150.0)
            with w2:
                F3M = st.number_input("F3M", min_value=0.0, value=150.0)
            with w3:
                SSN = st.number_input("SSN (sunspot number)", min_value=0.0, value=50.0)
            with w4:
                AP = st.number_input("AP (geomagnetic index)", min_value=0.0, value=10.0)

        submitted = st.form_submit_button("🔍 Predict this event", type="primary")

    if submitted:
        row = {
            "event_id": event_id, "time_to_tca": time_to_tca, "c_object_type": c_object_type,
            "miss_distance": miss_distance, "relative_speed": relative_speed,
            "mahalanobis_distance": mahalanobis_distance,
            "geocentric_latitude": geocentric_latitude, "azimuth": azimuth, "elevation": elevation,
            "t_j2k_inc": t_j2k_inc, "c_j2k_inc": c_j2k_inc,
            "t_rcs_estimate": t_rcs_estimate, "t_j2k_sma": t_j2k_sma, "t_sedr": t_sedr,
            "t_j2k_ecc": t_j2k_ecc, "t_span": t_span, "t_h_apo": t_h_apo, "t_h_per": t_h_per,
            "t_position_covariance_det": t_position_covariance_det,
            "c_rcs_estimate": c_rcs_estimate, "c_j2k_sma": c_j2k_sma, "c_sedr": c_sedr,
            "c_j2k_ecc": c_j2k_ecc, "c_span": c_span, "c_h_apo": c_h_apo, "c_h_per": c_h_per,
            "c_position_covariance_det": c_position_covariance_det,
            "t_sigma_r": t_sigma_r, "t_sigma_t": t_sigma_t, "t_sigma_n": t_sigma_n,
            "t_sigma_rdot": t_sigma_rdot, "t_sigma_tdot": t_sigma_tdot, "t_sigma_ndot": t_sigma_ndot,
            "c_sigma_r": c_sigma_r, "c_sigma_t": c_sigma_t, "c_sigma_n": c_sigma_n,
            "c_sigma_rdot": c_sigma_rdot, "c_sigma_tdot": c_sigma_tdot, "c_sigma_ndot": c_sigma_ndot,
            "F10": F10, "F3M": F3M, "SSN": SSN, "AP": AP,
        }

        try:
            probability = predict_via_api(row)
        except requests.exceptions.ConnectionError:
            st.error(
                "Couldn't reach the API. Make sure it's running:\n\n"
                "`uvicorn api:app --reload`"
            )
            st.stop()
        except Exception as e:
            st.error(
                "Prediction failed — this usually means a required raw "
                f"column is missing or malformed.\n\nDetails: {e}"
            )
            st.stop()

        band, css_class = risk_band(probability, threshold)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.plotly_chart(make_gauge(probability, threshold), use_container_width=True)
        with res_col2:
            st.markdown(f"### Risk level: <span class='risk-badge {css_class}'>{band}</span>", unsafe_allow_html=True)
            st.metric("Predicted probability", f"{probability:.5f}")
            st.metric("Decision threshold", f"{threshold:.5f}")
            st.write(
                "Flagged as **high-risk**" if probability >= threshold
                else "Not flagged at this threshold"
            )

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Model transparency
# ══════════════════════════════════════════════════════════════════════
with tab_model:
    st.subheader("What's driving predictions")

    try:
        importances = model.feature_importances_
        feature_names = preprocessor.expected_columns
        if feature_names and len(feature_names) == len(importances):
            imp_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False).head(15)
            fig_imp = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
            )
            fig_imp.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=450, margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature names and importances don't line up — showing raw importances only.")
            st.bar_chart(importances)
    except Exception as e:
        st.warning(f"Couldn't compute feature importances: {e}")

    st.markdown("---")
    st.subheader("Model configuration")
    st.json(
        {
            "model_type": "XGBoost Classifier",
            "max_depth": 3,
            "learning_rate": 0.01,
            "n_estimators": 200,
            "decision_threshold_default": DEFAULT_THRESHOLD,
            "training_class_balance": "~1 true collision per 1,300 events",
        }
    )
    st.caption(
        "⚠️ Heads up: `src/predict.py` currently hardcodes threshold=0.005, "
        f"while `src/config.py` defines decision_threshold={DEFAULT_THRESHOLD}. "
        "Worth pointing predict.py at the config value so they can't drift apart."
    )
