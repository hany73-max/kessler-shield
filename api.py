from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
import joblib
import sys

PROJECT_ROOT = Path(__file__).resolve().parent  
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from process import OrbitalPreprocessor  
import config

models_dir = PROJECT_ROOT / "models"
best_model = joblib.load(models_dir / "champion_xgboost.pkl")
preprocessor = joblib.load(models_dir / "orbital_preprocessor.pkl")

app = FastAPI()

class Conjunction_Event(BaseModel):
    event_id : str 
    time_to_tca: float
    miss_distance: float
    relative_speed : float
    t_rcs_estimate : float
    c_rcs_estimate : float
    t_sedr : float
    c_sedr : float
    t_j2k_sma : float
    c_j2k_sma : float
    t_j2k_ecc : float
    c_j2k_ecc : float
    t_j2k_inc : float
    c_j2k_inc : float
    t_span : float
    c_span : float
    t_h_apo : float
    t_h_per : float
    c_h_apo : float
    c_h_per : float
    geocentric_latitude : float
    azimuth : float
    elevation : float
    mahalanobis_distance : float
    t_position_covariance_det : float
    c_position_covariance_det : float
    t_sigma_r : float
    c_sigma_r : float
    t_sigma_t : float
    c_sigma_t : float
    t_sigma_n : float
    c_sigma_n : float
    t_sigma_rdot : float
    c_sigma_rdot : float
    t_sigma_tdot : float
    c_sigma_tdot : float
    t_sigma_ndot : float
    c_sigma_ndot : float
    F10 : float
    F3M : float
    SSN : float
    AP : float
    c_object_type : str

@app.post("/predict")
def predict(event: Conjunction_Event):
    try:
        df = pd.DataFrame([event.dict()])
        X_test_scaled, _, _ = preprocessor.transform_new_data(df)
        probability = best_model.predict_proba(X_test_scaled)
        y_pred_best = (probability[:, 1] >= config.decision_threshold).astype(int)

        return {
            "flagged_high_risk": bool(y_pred_best[0]),
            "risk_probability": float(probability[0][1]),
        }
    except Exception as e:
        # Temporary: surfaces the real error in the response instead of a
        # generic 500, so it shows up in Streamlit instead of only Railway's
        # logs. Fine for now while debugging — worth tightening later so it
        # doesn't leak internals once this is client-facing.
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

