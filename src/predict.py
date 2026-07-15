import pandas as pd
import joblib
from pathlib import Path
import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "test_data" / "test_data.csv"
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)
best_model = joblib.load(models_dir/ "champion_xgboost.pkl")
preprocessor = joblib.load(models_dir/ "orbital_preprocessor.pkl")

def predicting_pipeline():

    df_raw = pd.read_csv(DATA_PATH)
    
    X_test_scaled, y, event_ids = preprocessor.transform_new_data(df_raw)

    probability = best_model.predict_proba(X_test_scaled)

    y_pred_best = (probability[:, 1] >= config.decision_threshold).astype(int)

    print(f"sum of the hazards {sum(y_pred_best)}")

    return y_pred_best, y, probability, X_test_scaled

if __name__ == "__main__":
    predicting_pipeline()