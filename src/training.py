from config import best_model_params
from process import OrbitalPreprocessor

from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train_data" / "train_data.csv"
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)

def training_pipeline():
    # loading the data
    df_raw = pd.read_csv(DATA_PATH)

    # initiating the class
    preprocessor = OrbitalPreprocessor()
    # Run the fit_transform_train method
    X_train_scaled, y = preprocessor.fit_transform_train(df_raw)

    # applying the best model
    best_model= XGBClassifier(**best_model_params)
    best_model.fit(X_train_scaled, y)

    joblib.dump(best_model, models_dir / 'champion_xgboost.pkl')
    joblib.dump(preprocessor, models_dir / 'orbital_preprocessor.pkl')
    print("pipeline success")

if __name__ == "__main__":
    training_pipeline()