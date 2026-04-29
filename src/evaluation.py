from predict import predicting_pipeline
import matplotlib.pyplot as plt 
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, confusion_matrix, accuracy_score, f1_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)

def _feature_importance(X_test_scaled):
    best_model = joblib.load(models_dir / "champion_xgboost.pkl")
    importances = best_model.feature_importances_

    feature_names = X_test_scaled.columns 

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color="#e63946")
    plt.title('Champion Early-Warning System: Feature Importance')
    plt.xlabel('Relative Importance (Information Gain)')
    plt.ylabel('Telemetry Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def _precision_recall_curve(y, y_pred_best):
    precisions, recalls, thresholds = precision_recall_curve(y, y_pred_best)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='#e63946', linewidth=2)
    plt.title('Precision-Recall Curve (Test Data)')
    plt.xlabel('Recall (Catching all Collisions)')
    plt.ylabel('Precision (Avoiding False Alarms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def evaluation_pipeline():
    y_pred_best, y, probability, X_test_scaled = predicting_pipeline()

    best_model_results = {
        "Accuracy": accuracy_score(y, y_pred_best),
        "Precision": precision_score(y, y_pred_best, zero_division=0),
        "Recall": recall_score(y, y_pred_best, zero_division=0),
        "F1-Score": f1_score(y, y_pred_best, zero_division=0)
    }
    
    best_model_results_df = pd.DataFrame([best_model_results])
    print(best_model_results_df)

    Confusion_matrix = confusion_matrix(y, y_pred_best)
    print(Confusion_matrix)

    _feature_importance(X_test_scaled)
    _precision_recall_curve(y, probability[:, 1])

if __name__ == "__main__":
    print("Initializing Kessler-Shield Evaluation Pipeline...")
    evaluation_pipeline()