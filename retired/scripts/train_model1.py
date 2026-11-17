"""
Training Script for Model 1: Task Effort Regressor (XGBoost Quantile)
Predicts P25 (low_hours), P50 (median_hours), and P75 (high_hours) + Confidence.
Evaluates MAE, RMSE, R2 Score, and Prediction Interval Coverage (PICP).
Exports: models/artifacts/model_est.pkl
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

TASK_TYPES = ["feature", "bug", "infra", "refactor", "test", "docs"]


class EffortEstimatorModel:
    """Encapsulates vectorizer and XGBoost quantile regressors for inference."""
    def __init__(self, tfidf, model_low, model_mid, model_high, feature_cols):
        self.tfidf = tfidf
        self.model_low = model_low
        self.model_mid = model_mid
        self.model_high = model_high
        self.feature_cols = feature_cols

    def predict(self, title: str, task_type: str, story_points: float,
                dependency_count: int, historical_median: float, assignee_ratio: float) -> dict:
        text_vec = self.tfidf.transform([title]).toarray()
        type_vec = np.zeros(len(TASK_TYPES))
        if task_type in TASK_TYPES:
            type_vec[TASK_TYPES.index(task_type)] = 1.0
        num_vec = np.array([[story_points, dependency_count, historical_median, assignee_ratio]])
        X = np.hstack([num_vec, type_vec.reshape(1, -1), text_vec])

        low = float(self.model_low.predict(X)[0])
        mid = float(self.model_mid.predict(X)[0])
        high = float(self.model_high.predict(X)[0])

        low = max(0.5, round(min(low, mid), 1))
        high = max(low + 0.5, round(max(high, mid), 1))
        mid = round(min(max(mid, low), high), 1)

        rel_spread = (high - low) / max(mid, 1.0)
        confidence = float(np.clip(1.0 - (rel_spread * 0.35), 0.55, 0.95))

        return {
            "low_hours": low,
            "mid_hours": mid,
            "high_hours": high,
            "confidence": round(confidence, 2)
        }

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def preprocess(df, tfidf=None, is_train=True):
    if is_train:
        tfidf = TfidfVectorizer(max_features=32, stop_words="english")
        text_matrix = tfidf.fit_transform(df["title"]).toarray()
    else:
        text_matrix = tfidf.transform(df["title"]).toarray()

    type_matrices = []
    for t in TASK_TYPES:
        type_matrices.append((df["task_type"] == t).astype(float).values.reshape(-1, 1))
    type_matrix = np.hstack(type_matrices)

    num_matrix = df[["story_points", "dependency_count", "historical_median_hours", "assignee_user_ratio"]].values
    X = np.hstack([num_matrix, type_matrix, text_matrix])
    y = df["actual_hours"].values
    return X, y, tfidf


def train():
    print("Loading Model 1 datasets...")
    train_path = os.path.join(DATA_DIR, "model1_effort_train.csv")
    test_path = os.path.join(DATA_DIR, "model1_effort_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train, tfidf = preprocess(train_df, is_train=True)
    X_test, y_test, _ = preprocess(test_df, tfidf=tfidf, is_train=False)

    print("Training P25 Quantile Model (Low Hours)...")
    model_low = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.25,
        tree_method="hist",
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42
    )
    model_low.fit(X_train, y_train)

    print("Training P50 Quantile Model (Median Hours)...")
    model_mid = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.50,
        tree_method="hist",
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42
    )
    model_mid.fit(X_train, y_train)

    print("Training P75 Quantile Model (High Hours)...")
    model_high = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.75,
        tree_method="hist",
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42
    )
    model_high.fit(X_train, y_train)

    # Evaluate
    pred_low = model_low.predict(X_test)
    pred_mid = model_mid.predict(X_test)
    pred_high = model_high.predict(X_test)

    mae = mean_absolute_error(y_test, pred_mid)
    rmse = np.sqrt(mean_squared_error(y_test, pred_mid))
    r2 = r2_score(y_test, pred_mid)

    within_interval = (y_test >= pred_low) & (y_test <= pred_high)
    picp = float(np.mean(within_interval) * 100)

    metrics = {
        "model": "XGBoost Quantile Regressor",
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2_Score": round(float(r2), 4),
        "P25_P75_Coverage_Pct": round(picp, 1),
        "train_samples": len(train_df),
        "test_samples": len(test_df)
    }

    print("\n Model 1 Evaluation Metrics (Test Set):")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    bundle = EffortEstimatorModel(
        tfidf=tfidf,
        model_low=model_low,
        model_mid=model_mid,
        model_high=model_high,
        feature_cols=["story_points", "dependency_count", "historical_median_hours", "assignee_user_ratio"]
    )

    out_file = os.path.join(ARTIFACTS_DIR, "model_est.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(bundle, f)

    with open(os.path.join(ARTIFACTS_DIR, "model1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n Successfully exported Model 1 bundle to: {out_file}\n")


if __name__ == "__main__":
    train()
