"""
Training Script for Model 2: Slippage Risk Classifier (LightGBM + SHAP)
Predicts probability of task slippage and generates explainable reasoning_steps using SHAP TreeExplainer.
Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
Exports: models/artifacts/model_risk.pkl
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import lightgbm as lgb
import shap

FEATURE_NAMES = [
    "stalled_days", "dependency_depth", "active_blockers",
    "assignee_load_ratio", "logged_vs_estimated_ratio", "story_points"
]


class RiskPredictorModel:
    """Wraps LightGBM model and SHAP TreeExplainer for real-time inference and explanation."""
    def __init__(self, model, explainer, feature_names):
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names

    def predict_with_explanation(self, stalled_days: int, dependency_depth: int,
                                active_blockers: int, assignee_load_ratio: float,
                                logged_vs_estimated_ratio: float, story_points: float) -> dict:
        row = np.array([[
            stalled_days, dependency_depth, active_blockers,
            assignee_load_ratio, logged_vs_estimated_ratio, story_points
        ]])
        prob = float(self.model.predict_proba(row)[0, 1])
        if prob >= 0.75:
            severity = "high"
        elif prob >= 0.50:
            severity = "medium"
        else:
            severity = "low"

        shap_vals = self.explainer(row).values[0]
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[:, 1]

        contributions = []
        for name, val, shap_v in zip(self.feature_names, row[0], shap_vals):
            contributions.append({
                "feature": name,
                "value": float(val),
                "shap": float(shap_v)
            })

        contributions.sort(key=lambda x: x["shap"], reverse=True)
        reasoning_steps = []
        order = 1
        for item in contributions:
            if item["shap"] <= 0.05 and order > 1:
                break
            desc = f"Feature '{item['feature']}' contributed positively to risk score."
            reasoning_steps.append({
                "order": order,
                "description": desc,
                "evidence": {
                    item["feature"]: item["value"],
                    "shap_contribution": f"+{item['shap']:.2f}"
                }
            })
            order += 1
            if order > 3:
                break

        return {
            "confidence": round(prob, 2),
            "severity": severity,
            "is_risk": prob >= 0.50,
            "reasoning_steps": reasoning_steps
        }

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train():
    print("Loading Model 2 datasets...")
    train_path = os.path.join(DATA_DIR, "model2_risk_train.csv")
    test_path = os.path.join(DATA_DIR, "model2_risk_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["is_slipped"].values

    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["is_slipped"].values

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = neg_count / max(pos_count, 1)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        scale_pos_weight=imbalance_ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)

    print("Fitting SHAP TreeExplainer for Explainable AI...")
    explainer = shap.TreeExplainer(model)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model": "LightGBM Binary Classifier + SHAP",
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "ROC_AUC": round(float(roc_auc), 4),
        "Confusion_Matrix": cm,
        "train_samples": len(train_df),
        "test_samples": len(test_df)
    }

    print("\n Model 2 Evaluation Metrics (Test Set):")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    bundle = RiskPredictorModel(
        model=model,
        explainer=explainer,
        feature_names=FEATURE_NAMES
    )

    out_file = os.path.join(ARTIFACTS_DIR, "model_risk.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(bundle, f)

    with open(os.path.join(ARTIFACTS_DIR, "model2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n Successfully exported Model 2 bundle to: {out_file}\n")


if __name__ == "__main__":
    train()
