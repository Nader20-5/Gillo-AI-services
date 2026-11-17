"""
TAWOS Real-Data Benchmark Script for Model 1 and Model 2
Loads genuine Jira data from tawos_csv/
1. Model 2: Mid-sprint Carry-Over Classifier (LightGBM vs. Logistic Regression baseline)
   - Evaluated on a chronological temporal train/test split.
   - No target leakage: features use only information available at sprint midpoint.
2. Model 1: Cycle Time Regressor (XGBoost Quantile vs. Naive Median baselines)
   - Predicts real resolution cycle time in hours.
Outputs verified metric JSONs to models/artifacts/
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

TAWOS_DIR = os.path.join(os.path.dirname(__file__), "..", "tawos_csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def run_benchmark():
    print("=" * 60)
    print("STEP 1: Loading Sprints and Issues from TAWOS...")
    print("=" * 60)

    # 1. Load Sprints
    sprints = pd.read_csv(os.path.join(TAWOS_DIR, "sprint.csv"))
    sprints = sprints[sprints["State"] == "CLOSED"].dropna(subset=["Start_Date", "End_Date"])
    sprints["Start_Date"] = pd.to_datetime(sprints["Start_Date"], errors="coerce")
    sprints["End_Date"] = pd.to_datetime(sprints["End_Date"], errors="coerce")
    sprints = sprints.dropna(subset=["Start_Date", "End_Date"])
    sprints["duration_days"] = (sprints["End_Date"] - sprints["Start_Date"]).dt.total_seconds() / 86400.0
    valid_sprints = sprints[(sprints["duration_days"] >= 5) & (sprints["duration_days"] <= 35)].set_index("ID")

    print(f"Loaded {len(valid_sprints)} valid closed sprints.")

    # 2. Load Issue Links (for dependency count)
    print("Loading issue links for dependency features...")
    links_df = pd.read_csv(os.path.join(TAWOS_DIR, "issue_link.csv"), usecols=["Issue_ID"])
    dep_counts = links_df["Issue_ID"].value_counts().to_dict()

    # 3. Stream issues matching valid sprints
    sprint_ids_set = set(valid_sprints.index)
    cols = [
        "ID", "Issue_Key", "Title", "Type", "Priority", "Status",
        "Creation_Date", "Resolution_Date", "Story_Point",
        "Resolution_Time_Minutes", "Sprint_ID", "Project_ID"
    ]

    matched_chunks = []
    for chunk in pd.read_csv(os.path.join(TAWOS_DIR, "issue.csv"), usecols=cols, chunksize=50000):
        sub = chunk[chunk["Sprint_ID"].isin(sprint_ids_set)].copy()
        if len(sub) > 0:
            matched_chunks.append(sub)

    df = pd.concat(matched_chunks, ignore_index=True)
    print(f"Loaded {len(df)} issues belonging to valid sprints.")

    # Attach sprint metadata
    df["Sprint_Start"] = df["Sprint_ID"].map(valid_sprints["Start_Date"])
    df["Sprint_End"] = df["Sprint_ID"].map(valid_sprints["End_Date"])
    df["Sprint_Duration"] = df["Sprint_ID"].map(valid_sprints["duration_days"])
    df["Resolution_Date"] = pd.to_datetime(df["Resolution_Date"], errors="coerce")
    df["Creation_Date"] = pd.to_datetime(df["Creation_Date"], errors="coerce")

    # Define midpoint t_mid
    df["Midpoint_Date"] = df["Sprint_Start"] + (df["Sprint_End"] - df["Sprint_Start"]) / 2

    # Target: is_carried_over
    # Task slipped if unresolved OR resolved after sprint end
    slipped = df["Resolution_Date"].isna() | (df["Resolution_Date"] > df["Sprint_End"])
    df["is_carried_over"] = slipped.astype(int)

    # Feature 1: Days created before sprint start (backlog debt)
    df["backlog_age_days"] = np.clip((df["Sprint_Start"] - df["Creation_Date"]).dt.total_seconds() / 86400.0, 0, 365)
    df["backlog_age_days"] = df["backlog_age_days"].fillna(0)

    # Feature 2: Dependency count
    df["dependency_count"] = df["ID"].map(dep_counts).fillna(0).astype(int)

    # Feature 3: Story Points (impute median for missing)
    sp_median = df["Story_Point"].median() if df["Story_Point"].notna().any() else 3.0
    df["Story_Point_Clean"] = df["Story_Point"].fillna(sp_median)

    # Feature 4: Issue Type & Priority encoding
    top_types = df["Type"].value_counts().nlargest(5).index.tolist()
    for t in top_types:
        df[f"type_{t}"] = (df["Type"] == t).astype(float)

    top_prios = ["Blocker", "Critical", "Major", "Minor", "Trivial"]
    for p in top_prios:
        df[f"prio_{p}"] = (df["Priority"] == p).astype(float)

    feature_cols = [
        "Sprint_Duration", "backlog_age_days", "dependency_count", "Story_Point_Clean"
    ] + [f"type_{t}" for t in top_types] + [f"prio_{p}" for p in top_prios]

    # Chronological Temporal Split by Sprint_Start
    df = df.sort_values("Sprint_Start").reset_index(drop=True)
    split_idx = int(0.75 * len(df))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nChronological Split:")
    print(f"  - Train samples: {len(train_df)} (Sprints: {train_df['Sprint_Start'].min().date()} to {train_df['Sprint_Start'].max().date()})")
    print(f"  - Test samples:  {len(test_df)} (Sprints: {test_df['Sprint_Start'].min().date()} to {test_df['Sprint_Start'].max().date()})")
    print(f"  - Test Carry-Over Prevalence: {test_df['is_carried_over'].mean():.3f}")

    # =========================================================================
    # MODEL 2: CARRY-OVER RISK BENCHMARK (Real TAWOS)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Training Model 2 (LightGBM vs. Logistic Regression Baseline)...")
    print("=" * 60)

    X_train = train_df[feature_cols].values
    y_train = train_df["is_carried_over"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["is_carried_over"].values

    # Scaler for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Baseline: Logistic Regression
    logreg = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_roc = roc_auc_score(y_test, logreg_probs)
    p_lr, r_lr, _ = precision_recall_curve(y_test, logreg_probs)
    logreg_pr_auc = auc(r_lr, p_lr)
    logreg_brier = brier_score_loss(y_test, logreg_probs)

    # 2. Model 2: LightGBM
    pos_weight = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    lgb_preds = (lgb_probs >= 0.5).astype(int)

    lgb_roc = roc_auc_score(y_test, lgb_probs)
    p_lgb, r_lgb, _ = precision_recall_curve(y_test, lgb_probs)
    lgb_pr_auc = auc(r_lgb, p_lgb)
    lgb_brier = brier_score_loss(y_test, lgb_probs)
    lgb_acc = accuracy_score(y_test, lgb_preds)
    lgb_prec = precision_score(y_test, lgb_preds, zero_division=0)
    lgb_rec = recall_score(y_test, lgb_preds, zero_division=0)
    lgb_f1 = f1_score(y_test, lgb_preds, zero_division=0)
    cm = confusion_matrix(y_test, lgb_preds).tolist()

    # Brier Skill Score relative to null model
    p_base = float(np.mean(y_test))
    null_brier = p_base * (1.0 - p_base)
    bss = 1.0 - (lgb_brier / max(null_brier, 1e-6))

    m2_results = {
        "dataset": "TAWOS Real Jira Benchmark",
        "task": "Sprint Midpoint Carry-Over Prediction",
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "split_method": "Strict Temporal Split (by Sprint_Start Date)",
        "test_prevalence_carry_over": round(p_base, 4),
        "Baseline_LogisticRegression": {
            "ROC_AUC": round(float(logreg_roc), 4),
            "PR_AUC": round(float(logreg_pr_auc), 4),
            "Brier_Score": round(float(logreg_brier), 4)
        },
        "Model2_LightGBM": {
            "ROC_AUC": round(float(lgb_roc), 4),
            "PR_AUC": round(float(lgb_pr_auc), 4),
            "Accuracy": round(float(lgb_acc), 4),
            "Precision": round(float(lgb_prec), 4),
            "Recall": round(float(lgb_rec), 4),
            "F1_Score": round(float(lgb_f1), 4),
            "Brier_Score": round(float(lgb_brier), 4),
            "Brier_Skill_Score_Pct": round(float(bss * 100), 2),
            "Confusion_Matrix": cm
        }
    }

    out_m2 = os.path.join(ARTIFACTS_DIR, "model2_tawos_real_metrics.json")
    with open(out_m2, "w") as f:
        json.dump(m2_results, f, indent=2)

    print(f"Model 2 Results written to: {out_m2}")
    print(f"  Baseline Logistic Regression: ROC-AUC = {logreg_roc:.4f}, PR-AUC = {logreg_pr_auc:.4f}")
    print(f"  Model 2 LightGBM:             ROC-AUC = {lgb_roc:.4f}, PR-AUC = {lgb_pr_auc:.4f}, Recall = {lgb_rec:.4f}, Precision = {lgb_prec:.4f}")

    # =========================================================================
    # MODEL 1: CYCLE TIME REGRESSOR (Real TAWOS)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Training Model 1 (XGBoost Quantile vs. Naive Median Baselines)...")
    print("=" * 60)

    # Use resolved tasks with valid cycle time (Resolution_Time_Minutes > 0)
    resolved_df = df[df["Resolution_Time_Minutes"] > 0].copy()
    resolved_df["cycle_time_hours"] = np.clip(resolved_df["Resolution_Time_Minutes"] / 60.0, 0.5, 720.0)

    res_train = resolved_df[resolved_df["Sprint_Start"] <= train_df["Sprint_Start"].max()]
    res_test = resolved_df[resolved_df["Sprint_Start"] > train_df["Sprint_Start"].max()]

    print(f"Cycle time tasks: {len(res_train)} train, {len(res_test)} test.")

    y_train_c = res_train["cycle_time_hours"].values
    y_test_c = res_test["cycle_time_hours"].values

    X_train_c = res_train[feature_cols].values
    X_test_c = res_test[feature_cols].values

    # Baseline 1: Overall Median
    overall_median = float(np.median(y_train_c))
    base1_preds = np.full_like(y_test_c, overall_median)
    mae_b1 = mean_absolute_error(y_test_c, base1_preds)
    medae_b1 = median_absolute_error(y_test_c, base1_preds)

    # Baseline 2: Type Median
    type_medians = res_train.groupby("Type")["cycle_time_hours"].median().to_dict()
    base2_preds = res_test["Type"].map(type_medians).fillna(overall_median).values
    mae_b2 = mean_absolute_error(y_test_c, base2_preds)
    medae_b2 = median_absolute_error(y_test_c, base2_preds)

    # XGBoost Quantile Regressors (P25, P50, P75)
    xgb_low = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.25, n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_mid = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.50, n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_high = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.75, n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)

    xgb_low.fit(X_train_c, y_train_c)
    xgb_mid.fit(X_train_c, y_train_c)
    xgb_high.fit(X_train_c, y_train_c)

    p25 = xgb_low.predict(X_test_c)
    p50 = xgb_mid.predict(X_test_c)
    p75 = xgb_high.predict(X_test_c)

    mae_m1 = mean_absolute_error(y_test_c, p50)
    medae_m1 = median_absolute_error(y_test_c, p50)
    rmse_m1 = np.sqrt(mean_squared_error(y_test_c, p50))
    r2_m1 = r2_score(y_test_c, p50)

    pb25 = pinball_loss(y_test_c, p25, 0.25)
    pb50 = pinball_loss(y_test_c, p50, 0.50)
    pb75 = pinball_loss(y_test_c, p75, 0.75)
    cov = float(np.mean((y_test_c >= p25) & (y_test_c <= p75)) * 100)

    m1_results = {
        "dataset": "TAWOS Real Jira Benchmark",
        "target": "Cycle Time (Hours)",
        "train_samples": int(len(res_train)),
        "test_samples": int(len(res_test)),
        "split_method": "Strict Temporal Split (by Sprint_Start Date)",
        "Baseline_Overall_Median": {
            "MAE_hours": round(float(mae_b1), 2),
            "MedAE_hours": round(float(medae_b1), 2)
        },
        "Baseline_Type_Median": {
            "MAE_hours": round(float(mae_b2), 2),
            "MedAE_hours": round(float(medae_b2), 2)
        },
        "Model1_XGBoost_Quantile": {
            "MAE_hours": round(float(mae_m1), 2),
            "MedAE_hours": round(float(medae_m1), 2),
            "RMSE_hours": round(float(rmse_m1), 2),
            "R2_Score": round(float(r2_m1), 4),
            "Pinball_0.25": round(pb25, 2),
            "Pinball_0.50": round(pb50, 2),
            "Pinball_0.75": round(pb75, 2),
            "P25_P75_Coverage_Pct": round(cov, 1)
        }
    }

    out_m1 = os.path.join(ARTIFACTS_DIR, "model1_tawos_real_metrics.json")
    with open(out_m1, "w") as f:
        json.dump(m1_results, f, indent=2)

    print(f"Model 1 Results written to: {out_m1}")
    print(f"  Baseline Type Median:   MAE = {mae_b2:.2f}h, MedAE = {medae_b2:.2f}h")
    print(f"  Model 1 XGBoost (P50):  MAE = {mae_m1:.2f}h, MedAE = {medae_m1:.2f}h, R2 = {r2_m1:.4f}")
    print(f"  Coverage [P25, P75]:    {cov:.1f}%")
    print("\n Benchmark Completed Successfully!")


if __name__ == "__main__":
    run_benchmark()
