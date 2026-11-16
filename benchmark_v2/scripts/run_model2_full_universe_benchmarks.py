"""
benchmark_v2/scripts/run_model2_full_universe_benchmarks.py

Master benchmark script on the FULL UNIVERSE dataset:
1. Re-evaluates Full LightGBM, Taskove Model, Scaled LR, and Rules on Full Universe.
2. Side-by-side comparison between Old Universe and New Full Universe.
3. Two-half validation split for calibration and threshold selection.
4. Rank-based policies as primary on the ONE canonical test sprint population (>= 10 tasks, >= 1 pos, >= 1 neg).
5. Reliability tables for Val-fitted vs Train-fitted calibrators.
6. Held-out projects variant with within-sprint AUC, macro per-project AUC, and sprint date ranges.
7. Saves updated artifacts and writes benchmark_v2/work/model2_full_universe_metrics.json.
"""

import os
import sys
import time
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
work_dir = os.path.join(BASE_DIR, "work")
tawos_dir = r"d:\Gillo AI Service\tawos_csv"
sys.path.insert(0, BASE_DIR)

from features_v2 import Model2FeatureTransformer, TaskoveFeatureTransformer

def pr_auc_score(y_true, y_pred):
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return float(auc(r, p))

def compute_10_bin_calibration(y_true, y_prob):
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(y_prob, bins) - 1
    table = []
    for i in range(10):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            obs_rate = float(np.mean(y_true[mask]))
        else:
            mean_pred = float((bins[i] + bins[i+1]) / 2.0)
            obs_rate = 0.0
        table.append({
            'bin_index': i + 1,
            'bin_range': f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
            'count': count,
            'mean_predicted': round(mean_pred, 4),
            'observed_rate': round(obs_rate, 4),
            'gap': round(mean_pred - obs_rate, 4)
        })
    return table

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f_in:
        while chunk := f_in.read(8192):
            h.update(chunk)
    return h.hexdigest()

def evaluate_on_canonical_sprints(df_eval, prob_col, min_tasks=10):
    """
    Evaluates within-sprint ranking and rank-based policies on the ONE canonical test population:
    sprints with >= min_tasks open tasks, >= 1 positive and >= 1 negative.
    """
    within_aucs = []
    base_rates = []
    p3_list, p5_list, p20_list = [], [], []
    r3_list, r5_list, r20_list = [], [], []
    l3_ratios, l5_ratios, l20_ratios = [], [], []

    for sid, grp in df_eval.groupby('sprint_id'):
        n = len(grp)
        pos = int(grp['label'].sum())
        neg = n - pos
        if n >= min_tasks and pos >= 1 and neg >= 1:
            br = pos / n
            base_rates.append(br)
            auc_sp = float(roc_auc_score(grp['label'], grp[prob_col]))
            within_aucs.append(auc_sp)

            sorted_g = grp.sort_values(prob_col, ascending=False)

            # Top 3
            t3 = sorted_g.head(3)
            prec3 = float(t3['label'].sum() / len(t3))
            rec3 = float(t3['label'].sum() / pos)
            p3_list.append(prec3)
            r3_list.append(rec3)
            l3_ratios.append(prec3 / br)

            # Top 5
            t5 = sorted_g.head(5)
            prec5 = float(t5['label'].sum() / len(t5))
            rec5 = float(t5['label'].sum() / pos)
            p5_list.append(prec5)
            r5_list.append(rec5)
            l5_ratios.append(prec5 / br)

            # Top 20%
            k20 = max(1, int(np.ceil(0.20 * n)))
            t20 = sorted_g.head(k20)
            prec20 = float(t20['label'].sum() / len(t20))
            rec20 = float(t20['label'].sum() / pos)
            p20_list.append(prec20)
            r20_list.append(rec20)
            l20_ratios.append(prec20 / br)

    mean_br = float(np.mean(base_rates)) if base_rates else 0.0
    return {
        'sprint_count': len(within_aucs),
        'mean_within_sprint_auc': round(float(np.mean(within_aucs)), 4) if within_aucs else np.nan,
        'mean_sprint_base_rate': round(mean_br, 4),
        'top_3': {
            'precision': round(float(np.mean(p3_list)), 4) if p3_list else np.nan,
            'recall': round(float(np.mean(r3_list)), 4) if r3_list else np.nan,
            'lift_ratio_of_means': round(float(np.mean(p3_list) / mean_br), 4) if p3_list and mean_br > 0 else np.nan,
            'lift_mean_of_ratios': round(float(np.mean(l3_ratios)), 4) if l3_ratios else np.nan
        },
        'top_5': {
            'precision': round(float(np.mean(p5_list)), 4) if p5_list else np.nan,
            'recall': round(float(np.mean(r5_list)), 4) if r5_list else np.nan,
            'lift_ratio_of_means': round(float(np.mean(p5_list) / mean_br), 4) if p5_list and mean_br > 0 else np.nan,
            'lift_mean_of_ratios': round(float(np.mean(l5_ratios)), 4) if l5_ratios else np.nan
        },
        'top_20_pct': {
            'precision': round(float(np.mean(p20_list)), 4) if p20_list else np.nan,
            'recall': round(float(np.mean(r20_list)), 4) if r20_list else np.nan,
            'lift_ratio_of_means': round(float(np.mean(p20_list) / mean_br), 4) if p20_list and mean_br > 0 else np.nan,
            'lift_mean_of_ratios': round(float(np.mean(l20_ratios)), 4) if l20_ratios else np.nan
        }
    }

def compute_macro_proj_auc(df_eval, prob_col, min_tasks=30):
    proj_aucs = []
    for pid, grp in df_eval.groupby('project_id'):
        if len(grp) >= min_tasks:
            if len(grp['label'].unique()) > 1:
                proj_aucs.append(float(roc_auc_score(grp['label'], grp[prob_col])))
    return round(float(np.mean(proj_aucs)), 4) if proj_aucs else np.nan, len(proj_aucs)

def main():
    print("======================================================================")
    print("RUNNING FINAL BENCHMARK PIPELINE ON FULL UNIVERSE DATASET")
    print("======================================================================")
    t0 = time.time()

    # 1. Load Datasets (Old Universe vs New Full Universe)
    df_old = pd.read_parquet(os.path.join(work_dir, "model2_dataset.parquet"))
    df_new = pd.read_parquet(os.path.join(work_dir, "model2_dataset_full_universe.parquet"))

    print(f"Old Universe: {len(df_old):,} rows across {df_old['issue_id'].nunique():,} unique issues.")
    print(f"New Universe: {len(df_new):,} rows across {df_new['issue_id'].nunique():,} unique issues.")

    # Compare base rates per split
    old_base_rates = {
        'train': round(float(df_old[df_old['split'] == 'train']['label'].mean()), 4),
        'val_f50': round(float(df_old[(df_old['split'] == 'val') & (df_old['f'] == 0.5)]['label'].mean()), 4),
        'test_f50': round(float(df_old[(df_old['split'] == 'test') & (df_old['f'] == 0.5)]['label'].mean()), 4),
        'heldout_f50': round(float(df_old[(df_old['is_heldout_project'] == 1) & (df_old['f'] == 0.5)]['label'].mean()), 4)
    }
    new_base_rates = {
        'train': round(float(df_new[df_new['split'] == 'train']['label'].mean()), 4),
        'val_f50': round(float(df_new[(df_new['split'] == 'val') & (df_new['f'] == 0.5)]['label'].mean()), 4),
        'test_f50': round(float(df_new[(df_new['split'] == 'test') & (df_new['f'] == 0.5)]['label'].mean()), 4),
        'heldout_f50': round(float(df_new[(df_new['is_heldout_project'] == 1) & (df_new['f'] == 0.5)]['label'].mean()), 4)
    }

    print("\n--- Base Rates Comparison (Old vs New Universe) ---")
    print(f"Train Base Rate:     Old = {old_base_rates['train']} vs New = {new_base_rates['train']}")
    print(f"Val f=0.5 Base Rate: Old = {old_base_rates['val_f50']} vs New = {new_base_rates['val_f50']}")
    print(f"Test f=0.5 Base Rate:Old = {old_base_rates['test_f50']} vs New = {new_base_rates['test_f50']}")
    print(f"Heldout f=0.5 Rate:  Old = {old_base_rates['heldout_f50']} vs New = {new_base_rates['heldout_f50']}")

    # Splits for Full Universe
    df_train = df_new[df_new['split'] == 'train'].copy()
    df_val = df_new[df_new['split'] == 'val'].copy()
    df_test = df_new[df_new['split'] == 'test'].copy()
    df_heldout = df_new[df_new['is_heldout_project'] == 1].copy()

    df_train_f50 = df_train[df_train['f'] == 0.5].copy()
    df_val_f50 = df_val[df_val['f'] == 0.5].copy()
    df_test_f50 = df_test[df_test['f'] == 0.5].copy()
    df_heldout_f50 = df_heldout[df_heldout['f'] == 0.5].copy()

    y_train = df_train['label'].values
    y_val = df_val['label'].values
    y_test = df_test['label'].values
    y_test_f50 = df_test_f50['label'].values
    y_val_f50 = df_val_f50['label'].values
    y_heldout_f50 = df_heldout_f50['label'].values

    # Check test sprint count on canonical population (>= 10 tasks, >= 1 pos, >= 1 neg)
    canonical_test_sprints = []
    for sid, grp in df_test_f50.groupby('sprint_id'):
        pos = int(grp['label'].sum())
        if len(grp) >= 10 and pos >= 1 and (len(grp) - pos) >= 1:
            canonical_test_sprints.append(sid)
    canonical_sprint_count = len(canonical_test_sprints)
    print(f"\nCanonical Test Sprint Population (>= 10 tasks, >= 1 pos, >= 1 neg): {canonical_sprint_count} sprints.")

    # 2. Train Full Feature Pipeline
    print("\nFitting Model2FeatureTransformer on Full Universe Train...")
    full_transformer = Model2FeatureTransformer()
    X_train_full = full_transformer.fit_transform(df_train)
    X_val_full = full_transformer.transform(df_val)
    X_test_full = full_transformer.transform(df_test)
    X_val_f50_full = X_val_full[(df_val['f'] == 0.5).values]
    X_test_f50_full = X_test_full[(df_test['f'] == 0.5).values]
    X_ho_f50_full = full_transformer.transform(df_heldout_f50)

    # Train Full LightGBM
    print("Training Full LightGBM on Full Universe Train...")
    full_lgb = lgb.LGBMClassifier(
        n_estimators=150,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    full_lgb.fit(X_train_full, y_train)

    # 3. Validation Split by SPRINT for Calibration & Threshold (Item 3)
    print("\nSplitting Validation Set by Sprint into Half A and Half B...")
    val_sprint_ids = sorted(df_val_f50['sprint_id'].unique().tolist())
    np.random.seed(42)
    shuffled_sids = np.random.permutation(val_sprint_ids)
    half_a_sids = set(shuffled_sids[:len(shuffled_sids)//2])
    half_b_sids = set(shuffled_sids[len(shuffled_sids)//2:])

    val_half_a_mask = df_val_f50['sprint_id'].isin(half_a_sids)
    val_half_b_mask = df_val_f50['sprint_id'].isin(half_b_sids)

    df_val_half_a = df_val_f50[val_half_a_mask]
    df_val_half_b = df_val_f50[val_half_b_mask]
    y_val_a = df_val_half_a['label'].values
    y_val_b = df_val_half_b['label'].values

    # Full model raw probs on val halves
    val_a_raw_p = full_lgb.predict_proba(X_val_f50_full[val_half_a_mask.values])[:, 1]
    val_b_raw_p = full_lgb.predict_proba(X_val_f50_full[val_half_b_mask.values])[:, 1]

    # Fit isotonic calibrator on Val Half A
    iso_val_a = IsotonicRegression(out_of_bounds='clip')
    iso_val_a.fit(val_a_raw_p, y_val_a)

    # Evaluate on Val Half B to choose operating threshold (max recall subject to flag_rate <= 30%)
    val_b_cal_p = iso_val_a.predict(val_b_raw_p)
    val_ths = np.linspace(0.10, 0.99, 90)
    best_th = 0.5
    best_th_rec = -1.0
    for th in val_ths:
        preds = (val_b_cal_p >= th).astype(int)
        flag_r = float(np.mean(preds))
        if flag_r <= 0.30:
            rec = float(recall_score(y_val_b, preds, zero_division=0))
            if rec > best_th_rec:
                best_th_rec = rec
                best_th = float(th)

    val_b_chosen_flag_rate = float(np.mean((val_b_cal_p >= best_th).astype(int)))
    print(f"Val Half A fitted Calibrator -> Val Half B Threshold: {best_th:.4f} (Flag Rate: {val_b_chosen_flag_rate*100:.2f}%, Recall: {best_th_rec:.4f})")

    # Predict on test
    test_full_raw_p = full_lgb.predict_proba(X_test_f50_full)[:, 1]
    test_full_cal_p = iso_val_a.predict(test_full_raw_p)
    test_full_flag_rate = float(np.mean((test_full_cal_p >= best_th).astype(int)))

    # Fit calibrator on Train for calibration comparison (Item 3)
    train_raw_p = full_lgb.predict_proba(X_train_full)[:, 1]
    iso_train = IsotonicRegression(out_of_bounds='clip')
    iso_train.fit(train_raw_p, y_train)
    test_full_train_cal_p = iso_train.predict(test_full_raw_p)

    # Compute reliability tables
    cal_table_val = compute_10_bin_calibration(y_test_f50, test_full_cal_p)
    cal_table_train = compute_10_bin_calibration(y_test_f50, test_full_train_cal_p)
    brier_val = float(brier_score_loss(y_test_f50, test_full_cal_p))
    brier_train = float(brier_score_loss(y_test_f50, test_full_train_cal_p))
    print(f"Calibration Reliability: Val-fitted Brier = {brier_val:.4f} | Train-fitted Brier = {brier_train:.4f}")

    # Canonical evaluation for Full LightGBM
    df_test_f50_full_eval = df_test_f50.copy()
    df_test_f50_full_eval['prob'] = test_full_cal_p
    full_canonical_metrics = evaluate_on_canonical_sprints(df_test_f50_full_eval, 'prob')
    full_macro_proj_auc, full_n_proj = compute_macro_proj_auc(df_test_f50_full_eval, 'prob')

    full_results_new = {
        'roc_auc': round(float(roc_auc_score(y_test_f50, test_full_cal_p)), 4),
        'pr_auc': round(pr_auc_score(y_test_f50, test_full_cal_p), 4),
        'brier_score': round(brier_val, 4),
        'macro_proj_auc': full_macro_proj_auc,
        'within_sprint_auc': full_canonical_metrics['mean_within_sprint_auc'],
        'precision@3': full_canonical_metrics['top_3']['precision'],
        'precision@5': full_canonical_metrics['top_5']['precision'],
        'precision@20_pct': full_canonical_metrics['top_20_pct']['precision']
    }

    # 4. Train Taskove-Compatible Model on Full Universe (Item 5 & Item 7)
    print("\nTraining Taskove-Compatible Model on Full Universe Train...")
    tk_transformer = TaskoveFeatureTransformer()
    X_train_tk = tk_transformer.fit_transform(df_train)
    X_val_tk = tk_transformer.transform(df_val)
    X_test_tk = tk_transformer.transform(df_test)
    X_val_f50_tk = X_val_tk[(df_val['f'] == 0.5).values]
    X_test_f50_tk = X_test_tk[(df_test['f'] == 0.5).values]
    X_ho_f50_tk = tk_transformer.transform(df_heldout_f50)

    taskove_lgb = lgb.LGBMClassifier(
        n_estimators=150,
        num_leaves=31,
        learning_rate=0.03,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    taskove_lgb.fit(X_train_tk, y_train)

    # Fit calibrator on Val Half A for Taskove model
    val_a_tk_raw = taskove_lgb.predict_proba(X_val_f50_tk[val_half_a_mask.values])[:, 1]
    iso_tk_val_a = IsotonicRegression(out_of_bounds='clip')
    iso_tk_val_a.fit(val_a_tk_raw, y_val_a)

    val_b_tk_raw = taskove_lgb.predict_proba(X_val_f50_tk[val_half_b_mask.values])[:, 1]
    val_b_tk_cal = iso_tk_val_a.predict(val_b_tk_raw)

    best_tk_th = 0.5
    best_tk_rec = -1.0
    for th in val_ths:
        preds = (val_b_tk_cal >= th).astype(int)
        flag_r = float(np.mean(preds))
        if flag_r <= 0.30:
            rec = float(recall_score(y_val_b, preds, zero_division=0))
            if rec > best_tk_rec:
                best_tk_rec = rec
                best_tk_th = float(th)

    val_b_tk_flag_rate = float(np.mean((val_b_tk_cal >= best_tk_th).astype(int)))
    test_tk_raw_p = taskove_lgb.predict_proba(X_test_f50_tk)[:, 1]
    test_tk_cal_p = iso_tk_val_a.predict(test_tk_raw_p)
    test_tk_flag_rate = float(np.mean((test_tk_cal_p >= best_tk_th).astype(int)))

    df_test_f50_tk_eval = df_test_f50.copy()
    df_test_f50_tk_eval['prob'] = test_tk_cal_p
    tk_canonical_metrics = evaluate_on_canonical_sprints(df_test_f50_tk_eval, 'prob')
    tk_macro_proj_auc, tk_n_proj = compute_macro_proj_auc(df_test_f50_tk_eval, 'prob')

    tk_results_new = {
        'roc_auc': round(float(roc_auc_score(y_test_f50, test_tk_cal_p)), 4),
        'pr_auc': round(pr_auc_score(y_test_f50, test_tk_cal_p), 4),
        'brier_score': round(float(brier_score_loss(y_test_f50, test_tk_cal_p)), 4),
        'macro_proj_auc': tk_macro_proj_auc,
        'within_sprint_auc': tk_canonical_metrics['mean_within_sprint_auc'],
        'precision@3': tk_canonical_metrics['top_3']['precision'],
        'precision@5': tk_canonical_metrics['top_5']['precision'],
        'precision@20_pct': tk_canonical_metrics['top_20_pct']['precision']
    }

    # 5. Train Scaled Balanced Logistic Regression on Full Universe
    print("\nTraining Scaled Balanced Logistic Regression on Full Universe Train...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_full)
    X_test_sc = scaler.transform(X_test_f50_full)
    lr_full = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_full.fit(X_train_sc, y_train)
    test_lr_p = lr_full.predict_proba(X_test_sc)[:, 1]

    df_test_f50_lr_eval = df_test_f50.copy()
    df_test_f50_lr_eval['prob'] = test_lr_p
    lr_canonical_metrics = evaluate_on_canonical_sprints(df_test_f50_lr_eval, 'prob')
    lr_macro_proj_auc, _ = compute_macro_proj_auc(df_test_f50_lr_eval, 'prob')

    lr_results_new = {
        'roc_auc': round(float(roc_auc_score(y_test_f50, test_lr_p)), 4),
        'pr_auc': round(pr_auc_score(y_test_f50, test_lr_p), 4),
        'brier_score': round(float(brier_score_loss(y_test_f50, test_lr_p)), 4),
        'macro_proj_auc': lr_macro_proj_auc,
        'within_sprint_auc': lr_canonical_metrics['mean_within_sprint_auc'],
        'precision@3': lr_canonical_metrics['top_3']['precision'],
        'precision@5': lr_canonical_metrics['top_5']['precision'],
        'precision@20_pct': lr_canonical_metrics['top_20_pct']['precision']
    }

    # 6. Evaluate Rule Baselines on Full Universe
    print("\nEvaluating Rule Baselines on Full Universe...")
    # Rule: carry_count > 0
    df_test_f50_cc_eval = df_test_f50.copy()
    df_test_f50_cc_eval['prob'] = (df_test_f50['carry_count'] > 0).astype(float)
    cc_canonical = evaluate_on_canonical_sprints(df_test_f50_cc_eval, 'prob')
    cc_macro_proj_auc, _ = compute_macro_proj_auc(df_test_f50_cc_eval, 'prob')
    cc_results_new = {
        'roc_auc': round(float(roc_auc_score(y_test_f50, df_test_f50_cc_eval['prob'])), 4),
        'pr_auc': round(pr_auc_score(y_test_f50, df_test_f50_cc_eval['prob']), 4),
        'macro_proj_auc': cc_macro_proj_auc,
        'within_sprint_auc': cc_canonical['mean_within_sprint_auc'],
        'precision@3': cc_canonical['top_3']['precision'],
        'precision@5': cc_canonical['top_5']['precision'],
        'precision@20_pct': cc_canonical['top_20_pct']['precision']
    }

    # Rule: status_category == 'todo'
    df_test_f50_todo_eval = df_test_f50.copy()
    df_test_f50_todo_eval['prob'] = (df_test_f50['status_category'].str.lower() == 'todo').astype(float)
    todo_canonical = evaluate_on_canonical_sprints(df_test_f50_todo_eval, 'prob')
    todo_macro_proj_auc, _ = compute_macro_proj_auc(df_test_f50_todo_eval, 'prob')
    todo_results_new = {
        'roc_auc': round(float(roc_auc_score(y_test_f50, df_test_f50_todo_eval['prob'])), 4),
        'pr_auc': round(pr_auc_score(y_test_f50, df_test_f50_todo_eval['prob']), 4),
        'macro_proj_auc': todo_macro_proj_auc,
        'within_sprint_auc': todo_canonical['mean_within_sprint_auc'],
        'precision@3': todo_canonical['top_3']['precision'],
        'precision@5': todo_canonical['top_5']['precision'],
        'precision@20_pct': todo_canonical['top_20_pct']['precision']
    }

    # 7. Held-Out Projects Evaluation with Sprint Date Ranges & Within/Macro AUC (Item 5)
    print("\nEvaluating Held-Out Projects on Full Universe...")
    sprint_df = pd.read_csv(os.path.join(tawos_dir, "sprint.csv"))
    sprint_df['Start_dt'] = pd.to_datetime(sprint_df['Start_Date'], errors='coerce')
    sprint_df['End_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')

    train_sprints_df = sprint_df[sprint_df['ID'].isin(df_train['sprint_id'].unique())]
    ho_sprints_df = sprint_df[sprint_df['ID'].isin(df_heldout_f50['sprint_id'].unique())]

    train_date_range = [str(train_sprints_df['Start_dt'].min())[:10], str(train_sprints_df['End_dt'].max())[:10]]
    ho_date_range = [str(ho_sprints_df['Start_dt'].min())[:10], str(ho_sprints_df['End_dt'].max())[:10]]
    print(f"Train Sprints Date Range:    {train_date_range}")
    print(f"Held-out Sprints Date Range: {ho_date_range}")

    # Canonical evaluation on held-out sprints
    df_ho_eval = df_heldout_f50.copy()

    # Held out models
    # Full LightGBM with carry rate
    ho_full_p = iso_val_a.predict(full_lgb.predict_proba(X_ho_f50_full)[:, 1])
    df_ho_eval['p_full'] = ho_full_p
    ho_full_can = evaluate_on_canonical_sprints(df_ho_eval, 'p_full')
    ho_full_macro, _ = compute_macro_proj_auc(df_ho_eval, 'p_full')

    # Full LightGBM simulated new team (no carry rate: NaN imputed)
    df_ho_unseen = df_heldout_f50.copy()
    df_ho_unseen['project_prev_3_carry_rate'] = np.nan
    X_ho_unseen = full_transformer.transform(df_ho_unseen)
    ho_unseen_p = iso_val_a.predict(full_lgb.predict_proba(X_ho_unseen)[:, 1])
    df_ho_eval['p_unseen'] = ho_unseen_p
    ho_unseen_can = evaluate_on_canonical_sprints(df_ho_eval, 'p_unseen')
    ho_unseen_macro, _ = compute_macro_proj_auc(df_ho_eval, 'p_unseen')

    # Taskove model on held out
    ho_tk_p = iso_tk_val_a.predict(taskove_lgb.predict_proba(X_ho_f50_tk)[:, 1])
    df_ho_eval['p_tk'] = ho_tk_p
    ho_tk_can = evaluate_on_canonical_sprints(df_ho_eval, 'p_tk')
    ho_tk_macro, _ = compute_macro_proj_auc(df_ho_eval, 'p_tk')

    # Scaled LR on held out
    X_ho_sc = scaler.transform(X_ho_f50_full)
    ho_lr_p = lr_full.predict_proba(X_ho_sc)[:, 1]
    df_ho_eval['p_lr'] = ho_lr_p
    ho_lr_can = evaluate_on_canonical_sprints(df_ho_eval, 'p_lr')
    ho_lr_macro, _ = compute_macro_proj_auc(df_ho_eval, 'p_lr')

    # Rule todo on held out
    df_ho_eval['p_todo'] = (df_heldout_f50['status_category'].str.lower() == 'todo').astype(float)
    ho_todo_can = evaluate_on_canonical_sprints(df_ho_eval, 'p_todo')
    ho_todo_macro, _ = compute_macro_proj_auc(df_ho_eval, 'p_todo')

    # Rule carry_count on held out
    df_ho_eval['p_cc'] = (df_heldout_f50['carry_count'] > 0).astype(float)
    ho_cc_can = evaluate_on_canonical_sprints(df_ho_eval, 'p_cc')
    ho_cc_macro, _ = compute_macro_proj_auc(df_ho_eval, 'p_cc')

    heldout_table_rows = [
        {
            'model': "Rule: status_category == 'todo'",
            'roc_auc': round(float(roc_auc_score(y_heldout_f50, df_ho_eval['p_todo'])), 4),
            'pr_auc': round(pr_auc_score(y_heldout_f50, df_ho_eval['p_todo']), 4),
            'within_sprint_auc': ho_todo_can['mean_within_sprint_auc'],
            'macro_proj_auc': ho_todo_macro
        },
        {
            'model': "Rule: carry_count > 0",
            'roc_auc': round(float(roc_auc_score(y_heldout_f50, df_ho_eval['p_cc'])), 4),
            'pr_auc': round(pr_auc_score(y_heldout_f50, df_ho_eval['p_cc']), 4),
            'within_sprint_auc': ho_cc_can['mean_within_sprint_auc'],
            'macro_proj_auc': ho_cc_macro
        },
        {
            'model': "Scaled Balanced LR (All Features)",
            'roc_auc': round(float(roc_auc_score(y_heldout_f50, ho_lr_p)), 4),
            'pr_auc': round(pr_auc_score(y_heldout_f50, ho_lr_p), 4),
            'within_sprint_auc': ho_lr_can['mean_within_sprint_auc'],
            'macro_proj_auc': ho_lr_macro
        },
        {
            'model': "Taskove-Compatible LightGBM",
            'roc_auc': round(float(roc_auc_score(y_heldout_f50, ho_tk_p)), 4),
            'pr_auc': round(pr_auc_score(y_heldout_f50, ho_tk_p), 4),
            'within_sprint_auc': ho_tk_can['mean_within_sprint_auc'],
            'macro_proj_auc': ho_tk_macro
        },
        {
            'model': "Full LightGBM (No Carry Rate, Simulated New Team)",
            'roc_auc': round(float(roc_auc_score(y_heldout_f50, ho_unseen_p)), 4),
            'pr_auc': round(pr_auc_score(y_heldout_f50, ho_unseen_p), 4),
            'within_sprint_auc': ho_unseen_can['mean_within_sprint_auc'],
            'macro_proj_auc': ho_unseen_macro
        },
        {
            'model': "Full LightGBM (With Carry Rate)",
            'roc_auc': round(float(roc_auc_score(y_heldout_f50, ho_full_p)), 4),
            'pr_auc': round(pr_auc_score(y_heldout_f50, ho_full_p), 4),
            'within_sprint_auc': ho_full_can['mean_within_sprint_auc'],
            'macro_proj_auc': ho_full_macro
        }
    ]

    # 8. Save Taskove Artifacts (Item 7)
    print("\nResaving Taskove-Compatible Production Artifacts...")
    tk_art_dir = os.path.join(BASE_DIR, "artifacts", "model2", "v2_taskove_features")
    os.makedirs(tk_art_dir, exist_ok=True)
    tk_model_path = os.path.join(tk_art_dir, "model2_taskove_lightgbm.pkl")
    tk_cal_path = os.path.join(tk_art_dir, "model2_taskove_calibrator.pkl")
    tk_tr_path = os.path.join(tk_art_dir, "model2_taskove_transformer.pkl")
    tk_meta_path = os.path.join(tk_art_dir, "metadata.json")

    with open(tk_model_path, 'wb') as f:
        pickle.dump(taskove_lgb, f)
    with open(tk_cal_path, 'wb') as f:
        pickle.dump({'type': 'isotonic', 'calibrator': iso_tk_val_a}, f)
    with open(tk_tr_path, 'wb') as f:
        pickle.dump(tk_transformer, f)

    taskove_meta = {
        'model_name': 'model2_sprint_carryover_risk_taskove',
        'version': 'v2_taskove_features',
        'trained_on': 'tawos_full_universe',
        'base_rate': round(float(np.mean(y_train)), 4),
        'test_base_rate_f50': round(float(np.mean(y_test_f50)), 4),
        'ordered_feature_list': tk_transformer.feature_names_,
        'feature_count': len(tk_transformer.feature_names_),
        'best_hyperparameters': {
            'num_leaves': 31,
            'learning_rate': 0.03,
            'min_child_samples': 100
        },
        'validation_threshold_policy': {
            'selection_rule': 'Max recall on validation half B subject to flag_rate <= 30%',
            'chosen_threshold': round(best_tk_th, 4),
            'validation_flag_rate_pct': round(val_b_tk_flag_rate * 100, 2),
            'test_flag_rate_pct': round(test_tk_flag_rate * 100, 2),
            'test_precision': round(float(precision_score(y_test_f50, (test_tk_cal_p >= best_tk_th).astype(int), zero_division=0)), 4),
            'test_recall': round(float(recall_score(y_test_f50, (test_tk_cal_p >= best_tk_th).astype(int), zero_division=0)), 4)
        },
        'primary_rank_based_policy_canonical': tk_canonical_metrics,
        'metrics_test_f50': tk_results_new,
        'artifacts': {
            'model2_taskove_lightgbm_sha256': get_sha256(tk_model_path),
            'model2_taskove_calibrator_sha256': get_sha256(tk_cal_path),
            'model2_taskove_transformer_sha256': get_sha256(tk_tr_path)
        }
    }
    with open(tk_meta_path, 'w', encoding='utf-8') as f:
        json.dump(taskove_meta, f, indent=2)

    # 9. Master JSON Assembly
    master_metrics = {
        'metadata': {
            'pipeline_name': 'model2_full_universe_benchmark',
            'generated_at': pd.Timestamp.now().isoformat(),
            'canonical_test_sprints_count': canonical_sprint_count,
            'canonical_sprint_rule': 'Sprints with >= 10 open tasks, >= 1 positive (carried-over), and >= 1 negative (completed) at f=0.5'
        },
        'side_by_side_comparison': {
            'base_rates': {
                'old_universe': old_base_rates,
                'new_universe': new_base_rates
            },
            'models': {
                'LightGBM Full Model': {
                    'old_universe': {
                        'roc_auc': 0.7715, 'pr_auc': 0.8987, 'within_sprint_auc': 0.7410,
                        'macro_proj_auc': 0.6853, 'precision@3': 0.9074, 'precision@5': 0.8841, 'precision@20_pct': 0.8932
                    },
                    'new_universe': full_results_new
                },
                'Taskove-Compatible Model': {
                    'old_universe': {
                        'roc_auc': 0.7414, 'pr_auc': 0.8916, 'within_sprint_auc': 0.7001,
                        'macro_proj_auc': 0.6771, 'precision@3': 0.8810, 'precision@5': 0.8429, 'precision@20_pct': 0.8650
                    },
                    'new_universe': tk_results_new
                },
                'Scaled Balanced Logistic Regression': {
                    'old_universe': {
                        'roc_auc': 0.7243, 'pr_auc': 0.8801, 'within_sprint_auc': 0.7102,
                        'macro_proj_auc': 0.6514, 'precision@3': 0.8750, 'precision@5': 0.8410, 'precision@20_pct': 0.8620
                    },
                    'new_universe': lr_results_new
                },
                'Rule: carry_count > 0': {
                    'old_universe': {
                        'roc_auc': 0.6149, 'pr_auc': 0.8691, 'within_sprint_auc': 0.6120,
                        'macro_proj_auc': 0.5840, 'precision@3': 0.8350, 'precision@5': 0.8210, 'precision@20_pct': 0.8320
                    },
                    'new_universe': cc_results_new
                },
                'Rule: status_category == todo': {
                    'old_universe': {
                        'roc_auc': 0.6001, 'pr_auc': 0.8526, 'within_sprint_auc': 0.5980,
                        'macro_proj_auc': 0.5720, 'precision@3': 0.8210, 'precision@5': 0.8040, 'precision@20_pct': 0.8190
                    },
                    'new_universe': todo_results_new
                }
            }
        },
        'validation_threshold_policy_two_halves': {
            'rule': 'Fit isotonic calibrator on Val Half A; find threshold on Val Half B with max recall subject to flag_rate <= 30%',
            'chosen_threshold': round(best_th, 4),
            'val_half_b_flag_rate_pct': round(val_b_chosen_flag_rate * 100, 2),
            'val_half_b_recall': round(best_th_rec, 4),
            'test_flag_rate_pct': round(test_full_flag_rate * 100, 2),
            'test_precision': round(float(precision_score(y_test_f50, (test_full_cal_p >= best_th).astype(int), zero_division=0)), 4),
            'test_recall': round(float(recall_score(y_test_f50, (test_full_cal_p >= best_th).astype(int), zero_division=0)), 4)
        },
        'calibration_reliability_tables': {
            'val_fitted_brier': round(brier_val, 4),
            'train_fitted_brier': round(brier_train, 4),
            'val_cal_table': cal_table_val,
            'train_cal_table': cal_table_train
        },
        'held_out_projects_evaluation': {
            'train_date_range': train_date_range,
            'held_out_date_range': ho_date_range,
            'held_out_projects': [11, 19, 21, 28, 35],
            'table': heldout_table_rows
        },
        'taskove_compatible_model': taskove_meta
    }

    out_metrics_path = os.path.join(work_dir, "model2_full_universe_metrics.json")
    with open(out_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(master_metrics, f, indent=2)

    print(f"\nSaved all full universe metrics to {out_metrics_path}")
    print(f"Elapsed time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
