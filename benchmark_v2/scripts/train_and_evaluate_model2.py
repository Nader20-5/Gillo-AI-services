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
sys.path.insert(0, BASE_DIR)

from features_v2 import Model2FeatureTransformer

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
            'observed_rate': round(obs_rate, 4)
        })
    return table

def compute_sprint_precision_lift(df_sub, y_prob_series, top_k_list=[5, 10], min_tasks=10):
    """
    Computes precision@k and lift for sprints with >= min_tasks.
    """
    sprint_metrics = []
    df_eval = df_sub.copy()
    df_eval['y_prob'] = y_prob_series.values

    for sid, group in df_eval.groupby('sprint_id'):
        n_tasks = len(group)
        if n_tasks < min_tasks:
            continue
        tot_pos = group['label'].sum()
        sprint_base_rate = float(tot_pos / n_tasks)
        if sprint_base_rate == 0.0:
            continue

        sorted_group = group.sort_values('y_prob', ascending=False)
        rec = {
            'sprint_id': int(sid),
            'tasks': n_tasks,
            'base_rate': sprint_base_rate
        }
        for k in top_k_list:
            top_k = sorted_group.head(k)
            prec_k = float(top_k['label'].sum() / len(top_k))
            lift_k = float(prec_k / sprint_base_rate)
            rec[f'precision@{k}'] = prec_k
            rec[f'lift@{k}'] = lift_k

        sprint_metrics.append(rec)

    if not sprint_metrics:
        return {'eligible_sprints': 0}

    df_sp_res = pd.DataFrame(sprint_metrics)
    res = {
        'eligible_sprints': len(df_sp_res),
        'mean_sprint_base_rate': round(float(df_sp_res['base_rate'].mean()), 4)
    }
    for k in top_k_list:
        res[f'mean_precision@{k}'] = round(float(df_sp_res[f'precision@{k}'].mean()), 4)
        res[f'mean_lift@{k}'] = round(float(df_sp_res[f'lift@{k}'].mean()), 4)
    return res

def main():
    print("=== Training & Evaluating Model 2 Pipeline ===")
    t0 = time.time()
    work_dir = os.path.join(BASE_DIR, "work")
    data_path = os.path.join(work_dir, "model2_dataset.parquet")
    df = pd.read_parquet(data_path)
    print(f"Loaded dataset: {df.shape}")

    # Separate splits
    df_train = df[df['split'] == 'train'].copy()
    df_val = df[df['split'] == 'val'].copy()
    df_test = df[df['split'] == 'test'].copy()
    df_heldout = df[df['is_heldout_project'] == 1].copy()

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}, Held-out 5 Projs: {len(df_heldout)}")

    # 1. Feature Transformation (fit on train only)
    print("\nFitting Model2FeatureTransformer on Train only...")
    transformer = Model2FeatureTransformer()
    X_train = transformer.fit_transform(df_train)
    X_val = transformer.transform(df_val)
    X_test = transformer.transform(df_test)
    X_heldout = transformer.transform(df_heldout)

    y_train = df_train['label'].values
    y_val = df_val['label'].values
    y_test = df_test['label'].values
    y_heldout = df_heldout['label'].values

    feature_names = transformer.get_feature_names()
    print(f"Engineered {len(feature_names)} features. Sample: {feature_names[:8]}")

    # Focus slices at f = 0.5
    test_f50_mask = (df_test['f'] == 0.5).values
    val_f50_mask = (df_val['f'] == 0.5).values
    train_f50_mask = (df_train['f'] == 0.5).values
    heldout_f50_mask = (df_heldout['f'] == 0.5).values

    df_test_f50 = df_test[df_test['f'] == 0.5].copy()
    y_test_f50 = y_test[test_f50_mask]
    X_test_f50 = X_test[test_f50_mask]

    # 2. STEP 5: Evaluate Baselines on Test Set (at f=0.5 and test overall)
    print("\n=== STEP 5: Evaluating Baselines on Test Set ===")
    p_train_base = float(np.mean(y_train))
    p_train_f50_base = float(np.mean(y_train[train_f50_mask]))
    test_base_rate_f50 = float(np.mean(y_test_f50))
    test_brier_base_f50 = float(brier_score_loss(y_test_f50, np.full_like(y_test_f50, test_base_rate_f50, dtype=float)))

    baselines_results = {}

    def eval_baseline(name, preds_f50, preds_all):
        auc_f50 = float(roc_auc_score(y_test_f50, preds_f50))
        pr_f50 = pr_auc_score(y_test_f50, preds_f50)
        brier_f50 = float(brier_score_loss(y_test_f50, np.clip(preds_f50, 0.0, 1.0)))
        skill_f50 = float(1.0 - (brier_f50 / test_brier_base_f50))

        auc_all = float(roc_auc_score(y_test, preds_all))
        pr_all = pr_auc_score(y_test, preds_all)

        res = {
            'roc_auc_f50': round(auc_f50, 4),
            'pr_auc_f50': round(pr_f50, 4),
            'brier_f50': round(brier_f50, 4),
            'brier_skill_f50': round(skill_f50, 4),
            'roc_auc_all': round(auc_all, 4),
            'pr_auc_all': round(pr_all, 4)
        }
        baselines_results[name] = res
        print(f"  {name:40s} | f=0.5 AUC: {auc_f50:.4f} | PR-AUC: {pr_f50:.4f} | Brier: {brier_f50:.4f} | Skill: {skill_f50:.4f}")
        return res

    # 1. Base Rate
    eval_baseline("Base Rate (Train Prior)", np.full_like(y_test_f50, p_train_f50_base, dtype=float), np.full_like(y_test, p_train_base, dtype=float))

    # 2. Rule: status_category == todo
    todo_f50 = (df_test_f50['status_category'].str.lower() == 'todo').astype(float).values
    todo_all = (df_test['status_category'].str.lower() == 'todo').astype(float).values
    eval_baseline("Rule: status_category == 'todo'", todo_f50, todo_all)

    # 3. Rule: carry_count > 0
    cc_f50 = (df_test_f50['carry_count'] > 0).astype(float).values
    cc_all = (df_test['carry_count'] > 0).astype(float).values
    eval_baseline("Rule: carry_count > 0", cc_f50, cc_all)

    # 4. Project previous 3 carry rate
    proj_rate_f50 = X_test_f50['project_prev_3_carry_rate'].values
    proj_rate_all = X_test['project_prev_3_carry_rate'].values
    eval_baseline("Project Previous-3 Carry Rate", proj_rate_f50, proj_rate_all)

    # 5. Logistic Regression on SP only
    lr_sp = LogisticRegression()
    lr_sp.fit(X_train[['story_points', 'has_sp']], y_train)
    sp_f50 = lr_sp.predict_proba(X_test_f50[['story_points', 'has_sp']])[:, 1]
    sp_all = lr_sp.predict_proba(X_test[['story_points', 'has_sp']])[:, 1]
    eval_baseline("Logistic Regression (SP Only)", sp_f50, sp_all)

    # 6. Logistic Regression on fraction_points_done only
    lr_fd = LogisticRegression()
    lr_fd.fit(X_train[['fraction_points_done']], y_train)
    fd_f50 = lr_fd.predict_proba(X_test_f50[['fraction_points_done']])[:, 1]
    fd_all = lr_fd.predict_proba(X_test[['fraction_points_done']])[:, 1]
    eval_baseline("Logistic Regression (Points Done % Only)", fd_f50, fd_all)

    # 7. Scaled Balanced Logistic Regression on ALL features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_f50_sc = scaler.transform(X_test_f50)
    X_test_sc = scaler.transform(X_test)

    lr_all = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_all.fit(X_train_sc, y_train)
    lr_all_f50 = lr_all.predict_proba(X_test_f50_sc)[:, 1]
    lr_all_all = lr_all.predict_proba(X_test_sc)[:, 1]
    eval_baseline("Scaled Balanced LR (All Features)", lr_all_f50, lr_all_all)

    # 3. STEP 6: LightGBM Tuning on Validation Only
    print("\n=== STEP 6: LightGBM Tuning on Validation Only ===")
    grid = [
        {'num_leaves': nl, 'learning_rate': lr, 'min_child_samples': mcs}
        for nl in [15, 31, 63]
        for lr in [0.03, 0.05, 0.1]
        for mcs in [20, 50, 100]
    ]

    best_val_auc = -1.0
    best_params = None
    grid_results = []

    print(f"Searching over {len(grid)} hyperparameter candidates on Validation...")
    for idx, params in enumerate(grid):
        model_cand = lgb.LGBMClassifier(
            n_estimators=150,
            num_leaves=params['num_leaves'],
            learning_rate=params['learning_rate'],
            min_child_samples=params['min_child_samples'],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model_cand.fit(X_train, y_train)
        val_probs = model_cand.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_probs))
        val_f50_auc = float(roc_auc_score(y_val[val_f50_mask], val_probs[val_f50_mask]))

        rec = dict(params)
        rec['val_auc_all'] = round(val_auc, 4)
        rec['val_auc_f50'] = round(val_f50_auc, 4)
        grid_results.append(rec)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_params = params

    print(f"Best Hyperparameters: {best_params} | Best Val AUC: {best_val_auc:.4f}")

    # Fit best model
    best_lgb = lgb.LGBMClassifier(
        n_estimators=150,
        num_leaves=best_params['num_leaves'],
        learning_rate=best_params['learning_rate'],
        min_child_samples=best_params['min_child_samples'],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    best_lgb.fit(X_train, y_train)

    # Validation predictions
    val_raw_probs = best_lgb.predict_proba(X_val)[:, 1]

    # Select decision threshold on Validation only (maximizes F1)
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thresh = 0.5
    best_val_f1 = -1.0
    for th in thresholds:
        f1 = f1_score(y_val, (val_raw_probs >= th).astype(int))
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = float(th)

    print(f"Selected Validation Threshold: {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")

    # Fit probability calibrators on validation set
    print("\nFitting Probability Calibrators on Validation...")
    # Platt (Sigmoid)
    platt_lr = LogisticRegression()
    platt_lr.fit(val_raw_probs.reshape(-1, 1), y_val)
    val_platt_probs = platt_lr.predict_proba(val_raw_probs.reshape(-1, 1))[:, 1]

    # Isotonic
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_raw_probs, y_val)
    val_iso_probs = iso.predict(val_raw_probs)

    brier_val_uncal = float(brier_score_loss(y_val, val_raw_probs))
    brier_val_platt = float(brier_score_loss(y_val, val_platt_probs))
    brier_val_iso = float(brier_score_loss(y_val, val_iso_probs))

    print(f"Validation Brier - Uncalibrated: {brier_val_uncal:.4f} | Platt: {brier_val_platt:.4f} | Isotonic: {brier_val_iso:.4f}")

    if brier_val_iso <= brier_val_platt:
        chosen_calibrator_type = "isotonic"
        chosen_calibrator = iso
        val_cal_probs = val_iso_probs
    else:
        chosen_calibrator_type = "platt"
        chosen_calibrator = platt_lr
        val_cal_probs = val_platt_probs

    print(f"Chosen Calibrator: {chosen_calibrator_type}")

    # Calibration table on validation before and after
    val_cal_table_before = compute_10_bin_calibration(y_val, val_raw_probs)
    val_cal_table_after = compute_10_bin_calibration(y_val, val_cal_probs)

    # 4. STEP 7: Test Set Evaluation
    print("\n=== STEP 7: Comprehensive Evaluation on Test Set ===")
    test_raw_probs = best_lgb.predict_proba(X_test)[:, 1]
    if chosen_calibrator_type == "isotonic":
        test_cal_probs = chosen_calibrator.predict(test_raw_probs)
    else:
        test_cal_probs = chosen_calibrator.predict_proba(test_raw_probs.reshape(-1, 1))[:, 1]

    test_cal_probs_f50 = test_cal_probs[test_f50_mask]

    # Headline metrics at f = 0.5
    test_auc_f50 = float(roc_auc_score(y_test_f50, test_cal_probs_f50))
    test_pr_f50 = pr_auc_score(y_test_f50, test_cal_probs_f50)
    test_brier_f50 = float(brier_score_loss(y_test_f50, test_cal_probs_f50))
    test_bss_f50 = float(1.0 - (test_brier_f50 / test_brier_base_f50))

    # Confusion matrix at validation threshold
    test_preds_binary_f50 = (test_cal_probs_f50 >= best_thresh).astype(int)
    cm_f50 = confusion_matrix(y_test_f50, test_preds_binary_f50)
    tn_count, fp_count, fn_count, tp_count = [int(v) for v in cm_f50.ravel()]
    acc_f50 = float(accuracy_score(y_test_f50, test_preds_binary_f50))
    prec_f50 = float(precision_score(y_test_f50, test_preds_binary_f50))
    rec_f50 = float(recall_score(y_test_f50, test_preds_binary_f50))
    f1_f50 = float(f1_score(y_test_f50, test_preds_binary_f50))

    # 10-bin calibration table on test at f = 0.5
    test_cal_table_f50 = compute_10_bin_calibration(y_test_f50, test_cal_probs_f50)

    # Precision@5 and @10 Lift per sprint at f = 0.5
    sprint_lift_f50 = compute_sprint_precision_lift(df_test_f50, pd.Series(test_cal_probs_f50, index=df_test_f50.index))

    # Per-project AUC on test at f = 0.5 (top 10 projects)
    df_test_f50_eval = df_test_f50.copy()
    df_test_f50_eval['cal_prob'] = test_cal_probs_f50
    proj_auc_list = []
    top_10_test_pids = df_test_f50_eval['project_id'].value_counts().head(10).index.tolist()

    for pid in top_10_test_pids:
        sub_p = df_test_f50_eval[df_test_f50_eval['project_id'] == pid]
        if len(sub_p['label'].unique()) > 1:
            p_auc = float(roc_auc_score(sub_p['label'], sub_p['cal_prob']))
        else:
            p_auc = np.nan
        proj_auc_list.append({
            'project_id': int(pid),
            'tasks': len(sub_p),
            'base_rate': round(float(sub_p['label'].mean()), 4),
            'roc_auc': round(p_auc, 4) if not np.isnan(p_auc) else None
        })

    # Macro-average AUC across all test projects with >1 class
    all_proj_aucs = []
    for pid, sub_p in df_test_f50_eval.groupby('project_id'):
        if len(sub_p['label'].unique()) > 1:
            all_proj_aucs.append(float(roc_auc_score(sub_p['label'], sub_p['cal_prob'])))
    macro_avg_auc = float(np.mean(all_proj_aucs)) if all_proj_aucs else np.nan

    # Bootstrap 95% CIs for AUC and PR-AUC by resampling SPRINTS (1,000 draws, seed 42)
    print("\nRunning Sprint-Resampled Bootstrap (1000 draws, seed 42)...")
    unique_sprints_f50 = df_test_f50['sprint_id'].unique()
    np.random.seed(42)
    boot_aucs = []
    boot_praucs = []

    sprint_groups = {sid: group[['label']].copy() for sid, group in df_test_f50.groupby('sprint_id')}
    sprint_probs = {sid: test_cal_probs_f50[df_test_f50['sprint_id'] == sid] for sid in unique_sprints_f50}

    for b in range(1000):
        sampled_sids = np.random.choice(unique_sprints_f50, size=len(unique_sprints_f50), replace=True)
        b_labels = np.concatenate([sprint_groups[sid]['label'].values for sid in sampled_sids])
        b_probs = np.concatenate([sprint_probs[sid] for sid in sampled_sids])

        if len(np.unique(b_labels)) > 1:
            boot_aucs.append(roc_auc_score(b_labels, b_probs))
            boot_praucs.append(pr_auc_score(b_labels, b_probs))

    ci_auc_low, ci_auc_high = float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))
    ci_pr_low, ci_pr_high = float(np.percentile(boot_praucs, 2.5)), float(np.percentile(boot_praucs, 97.5))

    print(f"Headline at f=0.5: ROC-AUC: {test_auc_f50:.4f} (95% CI [{ci_auc_low:.4f}, {ci_auc_high:.4f}])")
    print(f"                   PR-AUC:  {test_pr_f50:.4f} (95% CI [{ci_pr_low:.4f}, {ci_pr_high:.4f}])")
    print(f"                   Brier:   {test_brier_f50:.4f} (Skill: {test_bss_f50:.4f})")

    # Feature Importances (Top 10)
    split_importances = best_lgb.booster_.feature_importance(importance_type='split')
    gain_importances = best_lgb.booster_.feature_importance(importance_type='gain')
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'split': split_importances,
        'gain': gain_importances
    }).sort_values('gain', ascending=False)
    top_10_features = feat_imp_df.head(10).to_dict('records')
    top_feature_name = feat_imp_df.iloc[0]['feature']
    print(f"\nTop feature by gain: {top_feature_name}")

    # Ablations
    print("\nRunning Ablation Studies on Test at f=0.5...")
    # 1. Drop top feature
    feats_no_top = [f for f in feature_names if f != top_feature_name]
    m_no_top = lgb.LGBMClassifier(**best_params, n_estimators=150, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m_no_top.fit(X_train[feats_no_top], y_train)
    auc_drop_top = float(roc_auc_score(y_test_f50, m_no_top.predict_proba(X_test_f50[feats_no_top])[:, 1]))

    # 2. Only task-level features
    task_feats = transformer.get_task_feature_names()
    m_task = lgb.LGBMClassifier(**best_params, n_estimators=150, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m_task.fit(X_train[task_feats], y_train)
    auc_task_only = float(roc_auc_score(y_test_f50, m_task.predict_proba(X_test_f50[task_feats])[:, 1]))

    # 3. Only team-level features
    team_feats = transformer.get_team_feature_names()
    m_team = lgb.LGBMClassifier(**best_params, n_estimators=150, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m_team.fit(X_train[team_feats], y_train)
    auc_team_only = float(roc_auc_score(y_test_f50, m_team.predict_proba(X_test_f50[team_feats])[:, 1]))

    # 4. Without carry_count
    feats_no_cc = [f for f in feature_names if f != 'carry_count']
    m_no_cc = lgb.LGBMClassifier(**best_params, n_estimators=150, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m_no_cc.fit(X_train[feats_no_cc], y_train)
    auc_no_carry = float(roc_auc_score(y_test_f50, m_no_cc.predict_proba(X_test_f50[feats_no_cc])[:, 1]))

    ablations = {
        'drop_top_feature': {'dropped': top_feature_name, 'test_auc': round(auc_drop_top, 4)},
        'task_only_features': {'feature_count': len(task_feats), 'test_auc': round(auc_task_only, 4)},
        'team_only_features': {'feature_count': len(team_feats), 'test_auc': round(auc_team_only, 4)},
        'without_carry_count': {'test_auc': round(auc_no_carry, 4)}
    }
    print(f"Ablations: Drop Top ({top_feature_name}): {auc_drop_top:.4f} | Task Only: {auc_task_only:.4f} | Team Only: {auc_team_only:.4f} | No carry_count: {auc_no_carry:.4f}")

    # Metrics per fraction f in {0.25, 0.5, 0.75}
    table_per_f = {}
    for f_val in [0.25, 0.5, 0.75]:
        f_mask = (df_test['f'] == f_val).values
        df_sub = df_test[f_mask]
        y_sub = y_test[f_mask]
        p_sub = test_cal_probs[f_mask]

        auc_f = float(roc_auc_score(y_sub, p_sub))
        pr_f = pr_auc_score(y_sub, p_sub)
        brier_f = float(brier_score_loss(y_sub, p_sub))
        b_base = float(brier_score_loss(y_sub, np.full_like(y_sub, np.mean(y_sub), dtype=float)))
        bss_f = float(1.0 - (brier_f / b_base))
        lift_res = compute_sprint_precision_lift(df_sub, pd.Series(p_sub, index=df_sub.index))

        table_per_f[str(f_val)] = {
            'tasks': len(y_sub),
            'sprints': df_sub['sprint_id'].nunique(),
            'base_rate': round(float(np.mean(y_sub)), 4),
            'roc_auc': round(auc_f, 4),
            'pr_auc': round(pr_f, 4),
            'brier': round(brier_f, 4),
            'brier_skill': round(bss_f, 4),
            'precision@5': lift_res.get('mean_precision@5'),
            'lift@5': lift_res.get('mean_lift@5'),
            'precision@10': lift_res.get('mean_precision@10'),
            'lift@10': lift_res.get('mean_lift@10')
        }

    # Held-out Projects Variant (f = 0.5)
    print("\nEvaluating on Held-Out 5 Projects (f = 0.5)...")
    heldout_f50_df = df_heldout[df_heldout['f'] == 0.5].copy()
    X_heldout_f50 = X_heldout[heldout_f50_mask]
    y_heldout_f50 = y_heldout[heldout_f50_mask]

    heldout_raw_probs = best_lgb.predict_proba(X_heldout_f50)[:, 1]
    if chosen_calibrator_type == "isotonic":
        heldout_cal_probs = chosen_calibrator.predict(heldout_raw_probs)
    else:
        heldout_cal_probs = chosen_calibrator.predict_proba(heldout_raw_probs.reshape(-1, 1))[:, 1]

    heldout_auc = float(roc_auc_score(y_heldout_f50, heldout_cal_probs))
    heldout_pr = pr_auc_score(y_heldout_f50, heldout_cal_probs)
    heldout_brier = float(brier_score_loss(y_heldout_f50, heldout_cal_probs))
    heldout_base = float(np.mean(y_heldout_f50))
    heldout_b_base = float(brier_score_loss(y_heldout_f50, np.full_like(y_heldout_f50, heldout_base, dtype=float)))
    heldout_bss = float(1.0 - (heldout_brier / heldout_b_base))
    heldout_lift = compute_sprint_precision_lift(heldout_f50_df, pd.Series(heldout_cal_probs, index=heldout_f50_df.index))

    heldout_headline = {
        'tasks': len(y_heldout_f50),
        'sprints': heldout_f50_df['sprint_id'].nunique(),
        'base_rate': round(heldout_base, 4),
        'roc_auc': round(heldout_auc, 4),
        'pr_auc': round(heldout_pr, 4),
        'brier': round(heldout_brier, 4),
        'brier_skill': round(heldout_bss, 4),
        'precision@5': heldout_lift.get('mean_precision@5'),
        'lift@5': heldout_lift.get('mean_lift@5')
    }
    print(f"Held-out Projects Headline: AUC: {heldout_auc:.4f} | PR-AUC: {heldout_pr:.4f} | Brier: {heldout_brier:.4f} | Skill: {heldout_bss:.4f}")

    # 5. STEP 8: Save Artifacts
    print("\n=== STEP 8: Saving Production Artifacts ===")
    artifact_dir = os.path.join(BASE_DIR, "artifacts", "model2", "v1_tawos")
    os.makedirs(artifact_dir, exist_ok=True)

    model_path = os.path.join(artifact_dir, "model2_lightgbm.pkl")
    calibrator_path = os.path.join(artifact_dir, "model2_calibrator.pkl")
    transformer_path = os.path.join(artifact_dir, "model2_transformer.pkl")

    with open(model_path, 'wb') as f_out:
        pickle.dump(best_lgb, f_out)
    with open(calibrator_path, 'wb') as f_out:
        pickle.dump({'type': chosen_calibrator_type, 'calibrator': chosen_calibrator}, f_out)
    with open(transformer_path, 'wb') as f_out:
        pickle.dump(transformer, f_out)

    # Compute sha256
    def get_sha256(filepath):
        h = hashlib.sha256()
        with open(filepath, 'rb') as f_in:
            while chunk := f_in.read(8192):
                h.update(chunk)
        return h.hexdigest()

    metadata = {
        'model_name': 'model2_sprint_carryover_risk',
        'version': 'v1_tawos',
        'trained_on': 'tawos',
        'training_dataset_sha256': get_sha256(data_path),
        'artifacts': {
            'model2_lightgbm_sha256': get_sha256(model_path),
            'model2_calibrator_sha256': get_sha256(calibrator_path),
            'model2_transformer_sha256': get_sha256(transformer_path)
        },
        'library_versions': {
            'python': sys.version.split()[0],
            'lightgbm': lgb.__version__,
            'scikit_learn': sklearn.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__
        },
        'best_hyperparameters': best_params,
        'decision_threshold': round(best_thresh, 4),
        'train_base_rate': round(p_train_base, 4),
        'test_base_rate_f50': round(test_base_rate_f50, 4),
        'calibrator_type': chosen_calibrator_type,
        'ordered_feature_list': feature_names,
        'feature_count': len(feature_names),
        'metrics_test_f50': {
            'roc_auc': round(test_auc_f50, 4),
            'pr_auc': round(test_pr_f50, 4),
            'brier_score': round(test_brier_f50, 4),
            'brier_skill_score': round(test_bss_f50, 4),
            'ci_95_roc_auc': [round(ci_auc_low, 4), round(ci_auc_high, 4)],
            'ci_95_pr_auc': [round(ci_pr_low, 4), round(ci_pr_high, 4)],
            'accuracy': round(acc_f50, 4),
            'precision': round(prec_f50, 4),
            'recall': round(rec_f50, 4),
            'f1_score': round(f1_f50, 4)
        }
    }

    meta_path = os.path.join(artifact_dir, "metadata.json")
    with open(meta_path, 'w') as f_out:
        json.dump(metadata, f_out, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Convert top_10_features types to python native types
    top_10_features_clean = []
    for r in top_10_features:
        top_10_features_clean.append({
            'feature': str(r['feature']),
            'split': int(r['split']),
            'gain': round(float(r['gain']), 2)
        })

    # Full Metrics JSON
    full_metrics = {
        'metadata': metadata,
        'baselines_test': baselines_results,
        'grid_search_validation': grid_results,
        'validation_threshold_search': {
            'best_threshold': round(best_thresh, 4),
            'best_val_f1': round(best_val_f1, 4)
        },
        'validation_calibration': {
            'uncalibrated_brier': round(brier_val_uncal, 4),
            'platt_brier': round(brier_val_platt, 4),
            'isotonic_brier': round(brier_val_iso, 4),
            'cal_table_before': val_cal_table_before,
            'cal_table_after': val_cal_table_after
        },
        'test_headline_f50': {
            'roc_auc': round(test_auc_f50, 4),
            'pr_auc': round(test_pr_f50, 4),
            'brier_score': round(test_brier_f50, 4),
            'brier_skill_score': round(test_bss_f50, 4),
            'bootstrap_ci_95_auc': [round(ci_auc_low, 4), round(ci_auc_high, 4)],
            'bootstrap_ci_95_prauc': [round(ci_pr_low, 4), round(ci_pr_high, 4)],
            'confusion_matrix': {
                'tn': tn_count, 'fp': fp_count, 'fn': fn_count, 'tp': tp_count,
                'accuracy': round(acc_f50, 4),
                'precision': round(prec_f50, 4),
                'recall': round(rec_f50, 4),
                'f1_score': round(f1_f50, 4)
            },
            'calibration_table': test_cal_table_f50,
            'sprint_lift': sprint_lift_f50,
            'top_10_projects_auc': proj_auc_list,
            'macro_avg_project_auc': round(macro_avg_auc, 4) if not np.isnan(macro_avg_auc) else None
        },
        'feature_importances_top10': top_10_features_clean,
        'ablations': ablations,
        'metrics_per_fraction': table_per_f,
        'heldout_projects_variant': heldout_headline,
        'execution_time_seconds': round(time.time() - t0, 2)
    }

    full_metrics_path = os.path.join(work_dir, "model2_bench_metrics.json")
    with open(full_metrics_path, 'w') as f_out:
        json.dump(full_metrics, f_out, indent=2)
    print(f"Saved full benchmark metrics to {full_metrics_path}")
    print(f"=== Model 2 Benchmark Pipeline Finished in {time.time()-t0:.1f}s ===")

if __name__ == "__main__":
    import sklearn
    main()
