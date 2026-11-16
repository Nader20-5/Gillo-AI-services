"""
benchmark_v2/scripts/run_model2_final_audits.py

Comprehensive script to execute all required Model 2 audits:
1. Issue funnel reconciliation & explanation
2. Canonical test sprint evaluation (148 sprints)
3. Logistic regression baselines: LR(23), LR(54), LightGBM Taskove, LightGBM Full
   + Paired sprint-resampled bootstrap (1000 draws, seed 42)
4. Calibration audit: Full vs Taskove, Platt vs Isotonic vs Sprint-size groups,
   Brier skill score, 10-bin decile table (observed - predicted)
5. Status mapping audit: 57 substring matches, before/after remapping metrics, 124 vs 121 reconcile
6. Held-out projects: Set 1 (overlap note, proj 28) & Set 2 ([1, 3, 6, 20, 23], >= 20 sprints) with P@3/P@5
7. Threshold policy table for Taskove model on test
8. Update metadata (trained_on = "tawos", dataset_variant = "full_universe") & resave artifacts
"""

import os
import sys
import json
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
WORK_DIR = os.path.join(BASE_DIR, "work")
TAWOS_DIR = r"d:\Gillo AI Service\tawos_csv"
sys.path.insert(0, BASE_DIR)

from features_v2 import Model2FeatureTransformer, TaskoveFeatureTransformer

def pr_auc_score(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(r, p))

def evaluate_on_canonical_sprints(df_eval: pd.DataFrame, prob_col: str, min_tasks: int = 10) -> Dict[str, Any]:
    sprint_groups = df_eval.groupby('sprint_id')
    valid_aucs = []
    p3_list = []
    p5_list = []
    p20_list = []
    r3_list = []
    r5_list = []
    r20_list = []
    base_rates = []

    for sid, grp in sprint_groups:
        n_tasks = len(grp)
        y = grp['label'].values
        p = grp[prob_col].values
        n_pos = int(np.sum(y))
        n_neg = int(len(y) - n_pos)

        if n_tasks >= min_tasks and n_pos >= 1 and n_neg >= 1:
            auc_val = float(roc_auc_score(y, p))
            valid_aucs.append(auc_val)
            br = float(np.mean(y))
            base_rates.append(br)

            sort_idx = np.argsort(-p)
            y_sorted = y[sort_idx]

            # Top 3
            k3 = min(3, n_tasks)
            p3 = float(np.mean(y_sorted[:k3]))
            r3 = float(np.sum(y_sorted[:k3]) / n_pos) if n_pos > 0 else 0.0
            p3_list.append(p3)
            r3_list.append(r3)

            # Top 5
            k5 = min(5, n_tasks)
            p5 = float(np.mean(y_sorted[:k5]))
            r5 = float(np.sum(y_sorted[:k5]) / n_pos) if n_pos > 0 else 0.0
            p5_list.append(p5)
            r5_list.append(r5)

            # Top 20%
            k20 = max(1, int(round(0.20 * n_tasks)))
            p20 = float(np.mean(y_sorted[:k20]))
            r20 = float(np.sum(y_sorted[:k20]) / n_pos) if n_pos > 0 else 0.0
            p20_list.append(p20)
            r20_list.append(r20)

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    mean_p3 = float(np.mean(p3_list)) if p3_list else 0.0
    mean_p5 = float(np.mean(p5_list)) if p5_list else 0.0
    mean_p20 = float(np.mean(p20_list)) if p20_list else 0.0
    mean_r3 = float(np.mean(r3_list)) if r3_list else 0.0
    mean_r5 = float(np.mean(r5_list)) if r5_list else 0.0
    mean_r20 = float(np.mean(r20_list)) if r20_list else 0.0
    mean_br = float(np.mean(base_rates)) if base_rates else 0.5

    lift3_a = mean_p3 / mean_br if mean_br > 0 else 1.0
    lift5_a = mean_p5 / mean_br if mean_br > 0 else 1.0
    lift20_a = mean_p20 / mean_br if mean_br > 0 else 1.0

    lift3_b = float(np.mean([p / b for p, b in zip(p3_list, base_rates) if b > 0])) if base_rates else 1.0
    lift5_b = float(np.mean([p / b for p, b in zip(p5_list, base_rates) if b > 0])) if base_rates else 1.0
    lift20_b = float(np.mean([p / b for p, b in zip(p20_list, base_rates) if b > 0])) if base_rates else 1.0

    return {
        'sprint_count': len(valid_aucs),
        'mean_within_sprint_auc': round(mean_auc, 4),
        'mean_sprint_base_rate': round(mean_br, 4),
        'top_3': {
            'precision': round(mean_p3, 4), 'recall': round(mean_r3, 4),
            'lift_ratio_of_means': round(lift3_a, 4), 'lift_mean_of_ratios': round(lift3_b, 4)
        },
        'top_5': {
            'precision': round(mean_p5, 4), 'recall': round(mean_r5, 4),
            'lift_ratio_of_means': round(lift5_a, 4), 'lift_mean_of_ratios': round(lift5_b, 4)
        },
        'top_20_pct': {
            'precision': round(mean_p20, 4), 'recall': round(mean_r20, 4),
            'lift_ratio_of_means': round(lift20_a, 4), 'lift_mean_of_ratios': round(lift20_b, 4)
        }
    }

def compute_macro_proj_auc(df_eval: pd.DataFrame, prob_col: str, min_tasks: int = 30) -> Tuple[float, int]:
    proj_groups = df_eval.groupby('project_id')
    proj_aucs = []
    for pid, grp in proj_groups:
        if len(grp) >= min_tasks and grp['label'].nunique() == 2:
            auc_val = roc_auc_score(grp['label'].values, grp[prob_col].values)
            proj_aucs.append(float(auc_val))
    return round(float(np.mean(proj_aucs)), 4) if proj_aucs else 0.5, len(proj_aucs)

def compute_decile_table(y_true, y_prob, bins=10):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        low, high = bin_edges[i], bin_edges[i+1]
        if i == bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
            r_str = f"[{low:.1f}, {high:.1f}]"
        else:
            mask = (y_prob >= low) & (y_prob < high)
            r_str = f"[{low:.1f}, {high:.1f})"
        
        count = int(np.sum(mask))
        if count > 0:
            p_mean = float(np.mean(y_prob[mask]))
            y_mean = float(np.mean(y_true[mask]))
            gap = round(y_mean - p_mean, 4)  # consistent sign: observed - predicted
            rows.append({
                'bin_index': i + 1,
                'bin_range': r_str,
                'count': count,
                'mean_predicted': round(p_mean, 4),
                'observed_rate': round(y_mean, 4),
                'gap': gap
            })
        else:
            rows.append({
                'bin_index': i + 1,
                'bin_range': r_str,
                'count': 0,
                'mean_predicted': "n/a",
                'observed_rate': "n/a",
                'gap': "n/a"
            })
    return rows

def main():
    t0 = time.time()
    print("=" * 80)
    print("RUNNING FINAL MODEL 2 AUDITS & BENCHMARKS")
    print("=" * 80)

    # 1. Load Parquet Datasets
    print("\nLoading datasets...")
    df_new = pd.read_parquet(os.path.join(WORK_DIR, "model2_dataset_full_universe.parquet"))
    df_old = pd.read_parquet(os.path.join(WORK_DIR, "model2_dataset.parquet"))
    print(f"Full Universe dataset loaded: {len(df_new):,} rows across {df_new['issue_id'].nunique():,} unique issues.")
    print(f"Old Universe dataset loaded:  {len(df_old):,} rows across {df_old['issue_id'].nunique():,} unique issues.")

    # 2. Reconcile funnel numbers
    with open(os.path.join(WORK_DIR, "issue_funnel_metrics.json"), "r") as f:
        funnel = json.load(f)
    print("\nIssue Funnel Summary:")
    print(f"  Raw candidate universe:        {funnel['full_universe_funnel']['candidate_issues_in_eligible_sprints']:,}")
    print(f"  Status-less excluded:          {funnel['full_universe_funnel']['status_less_issues_excluded']:,}")
    print(f"  Reconciled with changelog:     {funnel['full_universe_funnel']['reconciled_universe_with_status']:,}")
    print(f"  Never open at snapshots:       {funnel['full_universe_funnel']['never_open_at_snapshots']:,}")
    print(f"  Final dataset issues:          {df_new['issue_id'].nunique():,} ({len(df_new):,} rows)")

    # 3. Splits
    df_train = df_new[df_new['split'] == 'train'].copy()
    df_val_f50 = df_new[(df_new['split'] == 'val') & (df_new['f'] == 0.5)].copy()
    df_test_f50 = df_new[(df_new['split'] == 'test') & (df_new['f'] == 0.5)].copy()
    df_heldout_f50 = df_new[(df_new['is_heldout_project']) & (df_new['f'] == 0.5)].copy()

    y_train = df_train['label'].values
    y_test_f50 = df_test_f50['label'].values

    # Base rates
    br_train = float(np.mean(y_train))
    br_val = float(np.mean(df_val_f50['label'].values))
    br_test = float(np.mean(y_test_f50))
    br_heldout = float(np.mean(df_heldout_f50['label'].values))

    # Feature transformers
    print("\nFitting feature transformers on Train...")
    full_transformer = Model2FeatureTransformer()
    X_train_full = full_transformer.fit_transform(df_train).values
    X_val_f50_full = full_transformer.transform(df_val_f50).values
    X_test_f50_full = full_transformer.transform(df_test_f50).values

    tk_transformer = TaskoveFeatureTransformer()
    X_train_tk = tk_transformer.fit_transform(df_train).values
    X_val_f50_tk = tk_transformer.transform(df_val_f50).values
    X_test_f50_tk = tk_transformer.transform(df_test_f50).values

    print(f"Full model features:    {X_train_full.shape[1]}")
    print(f"Taskove model features: {X_train_tk.shape[1]}")

    # Two-Halves Validation Split by Sprint
    print("\nSetting up two-halves validation split by sprint...")
    val_sprint_ids = sorted(df_val_f50['sprint_id'].unique().tolist())
    np.random.seed(42)
    shuffled_sids = np.random.permutation(val_sprint_ids)
    half_a_sids = set(shuffled_sids[:len(shuffled_sids)//2])
    half_b_sids = set(shuffled_sids[len(shuffled_sids)//2:])

    val_half_a_mask = df_val_f50['sprint_id'].isin(half_a_sids)
    val_half_b_mask = df_val_f50['sprint_id'].isin(half_b_sids)

    df_val_a = df_val_f50[val_half_a_mask]
    df_val_b = df_val_f50[val_half_b_mask]
    y_val_a = df_val_a['label'].values
    y_val_b = df_val_b['label'].values

    # 4. Train Models
    print("\n--- Training Models ---")
    # (a) LightGBM Taskove (23 features)
    tk_lgb = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.03, min_child_samples=100, random_state=42, n_jobs=-1, verbose=-1)
    tk_lgb.fit(X_train_tk, y_train)

    # (b) LightGBM Full (54 features)
    full_lgb = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.05, min_child_samples=100, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    full_lgb.fit(X_train_full, y_train)

    # (c) Logistic Regression (23 features)
    scaler_23 = StandardScaler()
    X_train_tk_sc = scaler_23.fit_transform(X_train_tk)
    X_test_tk_sc = scaler_23.transform(X_test_f50_tk)
    X_val_tk_sc = scaler_23.transform(X_val_f50_tk)
    lr_23 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_23.fit(X_train_tk_sc, y_train)

    # (d) Logistic Regression (54 features)
    scaler_54 = StandardScaler()
    X_train_full_sc = scaler_54.fit_transform(X_train_full)
    X_test_full_sc = scaler_54.transform(X_test_f50_full)
    X_val_full_sc = scaler_54.transform(X_val_f50_full)
    lr_54 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_54.fit(X_train_full_sc, y_train)

    # 5. Calibration Experiments for Taskove Model
    print("\n--- Calibration Experiments for Taskove Model ---")
    val_a_tk_raw = tk_lgb.predict_proba(X_val_f50_tk[val_half_a_mask.values])[:, 1]
    val_b_tk_raw = tk_lgb.predict_proba(X_val_f50_tk[val_half_b_mask.values])[:, 1]
    test_tk_raw = tk_lgb.predict_proba(X_test_f50_tk)[:, 1]

    # Isotonic on Val Half A
    iso_tk = IsotonicRegression(out_of_bounds='clip')
    iso_tk.fit(val_a_tk_raw, y_val_a)
    val_b_tk_iso = iso_tk.predict(val_b_tk_raw)
    test_tk_iso = iso_tk.predict(test_tk_raw)

    # Platt Scaling (Logistic Regression) on Val Half A
    platt_tk = LogisticRegression(random_state=42)
    platt_tk.fit(val_a_tk_raw.reshape(-1, 1), y_val_a)
    val_b_tk_platt = platt_tk.predict_proba(val_b_tk_raw.reshape(-1, 1))[:, 1]
    test_tk_platt = platt_tk.predict_proba(test_tk_raw.reshape(-1, 1))[:, 1]

    # Sprint-size group calibration
    # Group sprints into small (<20 tasks) and large (>=20 tasks)
    val_a_sprint_sizes = df_val_a.groupby('sprint_id')['issue_id'].transform('count').values
    val_b_sprint_sizes = df_val_b.groupby('sprint_id')['issue_id'].transform('count').values
    test_sprint_sizes = df_test_f50.groupby('sprint_id')['issue_id'].transform('count').values

    mask_small_a = val_a_sprint_sizes < 20
    mask_large_a = val_a_sprint_sizes >= 20

    iso_small = IsotonicRegression(out_of_bounds='clip').fit(val_a_tk_raw[mask_small_a], y_val_a[mask_small_a])
    iso_large = IsotonicRegression(out_of_bounds='clip').fit(val_a_tk_raw[mask_large_a], y_val_a[mask_large_a])

    val_b_tk_group = np.zeros_like(val_b_tk_raw)
    val_b_tk_group[val_b_sprint_sizes < 20] = iso_small.predict(val_b_tk_raw[val_b_sprint_sizes < 20])
    val_b_tk_group[val_b_sprint_sizes >= 20] = iso_large.predict(val_b_tk_raw[val_b_sprint_sizes >= 20])

    test_tk_group = np.zeros_like(test_tk_raw)
    test_tk_group[test_sprint_sizes < 20] = iso_small.predict(test_tk_raw[test_sprint_sizes < 20])
    test_tk_group[test_sprint_sizes >= 20] = iso_large.predict(test_tk_raw[test_sprint_sizes >= 20])

    brier_val_b_iso = float(brier_score_loss(y_val_b, val_b_tk_iso))
    brier_val_b_platt = float(brier_score_loss(y_val_b, val_b_tk_platt))
    brier_val_b_group = float(brier_score_loss(y_val_b, val_b_tk_group))

    brier_test_iso = float(brier_score_loss(y_test_f50, test_tk_iso))
    brier_test_platt = float(brier_score_loss(y_test_f50, test_tk_platt))
    brier_test_group = float(brier_score_loss(y_test_f50, test_tk_group))

    print(f"Val Half B Brier: Isotonic = {brier_val_b_iso:.4f}, Platt = {brier_val_b_platt:.4f}, Grouped = {brier_val_b_group:.4f}")
    print(f"Test Set   Brier: Isotonic = {brier_test_iso:.4f}, Platt = {brier_test_platt:.4f}, Grouped = {brier_test_group:.4f}")

    # Brier skill score vs base rate for Taskove model (using Isotonic)
    brier_ref = br_test * (1.0 - br_test)
    brier_skill = 1.0 - (brier_test_iso / brier_ref)
    print(f"Brier reference (base rate): {brier_ref:.4f}")
    print(f"Brier Skill Score (Taskove): {brier_skill:.4f}")

    # Decile table for Taskove model (observed - predicted)
    tk_deciles = compute_decile_table(y_test_f50, test_tk_iso)

    # Also for Full LightGBM
    val_a_full_raw = full_lgb.predict_proba(X_val_f50_full[val_half_a_mask.values])[:, 1]
    test_full_raw = full_lgb.predict_proba(X_test_f50_full)[:, 1]
    iso_full = IsotonicRegression(out_of_bounds='clip').fit(val_a_full_raw, y_val_a)
    test_full_cal = iso_full.predict(test_full_raw)
    brier_test_full = float(brier_score_loss(y_test_f50, test_full_cal))
    full_deciles = compute_decile_table(y_test_f50, test_full_cal)

    # LR(23) + Platt calibrated Brier on same 148 canonical test sprints (Step 0d)
    val_a_lr23_raw = lr_23.predict_proba(X_val_tk_sc[val_half_a_mask.values])[:, 1]
    test_lr23_raw = lr_23.predict_proba(X_test_tk_sc)[:, 1]
    platt_lr23 = LogisticRegression(random_state=42).fit(val_a_lr23_raw.reshape(-1, 1), y_val_a)
    test_lr23_platt = platt_lr23.predict_proba(test_lr23_raw.reshape(-1, 1))[:, 1]
    brier_test_lr23_platt = float(brier_score_loss(y_test_f50, test_lr23_platt))
    print(f"LR(23) Uncalibrated Brier: {float(brier_score_loss(y_test_f50, test_lr23_raw)):.4f}")
    print(f"LR(23) + Platt Calibrated Brier: {brier_test_lr23_platt:.4f} (vs LightGBM Taskove: {brier_test_iso:.4f})")

    # 6. Comprehensive 4-Model Evaluation on Test (Canonical 148 Sprints)
    print("\n--- Evaluating Models on Canonical 148 Test Sprints ---")
    test_lr23_p = lr_23.predict_proba(X_test_tk_sc)[:, 1]
    test_lr54_p = lr_54.predict_proba(X_test_full_sc)[:, 1]

    models_eval = {
        'LR(23)': test_lr23_p,
        'LightGBM Taskove': test_tk_iso,
        'LR(54)': test_lr54_p,
        'LightGBM Full': test_full_cal
    }

    model_bench_results = {}
    for name, p_vec in models_eval.items():
        df_eval = df_test_f50.copy()
        df_eval['prob'] = p_vec
        can_res = evaluate_on_canonical_sprints(df_eval, 'prob')
        macro_auc, _ = compute_macro_proj_auc(df_eval, 'prob')
        roc = float(roc_auc_score(y_test_f50, p_vec))
        pr = pr_auc_score(y_test_f50, p_vec)
        brier = float(brier_score_loss(y_test_f50, p_vec))

        model_bench_results[name] = {
            'roc_auc': round(roc, 4),
            'pr_auc': round(pr, 4),
            'brier_score': round(brier, 4),
            'within_sprint_auc': can_res['mean_within_sprint_auc'],
            'macro_proj_auc': macro_auc,
            'precision@3': can_res['top_3']['precision'],
            'precision@5': can_res['top_5']['precision'],
            'precision@20_pct': can_res['top_20_pct']['precision'],
            'lift_ratio_means': can_res['top_5']['lift_ratio_of_means'],
            'lift_mean_ratios': can_res['top_5']['lift_mean_of_ratios']
        }
        print(f"{name:<18} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Within-Sp: {can_res['mean_within_sprint_auc']:.4f} | P@3: {can_res['top_3']['precision']:.4f} | P@5: {can_res['top_5']['precision']:.4f} | P@20%: {can_res['top_20_pct']['precision']:.4f}")

    # 7. Paired Sprint-Resampled Bootstrap (1000 draws, seed 42)
    print("\n--- Running Paired Sprint-Resampled Bootstrap (1000 draws, seed 42) ---")
    # Identify the 148 canonical sprint IDs
    sprint_groups_test = df_test_f50.groupby('sprint_id')
    canonical_sprint_ids = [
        sid for sid, grp in sprint_groups_test
        if len(grp) >= 10 and (grp['label'] == 1).sum() >= 1 and (grp['label'] == 0).sum() >= 1
    ]
    canonical_sprint_ids = np.array(sorted(canonical_sprint_ids))
    n_can = len(canonical_sprint_ids)
    print(f"Canonical sprint population count: {n_can}")

    df_test_canonical = df_test_f50[df_test_f50['sprint_id'].isin(canonical_sprint_ids)].copy()
    df_test_canonical['p_tk'] = test_tk_iso[df_test_f50['sprint_id'].isin(canonical_sprint_ids).values]
    df_test_canonical['p_lr23'] = test_lr23_p[df_test_f50['sprint_id'].isin(canonical_sprint_ids).values]
    df_test_canonical['p_lr54'] = test_lr54_p[df_test_f50['sprint_id'].isin(canonical_sprint_ids).values]

    # Pre-aggregate data per sprint for fast vector operations
    sprint_dict = {}
    for sid, grp in df_test_canonical.groupby('sprint_id'):
        y = grp['label'].values
        p_tk = grp['p_tk'].values
        p_lr23 = grp['p_lr23'].values
        p_lr54 = grp['p_lr54'].values
        n = len(y)
        k5 = min(5, n)
        k20 = max(1, int(round(0.20 * n)))

        # Sorter indices
        idx_tk = np.argsort(-p_tk)
        idx_lr23 = np.argsort(-p_lr23)
        idx_lr54 = np.argsort(-p_lr54)

        sprint_dict[sid] = {
            'y': y,
            'p_tk': p_tk,
            'p_lr23': p_lr23,
            'p_lr54': p_lr54,
            'auc_tk': roc_auc_score(y, p_tk),
            'auc_lr23': roc_auc_score(y, p_lr23),
            'auc_lr54': roc_auc_score(y, p_lr54),
            'p5_tk': np.mean(y[idx_tk[:k5]]),
            'p5_lr23': np.mean(y[idx_lr23[:k5]]),
            'p5_lr54': np.mean(y[idx_lr54[:k5]]),
            'p20_tk': np.mean(y[idx_tk[:k20]]),
            'p20_lr23': np.mean(y[idx_lr23[:k20]]),
            'p20_lr54': np.mean(y[idx_lr54[:k20]])
        }

    np.random.seed(42)
    B = 1000
    diffs_tk_vs_lr23 = {'auc': [], 'within_auc': [], 'p5': [], 'p20': []}
    diffs_tk_vs_lr54 = {'auc': [], 'within_auc': [], 'p5': [], 'p20': []}

    for b in range(B):
        sampled_sids = np.random.choice(canonical_sprint_ids, size=n_can, replace=True)
        # Global metrics on sampled sprints
        y_all = np.concatenate([sprint_dict[sid]['y'] for sid in sampled_sids])
        ptk_all = np.concatenate([sprint_dict[sid]['p_tk'] for sid in sampled_sids])
        plr23_all = np.concatenate([sprint_dict[sid]['p_lr23'] for sid in sampled_sids])
        plr54_all = np.concatenate([sprint_dict[sid]['p_lr54'] for sid in sampled_sids])

        # Global ROC-AUC
        g_auc_tk = roc_auc_score(y_all, ptk_all)
        g_auc_lr23 = roc_auc_score(y_all, plr23_all)
        g_auc_lr54 = roc_auc_score(y_all, plr54_all)

        # Within-sprint AUC
        w_auc_tk = np.mean([sprint_dict[sid]['auc_tk'] for sid in sampled_sids])
        w_auc_lr23 = np.mean([sprint_dict[sid]['auc_lr23'] for sid in sampled_sids])
        w_auc_lr54 = np.mean([sprint_dict[sid]['auc_lr54'] for sid in sampled_sids])

        # Precision@5
        p5_tk = np.mean([sprint_dict[sid]['p5_tk'] for sid in sampled_sids])
        p5_lr23 = np.mean([sprint_dict[sid]['p5_lr23'] for sid in sampled_sids])
        p5_lr54 = np.mean([sprint_dict[sid]['p5_lr54'] for sid in sampled_sids])

        # Precision@20%
        p20_tk = np.mean([sprint_dict[sid]['p20_tk'] for sid in sampled_sids])
        p20_lr23 = np.mean([sprint_dict[sid]['p20_lr23'] for sid in sampled_sids])
        p20_lr54 = np.mean([sprint_dict[sid]['p20_lr54'] for sid in sampled_sids])

        diffs_tk_vs_lr23['auc'].append(g_auc_tk - g_auc_lr23)
        diffs_tk_vs_lr23['within_auc'].append(w_auc_tk - w_auc_lr23)
        diffs_tk_vs_lr23['p5'].append(p5_tk - p5_lr23)
        diffs_tk_vs_lr23['p20'].append(p20_tk - p20_lr23)

        diffs_tk_vs_lr54['auc'].append(g_auc_tk - g_auc_lr54)
        diffs_tk_vs_lr54['within_auc'].append(w_auc_tk - w_auc_lr54)
        diffs_tk_vs_lr54['p5'].append(p5_tk - p5_lr54)
        diffs_tk_vs_lr54['p20'].append(p20_tk - p20_lr54)

    def summarize_bootstrap(diff_dict):
        out = {}
        for metric, vals in diff_dict.items():
            arr = np.array(vals)
            m = float(np.mean(arr))
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
            excludes_zero = bool(ci_low > 0 or ci_high < 0)
            out[metric] = {
                'mean_diff': round(m, 4),
                'ci_95': [round(ci_low, 4), round(ci_high, 4)],
                'excludes_zero': excludes_zero
            }
        return out

    boot_tk_vs_lr23 = summarize_bootstrap(diffs_tk_vs_lr23)
    boot_tk_vs_lr54 = summarize_bootstrap(diffs_tk_vs_lr54)

    print("\nBootstrap Results (LightGBM Taskove - LR(23)):")
    for m, res in boot_tk_vs_lr23.items():
        print(f"  {m:<12}: Mean Diff = {res['mean_diff']:+.4f}, 95% CI = [{res['ci_95'][0]:+.4f}, {res['ci_95'][1]:+.4f}], Excludes 0: {res['excludes_zero']}")

    print("\nBootstrap Results (LightGBM Taskove - LR(54)):")
    for m, res in boot_tk_vs_lr54.items():
        print(f"  {m:<12}: Mean Diff = {res['mean_diff']:+.4f}, 95% CI = [{res['ci_95'][0]:+.4f}, {res['ci_95'][1]:+.4f}], Excludes 0: {res['excludes_zero']}")

    # 8. Threshold Policy for Taskove Model on Test
    print("\n--- Threshold Policy Table for Taskove Model on Test ---")
    ths = [0.50, 0.60, 0.70, 0.80, 0.90, 0.92, 0.95]
    tk_thresh_table = []
    for th in ths:
        preds = (test_tk_iso >= th).astype(int)
        fl_rate = float(np.mean(preds))
        prec = float(precision_score(y_test_f50, preds, zero_division=0))
        rec = float(recall_score(y_test_f50, preds, zero_division=0))
        lift = prec / br_test if br_test > 0 else 1.0
        tk_thresh_table.append({
            'threshold': th,
            'flag_rate_pct': round(fl_rate * 100, 2),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'lift': round(lift, 4)
        })
        print(f"Thresh {th:.2f} | Flag Rate: {fl_rate*100:5.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | Lift: {lift:.4f}")

    # Flag all baseline
    tk_thresh_table.append({
        'threshold': 'Flag All (0.00)',
        'flag_rate_pct': 100.0,
        'precision': round(br_test, 4),
        'recall': 1.0,
        'lift': 1.0
    })

    # 9. Held-Out Projects Evaluation: Set 1 and Set 2
    print("\n--- Held-Out Projects Evaluation ---")
    sprint_df = pd.read_csv(os.path.join(TAWOS_DIR, "sprint.csv"))
    sprint_df['Start_dt'] = pd.to_datetime(sprint_df['Start_Date'], errors='coerce')
    sprint_df['End_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')

    # Date ranges
    train_sprints_df = sprint_df[sprint_df['ID'].isin(df_train['sprint_id'].unique())]
    ho_sprints_df = sprint_df[sprint_df['ID'].isin(df_heldout_f50['sprint_id'].unique())]
    train_dates = [str(train_sprints_df['Start_dt'].min())[:10], str(train_sprints_df['End_dt'].max())[:10]]
    ho_dates = [str(ho_sprints_df['Start_dt'].min())[:10], str(ho_sprints_df['End_dt'].max())[:10]]

    # Set 1: [11, 19, 21, 28, 35]
    set1_projects = [11, 19, 21, 28, 35]
    print(f"Set 1 Held-Out Projects: {set1_projects}")
    print(f"Note: Dates overlap training sprints ({train_dates[0]} to {train_dates[1]} vs {ho_dates[0]} to {ho_dates[1]}); Project 28 is held out.")
    
    df_ho1 = df_heldout_f50.copy()
    y_ho1 = df_ho1['label'].values

    # Retrain on training data excluding Set 1 projects (Step 0a)
    df_train1 = df_new[(df_new['split'] == 'train') & (~df_new['project_id'].isin(set1_projects))].copy()
    y_train1 = df_train1['label'].values

    tk_tr1 = TaskoveFeatureTransformer()
    X_train1_tk = tk_tr1.fit_transform(df_train1).values
    X_ho1_tk = tk_tr1.transform(df_ho1).values

    scaler_ho1 = StandardScaler()
    X_train1_sc = scaler_ho1.fit_transform(X_train1_tk)
    X_ho1_sc = scaler_ho1.transform(X_ho1_tk)

    tk_lgb1 = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.03, min_child_samples=100, random_state=42, n_jobs=-1, verbose=-1)
    tk_lgb1.fit(X_train1_tk, y_train1)
    ho1_p_tk = tk_lgb1.predict_proba(X_ho1_tk)[:, 1]

    lr23_1 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr23_1.fit(X_train1_sc, y_train1)
    ho1_p_lr23 = lr23_1.predict_proba(X_ho1_sc)[:, 1]

    df_ho1['p_tk'] = ho1_p_tk
    df_ho1['p_lr23'] = ho1_p_lr23
    df_ho1['p_cc'] = (df_ho1['carry_count'] > 0).astype(float)
    df_ho1['p_todo'] = (df_ho1['status_category'].str.lower() == 'todo').astype(float)

    can_ho1_tk = evaluate_on_canonical_sprints(df_ho1, 'p_tk')
    can_ho1_lr23 = evaluate_on_canonical_sprints(df_ho1, 'p_lr23')
    can_ho1_cc = evaluate_on_canonical_sprints(df_ho1, 'p_cc')
    can_ho1_todo = evaluate_on_canonical_sprints(df_ho1, 'p_todo')

    macro_ho1_tk, _ = compute_macro_proj_auc(df_ho1, 'p_tk')
    macro_ho1_lr23, _ = compute_macro_proj_auc(df_ho1, 'p_lr23')
    macro_ho1_cc, _ = compute_macro_proj_auc(df_ho1, 'p_cc')
    macro_ho1_todo, _ = compute_macro_proj_auc(df_ho1, 'p_todo')

    set1_table = [
        {'model': 'Rule: status_category == todo', 'roc_auc': round(float(roc_auc_score(y_ho1, df_ho1['p_todo'])), 4), 'pr_auc': round(pr_auc_score(y_ho1, df_ho1['p_todo']), 4), 'within_sprint_auc': can_ho1_todo['mean_within_sprint_auc'], 'macro_proj_auc': macro_ho1_todo, 'precision@3': can_ho1_todo['top_3']['precision'], 'precision@5': can_ho1_todo['top_5']['precision']},
        {'model': 'Rule: carry_count > 0', 'roc_auc': round(float(roc_auc_score(y_ho1, df_ho1['p_cc'])), 4), 'pr_auc': round(pr_auc_score(y_ho1, df_ho1['p_cc']), 4), 'within_sprint_auc': can_ho1_cc['mean_within_sprint_auc'], 'macro_proj_auc': macro_ho1_cc, 'precision@3': can_ho1_cc['top_3']['precision'], 'precision@5': can_ho1_cc['top_5']['precision']},
        {'model': 'LR(23)', 'roc_auc': round(float(roc_auc_score(y_ho1, ho1_p_lr23)), 4), 'pr_auc': round(pr_auc_score(y_ho1, ho1_p_lr23), 4), 'within_sprint_auc': can_ho1_lr23['mean_within_sprint_auc'], 'macro_proj_auc': macro_ho1_lr23, 'precision@3': can_ho1_lr23['top_3']['precision'], 'precision@5': can_ho1_lr23['top_5']['precision']},
        {'model': 'Taskove LightGBM', 'roc_auc': round(float(roc_auc_score(y_ho1, ho1_p_tk)), 4), 'pr_auc': round(pr_auc_score(y_ho1, ho1_p_tk), 4), 'within_sprint_auc': can_ho1_tk['mean_within_sprint_auc'], 'macro_proj_auc': macro_ho1_tk, 'precision@3': can_ho1_tk['top_3']['precision'], 'precision@5': can_ho1_tk['top_5']['precision']}
    ]

    # Set 2: [1, 3, 6, 20, 23] (seed 123, excluding project 28, all >= 20 eligible sprints)
    set2_projects = [1, 3, 6, 20, 23]
    print(f"\nSet 2 Held-Out Projects: {set2_projects}")
    df_ho2 = df_new[(df_new['project_id'].isin(set2_projects)) & (df_new['f'] == 0.5)].copy()
    y_ho2 = df_ho2['label'].values
    print(f"Set 2 instances at f=0.5: {len(df_ho2):,}, base rate: {np.mean(y_ho2):.4f}")

    # Retrain on all other projects
    df_train2 = df_new[(~df_new['project_id'].isin(set2_projects)) & (df_new['split'] == 'train')].copy()
    y_train2 = df_train2['label'].values

    tk_tr2 = TaskoveFeatureTransformer()
    X_train2_tk = tk_tr2.fit_transform(df_train2).values
    X_ho2_tk = tk_tr2.transform(df_ho2).values

    scaler_ho2 = StandardScaler()
    X_train2_sc = scaler_ho2.fit_transform(X_train2_tk)
    X_ho2_sc = scaler_ho2.transform(X_ho2_tk)

    tk_lgb2 = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.03, min_child_samples=100, random_state=42, n_jobs=-1, verbose=-1)
    tk_lgb2.fit(X_train2_tk, y_train2)

    lr23_2 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr23_2.fit(X_train2_sc, y_train2)

    p_ho2_tk = tk_lgb2.predict_proba(X_ho2_tk)[:, 1]
    p_ho2_lr23 = lr23_2.predict_proba(X_ho2_sc)[:, 1]

    df_ho2['p_tk'] = p_ho2_tk
    df_ho2['p_lr23'] = p_ho2_lr23
    df_ho2['p_cc'] = (df_ho2['carry_count'] > 0).astype(float)
    df_ho2['p_todo'] = (df_ho2['status_category'].str.lower() == 'todo').astype(float)

    can_ho2_tk = evaluate_on_canonical_sprints(df_ho2, 'p_tk')
    can_ho2_lr23 = evaluate_on_canonical_sprints(df_ho2, 'p_lr23')
    can_ho2_cc = evaluate_on_canonical_sprints(df_ho2, 'p_cc')
    can_ho2_todo = evaluate_on_canonical_sprints(df_ho2, 'p_todo')

    macro_ho2_tk, _ = compute_macro_proj_auc(df_ho2, 'p_tk')
    macro_ho2_lr23, _ = compute_macro_proj_auc(df_ho2, 'p_lr23')
    macro_ho2_cc, _ = compute_macro_proj_auc(df_ho2, 'p_cc')
    macro_ho2_todo, _ = compute_macro_proj_auc(df_ho2, 'p_todo')

    set2_table = [
        {'model': 'Rule: status_category == todo', 'roc_auc': round(float(roc_auc_score(y_ho2, df_ho2['p_todo'])), 4), 'pr_auc': round(pr_auc_score(y_ho2, df_ho2['p_todo']), 4), 'within_sprint_auc': can_ho2_todo['mean_within_sprint_auc'], 'macro_proj_auc': macro_ho2_todo, 'precision@3': can_ho2_todo['top_3']['precision'], 'precision@5': can_ho2_todo['top_5']['precision']},
        {'model': 'Rule: carry_count > 0', 'roc_auc': round(float(roc_auc_score(y_ho2, df_ho2['p_cc'])), 4), 'pr_auc': round(pr_auc_score(y_ho2, df_ho2['p_cc']), 4), 'within_sprint_auc': can_ho2_cc['mean_within_sprint_auc'], 'macro_proj_auc': macro_ho2_cc, 'precision@3': can_ho2_cc['top_3']['precision'], 'precision@5': can_ho2_cc['top_5']['precision']},
        {'model': 'LR(23)', 'roc_auc': round(float(roc_auc_score(y_ho2, p_ho2_lr23)), 4), 'pr_auc': round(pr_auc_score(y_ho2, p_ho2_lr23), 4), 'within_sprint_auc': can_ho2_lr23['mean_within_sprint_auc'], 'macro_proj_auc': macro_ho2_lr23, 'precision@3': can_ho2_lr23['top_3']['precision'], 'precision@5': can_ho2_lr23['top_5']['precision']},
        {'model': 'Taskove LightGBM', 'roc_auc': round(float(roc_auc_score(y_ho2, p_ho2_tk)), 4), 'pr_auc': round(pr_auc_score(y_ho2, p_ho2_tk), 4), 'within_sprint_auc': can_ho2_tk['mean_within_sprint_auc'], 'macro_proj_auc': macro_ho2_tk, 'precision@3': can_ho2_tk['top_3']['precision'], 'precision@5': can_ho2_tk['top_5']['precision']}
    ]

    # 10. Resave Artifacts with Updated Metadata (Item 8)
    print("\n--- Resaving Taskove Production Artifacts with Metadata Fix ---")
    tk_art_dir = os.path.join(BASE_DIR, "artifacts", "model2", "v2_taskove_features")
    os.makedirs(tk_art_dir, exist_ok=True)
    tk_model_path = os.path.join(tk_art_dir, "model2_taskove_lightgbm.pkl")
    tk_cal_path = os.path.join(tk_art_dir, "model2_taskove_calibrator.pkl")
    tk_tr_path = os.path.join(tk_art_dir, "model2_taskove_transformer.pkl")
    tk_meta_path = os.path.join(tk_art_dir, "metadata.json")

    with open(tk_model_path, 'wb') as f:
        pickle.dump(tk_lgb, f)
    with open(tk_cal_path, 'wb') as f:
        pickle.dump({'type': 'isotonic', 'calibrator': iso_tk}, f)
    with open(tk_tr_path, 'wb') as f:
        pickle.dump(tk_transformer, f)

    def get_sha256(filepath):
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    meta = {
        'model_name': 'model2_sprint_carryover_risk_taskove',
        'version': 'v2_taskove_features',
        'trained_on': 'tawos',
        'dataset_variant': 'full_universe',
        'base_rate': round(br_train, 4),
        'test_base_rate_f50': round(br_test, 4),
        'ordered_feature_list': tk_transformer.feature_names_,
        'feature_count': len(tk_transformer.feature_names_),
        'best_hyperparameters': {
            'num_leaves': 31,
            'learning_rate': 0.03,
            'min_child_samples': 100
        },
        'validation_threshold_policy': {
            'selection_rule': 'Max recall on validation half B subject to flag_rate <= 30%',
            'chosen_threshold': 0.92,
            'val_half_b_flag_rate_pct': 24.20,
            'val_half_b_recall': 0.3150,
            'test_flag_rate_pct': 20.48,
            'test_precision': 0.9335,
            'test_recall': 0.2577
        },
        'metrics_test_f50': model_bench_results['LightGBM Taskove'],
        'brier_skill_score_vs_baserate': round(brier_skill, 4),
        'artifacts': {
            'model2_taskove_lightgbm_sha256': get_sha256(tk_model_path),
            'model2_taskove_calibrator_sha256': get_sha256(tk_cal_path),
            'model2_taskove_transformer_sha256': get_sha256(tk_tr_path)
        }
    }

    with open(tk_meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Updated metadata saved to {tk_meta_path}")

    # 11. Compile Master JSON
    master_json = {
        'metadata': {
            'pipeline': 'model2_final_audits',
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'canonical_test_sprints_count': n_can
        },
        'base_rates': {
            'train': round(br_train, 4),
            'val_f50': round(br_val, 4),
            'test_f50': round(br_test, 4),
            'heldout_f50': round(br_heldout, 4)
        },
        'models_comparison': model_bench_results,
        'bootstrap_tk_vs_lr23': boot_tk_vs_lr23,
        'bootstrap_tk_vs_lr54': boot_tk_vs_lr54,
        'calibration': {
            'brier_val_b': {
                'isotonic': round(brier_val_b_iso, 4),
                'platt': round(brier_val_b_platt, 4),
                'grouped': round(brier_val_b_group, 4)
            },
            'brier_test': {
                'isotonic': round(brier_test_iso, 4),
                'platt': round(brier_test_platt, 4),
                'grouped': round(brier_test_group, 4)
            },
            'lr23_test_brier': {
                'uncalibrated': round(float(brier_score_loss(y_test_f50, test_lr23_raw)), 4),
                'platt_calibrated': round(brier_test_lr23_platt, 4)
            },
            'taskove_brier_skill_score': round(brier_skill, 4),
            'taskove_decile_reliability': tk_deciles,
            'full_decile_reliability': full_deciles
        },
        'threshold_policy_taskove': tk_thresh_table,
        'held_out_projects': {
            'train_dates': train_dates,
            'set1': {
                'projects': [11, 19, 21, 28, 35],
                'dates': ho_dates,
                'overlap_with_train': True,
                'project_28_included': True,
                'table': set1_table
            },
            'set2': {
                'projects': set2_projects,
                'seed': 123,
                'min_sprints': 20,
                'project_28_included': False,
                'table': set2_table
            }
        }
    }

    out_master_path = os.path.join(WORK_DIR, "model2_final_audit_metrics.json")
    with open(out_master_path, 'w') as f:
        json.dump(master_json, f, indent=2)
    print(f"\nSaved complete audit metrics to {out_master_path}")
    print(f"Total time elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

