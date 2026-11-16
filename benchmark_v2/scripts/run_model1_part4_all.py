import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
import lightgbm as lgb

TAWOS_DIR = r"d:\Gillo AI Service\tawos_csv"
BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
WORK_DIR = os.path.join(BASE_DIR, "work")
V2_ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "model1_sp", "v2_taskove")

FIBONACCI_SCALE = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0])

def snap_to_scale(arr, scale=FIBONACCI_SCALE):
    diffs = np.abs(arr[:, None] - scale[None, :])
    best_idx = np.argmin(diffs, axis=1)
    return scale[best_idx]

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def prep_text(df):
    return (df['Title'].fillna('') + ' ' + df['Description_Text'].fillna('')).str.slice(0, 2000).fillna('')

def main():
    print("=" * 80)
    print("RUNNING BENCHMARK V2 - PART 4 AUDIT CALCULATIONS")
    print("=" * 80)

    # 1. Load Clean Dataset
    print("\n[1] Loading issue.csv and creating chronological split...")
    cols = [
        "ID", "Title", "Description_Text", "Type", "Priority",
        "Story_Point", "Estimation_Date", "Creation_Date", "Project_ID",
        "Story_Point_Changed_After_Estimation",
        "Title_Changed_After_Estimation",
        "Description_Changed_After_Estimation",
        "Resolution"
    ]
    df_raw = pd.read_csv(os.path.join(TAWOS_DIR, "issue.csv"), usecols=cols, low_memory=False)
    df_raw.columns = [c.lstrip('\ufeff') for c in df_raw.columns]

    df_sp = df_raw[df_raw['Story_Point'].notna() & (df_raw['Story_Point'] > 0) & (df_raw['Story_Point'] <= 100.0)].copy()
    excluded_resolutions = {"won't fix", "wont fix", "wontfix", "invalid", "duplicate", "cannot reproduce"}
    res_clean = df_sp['Resolution'].fillna("").str.lower().str.strip()
    df_valid = df_sp[~res_clean.isin(excluded_resolutions)].copy()

    clean_mask = (
        (df_valid['Story_Point_Changed_After_Estimation'] == 0) &
        (df_valid['Title_Changed_After_Estimation'] == 0) &
        (df_valid['Description_Changed_After_Estimation'] == 0)
    )
    df_clean = df_valid[clean_mask].copy()

    df_clean['Timestamp_Str'] = df_clean['Estimation_Date'].fillna(df_clean['Creation_Date'])
    df_clean['Timestamp_dt'] = pd.to_datetime(df_clean['Timestamp_Str'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Timestamp_dt']).sort_values('Timestamp_dt').reset_index(drop=True)

    N_all = len(df_clean)
    idx_tr = int(N_all * 0.70)
    idx_va = int(N_all * 0.85)

    df_train_all = df_clean.iloc[:idx_tr].copy().reset_index(drop=True)
    df_val_all = df_clean.iloc[idx_tr:idx_va].copy().reset_index(drop=True)
    df_test_all = df_clean.iloc[idx_va:].copy().reset_index(drop=True)

    # Epic-excluded splits
    df_train = df_train_all[df_train_all['Type'] != 'Epic'].copy().reset_index(drop=True)
    df_val = df_val_all[df_val_all['Type'] != 'Epic'].copy().reset_index(drop=True)
    df_test = df_test_all[df_test_all['Type'] != 'Epic'].copy().reset_index(drop=True)
    df_ne_all = df_clean[df_clean['Type'] != 'Epic'].copy().reset_index(drop=True)

    print(f"Total Epic-excluded issues: {len(df_ne_all):,} (Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,})")

    y_train = df_train['Story_Point'].values
    y_val = df_val['Story_Point'].values
    y_test = df_test['Story_Point'].values

    # Base hyperparameters (LightGBM L1)
    base_lgb_params = {
        'objective': 'regression_l1',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 200,
        'n_jobs': -1,
        'verbose': -1
    }

    # =========================================================================
    # ITEM 1: TASKOVE TYPES AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 1: TASKOVE TYPES CODEBASE AUDIT")
    print("=" * 80)
    # List of types defined in codebase:
    # app/models/read_models.py: task_type: str = "feature"
    # scripts/seed_data.py: "infra", "backend", "ml", "frontend", "qa"
    # tests/ (conftest.py, test_hardening_fixes.py, test_registry.py): "backend", "ml"
    # app/services/llm_provider.py & scripts/prepare_ml_datasets.py: "feature", "bug", "infra", "refactor", "test", "docs"
    # chore: INVENTED (0 occurrences across the entire repo)

    taskove_types_audit = {
        'sources': {
            'app_models_read_models_py': ['feature'],
            'scripts_seed_data_py': ['infra', 'backend', 'ml', 'frontend', 'qa'],
            'tests': ['backend', 'ml'],
            'app_services_llm_provider_py': ['feature', 'bug', 'infra', 'refactor'],
            'scripts_prepare_ml_datasets_py': ['feature', 'bug', 'infra', 'refactor', 'test', 'docs']
        },
        'distinct_real_taskove_types': [
            'feature', 'bug', 'infra', 'refactor', 'test', 'docs', 'backend', 'frontend', 'ml', 'qa'
        ],
        'invented_types_in_previous_table': ['chore'],
        'omitted_real_types_in_previous_table': ['qa'],
        'updated_taskove_type_mapping': {
            'feature': {'tawos_type': 'Story', 'fidelity': 'approximate'},
            'bug': {'tawos_type': 'Bug', 'fidelity': 'exact'},
            'infra': {'tawos_type': 'Technical task', 'fidelity': 'approximate'},
            'refactor': {'tawos_type': 'Improvement', 'fidelity': 'approximate'},
            'test': {'tawos_type': 'Technical task', 'fidelity': 'approximate'},
            'docs': {'tawos_type': 'Documentation', 'fidelity': 'exact'},
            'backend': {'tawos_type': 'Technical task', 'fidelity': 'approximate'},
            'frontend': {'tawos_type': 'Technical task', 'fidelity': 'approximate'},
            'ml': {'tawos_type': 'Task', 'fidelity': 'approximate'},
            'qa': {'tawos_type': 'Technical task', 'fidelity': 'approximate'}
        }
    }
    print(f"Audited Taskove types. Real types count: {len(taskove_types_audit['distinct_real_taskove_types'])}")
    print("Invented type removed: 'chore'")
    print("Real type added: 'qa'")

    # =========================================================================
    # ITEM 2: TYPE-MAPPING SENSITIVITY FOR FINAL MODEL (TF-IDF + TYPE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 2: TYPE-MAPPING SENSITIVITY EVALUATION")
    print("=" * 80)
    # Mappings:
    # (a) Current mapping: TAWOS types unchanged
    # (b) backend/frontend/infra/test mapped to Task instead of Technical task
    #     -> in TAWOS data, Type == 'Technical task' mapped to 'Task'
    # (c) Type collapsed to {Bug, non-Bug}
    #     -> in TAWOS data, 'Bug' stays 'Bug', all other types mapped to 'non-Bug'
    # (d) all types set to Task
    #     -> in TAWOS data, all types mapped to 'Task'

    mapping_variants = {
        'a_current': lambda t: t,
        'b_tech_to_task': lambda t: 'Task' if t == 'Technical task' else t,
        'c_bug_vs_nonbug': lambda t: 'Bug' if t == 'Bug' else 'non-Bug',
        'd_all_task': lambda t: 'Task'
    }

    # Prepare GroupKFold by project
    all_projects = df_ne_all['Project_ID'].values
    gkf = GroupKFold(n_splits=5)
    folds_splits = list(gkf.split(df_ne_all, groups=all_projects))

    # Load pre-trained v2_taskove model, tfidf vectorizer, and ohe encoder
    with open(os.path.join(V2_ARTIFACTS_DIR, "model_sp_final.pkl"), "rb") as f:
        v2_model = pickle.load(f)
    with open(os.path.join(V2_ARTIFACTS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        v2_tfidf = pickle.load(f)
    with open(os.path.join(V2_ARTIFACTS_DIR, "ohe_encoder.pkl"), "rb") as f:
        v2_ohe = pickle.load(f)

    # Compute TF-IDF on chronological splits
    text_tr = prep_text(df_train)
    text_te = prep_text(df_test)
    X_tr_tfidf = v2_tfidf.transform(text_tr)
    X_te_tfidf = v2_tfidf.transform(text_te)

    type_sensitivity_results = {}

    for m_key, m_func in mapping_variants.items():
        print(f"\nEvaluating Mapping Variant: {m_key}...")
        
        # 1. Chronological Test Evaluation:
        # A) Inference-time with pre-trained v2_taskove model
        df_test_mapped = df_test.copy()
        df_test_mapped['Type'] = df_test_mapped['Type'].apply(m_func)
        X_te_cat_inf = v2_ohe.transform(df_test_mapped[['Type']].fillna('Missing'))
        X_te_feat_inf = sparse.hstack([X_te_tfidf, X_te_cat_inf], format='csr')
        p_te_inf = np.clip(v2_model.predict(X_te_feat_inf.astype(np.float32)), 0.5, 100.0)
        mae_te_inf = float(mean_absolute_error(y_test, p_te_inf))

        # B) Retrained model on chronological train split with mapped types
        df_train_mapped = df_train.copy()
        df_train_mapped['Type'] = df_train_mapped['Type'].apply(m_func)
        ohe_retrain = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_tr_cat_ret = ohe_retrain.fit_transform(df_train_mapped[['Type']].fillna('Missing'))
        X_te_cat_ret = ohe_retrain.transform(df_test_mapped[['Type']].fillna('Missing'))
        X_tr_feat_ret = sparse.hstack([X_tr_tfidf, X_tr_cat_ret], format='csr')
        X_te_feat_ret = sparse.hstack([X_te_tfidf, X_te_cat_ret], format='csr')

        m_retrain = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_feat_ret.astype(np.float32), y_train)
        p_te_ret = np.clip(m_retrain.predict(X_te_feat_ret.astype(np.float32)), 0.5, 100.0)
        mae_te_ret = float(mean_absolute_error(y_test, p_te_ret))

        # 2. Cold-Start 5-Fold Cross-Validation (Retrained per fold)
        fold_maes = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds_splits):
            f_tr = df_ne_all.iloc[train_idx].copy().reset_index(drop=True)
            f_te = df_ne_all.iloc[test_idx].copy().reset_index(drop=True)
            y_f_tr = f_tr['Story_Point'].values
            y_f_te = f_te['Story_Point'].values

            f_tr['Type'] = f_tr['Type'].apply(m_func)
            f_te['Type'] = f_te['Type'].apply(m_func)

            tfidf_f = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=3)
            X_f_tr_text = tfidf_f.fit_transform(prep_text(f_tr))
            X_f_te_text = tfidf_f.transform(prep_text(f_te))

            ohe_f = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            X_f_tr_cat = ohe_f.fit_transform(f_tr[['Type']].fillna('Missing'))
            X_f_te_cat = ohe_f.transform(f_te[['Type']].fillna('Missing'))

            X_f_tr = sparse.hstack([X_f_tr_text, X_f_tr_cat], format='csr')
            X_f_te = sparse.hstack([X_f_te_text, X_f_te_cat], format='csr')

            m_f = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_f_tr.astype(np.float32), y_f_tr)
            p_f = np.clip(m_f.predict(X_f_te.astype(np.float32)), 0.5, 100.0)
            mae_f = float(mean_absolute_error(y_f_te, p_f))
            fold_maes.append(mae_f)

        type_sensitivity_results[m_key] = {
            'test_MAE_inference_pretrained': round(mae_te_inf, 4),
            'test_MAE_retrained': round(mae_te_ret, 4),
            'cold_start_fold_maes': [round(x, 4) for x in fold_maes],
            'cold_start_mean_MAE': round(float(np.mean(fold_maes)), 4),
            'cold_start_std_MAE': round(float(np.std(fold_maes, ddof=1)), 4)
        }
        print(f"  Test MAE (Inference): {mae_te_inf:.4f} | Test MAE (Retrained): {mae_te_ret:.4f} | Cold-Start Mean MAE: {np.mean(fold_maes):.4f}")

    # =========================================================================
    # ITEM 3: PRACTICAL METRICS VS PER-TYPE MEDIAN
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 3: PRACTICAL METRICS & WIN COUNTS")
    print("=" * 80)

    # Predictions of final model (continuous) on chronological test set
    p_final_te = np.clip(v2_model.predict(X_te_feat_ret.astype(np.float32) if False else sparse.hstack([X_te_tfidf, v2_ohe.transform(df_test[['Type']].fillna('Missing'))], format='csr').astype(np.float32)), 0.5, 100.0)
    p_final_te_snapped = snap_to_scale(p_final_te)

    # Baseline: per-type median
    type_meds = df_train.groupby('Type')['Story_Point'].median().to_dict()
    glob_med = float(np.median(y_train))
    p_base_te = np.array([type_meds.get(t, glob_med) for t in df_test['Type']])
    p_base_te_snapped = snap_to_scale(p_base_te)

    # Metric calculations
    mae_final_raw = float(mean_absolute_error(y_test, p_final_te))
    mdae_final_raw = float(median_absolute_error(y_test, p_final_te))
    mae_final_snap = float(mean_absolute_error(y_test, p_final_te_snapped))
    mdae_final_snap = float(median_absolute_error(y_test, p_final_te_snapped))

    mae_base_raw = float(mean_absolute_error(y_test, p_base_te))
    mdae_base_raw = float(median_absolute_error(y_test, p_base_te))
    mae_base_snap = float(mean_absolute_error(y_test, p_base_te_snapped))
    mdae_base_snap = float(median_absolute_error(y_test, p_base_te_snapped))

    # Fraction abs error <= 1 point
    frac_le1_final_raw = float(np.mean(np.abs(y_test - p_final_te) <= 1.0))
    frac_le1_final_snap = float(np.mean(np.abs(y_test - p_final_te_snapped) <= 1.0))
    frac_le1_base_raw = float(np.mean(np.abs(y_test - p_base_te) <= 1.0))
    frac_le1_base_snap = float(np.mean(np.abs(y_test - p_base_te_snapped) <= 1.0))

    # Cold-Start Wins: Per-fold and Per-project
    # Fold wins
    # Part 3 fold results:
    # Fold 1: tfidf 2.4985 vs per-type 2.5783 (Win)
    # Fold 2: tfidf 1.4696 vs per-type 1.7452 (Win)
    # Fold 3: tfidf 2.2125 vs per-type 2.2672 (Win)
    # Fold 4: tfidf 3.4104 vs per-type 3.6813 (Win)
    # Fold 5: tfidf 1.4728 vs per-type 1.5599 (Win)
    fold_wins = 5
    total_folds = 5

    # Per-project wins across all 39 distinct projects in cold-start CV
    project_cold_start_records = []
    proj_wins = 0
    proj_losses = 0
    proj_ties = 0

    for fold_idx, (train_idx, test_idx) in enumerate(folds_splits):
        f_tr = df_ne_all.iloc[train_idx].copy().reset_index(drop=True)
        f_te = df_ne_all.iloc[test_idx].copy().reset_index(drop=True)
        y_f_tr = f_tr['Story_Point'].values
        y_f_te = f_te['Story_Point'].values

        f_type_meds = f_tr.groupby('Type')['Story_Point'].median().to_dict()
        f_glob_med = float(np.median(y_f_tr))
        p_f_type = np.array([f_type_meds.get(t, f_glob_med) for t in f_te['Type']])

        tfidf_f = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=3)
        X_f_tr_text = tfidf_f.fit_transform(prep_text(f_tr))
        X_f_te_text = tfidf_f.transform(prep_text(f_te))

        ohe_f = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_f_tr_cat = ohe_f.fit_transform(f_tr[['Type']].fillna('Missing'))
        X_f_te_cat = ohe_f.transform(f_te[['Type']].fillna('Missing'))

        X_f_tr = sparse.hstack([X_f_tr_text, X_f_tr_cat], format='csr')
        X_f_te = sparse.hstack([X_f_te_text, X_f_te_cat], format='csr')

        m_f = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_f_tr.astype(np.float32), y_f_tr)
        p_f_final = np.clip(m_f.predict(X_f_te.astype(np.float32)), 0.5, 100.0)

        # Evaluate per project in this fold
        for pid in sorted(f_te['Project_ID'].unique()):
            p_mask = (f_te['Project_ID'] == pid).values
            y_proj = y_f_te[p_mask]
            p_final_proj = p_f_final[p_mask]
            p_type_proj = p_f_type[p_mask]

            mae_final_p = float(mean_absolute_error(y_proj, p_final_proj))
            mae_type_p = float(mean_absolute_error(y_proj, p_type_proj))

            if mae_final_p < mae_type_p:
                status = "WIN"
                proj_wins += 1
            elif mae_final_p > mae_type_p:
                status = "LOSS"
                proj_losses += 1
            else:
                status = "TIE"
                proj_ties += 1

            project_cold_start_records.append({
                'project_id': int(pid),
                'fold': fold_idx + 1,
                'tickets': int(np.sum(p_mask)),
                'MAE_final_model': round(mae_final_p, 4),
                'MAE_per_type_median': round(mae_type_p, 4),
                'diff_MAE': round(mae_final_p - mae_type_p, 4),
                'status': status
            })

    practical_metrics_summary = {
        'chronological_test_comparison': {
            'final_model': {
                'raw_MAE': round(mae_final_raw, 4),
                'raw_MdAE': round(mdae_final_raw, 4),
                'snapped_MAE': round(mae_final_snap, 4),
                'snapped_MdAE': round(mdae_final_snap, 4),
                'raw_fraction_abs_error_le_1': round(frac_le1_final_raw, 4),
                'snapped_fraction_abs_error_le_1': round(frac_le1_final_snap, 4)
            },
            'per_type_median': {
                'raw_MAE': round(mae_base_raw, 4),
                'raw_MdAE': round(mdae_base_raw, 4),
                'snapped_MAE': round(mae_base_snap, 4),
                'snapped_MdAE': round(mdae_base_snap, 4),
                'raw_fraction_abs_error_le_1': round(frac_le1_base_raw, 4),
                'snapped_fraction_abs_error_le_1': round(frac_le1_base_snap, 4)
            }
        },
        'cold_start_win_counts': {
            'fold_wins': f"{fold_wins}/{total_folds} ({fold_wins/total_folds*100:.1f}%)",
            'project_wins': f"{proj_wins}/{len(project_cold_start_records)} ({proj_wins/len(project_cold_start_records)*100:.1f}%)",
            'project_losses': f"{proj_losses}/{len(project_cold_start_records)}",
            'project_ties': f"{proj_ties}/{len(project_cold_start_records)}",
            'total_projects_evaluated': len(project_cold_start_records),
            'per_project_details': project_cold_start_records
        }
    }
    print(f"Snapped MAE: Final={mae_final_snap:.4f} vs Baseline={mae_base_snap:.4f}")
    print(f"Fraction Error <= 1pt: Final={frac_le1_final_raw:.4f} (Snapped: {frac_le1_final_snap:.4f}) vs Baseline={frac_le1_base_raw:.4f} (Snapped: {frac_le1_base_snap:.4f})")
    print(f"Cold-Start Wins: Folds {fold_wins}/{total_folds}, Projects {proj_wins}/{len(project_cold_start_records)}")

    # =========================================================================
    # ITEM 4: INTERVAL COVERAGE & MEAN WIDTH PER TOP-10 PROJECT & COLD-START FOLDS
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 4: INTERVAL COVERAGE & WIDTH (TOP 10 & COLD-START FOLDS)")
    print("=" * 80)

    # Load quantile models and conformal constant for v2_taskove
    with open(os.path.join(V2_ARTIFACTS_DIR, "model_sp_q25.pkl"), "rb") as f:
        v2_q25 = pickle.load(f)
    with open(os.path.join(V2_ARTIFACTS_DIR, "model_sp_q75.pkl"), "rb") as f:
        v2_q75 = pickle.load(f)

    with open(os.path.join(V2_ARTIFACTS_DIR, "metadata.json"), "r") as f:
        v2_meta = json.load(f)
    c_const = v2_meta['conformal_interval']['adjustment_constant']

    # Predict test quantiles
    X_te_feat = sparse.hstack([X_te_tfidf, v2_ohe.transform(df_test[['Type']].fillna('Missing'))], format='csr')
    q25_te = np.clip(v2_q25.predict(X_te_feat.astype(np.float32)), 0.5, 100.0)
    q75_te = np.clip(v2_q75.predict(X_te_feat.astype(np.float32)), 0.5, 100.0)

    # Conformal interval
    ci_low_te = np.maximum(0.5, q25_te - c_const)
    ci_high_te = q75_te + c_const
    covered_te = (y_test >= ci_low_te) & (y_test <= ci_high_te)
    width_te = ci_high_te - ci_low_te

    # Top 10 test projects
    top10_pids = df_test['Project_ID'].value_counts().head(10).index.tolist()
    top10_intervals = []
    for pid in top10_pids:
        mask = (df_test['Project_ID'] == pid).values
        cov_p = float(np.mean(covered_te[mask]) * 100.0)
        wid_p = float(np.mean(width_te[mask]))
        top10_intervals.append({
            'project_id': int(pid),
            'test_tickets': int(np.sum(mask)),
            'conformal_coverage_pct': round(cov_p, 2),
            'mean_interval_width': round(wid_p, 4)
        })

    # Cold-Start Folds Interval Coverage
    # Evaluate 50% intervals in each cold-start fold (retraining quantile regressors per fold + conformal tuning on fold train)
    cold_start_fold_intervals = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds_splits):
        f_tr = df_ne_all.iloc[train_idx].copy().reset_index(drop=True)
        f_te = df_ne_all.iloc[test_idx].copy().reset_index(drop=True)
        y_f_tr = f_tr['Story_Point'].values
        y_f_te = f_te['Story_Point'].values

        tfidf_f = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=3)
        X_f_tr_text = tfidf_f.fit_transform(prep_text(f_tr))
        X_f_te_text = tfidf_f.transform(prep_text(f_te))

        ohe_f = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_f_tr_cat = ohe_f.fit_transform(f_tr[['Type']].fillna('Missing'))
        X_f_te_cat = ohe_f.transform(f_te[['Type']].fillna('Missing'))

        X_f_tr = sparse.hstack([X_f_tr_text, X_f_tr_cat], format='csr')
        X_f_te = sparse.hstack([X_f_te_text, X_f_te_cat], format='csr')

        # Train Q25 and Q75
        p_q25 = {**base_lgb_params, 'objective': 'quantile', 'alpha': 0.25, 'random_state': 42}
        p_q75 = {**base_lgb_params, 'objective': 'quantile', 'alpha': 0.75, 'random_state': 42}
        m_q25 = lgb.LGBMRegressor(**p_q25).fit(X_f_tr.astype(np.float32), y_f_tr)
        m_q75 = lgb.LGBMRegressor(**p_q75).fit(X_f_tr.astype(np.float32), y_f_tr)

        # Train set predictions to compute conformal adjustment for nominal 50%
        pred_q25_tr = np.clip(m_q25.predict(X_f_tr.astype(np.float32)), 0.5, 100.0)
        pred_q75_tr = np.clip(m_q75.predict(X_f_tr.astype(np.float32)), 0.5, 100.0)
        conf_scores_tr = np.maximum(pred_q25_tr - y_f_tr, y_f_tr - pred_q75_tr)
        c_fold = float(np.percentile(conf_scores_tr, 50.0))

        # Test set evaluation
        pred_q25_te = np.clip(m_q25.predict(X_f_te.astype(np.float32)), 0.5, 100.0)
        pred_q75_te = np.clip(m_q75.predict(X_f_te.astype(np.float32)), 0.5, 100.0)

        ci_low_f = np.maximum(0.5, pred_q25_te - c_fold)
        ci_high_f = pred_q75_te + c_fold
        cov_f = float(np.mean((y_f_te >= ci_low_f) & (y_f_te <= ci_high_f)) * 100.0)
        wid_f = float(np.mean(ci_high_f - ci_low_f))

        # Also evaluate pre-trained v2_taskove quantiles zero-shot on this fold
        X_f_te_v2 = sparse.hstack([v2_tfidf.transform(prep_text(f_te)), v2_ohe.transform(f_te[['Type']].fillna('Missing'))], format='csr')
        v2_q25_f = np.clip(v2_q25.predict(X_f_te_v2.astype(np.float32)), 0.5, 100.0)
        v2_q75_f = np.clip(v2_q75.predict(X_f_te_v2.astype(np.float32)), 0.5, 100.0)
        v2_ci_low_f = np.maximum(0.5, v2_q25_f - c_const)
        v2_ci_high_f = v2_q75_f + c_const
        v2_cov_f = float(np.mean((y_f_te >= v2_ci_low_f) & (y_f_te <= v2_ci_high_f)) * 100.0)
        v2_wid_f = float(np.mean(v2_ci_high_f - v2_ci_low_f))

        cold_start_fold_intervals.append({
            'fold': fold_idx + 1,
            'test_tickets': len(f_te),
            'retrained_conformal_constant': round(c_fold, 4),
            'retrained_coverage_pct': round(cov_f, 2),
            'retrained_mean_width': round(wid_f, 4),
            'v2_zero_shot_coverage_pct': round(v2_cov_f, 2),
            'v2_zero_shot_mean_width': round(v2_wid_f, 4)
        })

    interval_metrics_summary = {
        'top_10_test_projects': top10_intervals,
        'cold_start_folds_50pct_interval': cold_start_fold_intervals
    }
    print("Top 10 test projects and cold start fold intervals computed successfully.")

    # =========================================================================
    # ITEM 5: UPDATE METADATA.JSON IN V2_TASKOVE (REMOVE INVENTED CHORE, ADD QA)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 5: UPDATING V2_TASKOVE METADATA.JSON")
    print("=" * 80)
    v2_meta_path = os.path.join(V2_ARTIFACTS_DIR, "metadata.json")
    with open(v2_meta_path, "r") as f:
        meta_to_update = json.load(f)

    meta_to_update['taskove_type_mapping'] = taskove_types_audit['updated_taskove_type_mapping']
    meta_to_update['taskove_types_audit'] = {
        'verified_sources': ['app/models/read_models.py', 'scripts/seed_data.py', 'tests', 'app/services/llm_provider.py', 'scripts/prepare_ml_datasets.py'],
        'distinct_real_types': taskove_types_audit['distinct_real_taskove_types'],
        'removed_invented_types': ['chore'],
        'added_real_types': ['qa']
    }

    with open(v2_meta_path, "w") as f:
        json.dump(meta_to_update, f, indent=2)
    print("Updated metadata.json with verified Taskove types (chore removed, qa added).")

    # =========================================================================
    # SAVE FULL PART 4 METRICS TO JSON
    # =========================================================================
    part4_metrics = {
        'item1_taskove_types_audit': taskove_types_audit,
        'item2_type_mapping_sensitivity': type_sensitivity_results,
        'item3_practical_metrics': practical_metrics_summary,
        'item4_interval_metrics': interval_metrics_summary
    }

    out_metrics_path = os.path.join(WORK_DIR, "model1_part4_metrics.json")
    with open(out_metrics_path, "w") as f:
        json.dump(part4_metrics, f, indent=2)
    print(f"\nAll Part 4 metrics successfully written to: {out_metrics_path}")

if __name__ == "__main__":
    main()

