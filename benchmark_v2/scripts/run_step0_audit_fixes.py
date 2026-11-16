import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, median_absolute_error
import lightgbm as lgb

TAWOS_DIR = r"d:\Gillo AI Service\tawos_csv"
BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
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
    print("STEP 0: AUDIT FIXES & HONEST CALIBRATION")
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
    df_test_all = df_clean.iloc[idx_va:].copy().reset_index(drop=True)

    # Epic-excluded splits
    df_train = df_train_all[df_train_all['Type'] != 'Epic'].copy().reset_index(drop=True)
    df_test = df_test_all[df_test_all['Type'] != 'Epic'].copy().reset_index(drop=True)
    df_ne_all = df_clean[df_clean['Type'] != 'Epic'].copy().reset_index(drop=True)

    y_train = df_train['Story_Point'].values
    y_test = df_test['Story_Point'].values

    # -------------------------------------------------------------------------
    # STEP 0a: Honest interval calibration from OUT-OF-PROJECT residuals
    # -------------------------------------------------------------------------
    print("\n[Step 0a] Computing conformal constant from OUT-OF-PROJECT residuals across 5 folds...")
    all_projects = df_ne_all['Project_ID'].values
    gkf = GroupKFold(n_splits=5)
    folds_splits = list(gkf.split(df_ne_all, groups=all_projects))

    base_lgb_params = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 200,
        'n_jobs': -1,
        'verbose': -1
    }

    pooled_scores = []
    fold_eval_data = []

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

        # Train Q25 and Q75 models for this fold
        m_q25 = lgb.LGBMRegressor(**{**base_lgb_params, 'objective': 'quantile', 'alpha': 0.25, 'random_state': 42}).fit(X_f_tr.astype(np.float32), y_f_tr)
        m_q75 = lgb.LGBMRegressor(**{**base_lgb_params, 'objective': 'quantile', 'alpha': 0.75, 'random_state': 42}).fit(X_f_tr.astype(np.float32), y_f_tr)

        p_q25_te = np.clip(m_q25.predict(X_f_te.astype(np.float32)), 0.5, 100.0)
        p_q75_te = np.clip(m_q75.predict(X_f_te.astype(np.float32)), 0.5, 100.0)

        # Residual / conformity scores on held-out out-of-project test fold
        # s_i = max(q25 - y, y - q75)
        s_fold = np.maximum(p_q25_te - y_f_te, y_f_te - p_q75_te)
        pooled_scores.extend(s_fold.tolist())

        fold_eval_data.append({
            'fold': fold_idx + 1,
            'y_test': y_f_te,
            'p_q25': p_q25_te,
            'p_q75': p_q75_te,
            'tickets': len(y_f_te)
        })

    pooled_scores_arr = np.array(pooled_scores)
    # Exact 50th percentile of out-of-project conformity scores
    c_cold_start = float(np.percentile(pooled_scores_arr, 50.0))
    print(f"Computed c_cold_start from {len(pooled_scores_arr):,} pooled out-of-project residuals: {c_cold_start:.4f}")

    # Evaluate per fold coverage at c_cold_start
    fold_coverages = []
    fold_widths = []
    for f in fold_eval_data:
        ci_low = np.maximum(0.5, f['p_q25'] - c_cold_start)
        ci_high = f['p_q75'] + c_cold_start
        cov = float(np.mean((f['y_test'] >= ci_low) & (f['y_test'] <= ci_high)) * 100.0)
        wid = float(np.mean(ci_high - ci_low))
        fold_coverages.append(round(cov, 2))
        fold_widths.append(round(wid, 4))
        print(f"  Fold {f['fold']}: Coverage = {cov:.2f}% | Mean Width = {wid:.4f} SP (Tickets: {f['tickets']:,})")

    pooled_cov = float(np.mean([cov for cov in fold_coverages]))
    mean_wid_all = float(np.mean([wid for wid in fold_widths]))

    # -------------------------------------------------------------------------
    # STEP 0b: Model vs. Per-Type Median under SAME Treatment
    # -------------------------------------------------------------------------
    print("\n[Step 0b] Evaluating Model vs Per-Type Median under the SAME treatment...")
    with open(os.path.join(V2_ARTIFACTS_DIR, "model_sp_final.pkl"), "rb") as f:
        v2_model = pickle.load(f)
    with open(os.path.join(V2_ARTIFACTS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        v2_tfidf = pickle.load(f)
    with open(os.path.join(V2_ARTIFACTS_DIR, "ohe_encoder.pkl"), "rb") as f:
        v2_ohe = pickle.load(f)

    X_te_feat = sparse.hstack([v2_tfidf.transform(prep_text(df_test)), v2_ohe.transform(df_test[['Type']].fillna('Missing'))], format='csr')
    p_model_raw = np.clip(v2_model.predict(X_te_feat.astype(np.float32)), 0.5, 100.0)
    p_model_snap = snap_to_scale(p_model_raw)

    type_meds = df_train.groupby('Type')['Story_Point'].median().to_dict()
    glob_med = float(np.median(y_train))
    p_base_raw = np.array([type_meds.get(t, glob_med) for t in df_test['Type']])
    p_base_snap = snap_to_scale(p_base_raw)

    same_treatment_results = {
        'continuous_raw': {
            'model_MAE': round(float(mean_absolute_error(y_test, p_model_raw)), 4),
            'model_MdAE': round(float(median_absolute_error(y_test, p_model_raw)), 4),
            'model_frac_le_1': round(float(np.mean(np.abs(y_test - p_model_raw) <= 1.0) * 100.0), 2),
            'baseline_MAE': round(float(mean_absolute_error(y_test, p_base_raw)), 4),
            'baseline_MdAE': round(float(median_absolute_error(y_test, p_base_raw)), 4),
            'baseline_frac_le_1': round(float(np.mean(np.abs(y_test - p_base_raw) <= 1.0) * 100.0), 2)
        },
        'snapped_fibonacci': {
            'model_MAE': round(float(mean_absolute_error(y_test, p_model_snap)), 4),
            'model_MdAE': round(float(median_absolute_error(y_test, p_model_snap)), 4),
            'model_frac_le_1': round(float(np.mean(np.abs(y_test - p_model_snap) <= 1.0) * 100.0), 2),
            'baseline_MAE': round(float(mean_absolute_error(y_test, p_base_snap)), 4),
            'baseline_MdAE': round(float(median_absolute_error(y_test, p_base_snap)), 4),
            'baseline_frac_le_1': round(float(np.mean(np.abs(y_test - p_base_snap) <= 1.0) * 100.0), 2)
        }
    }

    print("\nSAME TREATMENT COMPARISON (Chronological Test Set, N=6,167):")
    print(f"{'Treatment':<22} | {'Model MAE':<10} | {'Base MAE':<10} | {'Model MdAE':<11} | {'Base MdAE':<10} | {'Model <=1pt':<12} | {'Base <=1pt':<12}")
    print("-" * 100)
    r1 = same_treatment_results['continuous_raw']
    print(f"{'Continuous (Raw)':<22} | {r1['model_MAE']:<10.4f} | {r1['baseline_MAE']:<10.4f} | {r1['model_MdAE']:<11.4f} | {r1['baseline_MdAE']:<10.4f} | {r1['model_frac_le_1']:<11.2f}% | {r1['baseline_frac_le_1']:<11.2f}%")
    r2 = same_treatment_results['snapped_fibonacci']
    print(f"{'Snapped (Fibonacci)':<22} | {r2['model_MAE']:<10.4f} | {r2['baseline_MAE']:<10.4f} | {r2['model_MdAE']:<11.4f} | {r2['baseline_MdAE']:<10.4f} | {r2['model_frac_le_1']:<11.2f}% | {r2['baseline_frac_le_1']:<11.2f}%")

    # -------------------------------------------------------------------------
    # STEP 0c: Table of the 9 Losing Projects in Cold Start
    # -------------------------------------------------------------------------
    print("\n[Step 0c] Verifying the 9 losing projects in cold-start CV...")
    with open(os.path.join(BASE_DIR, "work", "model1_part4_metrics.json"), "r") as f:
        p4 = json.load(f)
    losses = [p for p in p4['item3_practical_metrics']['cold_start_win_counts']['per_project_details'] if p['status'] == 'LOSS']

    print(f"Total losing projects: {len(losses)}")
    print(f"{'Project ID':<12} | {'Fold':<6} | {'Tickets (N)':<12} | {'Model MAE':<11} | {'Base MAE':<10} | {'Diff (Model - Base)':<20}")
    print("-" * 80)
    for l in losses:
        print(f"{l['project_id']:<12} | {l['fold']:<6} | {l['tickets']:<12} | {l['MAE_final_model']:<11.4f} | {l['MAE_per_type_median']:<10.4f} | {l['diff_MAE']:<+20.4f}")

    # -------------------------------------------------------------------------
    # STEP 0d: Save Per-Type Median Baseline JSON Artifact & Update Metadata
    # -------------------------------------------------------------------------
    print("\n[Step 0d] Saving per-type median baseline artifact for service fallback...")
    baseline_payload = {
        "description": "Per-type median story points computed from TAWOS training data (Epic-excluded, N=29,330)",
        "global_median": float(glob_med),
        "type_medians": {str(k): float(v) for k, v in type_meds.items()}
    }

    baseline_artifact_path = os.path.join(V2_ARTIFACTS_DIR, "type_median_fallback.json")
    with open(baseline_artifact_path, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)

    base_sha256 = get_sha256(baseline_artifact_path)
    print(f"Saved {baseline_artifact_path} (SHA256: {base_sha256})")

    # Update metadata.json
    meta_path = os.path.join(V2_ARTIFACTS_DIR, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta['artifacts']['type_median_fallback_sha256'] = base_sha256
    meta['conformal_interval']['in_domain'] = {
        'adjustment_constant': meta['conformal_interval']['adjustment_constant'],
        'test_coverage_pct': meta['conformal_interval']['test_coverage_pct'],
        'test_mean_width_points': meta['conformal_interval']['test_mean_width_points']
    }
    meta['conformal_interval']['cold_start'] = {
        'c_cold_start': round(c_cold_start, 4),
        'pooled_out_of_project_coverage_pct': round(pooled_cov, 2),
        'per_fold_coverage_pct': fold_coverages,
        'mean_interval_width_points': round(mean_wid_all, 4)
    }
    meta['conformal_interval']['c_cold_start'] = round(c_cold_start, 4)
    meta['fallback_baseline'] = baseline_payload

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Updated metadata.json with c_cold_start and type_median_fallback_sha256.")

    # Save summary of Step 0 to work dir
    step0_summary = {
        'c_cold_start': round(c_cold_start, 4),
        'per_fold_coverage_pct': fold_coverages,
        'same_treatment_results': same_treatment_results,
        'losing_projects_table': losses,
        'type_median_fallback': baseline_payload,
        'type_median_fallback_sha256': base_sha256
    }
    with open(os.path.join(BASE_DIR, "work", "step0_audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(step0_summary, f, indent=2)
    print("Step 0 audit summary written to benchmark_v2/work/step0_audit_summary.json")

if __name__ == "__main__":
    main()

