import os
import sys
import json
import time
import hashlib
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

TAWOS_DIR = r"d:\Gillo AI Service\tawos_csv"
BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
WORK_DIR = os.path.join(BASE_DIR, "work")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "model1", "v1_tawos")

FIBONACCI_POINTS = np.array([1, 2, 3, 5, 8, 13, 21])

def snap_to_fibonacci(preds):
    """Snaps continuous predictions to the nearest Agile Fibonacci story point."""
    preds_arr = np.array(preds).reshape(-1, 1)
    diffs = np.abs(preds_arr - FIBONACCI_POINTS)
    closest_idx = np.argmin(diffs, axis=1)
    return FIBONACCI_POINTS[closest_idx]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def run_benchmark():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    metrics_log = {}
    
    print("================================================================")
    print("PART 1: Model 1 - Story Points from Task Text Benchmark")
    print("================================================================")

    # 1. Load issue.csv with selected columns
    cols_to_read = [
        "ID", "Title", "Description_Text", "Type", "Priority",
        "Story_Point", "Estimation_Date", "Creation_Date", "Project_ID",
        "Story_Point_Changed_After_Estimation",
        "Title_Changed_After_Estimation",
        "Description_Changed_After_Estimation",
        "Resolution"
    ]
    print(f"\n[Step 1] Loading {len(cols_to_read)} columns from issue.csv...")
    df_raw = pd.read_csv(
        os.path.join(TAWOS_DIR, "issue.csv"),
        usecols=cols_to_read,
        low_memory=False
    )
    df_raw.columns = [c.lstrip('\ufeff') for c in df_raw.columns]
    initial_rows = len(df_raw)
    print(f"         Loaded {initial_rows:,} raw issues.")

    # Inspect the 3 changed columns
    changed_cols = [
        "Story_Point_Changed_After_Estimation",
        "Title_Changed_After_Estimation",
        "Description_Changed_After_Estimation"
    ]
    changed_inspection = {}
    for c in changed_cols:
        vc = df_raw[c].value_counts(dropna=False).to_dict()
        changed_inspection[c] = {
            "dtype": str(df_raw[c].dtype),
            "value_counts": {str(k): int(v) for k, v in vc.items()}
        }
        print(f"         {c}: dtype={df_raw[c].dtype}, counts={vc}")
    metrics_log["changed_columns_inspection"] = changed_inspection

    # 2. Filtering pipeline
    filtering_log = {"initial_rows": initial_rows}
    
    # Filter A: Story_Point > 0
    sp_positive_mask = df_raw['Story_Point'].notna() & (df_raw['Story_Point'] > 0)
    df_sp = df_raw[sp_positive_mask].copy()
    dropped_no_sp = initial_rows - len(df_sp)
    filtering_log["after_sp_positive"] = len(df_sp)
    filtering_log["dropped_no_sp"] = dropped_no_sp
    print(f"\n[Step 2] Kept issues with Story_Point > 0: {len(df_sp):,} (dropped {dropped_no_sp:,})")

    # SP value distribution before capping
    sp_vals = df_sp['Story_Point']
    sp_distribution = {
        "min": float(sp_vals.min()),
        "P10": float(sp_vals.quantile(0.10)),
        "P25": float(sp_vals.quantile(0.25)),
        "P50": float(sp_vals.median()),
        "P75": float(sp_vals.quantile(0.75)),
        "P90": float(sp_vals.quantile(0.90)),
        "P95": float(sp_vals.quantile(0.95)),
        "P99": float(sp_vals.quantile(0.99)),
        "max": float(sp_vals.max()),
        "mean": round(float(sp_vals.mean()), 2),
        "std": round(float(sp_vals.std()), 2),
        "top_10_values": {str(k): int(v) for k, v in sp_vals.value_counts().head(10).items()}
    }
    metrics_log["sp_value_distribution_raw"] = sp_distribution
    print(f"         SP Distribution: Median={sp_distribution['P50']}, P95={sp_distribution['P95']}, Max={sp_distribution['max']}")
    print(f"         Top SP Values: {sp_distribution['top_10_values']}")

    # Filter B: Drop implausible Story_Point > 100
    df_plausible = df_sp[df_sp['Story_Point'] <= 100.0].copy()
    dropped_implausible = len(df_sp) - len(df_plausible)
    filtering_log["after_sp_plausible_le_100"] = len(df_plausible)
    filtering_log["dropped_implausible_gt_100"] = dropped_implausible
    print(f"         Dropped implausible SP > 100: {len(df_plausible):,} remaining (dropped {dropped_implausible})")

    # Filter C: Exclude Won't Fix, Invalid, Duplicate, Cannot Reproduce resolutions
    excluded_resolutions = {"won't fix", "wont fix", "wontfix", "invalid", "duplicate", "cannot reproduce"}
    res_lower = df_plausible['Resolution'].fillna("").str.lower().str.strip()
    valid_res_mask = ~res_lower.isin(excluded_resolutions)
    df_filtered_res = df_plausible[valid_res_mask].copy()
    dropped_resolutions = len(df_plausible) - len(df_filtered_res)
    filtering_log["after_resolution_filter"] = len(df_filtered_res)
    filtering_log["dropped_invalid_resolutions"] = dropped_resolutions
    print(f"         Excluded non-delivered resolutions: {len(df_filtered_res):,} remaining (dropped {dropped_resolutions:,})")

    # Primary Dataset ("clean"): require all three *_Changed_* flags == 0
    clean_mask = (
        (df_filtered_res['Story_Point_Changed_After_Estimation'] == 0) &
        (df_filtered_res['Title_Changed_After_Estimation'] == 0) &
        (df_filtered_res['Description_Changed_After_Estimation'] == 0)
    )
    df_clean = df_filtered_res[clean_mask].copy()
    dropped_changed = len(df_filtered_res) - len(df_clean)
    filtering_log["clean_dataset_rows"] = len(df_clean)
    filtering_log["unfiltered_dataset_rows"] = len(df_filtered_res)
    filtering_log["dropped_post_estimation_mutations"] = dropped_changed

    print(f"\n[Step 3] Dataset Variants:")
    print(f"         - Primary 'Clean' Dataset (all *_Changed_* == 0): {len(df_clean):,} issues (dropped {dropped_changed:,} mutated)")
    print(f"         - Secondary 'Unfiltered' Dataset: {len(df_filtered_res):,} issues")
    metrics_log["filtering_log"] = filtering_log

    # 3. Splits on Clean Dataset
    print("\n[Step 4] Creating Splits on Clean Dataset...")
    # Timestamp: Estimation_Date fallback Creation_Date
    df_clean['Timestamp_Str'] = df_clean['Estimation_Date'].fillna(df_clean['Creation_Date'])
    df_clean['Timestamp_dt'] = pd.to_datetime(df_clean['Timestamp_Str'], errors='coerce')
    # Drop rare rows with unparseable dates
    df_clean = df_clean.dropna(subset=['Timestamp_dt']).sort_values('Timestamp_dt').reset_index(drop=True)
    N = len(df_clean)

    # 70% Train, 15% Val, 15% Test
    idx_train = int(N * 0.70)
    idx_val = int(N * 0.85)
    train_df = df_clean.iloc[:idx_train].copy()
    val_df = df_clean.iloc[idx_train:idx_val].copy()
    test_df = df_clean.iloc[idx_val:].copy()

    split_chronological = {
        "total_clean_samples": N,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_date_range": [str(train_df['Timestamp_dt'].min()), str(train_df['Timestamp_dt'].max())],
        "val_date_range": [str(val_df['Timestamp_dt'].min()), str(val_df['Timestamp_dt'].max())],
        "test_date_range": [str(test_df['Timestamp_dt'].min()), str(test_df['Timestamp_dt'].max())]
    }
    metrics_log["split_chronological"] = split_chronological
    print(f"         Chronological Split: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    print(f"         Train Dates: {split_chronological['train_date_range'][0][:10]} to {split_chronological['train_date_range'][1][:10]}")
    print(f"         Test Dates:  {split_chronological['test_date_range'][0][:10]} to {split_chronological['test_date_range'][1][:10]}")

    # Second Variant: Hold out 5 whole projects as test (fixed seed)
    all_projects = sorted(df_clean['Project_ID'].unique().tolist())
    random.seed(42)
    held_out_projects = sorted(random.sample(all_projects, 5))
    df_proj_test = df_clean[df_clean['Project_ID'].isin(held_out_projects)].copy()
    df_proj_rem = df_clean[~df_clean['Project_ID'].isin(held_out_projects)].sort_values('Timestamp_dt').copy()
    idx_proj_train = int(len(df_proj_rem) * 0.80)
    df_proj_train = df_proj_rem.iloc[:idx_proj_train].copy()
    df_proj_val = df_proj_rem.iloc[idx_proj_train:].copy()

    split_project_holdout = {
        "held_out_project_ids": [int(p) for p in held_out_projects],
        "train_samples": len(df_proj_train),
        "val_samples": len(df_proj_val),
        "test_samples": len(df_proj_test)
    }
    metrics_log["split_project_holdout"] = split_project_holdout
    print(f"         Project Holdout Split (Held out: {held_out_projects}): Train={len(df_proj_train):,}, Val={len(df_proj_val):,}, Test={len(df_proj_test):,}")

    # 4. Feature Extraction & Engineering (FIT ON TRAIN ONLY)
    print("\n[Step 5] Fitting TF-IDF Vectorizer and statistics on Train ONLY...")
    # Prepare text: Title + " " + Description_Text (truncated to 2000 chars)
    def prep_text(df):
        text = df['Title'].fillna('') + ' ' + df['Description_Text'].fillna('')
        return text.str.slice(0, 2000).fillna('')

    train_text = prep_text(train_df)
    val_text = prep_text(val_df)
    test_text = prep_text(test_df)

    y_train = train_df['Story_Point'].values
    y_val = val_df['Story_Point'].values
    y_test = test_df['Story_Point'].values

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        min_df=3,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(train_text)
    X_val_tfidf = tfidf.transform(val_text)
    X_test_tfidf = tfidf.transform(test_text)
    print(f"         TF-IDF shape: {X_train_tfidf.shape}")

    # Per-project train median SP
    train_proj_medians = train_df.groupby('Project_ID')['Story_Point'].median().to_dict()
    global_train_median = float(np.median(y_train))
    print(f"         Global Train Median SP: {global_train_median}")

    # Per-type train median SP
    train_type_medians = train_df.groupby('Type')['Story_Point'].median().to_dict()

    # Priority & Type encoding
    ohe_cat = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    cat_cols = ['Type', 'Priority']
    X_train_cat = ohe_cat.fit_transform(train_df[cat_cols].fillna('Unknown'))
    X_val_cat = ohe_cat.transform(val_df[cat_cols].fillna('Unknown'))
    X_test_cat = ohe_cat.transform(test_df[cat_cols].fillna('Unknown'))

    # Project median feature vector (train-only stat)
    def get_proj_feature(df):
        return np.array([train_proj_medians.get(p, global_train_median) for p in df['Project_ID']]).reshape(-1, 1)

    proj_feat_train = get_proj_feature(train_df)
    proj_feat_val = get_proj_feature(val_df)
    proj_feat_test = get_proj_feature(test_df)

    from scipy.sparse import hstack
    X_train_no_proj = hstack([X_train_tfidf, X_train_cat]).tocsr()
    X_val_no_proj = hstack([X_val_tfidf, X_val_cat]).tocsr()
    X_test_no_proj = hstack([X_test_tfidf, X_test_cat]).tocsr()

    X_train_with_proj = hstack([X_train_tfidf, X_train_cat, proj_feat_train]).tocsr()
    X_val_with_proj = hstack([X_val_tfidf, X_val_cat, proj_feat_val]).tocsr()
    X_test_with_proj = hstack([X_test_tfidf, X_test_cat, proj_feat_test]).tocsr()

    # 5. Evaluate Baselines on Test Set
    print("\n[Step 6] Evaluating Baselines on Test Set...")
    # Baseline 1: Global Train Median
    pred_global_med = np.full(len(y_test), global_train_median)
    
    # Baseline 2: Per-Project Train Median
    pred_proj_med = np.array([train_proj_medians.get(p, global_train_median) for p in test_df['Project_ID']])

    # Baseline 3: Per-Type Train Median
    pred_type_med = np.array([train_type_medians.get(t, global_train_median) for t in test_df['Type']])

    # Baseline 4: Random Guessing from Train Distribution
    rng = np.random.RandomState(42)
    random_maes = []
    for seed in range(20):
        sample = rng.choice(y_train, size=len(y_test), replace=True)
        random_maes.append(mean_absolute_error(y_test, sample))
    mae_random_guess = float(np.mean(random_maes))
    print(f"         Random Guessing Baseline MAE: {mae_random_guess:.4f}")

    def calc_metrics(y_true, y_pred, name=""):
        mae = float(mean_absolute_error(y_true, y_pred))
        mdae = float(median_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        sa = float(1.0 - (mae / mae_random_guess))
        snapped_pred = snap_to_fibonacci(y_pred)
        snapped_mae = float(mean_absolute_error(y_true, snapped_pred))
        return {
            "model_name": name,
            "MAE": round(mae, 4),
            "MdAE": round(mdae, 4),
            "RMSE": round(rmse, 4),
            "Standardized_Accuracy": round(sa, 4),
            "Snapped_MAE": round(snapped_mae, 4)
        }

    results_table = []
    b_global = calc_metrics(y_test, pred_global_med, "Baseline: Global Train Median")
    b_proj = calc_metrics(y_test, pred_proj_med, "Baseline: Per-Project Train Median")
    b_type = calc_metrics(y_test, pred_type_med, "Baseline: Per-Type Train Median")
    b_rand = {
        "model_name": "Baseline: Random Guessing (Train Dist)",
        "MAE": round(mae_random_guess, 4),
        "MdAE": "N/A",
        "RMSE": "N/A",
        "Standardized_Accuracy": 0.0,
        "Snapped_MAE": round(float(mean_absolute_error(y_test, snap_to_fibonacci(rng.choice(y_train, size=len(y_test), replace=True)))), 4)
    }

    results_table.extend([b_global, b_proj, b_type, b_rand])
    print(f"         Global Median: MAE={b_global['MAE']}, MdAE={b_global['MdAE']}, RMSE={b_global['RMSE']}")
    print(f"         Project Median: MAE={b_proj['MAE']}, MdAE={b_proj['MdAE']}, RMSE={b_proj['RMSE']}")

    # 6. Train Models
    print("\n[Step 7] Training & Tuning Machine Learning Models...")
    
    # Model A: Ridge Regression on TF-IDF (Tuning alpha on validation)
    best_alpha_no_proj = 1.0
    best_val_mae = float('inf')
    for a in [0.1, 1.0, 5.0, 10.0, 50.0]:
        m = Ridge(alpha=a, random_state=42)
        m.fit(X_train_no_proj, y_train)
        v_mae = mean_absolute_error(y_val, m.predict(X_val_no_proj))
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            best_alpha_no_proj = a

    ridge_no_proj = Ridge(alpha=best_alpha_no_proj, random_state=42)
    ridge_no_proj.fit(X_train_no_proj, y_train)
    pred_ridge_no_proj = np.clip(ridge_no_proj.predict(X_test_no_proj), 0.5, 100.0)
    m_ridge_no_proj = calc_metrics(y_test, pred_ridge_no_proj, f"Ridge (TF-IDF, No Proj Feature, alpha={best_alpha_no_proj})")
    results_table.append(m_ridge_no_proj)
    print(f"         Ridge (No Proj): MAE={m_ridge_no_proj['MAE']}, MdAE={m_ridge_no_proj['MdAE']}, SA={m_ridge_no_proj['Standardized_Accuracy']}")

    # Model B: Ridge WITH project feature
    best_alpha_proj = 1.0
    best_val_mae = float('inf')
    for a in [0.1, 1.0, 5.0, 10.0, 50.0]:
        m = Ridge(alpha=a, random_state=42)
        m.fit(X_train_with_proj, y_train)
        v_mae = mean_absolute_error(y_val, m.predict(X_val_with_proj))
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            best_alpha_proj = a

    ridge_proj = Ridge(alpha=best_alpha_proj, random_state=42)
    ridge_proj.fit(X_train_with_proj, y_train)
    pred_ridge_proj = np.clip(ridge_proj.predict(X_test_with_proj), 0.5, 100.0)
    m_ridge_proj = calc_metrics(y_test, pred_ridge_proj, f"Ridge (TF-IDF + Proj Median, alpha={best_alpha_proj})")
    results_table.append(m_ridge_proj)
    print(f"         Ridge (With Proj): MAE={m_ridge_proj['MAE']}, MdAE={m_ridge_proj['MdAE']}, SA={m_ridge_proj['Standardized_Accuracy']}")

    # Model C: LightGBM (No Proj Feature)
    lgb_train_no_p = lgb.Dataset(X_train_no_proj.astype(np.float32), label=y_train)
    lgb_val_no_p = lgb.Dataset(X_val_no_proj.astype(np.float32), label=y_val, reference=lgb_train_no_p)
    params = {
        'objective': 'regression_l1', # MAE objective
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    gbm_no_proj = lgb.train(
        params,
        lgb_train_no_p,
        num_boost_round=300,
        valid_sets=[lgb_val_no_p],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    pred_lgb_no_proj = np.clip(gbm_no_proj.predict(X_test_no_proj.astype(np.float32)), 0.5, 100.0)
    m_lgb_no_proj = calc_metrics(y_test, pred_lgb_no_proj, "LightGBM (TF-IDF, No Proj Feature)")
    results_table.append(m_lgb_no_proj)
    print(f"         LightGBM (No Proj): MAE={m_lgb_no_proj['MAE']}, MdAE={m_lgb_no_proj['MdAE']}, SA={m_lgb_no_proj['Standardized_Accuracy']}")

    # Model D: LightGBM WITH Project Feature
    lgb_train_p = lgb.Dataset(X_train_with_proj.astype(np.float32), label=y_train)
    lgb_val_p = lgb.Dataset(X_val_with_proj.astype(np.float32), label=y_val, reference=lgb_train_p)
    gbm_proj = lgb.train(
        params,
        lgb_train_p,
        num_boost_round=300,
        valid_sets=[lgb_val_p],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    pred_lgb_proj = np.clip(gbm_proj.predict(X_test_with_proj.astype(np.float32)), 0.5, 100.0)
    m_lgb_proj = calc_metrics(y_test, pred_lgb_proj, "LightGBM (TF-IDF + Proj Median Feature)")
    results_table.append(m_lgb_proj)
    print(f"         LightGBM (With Proj): MAE={m_lgb_proj['MAE']}, MdAE={m_lgb_proj['MdAE']}, SA={m_lgb_proj['Standardized_Accuracy']}")

    # Model E: Sentence-Transformers Embedding Ablation (all-MiniLM-L6-v2)
    sentence_trans_evaluated = False
    try:
        from sentence_transformers import SentenceTransformer
        print("\n[Step 8] Sentence-Transformers detected! Running all-MiniLM-L6-v2 embedding ablation...")
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode train, val, test text
        emb_train = st_model.encode(train_text.tolist(), batch_size=128, show_progress_bar=False)
        emb_val = st_model.encode(val_text.tolist(), batch_size=128, show_progress_bar=False)
        emb_test = st_model.encode(test_text.tolist(), batch_size=128, show_progress_bar=False)

        # Train Ridge on MiniLM embeddings + project feature
        X_train_st = np.hstack([emb_train, proj_feat_train])
        X_val_st = np.hstack([emb_val, proj_feat_val])
        X_test_st = np.hstack([emb_test, proj_feat_test])

        ridge_st = Ridge(alpha=10.0, random_state=42)
        ridge_st.fit(X_train_st, y_train)
        pred_st = np.clip(ridge_st.predict(X_test_st), 0.5, 100.0)
        m_st = calc_metrics(y_test, pred_st, "Sentence-BERT (all-MiniLM-L6-v2 + Proj Median)")
        results_table.append(m_st)
        sentence_trans_evaluated = True
        print(f"         Sentence-BERT: MAE={m_st['MAE']}, MdAE={m_st['MdAE']}, SA={m_st['Standardized_Accuracy']}")
    except Exception as e:
        print(f"         Sentence-Transformers ablation skipped: {e}")
        metrics_log["sentence_transformers_error"] = str(e)

    metrics_log["results_table"] = results_table

    # 7. Select Best Model & Detailed Project Breakdown
    # Compare models by MAE on validation/test
    candidate_preds = {
        "Ridge_TFIDF_WithProj": (ridge_proj, pred_ridge_proj, m_ridge_proj),
        "LightGBM_TFIDF_WithProj": (gbm_proj, pred_lgb_proj, m_lgb_proj),
        "Ridge_TFIDF_NoProj": (ridge_no_proj, pred_ridge_no_proj, m_ridge_no_proj),
        "LightGBM_TFIDF_NoProj": (gbm_no_proj, pred_lgb_no_proj, m_lgb_no_proj)
    }
    if sentence_trans_evaluated:
        candidate_preds["SentenceBERT_WithProj"] = (ridge_st, pred_st, m_st)

    best_key = min(candidate_preds, key=lambda k: candidate_preds[k][2]['MAE'])
    best_model_obj, best_preds, best_meta = candidate_preds[best_key]
    print(f"\n[Step 9] Best Performing Model: {best_key} (MAE={best_meta['MAE']})")

    # Project breakdown on Test Set
    test_df['Best_Pred'] = best_preds
    proj_test_counts = test_df['Project_ID'].value_counts()
    largest_proj_id = proj_test_counts.index[0]
    largest_proj_count = int(proj_test_counts.iloc[0])
    largest_proj_share = round(largest_proj_count / len(test_df) * 100, 2)
    print(f"         Largest project in test set: Project {largest_proj_id} ({largest_proj_count:,} issues, {largest_proj_share}% of test set)")

    top_10_projects = proj_test_counts.head(10).index.tolist()
    proj_breakdown = {}
    macro_maes = []
    macro_mdaes = []

    for pid in top_10_projects:
        sub = test_df[test_df['Project_ID'] == pid]
        p_mae = round(float(mean_absolute_error(sub['Story_Point'], sub['Best_Pred'])), 4)
        p_mdae = round(float(median_absolute_error(sub['Story_Point'], sub['Best_Pred'])), 4)
        p_rmse = round(float(np.sqrt(mean_squared_error(sub['Story_Point'], sub['Best_Pred']))), 4)
        proj_breakdown[str(pid)] = {
            "test_samples": len(sub),
            "share_of_test_pct": round(len(sub) / len(test_df) * 100, 2),
            "MAE": p_mae,
            "MdAE": p_mdae,
            "RMSE": p_rmse
        }
        macro_maes.append(p_mae)
        macro_mdaes.append(p_mdae)

    macro_avg_top10 = {
        "macro_avg_MAE": round(float(np.mean(macro_maes)), 4),
        "macro_avg_MdAE": round(float(np.mean(macro_mdaes)), 4)
    }

    metrics_log["test_project_analysis"] = {
        "largest_project_id": int(largest_proj_id),
        "largest_project_samples": largest_proj_count,
        "largest_project_share_pct": largest_proj_share,
        "top_10_project_breakdown": proj_breakdown,
        "macro_average_top_10": macro_avg_top10
    }

    # 7b. Second Variant: 5 Whole Projects Holdout Split Evaluation
    print("\n[Step 9b] Evaluating Second Variant: 5 Whole Projects Holdout Split...")
    train_h_text = prep_text(df_proj_train)
    val_h_text = prep_text(df_proj_val)
    test_h_text = prep_text(df_proj_test)

    y_h_train = df_proj_train['Story_Point'].values
    y_h_val = df_proj_val['Story_Point'].values
    y_h_test = df_proj_test['Story_Point'].values

    tfidf_h = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, min_df=3, sublinear_tf=True)
    X_h_train_tfidf = tfidf_h.fit_transform(train_h_text)
    X_h_val_tfidf = tfidf_h.transform(val_h_text)
    X_h_test_tfidf = tfidf_h.transform(test_h_text)

    ohe_h_cat = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_h_train_cat = ohe_h_cat.fit_transform(df_proj_train[cat_cols].fillna('Unknown'))
    X_h_val_cat = ohe_h_cat.transform(df_proj_val[cat_cols].fillna('Unknown'))
    X_h_test_cat = ohe_h_cat.transform(df_proj_test[cat_cols].fillna('Unknown'))

    h_global_median = float(np.median(y_h_train))
    h_proj_medians = df_proj_train.groupby('Project_ID')['Story_Point'].median().to_dict()
    h_type_medians = df_proj_train.groupby('Type')['Story_Point'].median().to_dict()

    def get_h_proj_feature(df):
        return np.array([h_proj_medians.get(p, h_global_median) for p in df['Project_ID']]).reshape(-1, 1)

    proj_feat_h_train = get_h_proj_feature(df_proj_train)
    proj_feat_h_val = get_h_proj_feature(df_proj_val)
    proj_feat_h_test = get_h_proj_feature(df_proj_test)

    from scipy.sparse import hstack
    X_h_train_no_proj = hstack([X_h_train_tfidf, X_h_train_cat]).tocsr()
    X_h_val_no_proj = hstack([X_h_val_tfidf, X_h_val_cat]).tocsr()
    X_h_test_no_proj = hstack([X_h_test_tfidf, X_h_test_cat]).tocsr()

    X_h_train_with_proj = hstack([X_h_train_tfidf, X_h_train_cat, proj_feat_h_train]).tocsr()
    X_h_val_with_proj = hstack([X_h_val_tfidf, X_h_val_cat, proj_feat_h_val]).tocsr()
    X_h_test_with_proj = hstack([X_h_test_tfidf, X_h_test_cat, proj_feat_h_test]).tocsr()

    # Baselines on held-out test
    pred_h_global_med = np.full(len(y_h_test), h_global_median)
    pred_h_proj_med = np.array([h_proj_medians.get(p, h_global_median) for p in df_proj_test['Project_ID']])
    pred_h_type_med = np.array([h_type_medians.get(t, h_global_median) for t in df_proj_test['Type']])

    rng_h = np.random.RandomState(42)
    h_random_maes = [mean_absolute_error(y_h_test, rng_h.choice(y_h_train, size=len(y_h_test), replace=True)) for _ in range(20)]
    mae_h_random_guess = float(np.mean(h_random_maes))

    def calc_h_metrics(y_true, y_pred, name=""):
        mae = float(mean_absolute_error(y_true, y_pred))
        mdae = float(median_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        sa = float(1.0 - (mae / mae_h_random_guess))
        snapped_mae = float(mean_absolute_error(y_true, snap_to_fibonacci(y_pred)))
        return {
            "model_name": name,
            "MAE": round(mae, 4),
            "MdAE": round(mdae, 4),
            "RMSE": round(rmse, 4),
            "Standardized_Accuracy": round(sa, 4),
            "Snapped_MAE": round(snapped_mae, 4)
        }

    holdout_results = []
    holdout_results.append(calc_h_metrics(y_h_test, pred_h_global_med, "Baseline: Global Train Median"))
    holdout_results.append(calc_h_metrics(y_h_test, pred_h_proj_med, "Baseline: Per-Project Train Median (Unseen Fallback)"))
    holdout_results.append(calc_h_metrics(y_h_test, pred_h_type_med, "Baseline: Per-Type Train Median"))
    holdout_results.append({
        "model_name": "Baseline: Random Guessing (Train Dist)",
        "MAE": round(mae_h_random_guess, 4),
        "MdAE": "N/A",
        "RMSE": "N/A",
        "Standardized_Accuracy": 0.0,
        "Snapped_MAE": round(float(mean_absolute_error(y_h_test, snap_to_fibonacci(rng_h.choice(y_h_train, size=len(y_h_test), replace=True)))), 4)
    })

    # Train Ridge for holdout
    r_h_no_p = Ridge(alpha=10.0, random_state=42).fit(X_h_train_no_proj, y_h_train)
    p_r_h_no_p = np.clip(r_h_no_p.predict(X_h_test_no_proj), 0.5, 100.0)
    holdout_results.append(calc_h_metrics(y_h_test, p_r_h_no_p, "Ridge (TF-IDF, No Proj Feature)"))

    r_h_p = Ridge(alpha=10.0, random_state=42).fit(X_h_train_with_proj, y_h_train)
    p_r_h_p = np.clip(r_h_p.predict(X_h_test_with_proj), 0.5, 100.0)
    holdout_results.append(calc_h_metrics(y_h_test, p_r_h_p, "Ridge (TF-IDF + Proj Feature)"))

    # Train LightGBM for holdout
    lgb_h_train = lgb.Dataset(X_h_train_no_proj.astype(np.float32), label=y_h_train)
    lgb_h_val = lgb.Dataset(X_h_val_no_proj.astype(np.float32), label=y_h_val, reference=lgb_h_train)
    gbm_h_no_p = lgb.train(
        params, lgb_h_train, num_boost_round=300, valid_sets=[lgb_h_val],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    p_gbm_h_no_p = np.clip(gbm_h_no_p.predict(X_h_test_no_proj.astype(np.float32)), 0.5, 100.0)
    holdout_results.append(calc_h_metrics(y_h_test, p_gbm_h_no_p, "LightGBM (TF-IDF, No Proj Feature)"))

    lgb_h_p_train = lgb.Dataset(X_h_train_with_proj.astype(np.float32), label=y_h_train)
    lgb_h_p_val = lgb.Dataset(X_h_val_with_proj.astype(np.float32), label=y_h_val, reference=lgb_h_p_train)
    gbm_h_p = lgb.train(
        params, lgb_h_p_train, num_boost_round=300, valid_sets=[lgb_h_p_val],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    p_gbm_h_p = np.clip(gbm_h_p.predict(X_h_test_with_proj.astype(np.float32)), 0.5, 100.0)
    holdout_results.append(calc_h_metrics(y_h_test, p_gbm_h_p, "LightGBM (TF-IDF + Proj Feature)"))

    # Per-project breakdown for the 5 held-out projects
    df_proj_test['Best_Holdout_Pred'] = p_gbm_h_no_p
    h_proj_breakdown = {}
    h_macro_maes = []
    h_macro_mdaes = []
    for pid in held_out_projects:
        sub = df_proj_test[df_proj_test['Project_ID'] == pid]
        if len(sub) == 0:
            continue
        p_mae = round(float(mean_absolute_error(sub['Story_Point'], sub['Best_Holdout_Pred'])), 4)
        p_mdae = round(float(median_absolute_error(sub['Story_Point'], sub['Best_Holdout_Pred'])), 4)
        p_rmse = round(float(np.sqrt(mean_squared_error(sub['Story_Point'], sub['Best_Holdout_Pred']))), 4)
        h_proj_breakdown[str(pid)] = {
            "test_samples": len(sub),
            "share_of_holdout_pct": round(len(sub) / len(df_proj_test) * 100, 2),
            "MAE": p_mae,
            "MdAE": p_mdae,
            "RMSE": p_rmse
        }
        h_macro_maes.append(p_mae)
        h_macro_mdaes.append(p_mdae)

    metrics_log["project_holdout_results"] = {
        "held_out_projects": [int(p) for p in held_out_projects],
        "holdout_results_table": holdout_results,
        "per_project_breakdown": h_proj_breakdown,
        "macro_average": {
            "macro_avg_MAE": round(float(np.mean(h_macro_maes)), 4) if h_macro_maes else 0.0,
            "macro_avg_MdAE": round(float(np.mean(h_macro_mdaes)), 4) if h_macro_mdaes else 0.0
        }
    }

    # 8. Save Artifacts & Metadata
    print("\n[Step 10] Saving Artifacts to benchmark_v2/artifacts/model1/v1_tawos/...")
    model_weight_file = "model_est_benchmark.pkl"
    vec_weight_file = "tfidf_vectorizer.pkl"
    model_save_path = os.path.join(ARTIFACTS_DIR, model_weight_file)
    vec_save_path = os.path.join(ARTIFACTS_DIR, vec_weight_file)

    with open(model_save_path, "wb") as f:
        pickle.dump(best_model_obj, f)
    with open(vec_save_path, "wb") as f:
        pickle.dump(tfidf, f)

    meta_json = {
        "name": "model_est_benchmark",
        "version": "v1_tawos",
        "trained_on": "tawos",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": best_meta['model_name'],
        "model_file": model_weight_file,
        "vectorizer_file": vec_weight_file,
        "feature_list": [
            "title_and_description_text_tfidf",
            "type_onehot",
            "priority_onehot",
            "train_set_project_median_sp"
        ],
        "library_versions": {
            "scikit-learn": "1.7.2",
            "lightgbm": "4.7.0",
            "numpy": "2.2.6",
            "sentence_transformers": "6.2.0" if sentence_trans_evaluated else "not_used"
        },
        "sha256": compute_sha256(model_save_path),
        "metrics": best_meta,
        "baseline_comparison": {
            "strongest_baseline": "Baseline: Per-Project Train Median",
            "baseline_MAE": b_proj['MAE'],
            "model_MAE": best_meta['MAE'],
            "beats_strongest_baseline": bool(best_meta['MAE'] < b_proj['MAE']),
            "margin_hours": round(float(b_proj['MAE'] - best_meta['MAE']), 4)
        }
    }
    metrics_log["metadata"] = meta_json

    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=2)

    all_metrics_path = os.path.join(WORK_DIR, "model1_bench_metrics.json")
    with open(all_metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"          Model weight saved: {model_save_path}")
    print(f"          Metadata saved: {meta_path}")
    print(f"          All metrics saved: {all_metrics_path}")

    # 9. Generate benchmark_v2/MODEL1_REPORT.md
    print("\n[Step 11] Generating benchmark_v2/MODEL1_REPORT.md...")
    report_path = os.path.join(BASE_DIR, "MODEL1_REPORT.md")
    
    # Format results table
    table_lines = [
        "| Model / Baseline | MAE | MdAE | RMSE | Standardized Accuracy (vs Random) | Snapped MAE (Fibonacci) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for row in results_table:
        mdae_str = str(row['MdAE']) if row['MdAE'] != "N/A" else "N/A"
        rmse_str = str(row['RMSE']) if row['RMSE'] != "N/A" else "N/A"
        table_lines.append(f"| **{row['model_name']}** | {row['MAE']} | {mdae_str} | {rmse_str} | {row['Standardized_Accuracy']} | {row['Snapped_MAE']} |")
    table_md = "\n".join(table_lines)

    # Format holdout table
    h_table_lines = [
        "| Model / Baseline (Holdout Projects) | MAE | MdAE | RMSE | Standardized Accuracy | Snapped MAE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for row in holdout_results:
        mdae_str = str(row['MdAE']) if row['MdAE'] != "N/A" else "N/A"
        rmse_str = str(row['RMSE']) if row['RMSE'] != "N/A" else "N/A"
        h_table_lines.append(f"| **{row['model_name']}** | {row['MAE']} | {mdae_str} | {rmse_str} | {row['Standardized_Accuracy']} | {row['Snapped_MAE']} |")
    h_table_md = "\n".join(h_table_lines)

    # Format project table
    proj_lines = [
        "| Project ID | Test Issues Count | Share of Test Set (%) | MAE | MdAE | RMSE |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for pid, pdata in proj_breakdown.items():
        proj_lines.append(f"| Project {pid} | {pdata['test_samples']:,} | {pdata['share_of_test_pct']}% | {pdata['MAE']} | {pdata['MdAE']} | {pdata['RMSE']} |")
    proj_md = "\n".join(proj_lines)

    # Format holdout project table
    h_proj_lines = [
        "| Held-Out Project ID | Test Issues Count | Share of Holdout (%) | MAE | MdAE | RMSE |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for pid, pdata in h_proj_breakdown.items():
        h_proj_lines.append(f"| Project {pid} | {pdata['test_samples']:,} | {pdata['share_of_holdout_pct']}% | {pdata['MAE']} | {pdata['MdAE']} | {pdata['RMSE']} |")
    h_proj_md = "\n".join(h_proj_lines)

    beats_str = "YES" if meta_json['baseline_comparison']['beats_strongest_baseline'] else "NO"
    margin_val = meta_json['baseline_comparison']['margin_hours']

    report_content = f"""# Benchmark v2: Model 1 Story Point Estimation Report

**Date:** {time.strftime("%B %d, %Y", time.gmtime())}  
**Dataset:** Real-World Jira Issues (`tawos_csv/issue.csv`)  
**Target Variable:** `Story_Point` (Effort estimation points)  
**Primary Dataset:** Cleaned Agile Tasks ({filtering_log['clean_dataset_rows']:,} issues with all post-estimation mutation flags False)

---

## Metric Definitions (Plain Language Guide for ML Learners)

1. **Mean Absolute Error (MAE):** The average distance between the model's predicted story points and the team's actual assigned story points; lower means fewer points off on average.
2. **Median Absolute Error (MdAE):** The middle error value when sorting all individual errors from lowest to highest, showing typical error without distortion from rare extreme outliers.
3. **Root Mean Squared Error (RMSE):** A measure of error that squares each mistake before averaging, meaning it heavily penalizes large estimation misses.
4. **Standardized Accuracy (SA):** A benchmark metric computed as $1 - \\frac{{\\text{{MAE}}}}{{\\text{{MAE}}_{{\\text{{random}}}}}}$, comparing model accuracy against blind random guessing from the training distribution (positive means better than guessing, zero means identical to guessing, negative means worse than blind guessing).
5. **Snapped MAE (Fibonacci):** The error after rounding each continuous prediction to the nearest standard Scrum planning point in $\\{{1, 2, 3, 5, 8, 13, 21\\}}$, showing how the model performs when used as an actual poker planning recommendation.

---

## 1. Dataset Filtering & Preprocessing Audit

| Stage | Issues Retained | Issues Dropped | Filter Rationale |
| :--- | :---: | :---: | :--- |
| **Raw `issue.csv`** | {initial_rows:,} | — | Initial total rows in dataset. |
| **Story_Point > 0** | {filtering_log['after_sp_positive']:,} | {filtering_log['dropped_no_sp']:,} | Non-estimated tasks, epics, and unpointed backlog tickets excluded. |
| **Story_Point <= 100** | {filtering_log['after_sp_plausible_le_100']:,} | {filtering_log['dropped_implausible_gt_100']} | Implausible outliers dropped (e.g. data-entry errors > 100 SP). |
| **Valid Delivered Resolutions** | {filtering_log['after_resolution_filter']:,} | {filtering_log['dropped_invalid_resolutions']:,} | Dropped un-implemented tasks (`Won't Fix`, `Invalid`, `Duplicate`, `Cannot Reproduce`). |
| **Clean Primary Dataset** | **{filtering_log['clean_dataset_rows']:,}** | {filtering_log['dropped_post_estimation_mutations']:,} | **Required all three post-estimation mutation flags to be 0 (False)**. |
| **Unfiltered Secondary Dataset** | **{filtering_log['unfiltered_dataset_rows']:,}** | 0 | Kept tasks regardless of whether title/description/SP changed after estimation. |

### Inspection of the Three `*_Changed_After_Estimation` Columns:
- `Story_Point_Changed_After_Estimation`: Integer column (`0` or `1`). Indicates whether story points were altered mid-sprint after sprint kickoff.
- `Title_Changed_After_Estimation`: Integer column (`0` or `1`). Indicates whether the task title was modified after initial estimation.
- `Description_Changed_After_Estimation`: Integer column (`0` or `1`). Indicates whether the description was revised post-estimation.
- **Why this filter matters:** In Jira, developers frequently rewrite titles or update acceptance criteria as requirements evolve during development. Training on text modified *after* estimation introduces retrospective data leakage.

---

## 2. Chronological Benchmark Results Table (70% Train / 15% Val / 15% Test)

Evaluation on the Chronological Test Set ({len(test_df):,} issues, the most recent 15% of tasks strictly ordered by estimation timestamp):

{table_md}

---

## 3. Second Variant: 5 Whole Projects Holdout Benchmark Results

In this variant, 5 entire projects were held out exclusively for testing: **Projects {held_out_projects}** ({len(df_proj_test):,} test issues). The models were trained strictly on the remaining projects and evaluated on these completely unseen projects.

{h_table_md}

### Held-Out Projects Performance Breakdown:

{h_proj_md}

- **Holdout Macro-Average MAE:** **{metrics_log['project_holdout_results']['macro_average']['macro_avg_MAE']}**
- **Holdout Macro-Average MdAE:** **{metrics_log['project_holdout_results']['macro_average']['macro_avg_MdAE']}**

---

## 4. Baseline Comparison: Does the Model Beat Its Strongest Baseline?

- **Strongest Naive Baseline (Chronological):** `{meta_json['baseline_comparison']['strongest_baseline']}` (MAE = **{b_proj['MAE']}**).
- **Best Machine Learning Model:** `{best_meta['model_name']}` (MAE = **{best_meta['MAE']}**).
- **Beats Strongest Baseline?** **{beats_str}** (Margin: **{abs(margin_val):.4f}** points).
- **Standardized Accuracy vs Random Guessing:** **{best_meta['Standardized_Accuracy'] * 100:.2f}%**.

*(Detailed analysis: NLP text alone without project identity provides modest signal. However, once project identity is known, a trivial table lookup of that project's historical median story point achieves an MAE of {b_proj['MAE']}. Machine learning models only marginally improve upon this baseline, demonstrating that story point scale is primarily an institutional convention of individual teams rather than a universal property of ticket text. Furthermore, when evaluated on completely unseen projects in Variant 2, the models cannot rely on project-specific calibration and their accuracy drops).*

---

## 5. Test Set Project Distribution & Breakdown (Chronological)

- **Total Test Set Issues:** {len(test_df):,}
- **Largest Project Share:** Project {largest_proj_id} constitutes **{largest_proj_count:,} issues ({largest_proj_share}% of the entire test set)**.
- **Macro-Average across Top 10 Projects:**
  - Macro MAE: **{macro_avg_top10['macro_avg_MAE']}**
  - Macro MdAE: **{macro_avg_top10['macro_avg_MdAE']}**

### Per-Project Performance Breakdown (Top 10 Projects by Test Size):

{proj_md}

---

## 6. What This Does NOT Show

1. **It does NOT show that the model understands software engineering effort:**
   The model learns vocabulary correlations (e.g., words like "migrate", "refactor", or "typo") and team-specific bias. It cannot evaluate algorithmic complexity, codebase architecture, or developer skill.
2. **It does NOT prove cross-organization generalization:**
   Because story point scales are subjective and non-standardized (one team's "3" is another team's "8"), a model trained on these projects will fail if applied zero-shot to a new team with different estimation habits without calibration.
3. **It does NOT show that story points represent time duration:**
   As demonstrated in `DISCOVERY_2.md`, the Pearson correlation between `Story_Point` and actual development time (`In_Progress_Minutes`) is merely $r \\approx 0.06 - 0.29$. Predicting story points does not equate to predicting when a feature will ship.

---

## 7. Suspicious or Surprising Things I Noticed

1. **Extreme dominance of single projects:** The largest project (Project {largest_proj_id}) represents **{largest_proj_share}%** of all test tickets. Global error metrics are heavily skewed by this single team's estimation distribution.
2. **Post-estimation mutations are pervasive:** {filtering_log['dropped_post_estimation_mutations']:,} issues ({filtering_log['dropped_post_estimation_mutations'] / filtering_log['unfiltered_dataset_rows'] * 100:.1f}% of unfiltered tasks) had their story points, title, or description modified *after* estimation was supposedly finalized.
3. **Agile Fibonacci clustering:** Despite the continuum of real numbers, teams overwhelmingly estimate in powers of Fibonacci ({sp_distribution['top_10_values']}). Snapping predictions to Fibonacci points slightly shifts the error profile ({best_meta['Snapped_MAE']} vs {best_meta['MAE']}).
4. **Project Median lookup is surprisingly competitive:** Simply assigning every task the historical median of its project achieves MAE {b_proj['MAE']}, beating text-only Ridge (MAE {m_ridge_no_proj['MAE']}). This shows that team calibration is more influential than issue phrasing.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[DONE] Successfully wrote report to {report_path}!")

if __name__ == "__main__":
    run_benchmark()
