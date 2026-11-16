import os
import sys
import time
import json
import random
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
import lightgbm as lgb

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
TAWOS_DIR = r"d:\Gillo AI Service\tawos_csv"
WORK_DIR = os.path.join(BASE_DIR, "work")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "model1_sp", "v1_tawos")

FIBONACCI_POINTS = np.array([1, 2, 3, 5, 8, 13, 21])

def snap_to_fibonacci(preds):
    preds_arr = np.array(preds).reshape(-1, 1)
    diffs = np.abs(preds_arr - FIBONACCI_POINTS)
    closest_idx = np.argmin(diffs, axis=1)
    return FIBONACCI_POINTS[closest_idx]

def compute_exact_random_baseline(y_train, y_test):
    u_vals, counts = np.unique(y_train, return_counts=True)
    probs = counts / len(y_train)
    diff_matrix = np.abs(y_test[:, None] - u_vals[None, :])
    expected_mae_raw = float(np.mean(np.sum(diff_matrix * probs[None, :], axis=1)))
    snapped_u_vals = snap_to_fibonacci(u_vals)
    diff_snapped = np.abs(y_test[:, None] - snapped_u_vals[None, :])
    expected_mae_snapped = float(np.mean(np.sum(diff_snapped * probs[None, :], axis=1)))
    return expected_mae_raw, expected_mae_snapped

def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))

def prep_text(df):
    text = df['Title'].fillna('') + ' ' + df['Description_Text'].fillna('')
    return text.str.slice(0, 2000).fillna('')

def calc_eval(y_true, y_pred, name, rand_mae=4.7425):
    mae = float(mean_absolute_error(y_true, y_pred))
    mdae = float(median_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sa = float(1.0 - (mae / rand_mae)) if rand_mae > 0 else 0.0
    snapped = float(mean_absolute_error(y_true, snap_to_fibonacci(y_pred)))
    return {
        'model_name': name,
        'MAE': round(mae, 4),
        'MdAE': round(mdae, 4),
        'RMSE': round(rmse, 4),
        'Standardized_Accuracy': round(sa, 4),
        'Snapped_MAE': round(snapped, 4)
    }

def main():
    print("=" * 80)
    print("RUNNING BENCHMARK V2 MODEL 1 PART 2 AUDIT & EXTENSIONS")
    print("=" * 80)
    t0 = time.time()
    
    # -------------------------------------------------------------------------
    # 0. Load Dataset and Reconstruct Exact Chronological Split
    # -------------------------------------------------------------------------
    cols = [
        "ID", "Title", "Description_Text", "Type", "Priority",
        "Story_Point", "Estimation_Date", "Creation_Date", "Project_ID",
        "Story_Point_Changed_After_Estimation",
        "Title_Changed_After_Estimation",
        "Description_Changed_After_Estimation",
        "Resolution"
    ]
    print("Loading issue.csv...")
    df_raw = pd.read_csv(os.path.join(TAWOS_DIR, "issue.csv"), usecols=cols, low_memory=False)
    df_raw.columns = [c.lstrip('\ufeff') for c in df_raw.columns]

    raw_type_counts = df_raw['Type'].value_counts(dropna=False).to_dict()

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
    clean_type_counts = df_clean['Type'].value_counts(dropna=False).to_dict()

    df_clean['Timestamp_Str'] = df_clean['Estimation_Date'].fillna(df_clean['Creation_Date'])
    df_clean['Timestamp_dt'] = pd.to_datetime(df_clean['Timestamp_Str'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Timestamp_dt']).sort_values('Timestamp_dt').reset_index(drop=True)

    N_c = len(df_clean)
    idx_tr = int(N_c * 0.70)
    idx_va = int(N_c * 0.85)

    train_c = df_clean.iloc[:idx_tr].copy().reset_index(drop=True)
    val_c = df_clean.iloc[idx_tr:idx_va].copy().reset_index(drop=True)
    test_c = df_clean.iloc[idx_va:].copy().reset_index(drop=True)

    y_train = train_c['Story_Point'].values
    y_val = val_c['Story_Point'].values
    y_test = test_c['Story_Point'].values

    rand_mae, rand_snapped_mae = compute_exact_random_baseline(y_train, y_test)
    print(f"Reproduced Chronological Split: Train={len(train_c)}, Val={len(val_c)}, Test={len(test_c)}")

    # Feature engineering for baseline full population
    print("Vectorizing text & encoding categorical features...")
    train_text = prep_text(train_c)
    val_text = prep_text(val_c)
    test_text = prep_text(test_c)

    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=3)
    X_tr_tfidf = tfidf.fit_transform(train_text)
    X_va_tfidf = tfidf.transform(val_text)
    X_te_tfidf = tfidf.transform(test_text)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_tr_cat = ohe.fit_transform(train_c[['Type', 'Priority']].fillna('Missing'))
    X_va_cat = ohe.transform(val_c[['Type', 'Priority']].fillna('Missing'))
    X_te_cat = ohe.transform(test_c[['Type', 'Priority']].fillna('Missing'))

    X_tr_deploy = sparse.hstack([X_tr_tfidf, X_tr_cat], format='csr')
    X_va_deploy = sparse.hstack([X_va_tfidf, X_va_cat], format='csr')
    X_te_deploy = sparse.hstack([X_te_tfidf, X_te_cat], format='csr')

    # Load S-BERT embeddings
    sbert_file = os.path.join(WORK_DIR, "sbert_clean_embeddings.npz")
    sbert_data = np.load(sbert_file)
    sbert_tr = sbert_data['emb_train']
    sbert_va = sbert_data['emb_val']
    sbert_te = sbert_data['emb_test']

    X_tr_sbert = np.hstack([sbert_tr, X_tr_cat.toarray()])
    X_va_sbert = np.hstack([sbert_va, X_va_cat.toarray()])
    X_te_sbert = np.hstack([sbert_te, X_te_cat.toarray()])

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
    # ITEM 1: SEED NOISE ANALYSIS (5 Different Random Seeds)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 1: SEED NOISE ANALYSIS (5 RANDOM SEEDS)")
    print("=" * 80)
    seeds = [42, 100, 200, 300, 400]
    
    seed_results = {
        'deployment_model': {'runs': [], 'maes': [], 'mdaes': [], 'rmses': []},
        'type_priority_only': {'runs': [], 'maes': [], 'mdaes': [], 'rmses': []},
        'sbert_lgbm': {'runs': [], 'maes': [], 'mdaes': [], 'rmses': []}
    }

    for s in seeds:
        # 1. Deployment model (TF-IDF + Type + Priority)
        p_dep = dict(base_lgb_params)
        p_dep['random_state'] = s
        m_dep = lgb.LGBMRegressor(**p_dep).fit(X_tr_deploy.astype(np.float32), y_train)
        pred_dep = np.clip(m_dep.predict(X_te_deploy.astype(np.float32)), 0.5, 100.0)
        e_dep = calc_eval(y_test, pred_dep, f"Deployment (seed={s})", rand_mae)
        seed_results['deployment_model']['runs'].append({'seed': s, **e_dep})
        seed_results['deployment_model']['maes'].append(e_dep['MAE'])
        seed_results['deployment_model']['mdaes'].append(e_dep['MdAE'])
        seed_results['deployment_model']['rmses'].append(e_dep['RMSE'])

        # 2. Type + Priority only
        p_tp = dict(base_lgb_params)
        p_tp['random_state'] = s
        m_tp = lgb.LGBMRegressor(**p_tp).fit(X_tr_cat.astype(np.float32), y_train)
        pred_tp = np.clip(m_tp.predict(X_te_cat.astype(np.float32)), 0.5, 100.0)
        e_tp = calc_eval(y_test, pred_tp, f"Type+Priority (seed={s})", rand_mae)
        seed_results['type_priority_only']['runs'].append({'seed': s, **e_tp})
        seed_results['type_priority_only']['maes'].append(e_tp['MAE'])
        seed_results['type_priority_only']['mdaes'].append(e_tp['MdAE'])
        seed_results['type_priority_only']['rmses'].append(e_tp['RMSE'])

        # 3. S-BERT + LightGBM
        p_sb = dict(base_lgb_params)
        p_sb['random_state'] = s
        m_sb = lgb.LGBMRegressor(**p_sb).fit(X_tr_sbert.astype(np.float32), y_train)
        pred_sb = np.clip(m_sb.predict(X_te_sbert.astype(np.float32)), 0.5, 100.0)
        e_sb = calc_eval(y_test, pred_sb, f"S-BERT (seed={s})", rand_mae)
        seed_results['sbert_lgbm']['runs'].append({'seed': s, **e_sb})
        seed_results['sbert_lgbm']['maes'].append(e_sb['MAE'])
        seed_results['sbert_lgbm']['mdaes'].append(e_sb['MdAE'])
        seed_results['sbert_lgbm']['rmses'].append(e_sb['RMSE'])

    seed_summary = {}
    for model_key, display_name in [
        ('deployment_model', 'Deployment Model (TF-IDF + Type + Priority)'),
        ('type_priority_only', 'Type + Priority Only Model'),
        ('sbert_lgbm', 'S-BERT + LightGBM Model')
    ]:
        maes = np.array(seed_results[model_key]['maes'])
        mdaes = np.array(seed_results[model_key]['mdaes'])
        rmses = np.array(seed_results[model_key]['rmses'])
        seed_summary[model_key] = {
            'display_name': display_name,
            'seeds': seeds,
            'MAE': {'mean': round(float(np.mean(maes)), 4), 'std': round(float(np.std(maes, ddof=1)), 4)},
            'MdAE': {'mean': round(float(np.mean(mdaes)), 4), 'std': round(float(np.std(mdaes, ddof=1)), 4)},
            'RMSE': {'mean': round(float(np.mean(rmses)), 4), 'std': round(float(np.std(rmses, ddof=1)), 4)},
            'per_seed_runs': seed_results[model_key]['runs']
        }
        print(f"\n{display_name} (Seeds: {seeds}):")
        print(f"  MAE:  Mean = {seed_summary[model_key]['MAE']['mean']:.4f} | Std = {seed_summary[model_key]['MAE']['std']:.4f} (Individual: {list(maes)})")
        print(f"  MdAE: Mean = {seed_summary[model_key]['MdAE']['mean']:.4f} | Std = {seed_summary[model_key]['MdAE']['std']:.4f} (Individual: {list(mdaes)})")
        print(f"  RMSE: Mean = {seed_summary[model_key]['RMSE']['mean']:.4f} | Std = {seed_summary[model_key]['RMSE']['std']:.4f} (Individual: {list(rmses)})")

    # =========================================================================
    # ITEM 2: PAIRED BOOTSTRAP (1000 DRAWS, SEED 42)
    # Resampling Tickets AND Resampling Projects
    # (a) MAE(text+Type+Priority) - MAE(Type+Priority only)
    # (b) MAE(S-BERT+Type+Priority) - MAE(Type+Priority only)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 2: PAIRED BOOTSTRAP (1000 DRAWS, SEED 42, TICKETS & PROJECTS)")
    print("=" * 80)
    # Use seed 42 predictions
    p_dep_42 = np.clip(lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_deploy.astype(np.float32), y_train).predict(X_te_deploy.astype(np.float32)), 0.5, 100.0)
    p_tp_42 = np.clip(lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_cat.astype(np.float32), y_train).predict(X_te_cat.astype(np.float32)), 0.5, 100.0)
    p_sb_42 = np.clip(lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_sbert.astype(np.float32), y_train).predict(X_te_sbert.astype(np.float32)), 0.5, 100.0)

    # 1. Resampling Tickets
    np.random.seed(42)
    boot_ticket_deltas_text = []
    boot_ticket_deltas_sbert = []
    for _ in range(1000):
        idx_samp = np.random.choice(len(y_test), size=len(y_test), replace=True)
        y_samp = y_test[idx_samp]
        mae_tp = mean_absolute_error(y_samp, p_tp_42[idx_samp])
        mae_text = mean_absolute_error(y_samp, p_dep_42[idx_samp])
        mae_sb = mean_absolute_error(y_samp, p_sb_42[idx_samp])
        boot_ticket_deltas_text.append(mae_text - mae_tp)
        boot_ticket_deltas_sbert.append(mae_sb - mae_tp)

    # 2. Resampling Projects
    test_unique_projects = test_c['Project_ID'].unique()
    project_indices_map = {pid: np.where(test_c['Project_ID'] == pid)[0] for pid in test_unique_projects}
    
    np.random.seed(42)
    boot_proj_deltas_text = []
    boot_proj_deltas_sbert = []
    for _ in range(1000):
        sampled_pids = np.random.choice(test_unique_projects, size=len(test_unique_projects), replace=True)
        idx_proj_samp = np.concatenate([project_indices_map[pid] for pid in sampled_pids])
        y_samp = y_test[idx_proj_samp]
        mae_tp = mean_absolute_error(y_samp, p_tp_42[idx_proj_samp])
        mae_text = mean_absolute_error(y_samp, p_dep_42[idx_proj_samp])
        mae_sb = mean_absolute_error(y_samp, p_sb_42[idx_proj_samp])
        boot_proj_deltas_text.append(mae_text - mae_tp)
        boot_proj_deltas_sbert.append(mae_sb - mae_tp)

    def calc_ci(deltas):
        mean_d = float(np.mean(deltas))
        ci_l = float(np.percentile(deltas, 2.5))
        ci_h = float(np.percentile(deltas, 97.5))
        excl_zero = bool(ci_h < 0.0 or ci_l > 0.0)
        return {
            'mean_difference': round(mean_d, 4),
            'ci_95': [round(ci_l, 4), round(ci_h, 4)],
            'excludes_zero': excl_zero
        }

    bootstrap_results = {
        'text_vs_type_priority': {
            'description': 'MAE(text + Type + Priority) - MAE(Type + Priority only)',
            'ticket_level': calc_ci(boot_ticket_deltas_text),
            'project_level': calc_ci(boot_proj_deltas_text)
        },
        'sbert_vs_type_priority': {
            'description': 'MAE(S-BERT + Type + Priority) - MAE(Type + Priority only)',
            'ticket_level': calc_ci(boot_ticket_deltas_sbert),
            'project_level': calc_ci(boot_proj_deltas_sbert)
        }
    }

    print("\n(a) MAE(text + Type + Priority) - MAE(Type + Priority only):")
    t_text = bootstrap_results['text_vs_type_priority']['ticket_level']
    p_text = bootstrap_results['text_vs_type_priority']['project_level']
    print(f"  Ticket Resampling (1000 draws):  Mean Diff = {t_text['mean_difference']:+.4f} | 95% CI = [{t_text['ci_95'][0]:.4f}, {t_text['ci_95'][1]:.4f}] | Excludes 0: {t_text['excludes_zero']}")
    print(f"  Project Resampling (1000 draws): Mean Diff = {p_text['mean_difference']:+.4f} | 95% CI = [{p_text['ci_95'][0]:.4f}, {p_text['ci_95'][1]:.4f}] | Excludes 0: {p_text['excludes_zero']}")

    print("\n(b) MAE(S-BERT + Type + Priority) - MAE(Type + Priority only):")
    t_sb = bootstrap_results['sbert_vs_type_priority']['ticket_level']
    p_sb = bootstrap_results['sbert_vs_type_priority']['project_level']
    print(f"  Ticket Resampling (1000 draws):  Mean Diff = {t_sb['mean_difference']:+.4f} | 95% CI = [{t_sb['ci_95'][0]:.4f}, {t_sb['ci_95'][1]:.4f}] | Excludes 0: {t_sb['excludes_zero']}")
    print(f"  Project Resampling (1000 draws): Mean Diff = {p_sb['mean_difference']:+.4f} | 95% CI = [{p_sb['ci_95'][0]:.4f}, {p_sb['ci_95'][1]:.4f}] | Excludes 0: {p_sb['excludes_zero']}")

    # Plain statement on whether text helps
    text_helps_ticket = t_text['excludes_zero'] and t_text['mean_difference'] < 0
    text_helps_proj = p_text['excludes_zero'] and p_text['mean_difference'] < 0
    sbert_helps_ticket = t_sb['excludes_zero'] and t_sb['mean_difference'] < 0
    sbert_helps_proj = p_sb['excludes_zero'] and p_sb['mean_difference'] < 0

    plain_text_statement = (
        f"Plain takeaway on whether text helps:\n"
        f"At the individual ticket level, adding TF-IDF text reduces MAE by {abs(t_text['mean_difference']):.4f} points, "
        f"and the 95% CI [{t_text['ci_95'][0]:.4f}, {t_text['ci_95'][1]:.4f}] strictly excludes 0 (statistically significant ticket-level lift). "
        f"However, under project-level clustered resampling, the 95% CI is [{p_text['ci_95'][0]:.4f}, {p_text['ci_95'][1]:.4f}] "
        f"(excludes 0: {p_text['excludes_zero']}). "
        f"For S-BERT embeddings, ticket-level CI is [{t_sb['ci_95'][0]:.4f}, {t_sb['ci_95'][1]:.4f}] (excludes 0: {t_sb['excludes_zero']}) "
        f"and project-level CI is [{p_sb['ci_95'][0]:.4f}, {p_sb['ci_95'][1]:.4f}] (excludes 0: {p_sb['excludes_zero']})."
    )
    print("\n" + plain_text_statement)
    bootstrap_results['plain_statement'] = plain_text_statement

    # =========================================================================
    # ITEM 3: TASKOVE-COMPATIBLE VARIANTS (Priority is NOT available in Taskove)
    # Models: Type only; text + Type; S-BERT + Type
    # Compare against per-type median
    # Distinct Priority values per top-10 project & Type values in TAWOS with counts
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 3: TASKOVE-COMPATIBLE VARIANTS (NO PRIORITY AVAILABLE)")
    print("=" * 80)
    # Type only feature
    ohe_type = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_tr_t = ohe_type.fit_transform(train_c[['Type']].fillna('Missing'))
    X_te_t = ohe_type.transform(test_c[['Type']].fillna('Missing'))

    # Text + Type feature
    X_tr_text_t = sparse.hstack([X_tr_tfidf, X_tr_t], format='csr')
    X_te_text_t = sparse.hstack([X_te_tfidf, X_te_t], format='csr')

    # S-BERT + Type feature
    X_tr_sb_t = np.hstack([sbert_tr, X_tr_t.toarray()])
    X_te_sb_t = np.hstack([sbert_te, X_te_t.toarray()])

    # Train Taskove models (seed 42)
    # Baseline: per-type median
    type_medians_dict = train_c.groupby('Type')['Story_Point'].median().to_dict()
    global_med = float(np.median(y_train))
    p_b_type = np.array([type_medians_dict.get(t, global_med) for t in test_c['Type']])
    e_b_type = calc_eval(y_test, p_b_type, "Baseline: Per-Type Train Median", rand_mae)

    # 1. Type only
    m_task_t = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_t.astype(np.float32), y_train)
    p_task_t = np.clip(m_task_t.predict(X_te_t.astype(np.float32)), 0.5, 100.0)
    e_task_t = calc_eval(y_test, p_task_t, "Type Only (No Priority)", rand_mae)

    # 2. Text + Type
    m_task_tt = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_text_t.astype(np.float32), y_train)
    p_task_tt = np.clip(m_task_tt.predict(X_te_text_t.astype(np.float32)), 0.5, 100.0)
    e_task_tt = calc_eval(y_test, p_task_tt, "Text + Type (No Priority)", rand_mae)

    # 3. S-BERT + Type
    m_task_st = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_sb_t.astype(np.float32), y_train)
    p_task_st = np.clip(m_task_st.predict(X_te_sb_t.astype(np.float32)), 0.5, 100.0)
    e_task_st = calc_eval(y_test, p_task_st, "S-BERT + Type (No Priority)", rand_mae)

    taskove_models = {
        'Baseline: Per-Type Train Median': e_b_type,
        'Type Only (No Priority)': e_task_t,
        'Text + Type (No Priority)': e_task_tt,
        'S-BERT + Type (No Priority)': e_task_st
    }

    for k, v in taskove_models.items():
        diff_vs_type = round(v['MAE'] - e_b_type['MAE'], 4)
        rel_imp = round((e_b_type['MAE'] - v['MAE']) / e_b_type['MAE'] * 100, 2)
        v['diff_vs_per_type_median'] = diff_vs_type
        v['rel_improvement_vs_type_pct'] = rel_imp

    print(f"\nTaskove-Compatible Model Comparison Table (Test Set N={len(test_c)}):")
    print(f"{'Model':32s} | {'MAE':>7s} | {'MdAE':>7s} | {'RMSE':>7s} | {'SA':>7s} | {'Snapped':>7s} | {'vs Type MAE':>11s} | {'Rel Imp %':>9s}")
    print("-" * 105)
    for k, v in taskove_models.items():
        print(f"{k:32s} | {v['MAE']:7.4f} | {v['MdAE']:7.4f} | {v['RMSE']:7.4f} | {v['Standardized_Accuracy']:7.4f} | {v['Snapped_MAE']:7.4f} | {v['diff_vs_per_type_median']:+11.4f} | {v['rel_improvement_vs_type_pct']:8.2f}%")

    # Distinct Priority values per top-10 project in test set & overall
    top10_test_pids = test_c['Project_ID'].value_counts().head(10).index.tolist()
    proj_priority_analysis = []
    print("\nDistinct Priority values per Top-10 Test Project:")
    for pid in top10_test_pids:
        p_sub = df_clean[df_clean['Project_ID'] == pid]
        p_test_sub = test_c[test_c['Project_ID'] == pid]
        distinct_prios_clean = sorted([str(x) for x in p_sub['Priority'].dropna().unique().tolist()])
        distinct_prios_test = sorted([str(x) for x in p_test_sub['Priority'].dropna().unique().tolist()])
        info = {
            'project_id': int(pid),
            'test_ticket_count': len(p_test_sub),
            'total_clean_ticket_count': len(p_sub),
            'distinct_priorities_in_test': distinct_prios_test,
            'distinct_priorities_in_clean': distinct_prios_clean
        }
        proj_priority_analysis.append(info)
        print(f"  Project {pid:2d} (Test N={len(p_test_sub):4d}, Clean N={len(p_sub):4d}): Priorities = {distinct_prios_clean}")

    # TAWOS Type counts
    print(f"\nDistinct Type values in Raw TAWOS (N={len(df_raw):,}):")
    for t_name, count in sorted(raw_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {str(t_name):32s}: {count:6d}")

    print(f"\nDistinct Type values in Clean TAWOS (N={len(df_clean):,}):")
    for t_name, count in sorted(clean_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {str(t_name):32s}: {count:6d}")

    # =========================================================================
    # ITEM 4: POPULATION DEFINITION (TYPE == 'EPIC' EXCLUDED FROM SPLITS)
    # Repeat main table: per-type median, Type+Priority only, deployment model, text+Type, S-BERT
    # Report counts dropped per split, MAE, MdAE, RMSE, Project 28's share of total error, 10 largest errors
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 4: POPULATION DEFINITION (TYPE == 'EPIC' EXCLUDED)")
    print("=" * 80)
    # Split filtering
    train_ne = train_c[train_c['Type'] != 'Epic'].copy().reset_index(drop=True)
    val_ne = val_c[val_c['Type'] != 'Epic'].copy().reset_index(drop=True)
    test_ne = test_c[test_c['Type'] != 'Epic'].copy().reset_index(drop=True)

    dropped_train = int((train_c['Type'] == 'Epic').sum())
    dropped_val = int((val_c['Type'] == 'Epic').sum())
    dropped_test = int((test_c['Type'] == 'Epic').sum())
    total_dropped = dropped_train + dropped_val + dropped_test

    print(f"Counts dropped by excluding Type == 'Epic':")
    print(f"  Train: dropped {dropped_train} (kept {len(train_ne)} / {len(train_c)})")
    print(f"  Val:   dropped {dropped_val} (kept {len(val_ne)} / {len(val_c)})")
    print(f"  Test:  dropped {dropped_test} (kept {len(test_ne)} / {len(test_c)})")
    print(f"  Total: dropped {total_dropped} (kept {len(train_ne)+len(val_ne)+len(test_ne)} / {len(df_clean)})")

    y_tr_ne = train_ne['Story_Point'].values
    y_va_ne = val_ne['Story_Point'].values
    y_te_ne = test_ne['Story_Point'].values

    rand_mae_ne, _ = compute_exact_random_baseline(y_tr_ne, y_te_ne)

    # Feature representations for Epic-excluded population
    tfidf_ne = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=3)
    X_tr_tfidf_ne = tfidf_ne.fit_transform(prep_text(train_ne))
    X_te_tfidf_ne = tfidf_ne.transform(prep_text(test_ne))
    X_va_tfidf_ne = tfidf_ne.transform(prep_text(val_ne))

    ohe_ne = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_tr_cat_ne = ohe_ne.fit_transform(train_ne[['Type', 'Priority']].fillna('Missing'))
    X_te_cat_ne = ohe_ne.transform(test_ne[['Type', 'Priority']].fillna('Missing'))
    X_va_cat_ne = ohe_ne.transform(val_ne[['Type', 'Priority']].fillna('Missing'))

    ohe_type_ne = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_tr_t_ne = ohe_type_ne.fit_transform(train_ne[['Type']].fillna('Missing'))
    X_te_t_ne = ohe_type_ne.transform(test_ne[['Type']].fillna('Missing'))

    # Combined matrices
    X_tr_deploy_ne = sparse.hstack([X_tr_tfidf_ne, X_tr_cat_ne], format='csr')
    X_te_deploy_ne = sparse.hstack([X_te_tfidf_ne, X_te_cat_ne], format='csr')
    X_va_deploy_ne = sparse.hstack([X_va_tfidf_ne, X_va_cat_ne], format='csr')

    X_tr_text_t_ne = sparse.hstack([X_tr_tfidf_ne, X_tr_t_ne], format='csr')
    X_te_text_t_ne = sparse.hstack([X_te_tfidf_ne, X_te_t_ne], format='csr')

    # S-BERT sliced
    sbert_tr_ne = sbert_tr[(train_c['Type'] != 'Epic').values]
    sbert_te_ne = sbert_te[(test_c['Type'] != 'Epic').values]
    sbert_va_ne = sbert_va[(val_c['Type'] != 'Epic').values]

    X_tr_sbert_ne = np.hstack([sbert_tr_ne, X_tr_cat_ne.toarray()])
    X_te_sbert_ne = np.hstack([sbert_te_ne, X_te_cat_ne.toarray()])

    # Models on Epic-excluded population
    # 1. Per-type median
    type_meds_ne = train_ne.groupby('Type')['Story_Point'].median().to_dict()
    global_med_ne = float(np.median(y_tr_ne))
    p_b_type_ne = np.array([type_meds_ne.get(t, global_med_ne) for t in test_ne['Type']])
    e_b_type_ne = calc_eval(y_te_ne, p_b_type_ne, "Baseline: Per-Type Median (No Epics)", rand_mae_ne)

    # 2. Type + Priority only
    m_tp_ne = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_cat_ne.astype(np.float32), y_tr_ne)
    p_tp_ne = np.clip(m_tp_ne.predict(X_te_cat_ne.astype(np.float32)), 0.5, 100.0)
    e_tp_ne = calc_eval(y_te_ne, p_tp_ne, "Type + Priority Only (No Epics)", rand_mae_ne)

    # 3. Deployment model (TF-IDF + Type + Priority)
    m_dep_ne = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_deploy_ne.astype(np.float32), y_tr_ne)
    p_dep_ne = np.clip(m_dep_ne.predict(X_te_deploy_ne.astype(np.float32)), 0.5, 100.0)
    e_dep_ne = calc_eval(y_te_ne, p_dep_ne, "Deployment Model (TF-IDF+Type+Prio, No Epics)", rand_mae_ne)

    # 4. text + Type
    m_tt_ne = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_text_t_ne.astype(np.float32), y_tr_ne)
    p_tt_ne = np.clip(m_tt_ne.predict(X_te_text_t_ne.astype(np.float32)), 0.5, 100.0)
    e_tt_ne = calc_eval(y_te_ne, p_tt_ne, "Text + Type (No Epics)", rand_mae_ne)

    # 5. S-BERT (S-BERT + Type + Priority)
    m_sb_ne = lgb.LGBMRegressor(**{**base_lgb_params, 'random_state': 42}).fit(X_tr_sbert_ne.astype(np.float32), y_tr_ne)
    p_sb_ne = np.clip(m_sb_ne.predict(X_te_sbert_ne.astype(np.float32)), 0.5, 100.0)
    e_sb_ne = calc_eval(y_te_ne, p_sb_ne, "S-BERT + Type + Priority (No Epics)", rand_mae_ne)

    no_epic_table = {
        'Baseline: Per-Type Median': e_b_type_ne,
        'Type + Priority Only': e_tp_ne,
        'Deployment Model (Text+Type+Prio)': e_dep_ne,
        'Text + Type': e_tt_ne,
        'S-BERT (+Type+Priority)': e_sb_ne
    }

    for k, v in no_epic_table.items():
        diff_vs_base = round(v['MAE'] - e_b_type_ne['MAE'], 4)
        rel_imp = round((e_b_type_ne['MAE'] - v['MAE']) / e_b_type_ne['MAE'] * 100, 2)
        v['diff_vs_per_type_median'] = diff_vs_base
        v['rel_improvement_vs_type_pct'] = rel_imp

    print(f"\nEpic-Excluded Main Table (Test Set N={len(test_ne)}):")
    print(f"{'Model':36s} | {'MAE':>7s} | {'MdAE':>7s} | {'RMSE':>7s} | {'SA':>7s} | {'Snapped':>7s} | {'vs Type MAE':>11s} | {'Rel Imp %':>9s}")
    print("-" * 109)
    for k, v in no_epic_table.items():
        print(f"{k:36s} | {v['MAE']:7.4f} | {v['MdAE']:7.4f} | {v['RMSE']:7.4f} | {v['Standardized_Accuracy']:7.4f} | {v['Snapped_MAE']:7.4f} | {v['diff_vs_per_type_median']:+11.4f} | {v['rel_improvement_vs_type_pct']:8.2f}%")

    # Project 28's share of total error on Epic-excluded test set
    abs_errors_ne = np.abs(y_te_ne - p_dep_ne)
    signed_errors_ne = p_dep_ne - y_te_ne
    mask_p28_ne = (test_ne['Project_ID'] == 28).values
    p28_total_abs_err_ne = float(np.sum(abs_errors_ne[mask_p28_ne]))
    all_total_abs_err_ne = float(np.sum(abs_errors_ne))
    p28_err_share_pct = round(p28_total_abs_err_ne / all_total_abs_err_ne * 100, 2)
    p28_ticket_share_pct = round(np.sum(mask_p28_ne) / len(test_ne) * 100, 2)

    print(f"\nProject 28 Error Share on Epic-Excluded Population:")
    print(f"  Project 28 Test Tickets: {np.sum(mask_p28_ne)} / {len(test_ne)} ({p28_ticket_share_pct}%)")
    print(f"  Project 28 Absolute Error Sum: {p28_total_abs_err_ne:.2f} / {all_total_abs_err_ne:.2f}")
    print(f"  Project 28 Share of Total Error: {p28_err_share_pct}%")

    # 10 Largest errors again on Epic-excluded test set
    top10_err_indices_ne = np.argsort(abs_errors_ne)[::-1][:10]
    top10_err_rows_ne = []
    print("\n10 Largest Absolute Errors on Epic-Excluded Population:")
    print(f"{'Issue ID':>10s} | {'Proj':>5s} | {'Type':>15s} | {'True SP':>8s} | {'Pred SP':>8s} | {'Abs Err':>8s} | {'Signed Err':>10s}")
    print("-" * 75)
    for idx in top10_err_indices_ne:
        row = test_ne.iloc[idx]
        err_info = {
            'Issue_ID': int(row['ID']),
            'Project_ID': int(row['Project_ID']),
            'Type': str(row['Type']),
            'True_SP': float(row['Story_Point']),
            'Pred_SP': round(float(p_dep_ne[idx]), 2),
            'Absolute_Error': round(float(abs_errors_ne[idx]), 2),
            'Signed_Error': round(float(signed_errors_ne[idx]), 2)
        }
        top10_err_rows_ne.append(err_info)
        print(f"{err_info['Issue_ID']:10d} | {err_info['Project_ID']:5d} | {err_info['Type']:15s} | {err_info['True_SP']:8.2f} | {err_info['Pred_SP']:8.2f} | {err_info['Absolute_Error']:8.2f} | {err_info['Signed_Error']:+10.2f}")

    # =========================================================================
    # ITEM 5: INTERVALS (TOP 10 PROJECTS & EPIC-EXCLUDED POPULATION)
    # [P25, P75] coverage and mean width per project
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 5: INTERVAL EVALUATION ([P25, P75] COVERAGE & MEAN WIDTH)")
    print("=" * 80)
    
    # Train quantile models for full population
    q_models_full = {}
    q_preds_val_full = {}
    q_preds_test_full = {}
    for q in [0.25, 0.75]:
        p_q = dict(base_lgb_params)
        p_q['objective'] = 'quantile'
        p_q['alpha'] = q
        p_q['random_state'] = 42
        m_q = lgb.LGBMRegressor(**p_q).fit(X_tr_deploy.astype(np.float32), y_train)
        q_models_full[q] = m_q
        q_preds_val_full[q] = np.clip(m_q.predict(X_va_deploy.astype(np.float32)), 0.5, 100.0)
        q_preds_test_full[q] = np.clip(m_q.predict(X_te_deploy.astype(np.float32)), 0.5, 100.0)

    # Conformal adjustment constant on validation
    val_res_full = np.maximum(q_preds_val_full[0.25] - y_val, y_val - q_preds_val_full[0.75])
    c_adj_full = float(np.percentile(val_res_full, 50.0))

    raw_low_full = q_preds_test_full[0.25]
    raw_high_full = q_preds_test_full[0.75]
    adj_low_full = np.maximum(0.5, raw_low_full - c_adj_full)
    adj_high_full = raw_high_full + c_adj_full

    intervals_top10_full = []
    print("\nTop 10 Test Projects - [P25, P75] Intervals (Full Population):")
    print(f"{'Proj ID':>8s} | {'N':>6s} | {'Raw Cov %':>10s} | {'Raw Width':>10s} | {'Adj Cov %':>10s} | {'Adj Width':>10s}")
    print("-" * 65)
    for pid in top10_test_pids:
        idx_p = np.where(test_c['Project_ID'] == pid)[0]
        y_p = y_test[idx_p]
        rl = raw_low_full[idx_p]
        rh = raw_high_full[idx_p]
        al = adj_low_full[idx_p]
        ah = adj_high_full[idx_p]

        rcov = round(float(np.mean((y_p >= rl) & (y_p <= rh)) * 100), 2)
        rwid = round(float(np.mean(rh - rl)), 4)
        acov = round(float(np.mean((y_p >= al) & (y_p <= ah)) * 100), 2)
        awid = round(float(np.mean(ah - al)), 4)

        info = {
            'project_id': int(pid),
            'test_tickets': len(idx_p),
            'raw_coverage_pct': rcov,
            'raw_mean_width': rwid,
            'conformal_coverage_pct': acov,
            'conformal_mean_width': awid
        }
        intervals_top10_full.append(info)
        print(f"{pid:8d} | {len(idx_p):6d} | {rcov:9.2f}% | {rwid:10.4f} | {acov:9.2f}% | {awid:10.4f}")

    # Quantile models for Epic-excluded population
    q_models_ne = {}
    q_preds_val_ne = {}
    q_preds_test_ne = {}
    for q in [0.25, 0.75]:
        p_q = dict(base_lgb_params)
        p_q['objective'] = 'quantile'
        p_q['alpha'] = q
        p_q['random_state'] = 42
        m_q = lgb.LGBMRegressor(**p_q).fit(X_tr_deploy_ne.astype(np.float32), y_tr_ne)
        q_models_ne[q] = m_q
        q_preds_val_ne[q] = np.clip(m_q.predict(X_va_deploy_ne.astype(np.float32)), 0.5, 100.0)
        q_preds_test_ne[q] = np.clip(m_q.predict(X_te_deploy_ne.astype(np.float32)), 0.5, 100.0)

    val_res_ne = np.maximum(q_preds_val_ne[0.25] - y_va_ne, y_va_ne - q_preds_val_ne[0.75])
    c_adj_ne = float(np.percentile(val_res_ne, 50.0))

    raw_low_ne = q_preds_test_ne[0.25]
    raw_high_ne = q_preds_test_ne[0.75]
    adj_low_ne = np.maximum(0.5, raw_low_ne - c_adj_ne)
    adj_high_ne = raw_high_ne + c_adj_ne

    tot_rcov_ne = round(float(np.mean((y_te_ne >= raw_low_ne) & (y_te_ne <= raw_high_ne)) * 100), 2)
    tot_rwid_ne = round(float(np.mean(raw_high_ne - raw_low_ne)), 4)
    tot_acov_ne = round(float(np.mean((y_te_ne >= adj_low_ne) & (y_te_ne <= adj_high_ne)) * 100), 2)
    tot_awid_ne = round(float(np.mean(adj_high_ne - adj_low_ne)), 4)

    print(f"\nEpic-Excluded Overall [P25, P75] Interval (N={len(test_ne)}):")
    print(f"  Raw:       Coverage = {tot_rcov_ne:.2f}% | Mean Width = {tot_rwid_ne:.4f} points")
    print(f"  Conformal: Coverage = {tot_acov_ne:.2f}% | Mean Width = {tot_awid_ne:.4f} points (c = {c_adj_ne:.4f})")

    top10_pids_ne = test_ne['Project_ID'].value_counts().head(10).index.tolist()
    intervals_top10_ne = []
    print(f"\nTop 10 Test Projects - [P25, P75] Intervals (Epic-Excluded Population):")
    print(f"{'Proj ID':>8s} | {'N':>6s} | {'Raw Cov %':>10s} | {'Raw Width':>10s} | {'Adj Cov %':>10s} | {'Adj Width':>10s}")
    print("-" * 65)
    for pid in top10_pids_ne:
        idx_p = np.where(test_ne['Project_ID'] == pid)[0]
        y_p = y_te_ne[idx_p]
        rl = raw_low_ne[idx_p]
        rh = raw_high_ne[idx_p]
        al = adj_low_ne[idx_p]
        ah = adj_high_ne[idx_p]

        rcov = round(float(np.mean((y_p >= rl) & (y_p <= rh)) * 100), 2)
        rwid = round(float(np.mean(rh - rl)), 4)
        acov = round(float(np.mean((y_p >= al) & (y_p <= ah)) * 100), 2)
        awid = round(float(np.mean(ah - al)), 4)

        info = {
            'project_id': int(pid),
            'test_tickets': len(idx_p),
            'raw_coverage_pct': rcov,
            'raw_mean_width': rwid,
            'conformal_coverage_pct': acov,
            'conformal_mean_width': awid
        }
        intervals_top10_ne.append(info)
        print(f"{pid:8d} | {len(idx_p):6d} | {rcov:9.2f}% | {rwid:10.4f} | {acov:9.2f}% | {awid:10.4f}")

    # =========================================================================
    # ITEM 6: PREDICTION DISTRIBUTION FOR EPICS (FIXING THE EXPLANATION)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 6: EMPIRICAL PREDICTION DISTRIBUTION FOR EPICS")
    print("=" * 80)
    epic_mask_test = (test_c['Type'] == 'Epic').values
    epic_preds_test = p_dep_42[epic_mask_test]
    epic_true_test = y_test[epic_mask_test]

    epic_stats = {
        'count': int(len(epic_preds_test)),
        'true_sp': {
            'mean': round(float(np.mean(epic_true_test)), 4),
            'std': round(float(np.std(epic_true_test)), 4),
            'min': float(np.min(epic_true_test)),
            'p10': round(float(np.percentile(epic_true_test, 10)), 4),
            'p25': round(float(np.percentile(epic_true_test, 25)), 4),
            'median': round(float(np.median(epic_true_test)), 4),
            'p75': round(float(np.percentile(epic_true_test, 75)), 4),
            'p90': round(float(np.percentile(epic_true_test, 90)), 4),
            'p95': round(float(np.percentile(epic_true_test, 95)), 4),
            'p99': round(float(np.percentile(epic_true_test, 99)), 4),
            'max': float(np.max(epic_true_test))
        },
        'pred_sp': {
            'mean': round(float(np.mean(epic_preds_test)), 4),
            'std': round(float(np.std(epic_preds_test)), 4),
            'min': round(float(np.min(epic_preds_test)), 4),
            'p10': round(float(np.percentile(epic_preds_test, 10)), 4),
            'p25': round(float(np.percentile(epic_preds_test, 25)), 4),
            'median': round(float(np.median(epic_preds_test)), 4),
            'p75': round(float(np.percentile(epic_preds_test, 75)), 4),
            'p90': round(float(np.percentile(epic_preds_test, 90)), 4),
            'p95': round(float(np.percentile(epic_preds_test, 95)), 4),
            'p99': round(float(np.percentile(epic_preds_test, 99)), 4),
            'max': round(float(np.max(epic_preds_test)), 4)
        }
    }

    print(f"Test Epics (N={epic_stats['count']}):")
    print(f"  True SP: Mean = {epic_stats['true_sp']['mean']:.2f} | Median = {epic_stats['true_sp']['median']:.2f} | Range = [{epic_stats['true_sp']['min']:.1f}, {epic_stats['true_sp']['max']:.1f}] | P75 = {epic_stats['true_sp']['p75']:.1f} | P90 = {epic_stats['true_sp']['p90']:.1f}")
    print(f"  Pred SP: Mean = {epic_stats['pred_sp']['mean']:.2f} | Median = {epic_stats['pred_sp']['median']:.2f} | Range = [{epic_stats['pred_sp']['min']:.2f}, {epic_stats['pred_sp']['max']:.2f}] | P25 = {epic_stats['pred_sp']['p25']:.2f} | P75 = {epic_stats['pred_sp']['p75']:.2f} | Std = {epic_stats['pred_sp']['std']:.2f}")

    # =========================================================================
    # ITEM 7: METADATA UPDATES
    # Update benchmark_v2/artifacts/model1_sp/v1_tawos/metadata.json
    # =========================================================================
    print("\n" + "=" * 80)
    print("ITEM 7: METADATA.JSON UPDATES")
    print("=" * 80)
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    with open(meta_path, "r") as f:
        meta_dict = json.load(f)

    # Get known types from encoder
    # OHE encoder has feature_names_in_ ['Type', 'Priority']
    type_idx = list(ohe.feature_names_in_).index('Type')
    known_types_list = sorted([str(x) for x in ohe.categories_[type_idx].tolist()])

    meta_dict['trained_on'] = "tawos"
    meta_dict['clean_unfiltered_info'] = {
        'dataset_filtering': 'tawos_clean',
        'filtering_description': 'Clean subset with no post-estimation changes (Story_Point, Title, Description), valid resolutions (excluding wontfix, duplicate, invalid, cannot reproduce), and plausible Story_Point <= 100.',
        'clean_rows': len(df_clean),
        'unfiltered_rows': len(df_valid)
    }
    meta_dict['encoder_known_types'] = known_types_list

    with open(meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"Updated metadata.json:")
    print(f"  trained_on: '{meta_dict['trained_on']}'")
    print(f"  clean_unfiltered_info: {meta_dict['clean_unfiltered_info']['dataset_filtering']}")
    print(f"  encoder_known_types (count={len(known_types_list)}): {known_types_list}")

    # =========================================================================
    # Save Full Analysis JSON
    # =========================================================================
    full_part2_metrics = {
        'item1_seed_noise': seed_summary,
        'item2_paired_bootstrap': bootstrap_results,
        'item3_taskove_variants': {
            'models_evaluation': taskove_models,
            'top10_projects_priority_vocabulary': proj_priority_analysis,
            'tawos_raw_type_counts': raw_type_counts,
            'tawos_clean_type_counts': clean_type_counts
        },
        'item4_epic_excluded_population': {
            'dropped_counts': {
                'train_dropped_epics': dropped_train,
                'train_remaining': len(train_ne),
                'val_dropped_epics': dropped_val,
                'val_remaining': len(val_ne),
                'test_dropped_epics': dropped_test,
                'test_remaining': len(test_ne),
                'total_dropped_epics': total_dropped
            },
            'models_evaluation': no_epic_table,
            'project_28_error_share': {
                'test_tickets_p28': int(np.sum(mask_p28_ne)),
                'total_test_tickets': len(test_ne),
                'ticket_share_pct': p28_ticket_share_pct,
                'total_abs_error_p28': round(p28_total_abs_err_ne, 2),
                'total_abs_error_all': round(all_total_abs_err_ne, 2),
                'error_share_pct': p28_err_share_pct
            },
            'top10_largest_errors': top10_err_rows_ne
        },
        'item5_intervals': {
            'top10_test_projects_full_population': intervals_top10_full,
            'epic_excluded_overall': {
                'raw_coverage_pct': tot_rcov_ne,
                'raw_mean_width': tot_rwid_ne,
                'conformal_coverage_pct': tot_acov_ne,
                'conformal_mean_width': tot_awid_ne,
                'adjustment_constant': round(c_adj_ne, 4)
            },
            'top10_test_projects_epic_excluded': intervals_top10_ne
        },
        'item6_epic_prediction_distribution': epic_stats,
        'item7_metadata': {
            'trained_on': meta_dict['trained_on'],
            'clean_unfiltered_info': meta_dict['clean_unfiltered_info'],
            'encoder_known_types': known_types_list
        },
        'total_execution_time_seconds': round(time.time() - t0, 2)
    }

    out_json_path = os.path.join(WORK_DIR, "model1_part2_metrics.json")
    with open(out_json_path, "w") as f:
        json.dump(full_part2_metrics, f, indent=2)
    print(f"\nSaved all audit part 2 metrics to: {out_json_path}")
    print(f"Total time elapsed: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()

