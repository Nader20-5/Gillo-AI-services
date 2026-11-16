import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

TAWOS_DIR = r"d:\Gillo AI Service\tawos_csv"
BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "model1_sp", "v1_tawos")

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_artifacts():
    print("=" * 70)
    print("VERIFYING MODEL 1 PRODUCTION ARTIFACT (DEPLOYMENT MODEL)")
    print("=" * 70)
    
    # 1. Check metadata and artifacts
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    model_path = os.path.join(ARTIFACTS_DIR, "model_sp_deployment.pkl")
    q25_path = os.path.join(ARTIFACTS_DIR, "model_sp_q25.pkl")
    q50_path = os.path.join(ARTIFACTS_DIR, "model_sp_q50.pkl")
    q75_path = os.path.join(ARTIFACTS_DIR, "model_sp_q75.pkl")
    vec_path = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
    ohe_path = os.path.join(ARTIFACTS_DIR, "ohe_encoder.pkl")
    
    required_files = [meta_path, model_path, q25_path, q50_path, q75_path, vec_path, ohe_path]
    for p in required_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")
        print(f"Found artifact: {os.path.basename(p):30s} ({os.path.getsize(p):,} bytes)")
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    print(f"\nMetadata Model Name:      {meta['name']}")
    print(f"Metadata Version:         {meta['version']}")
    print(f"Metadata Target:          {meta['target']}")
    print(f"Metadata Trained On:      {meta['trained_on']}")
    print(f"Metadata Training Split:  {meta['training_dataset_split']}")
    print(f"Reported Test MAE:        {meta['test_metrics']['MAE']:.4f}")
    print(f"Reported Test MdAE:       {meta['test_metrics']['MdAE']:.4f}")
    print(f"Reported Test RMSE:       {meta['test_metrics']['RMSE']:.4f}")
    print(f"Conformal 50% Coverage:   {meta['conformal_interval']['test_coverage_pct']}% (width: {meta['conformal_interval']['test_mean_width_points']} pts)")
    
    # Verify Checksums
    assert get_sha256(model_path) == meta['artifacts']['model_sp_deployment_sha256'], "Deployment model SHA256 mismatch!"
    assert get_sha256(q25_path) == meta['artifacts']['model_sp_q25_sha256'], "Q25 model SHA256 mismatch!"
    assert get_sha256(q50_path) == meta['artifacts']['model_sp_q50_sha256'], "Q50 model SHA256 mismatch!"
    assert get_sha256(q75_path) == meta['artifacts']['model_sp_q75_sha256'], "Q75 model SHA256 mismatch!"
    assert get_sha256(vec_path) == meta['artifacts']['tfidf_vectorizer_sha256'], "TFIDF SHA256 mismatch!"
    assert get_sha256(ohe_path) == meta['artifacts']['ohe_encoder_sha256'], "OHE SHA256 mismatch!"
    print("\nAll artifact SHA256 checksums verified against metadata.json successfully!")
    
    # 2. Load model, vectorizer, and encoder
    print("\nLoading pickled artifacts from disk...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(ohe_path, "rb") as f:
        ohe = pickle.load(f)
        
    # 3. Load dataset and reproduce exact chronological test set
    print("\nLoading test partition from issue.csv...")
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
    
    N = len(df_clean)
    idx_val = int(N * 0.85)
    test_df = df_clean.iloc[idx_val:].copy().reset_index(drop=True)
    y_test = test_df['Story_Point'].values
    print(f"Reproduced exact test set: {len(test_df):,} issues.")
    
    # 4. Feature transformation using loaded vectorizer and encoder
    text_test = (test_df['Title'].fillna('') + ' ' + test_df['Description_Text'].fillna('')).str.slice(0, 2000).fillna('')
    X_test_tfidf = tfidf.transform(text_test)
    X_test_cat = ohe.transform(test_df[['Type', 'Priority']].fillna('Missing'))
    
    # Deployment model uses NO project feature
    X_test_deploy = hstack([X_test_tfidf, X_test_cat]).tocsr()
    
    # 5. Predict and verify MAE to 4 decimals
    preds_test = np.clip(model.predict(X_test_deploy.astype(np.float32)), 0.5, 100.0)
    reproduced_mae = round(float(mean_absolute_error(y_test, preds_test)), 4)
    reproduced_mdae = round(float(median_absolute_error(y_test, preds_test)), 4)
    reproduced_rmse = round(float(np.sqrt(mean_squared_error(y_test, preds_test))), 4)
    
    target_mae = meta['test_metrics']['MAE']
    target_mdae = meta['test_metrics']['MdAE']
    target_rmse = meta['test_metrics']['RMSE']
    
    print("\n" + "=" * 70)
    print("REPRODUCIBILITY RESULTS")
    print("=" * 70)
    print(f"Target Test MAE from metadata:     {target_mae:.4f}")
    print(f"Reproduced Test MAE from artifact: {reproduced_mae:.4f}")
    print(f"Target Test MdAE:                  {target_mdae:.4f}")
    print(f"Reproduced Test MdAE:              {reproduced_mdae:.4f}")
    print(f"Target Test RMSE:                  {target_rmse:.4f}")
    print(f"Reproduced Test RMSE:              {reproduced_rmse:.4f}")
    
    assert reproduced_mae == target_mae, f"MAE mismatch: {reproduced_mae} != {target_mae}"
    assert reproduced_mdae == target_mdae, f"MdAE mismatch: {reproduced_mdae} != {target_mdae}"
    assert reproduced_rmse == target_rmse, f"RMSE mismatch: {reproduced_rmse} != {target_rmse}"
    
    print("\nSUCCESS: All test metrics reproduced to 4 decimal places bit-for-bit from saved artifacts!")

if __name__ == "__main__":
    verify_artifacts()
