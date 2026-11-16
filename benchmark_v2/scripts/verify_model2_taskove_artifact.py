"""
benchmark_v2/scripts/verify_model2_taskove_artifact.py

Verification script for Model 2 Taskove-Compatible Production Artifacts:
1. Verifies SHA256 hashes of all artifacts against metadata.json.
2. Reloads pickled LightGBM model, Isotonic Calibrator, and Feature Transformer from disk.
3. Evaluates predictions on test set at f=0.5.
4. Asserts reproduced ROC-AUC matches metadata.json to 4 decimal places bit-for-bit.
"""

import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
sys.path.insert(0, BASE_DIR)
from features_v2 import TaskoveFeatureTransformer
art_dir = os.path.join(BASE_DIR, "artifacts", "model2", "v2_taskove_features")
work_dir = os.path.join(BASE_DIR, "work")

def pr_auc_score(y_true, y_pred):
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return float(auc(r, p))

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f_in:
        while chunk := f_in.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("======================================================================")
    print("VERIFYING MODEL 2 TASKOVE PRODUCTION ARTIFACTS")
    print("======================================================================")

    meta_path = os.path.join(art_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found.")
        sys.exit(1)

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    print(f"Loaded metadata.json: Model '{meta['model_name']}' version '{meta['version']}'")
    print(f"Trained on: {meta['trained_on']} | Feature count: {meta['feature_count']}")
    print(f"Reported Test ROC-AUC: {meta['metrics_test_f50']['roc_auc']}")
    print(f"Reported Test PR-AUC:  {meta['metrics_test_f50']['pr_auc']}")
    print(f"Reported Test Brier:   {meta['metrics_test_f50']['brier_score']}")

    # 1. Verify SHA256 Checksums
    print("\nVerifying artifact SHA256 checksums...")
    for art_key, expected_hash in meta['artifacts'].items():
        fname = art_key.replace('_sha256', '.pkl')
        fpath = os.path.join(art_dir, fname)
        if not os.path.exists(fpath):
            print(f"ERROR: Artifact file {fname} not found!")
            sys.exit(1)
        actual_hash = get_sha256(fpath)
        if actual_hash != expected_hash:
            print(f"ERROR: Checksum mismatch for {fname}!")
            print(f"  Expected: {expected_hash}")
            print(f"  Actual:   {actual_hash}")
            sys.exit(1)
        print(f"  {fname:35s} SHA256 verified ({os.path.getsize(fpath):,} bytes)")

    print("\nAll artifact checksums verified successfully!")

    # 2. Reload pickled objects
    print("\nLoading pickled artifacts from disk...")
    with open(os.path.join(art_dir, "model2_taskove_lightgbm.pkl"), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(art_dir, "model2_taskove_calibrator.pkl"), 'rb') as f:
        cal_obj = pickle.load(f)
        calibrator = cal_obj['calibrator']
    with open(os.path.join(art_dir, "model2_taskove_transformer.pkl"), 'rb') as f:
        transformer = pickle.load(f)

    # 3. Load test partition
    print("Loading test partition from model2_dataset_full_universe.parquet at f=0.5...")
    df = pd.read_parquet(os.path.join(work_dir, "model2_dataset_full_universe.parquet"))
    df_test_f50 = df[(df['split'] == 'test') & (df['f'] == 0.5)].copy()
    y_test_f50 = df_test_f50['label'].values
    print(f"Loaded test slice: {len(df_test_f50)} instances, base rate: {np.mean(y_test_f50):.4f}")

    # 4. Transform & Predict
    X_test_tk = transformer.transform(df_test_f50)
    raw_probs = model.predict_proba(X_test_tk)[:, 1]
    cal_probs = calibrator.predict(raw_probs)

    reproduced_auc = round(float(roc_auc_score(y_test_f50, cal_probs)), 4)
    reproduced_pr = round(pr_auc_score(y_test_f50, cal_probs), 4)
    reproduced_brier = round(float(brier_score_loss(y_test_f50, cal_probs)), 4)

    target_auc = meta['metrics_test_f50']['roc_auc']
    target_pr = meta['metrics_test_f50']['pr_auc']
    target_brier = meta['metrics_test_f50']['brier_score']

    print("\n======================================================================")
    print("REPRODUCIBILITY RESULTS")
    print("======================================================================")
    print(f"Target Test ROC-AUC:     {target_auc:.4f}")
    print(f"Reproduced Test ROC-AUC: {reproduced_auc:.4f}")
    print(f"Target Test PR-AUC:      {target_pr:.4f}")
    print(f"Reproduced Test PR-AUC:  {reproduced_pr:.4f}")
    print(f"Target Test Brier:       {target_brier:.4f}")
    print(f"Reproduced Test Brier:   {reproduced_brier:.4f}")

    assert reproduced_auc == target_auc, f"AUC mismatch: {reproduced_auc} vs {target_auc}"
    assert reproduced_pr == target_pr, f"PR-AUC mismatch: {reproduced_pr} vs {target_pr}"
    assert reproduced_brier == target_brier, f"Brier mismatch: {reproduced_brier} vs {target_brier}"

    print("\nSUCCESS: All test metrics reproduced to 4 decimal places bit-for-bit from saved artifacts!")

if __name__ == "__main__":
    main()

