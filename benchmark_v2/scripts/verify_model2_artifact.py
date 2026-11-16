import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
sys.path.insert(0, BASE_DIR)

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f_in:
        while chunk := f_in.read(8192):
            h.update(chunk)
    return h.hexdigest()

def pr_auc_score(y_true, y_pred):
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return float(auc(r, p))

def main():
    print("=== Verifying Model 2 Production Artifact ===")
    artifact_dir = os.path.join(BASE_DIR, "artifacts", "model2", "v1_tawos")
    meta_path = os.path.join(artifact_dir, "metadata.json")

    with open(meta_path, 'r') as fp:
        meta = json.load(fp)

    print(f"Loaded metadata.json: Model '{meta['model_name']}' version '{meta['version']}'")
    print(f"Trained on: {meta['trained_on']}")
    print(f"Target Decision Threshold: {meta['decision_threshold']}")
    print(f"Calibrator Type: {meta['calibrator_type']}")

    # 1. Verify Checksums
    model_path = os.path.join(artifact_dir, "model2_lightgbm.pkl")
    calibrator_path = os.path.join(artifact_dir, "model2_calibrator.pkl")
    transformer_path = os.path.join(artifact_dir, "model2_transformer.pkl")

    assert get_sha256(model_path) == meta['artifacts']['model2_lightgbm_sha256'], "Model SHA256 mismatch!"
    assert get_sha256(calibrator_path) == meta['artifacts']['model2_calibrator_sha256'], "Calibrator SHA256 mismatch!"
    assert get_sha256(transformer_path) == meta['artifacts']['model2_transformer_sha256'], "Transformer SHA256 mismatch!"
    print("All artifact SHA256 checksums verified successfully!")

    # 2. Reload Objects
    with open(model_path, 'rb') as fp:
        model = pickle.load(fp)
    with open(calibrator_path, 'rb') as fp:
        calibrator_info = pickle.load(fp)
        calibrator = calibrator_info['calibrator']
        cal_type = calibrator_info['type']
    with open(transformer_path, 'rb') as fp:
        transformer = pickle.load(fp)

    # 3. Load Test Data
    data_path = os.path.join(BASE_DIR, "work", "model2_dataset.parquet")
    df = pd.read_parquet(data_path)
    df_test_f50 = df[(df['split'] == 'test') & (df['f'] == 0.5)].copy()
    y_test_f50 = df_test_f50['label'].values
    print(f"Loaded test split at f=0.5: {len(df_test_f50)} instances, base rate: {np.mean(y_test_f50):.4f}")

    # 4. Transform features using reloaded transformer
    X_test_f50 = transformer.transform(df_test_f50)
    assert list(X_test_f50.columns) == meta['ordered_feature_list'], "Feature columns mismatch!"

    # 5. Predict using reloaded model and calibrator
    raw_probs = model.predict_proba(X_test_f50)[:, 1]
    if cal_type == 'isotonic':
        cal_probs = calibrator.predict(raw_probs)
    else:
        cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    # 6. Evaluate metrics
    reproduced_auc = float(roc_auc_score(y_test_f50, cal_probs))
    reproduced_prauc = pr_auc_score(y_test_f50, cal_probs)

    target_auc = meta['metrics_test_f50']['roc_auc']
    target_prauc = meta['metrics_test_f50']['pr_auc']

    print(f"\nTarget ROC-AUC from metadata:     {target_auc:.4f}")
    print(f"Reproduced ROC-AUC from reloaded: {reproduced_auc:.4f}")
    print(f"Target PR-AUC from metadata:      {target_prauc:.4f}")
    print(f"Reproduced PR-AUC from reloaded:  {reproduced_prauc:.4f}")

    assert round(reproduced_auc, 4) == target_auc, f"AUC mismatch: {round(reproduced_auc, 4)} != {target_auc}"
    assert round(reproduced_prauc, 4) == target_prauc, f"PR-AUC mismatch: {round(reproduced_prauc, 4)} != {target_prauc}"

    print("\nSUCCESS: Reloaded artifact reproduces test ROC-AUC and PR-AUC to 4 decimal places bit-for-bit!")

if __name__ == "__main__":
    main()

