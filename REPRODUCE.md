# Machine Learning Reproduction Guide (`REPRODUCE.md`)

This guide provides the exact commands, scripts, input data dependencies, and verification procedures required to reproduce the served production artifacts in `models/artifacts/` for **Model 1** (`model_sp_taskove`) and **Model 2** (`model2_sprint_carryover_risk_taskove`).

---

## 1. Prerequisites & Input Data Files

All models are trained exclusively on empirical, open-source Jira data from the TAWOS dataset. The raw CSV files must reside in `tawos_csv/`:

| Input File | Required Columns | Size (Approx) | Purpose |
|---|---|---|---|
| `tawos_csv/issue.csv` | `ID`, `Title`, `Description_Text`, `Type`, `Priority`, `Story_Point`, `Estimation_Date`, `Creation_Date`, `Project_ID`, `Story_Point_Changed_After_Estimation`, `Title_Changed_After_Estimation`, `Description_Changed_After_Estimation`, `Resolution` | ~1.1 GB | Issue backlog, text descriptions, and estimation outcomes for Model 1 and Model 2 |
| `tawos_csv/sprint.csv` | `Sprint_ID`, `Project_ID`, `Start_Date`, `End_Date`, `Complete_Date`, `State` | ~15 MB | Sprint boundary dates and metadata for Model 2 temporal replay |
| `tawos_csv/changelog.csv` | `Issue_ID`, `Creation_Date`, `Field`, `From_String`, `To_String` | ~2.5 GB | Event history transitions for Model 2 state reconstruction at time $t$ |

---

## 2. Model 1: Story Point Effort Regressor (`v2_taskove`)

### A. Artifact Inventory
Target serving directory: `models/artifacts/model_sp/v2_taskove/`  
Target benchmark directory: `benchmark_v2/artifacts/model1_sp/v2_taskove/`

- `model_sp_final.pkl`: LightGBM L1 point regressor (2,517 input dimensions: 2,500 TF-IDF + 17 OHE types).
- `model_sp_q25.pkl`: LightGBM pinball loss ($\alpha = 0.25$) quantile regressor.
- `model_sp_q50.pkl`: LightGBM pinball loss ($\alpha = 0.50$) quantile regressor.
- `model_sp_q75.pkl`: LightGBM pinball loss ($\alpha = 0.75$) quantile regressor.
- `tfidf_vectorizer.pkl`: Scikit-learn `TfidfVectorizer` (2,500 unigram/bigram features, English Jira vocabulary).
- `ohe_encoder.pkl`: Scikit-learn `OneHotEncoder` (17 TAWOS task types, handle_unknown='ignore').
- `type_median_fallback.json`: Empirical interquartile ranges $[Q25, Q75]$ and median points computed from 29,330 TAWOS training tasks.
- `metadata.json`: Model card, input contract `['title', 'description', 'type']`, out-of-project conformal constant ($c_{\text{cold-start}} = 0.1147$), test metrics, and SHA256 checksums.

### B. Reproduction Pipeline (In Order)

```bash
# Step 1: Train point model and quantile models, fit TF-IDF and OHE, and write initial artifacts
python benchmark_v2/scripts/run_model1_part3_all.py

# Step 2: Compute empirical fallback interquartile ranges [Q25, Q75] for each task type
python benchmark_v2/scripts/compute_type_quantiles.py

# Step 3: Compute out-of-project conformal constant c_cold_start across 5-fold CV and update metadata.json
python benchmark_v2/scripts/run_step0_audit_fixes.py

# Step 4: Deploy artifacts to runtime serving directory
cp benchmark_v2/artifacts/model1_sp/v2_taskove/* models/artifacts/model_sp/v2_taskove/
```

### C. Verification Scripts & Proof Outputs

```bash
# Verify artifact integrity, test metrics, and conformal coverage against TAWOS test set
python benchmark_v2/scripts/verify_model1_v2_taskove.py | Tee-Object reports/reproduce/verify_model1_output.txt

# Verify bit-for-bit parity of runtime served copy against benchmark artifact
pytest -v tests/test_model1_integration.py::test_model1_parity_against_benchmark_artifacts | Tee-Object -Append reports/reproduce/parity_tests_output.txt
```

**Expected Verification Metrics (from `reports/reproduce/verify_model1_output.txt`):**
- Test MAE: `1.7607`
- Test MdAE: `1.0932`
- Test RMSE: `3.2217`
- 50% Conformal Coverage: `51.6783%` (mean width: 2.9217 SP)
- Checksums: All 7 files match metadata SHA256 hashes bit-for-bit.

---

## 3. Model 2: Sprint Carry-Over Risk Classifier (`v2_taskove_features`)

### A. Artifact Inventory
Target serving directory: `models/artifacts/model2/v2_taskove_features/`  
Target benchmark directory: `benchmark_v2/artifacts/model2/v2_taskove_features/`

- `model2_taskove_lightgbm.pkl`: LightGBM binary classifier (23 canonical Taskove-compatible features).
- `model2_taskove_calibrator.pkl`: Isotonic regression probability calibrator.
- `model2_taskove_transformer.pkl`: Feature extraction pipeline transformer (ordered 23-feature vector).
- `metadata.json`: Model card, feature contract (23 features), evaluation metrics, and SHA256 checksums.

### B. Reproduction Pipeline (In Order)

```bash
# Step 1: Replay sprint changelogs to construct full-universe temporal replay dataset
python benchmark_v2/scripts/generate_model2_dataset_full_universe.py
# Output: benchmark_v2/work/model2_dataset_full_universe.parquet (62,569 rows)

# Step 2: Train LightGBM classifier on 23 Taskove-compatible features, fit isotonic calibrator, and save artifacts
python benchmark_v2/scripts/run_model2_final_audits.py

# Step 3: Deploy artifacts to runtime serving directory
cp benchmark_v2/artifacts/model2/v2_taskove_features/* models/artifacts/model2/v2_taskove_features/
```

### C. Verification Scripts & Proof Outputs

```bash
# Verify artifact integrity, calibration, ROC-AUC, PR-AUC, and Brier score
python benchmark_v2/scripts/verify_model2_taskove_artifact.py | Tee-Object reports/reproduce/verify_model2_output.txt

# Verify numerical parity on 200 held-out test rows between runtime and benchmark copies
pytest -v tests/test_model2_integration.py::test_model2_numerical_parity_200_rows | Tee-Object -Append reports/reproduce/parity_tests_output.txt
```

**Expected Verification Metrics (from `reports/reproduce/verify_model2_output.txt`):**
- Test ROC-AUC: `0.7222`
- Test PR-AUC: `0.8786`
- Test Brier Score: `0.1712`
- Checksums: All 3 files match metadata SHA256 hashes bit-for-bit.

---

## 4. Execution Feasibility & Provenance Summary

| Pipeline Component | Source Script | Required Inputs | Execution Feasibility | Proof Report File |
|---|---|---|---|---|
| **Model 1 Training** | `benchmark_v2/scripts/run_model1_part3_all.py` | `tawos_csv/issue.csv` | Feasible (CPU, ~4 min) | `reports/reproduce/verify_model1_output.txt` |
| **Model 1 Fallback IQRs** | `benchmark_v2/scripts/compute_type_quantiles.py` | `tawos_csv/issue.csv` | Feasible (CPU, ~30s) | `reports/reproduce/verify_model1_output.txt` |
| **Model 1 Out-of-Project Conformal** | `benchmark_v2/scripts/run_step0_audit_fixes.py` | `tawos_csv/issue.csv` | Feasible (CPU, ~3 min) | `reports/reproduce/verify_model1_output.txt` |
| **Model 1 Verification** | `benchmark_v2/scripts/verify_model1_v2_taskove.py` | Artifacts + `tawos_csv/issue.csv` | Executed & Verified | `reports/reproduce/verify_model1_output.txt` |
| **Model 2 Dataset Replay** | `benchmark_v2/scripts/generate_model2_dataset_full_universe.py` | `tawos_csv/` (all 3 CSVs) | Heavy I/O (~15 min; cached in `parquet`) | `reports/reproduce/verify_model2_output.txt` |
| **Model 2 Training & Auditing** | `benchmark_v2/scripts/run_model2_final_audits.py` | Replay parquet dataset | Feasible (CPU, ~2 min) | `reports/reproduce/verify_model2_output.txt` |
| **Model 2 Verification** | `benchmark_v2/scripts/verify_model2_taskove_artifact.py` | Artifacts + parquet dataset | Executed & Verified | `reports/reproduce/verify_model2_output.txt` |
| **Runtime Serving Parity** | `tests/test_model1_integration.py`, `tests/test_model2_integration.py` | `models/artifacts/` | Executed & Verified | `reports/reproduce/parity_tests_output.txt` |

*(Note: Legacy synthetic scripts `train_model1.py`, `train_model2.py`, and `prepare_ml_datasets.py` were early validation prototypes and are permanently retired under `legacy_synthetic/`; they are never used to produce served artifacts).*
