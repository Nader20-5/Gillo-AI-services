# Retired & Archived Files

This folder contains components from **Phase 1 (Synthetic Prototype)** and offline research benchmarks that are **not part of the active production runtime**.

The active production microservice uses:
- **Model 1 (Story Points):** `models/artifacts/model_sp/v2_taskove/` (LightGBM + TF-IDF on 29,330 TAWOS issues)
- **Model 2 (Carry-Over Risk):** `models/artifacts/model_risk/v2_taskove_features/` (LightGBM + Isotonic Calibration + TreeSHAP on 23 temporal features)
- **Model 3 (Agile Drafts & Chat):** Versioned prompt templates with Pydantic JSON validation in `prompts/` and `app/services/llm_provider.py`

---

## Folder Contents

### 1. `scripts/` (Legacy Phase 1 Scripts)
* `prepare_ml_datasets.py`: Synthetic dataset generator (superseded by real TAWOS benchmark data).
* `train_model1.py`: Synthetic XGBoost model 1 training script (superseded by `benchmark_v2/scripts/train_model1_benchmark.py`).
* `train_model2.py`: Synthetic LightGBM model 2 training script (superseded by `benchmark_v2/scripts/train_and_evaluate_model2.py`).
* `benchmark_tawos.py`: Legacy v1 benchmark script that contained temporal leakage (`backlog_age_days`), superseded by `benchmark_v2/`.

### 2. `notebooks/` (Exploratory & Unexecuted Notebooks)
* `01_effort_regressor.ipynb`: Initial exploration of synthetic effort regression.
* `02_risk_classifier_shap.ipynb`: Initial exploration of synthetic risk classification.
* `03_qlora_llm_finetuning.ipynb`: Unexecuted QLoRA fine-tuning notebook (runtime relies on prompt templates with Ollama/Cloud providers).

### 3. `data/` (Synthetic Datasets)
* `model1_effort_train.csv` & `model1_effort_test.csv`: Synthetic task effort data.
* `model2_risk_train.csv` & `model2_risk_test.csv`: Synthetic slippage risk data.
* `model3_instruction_tuning.jsonl`: Synthetic instruction tuning conversations.

### 4. `scratch_logs/` (Auditing Terminal Dumps)
* Static text output dumps from past terminal runs, grep searches, and pytest captures.
