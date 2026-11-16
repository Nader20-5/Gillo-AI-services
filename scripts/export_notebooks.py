"""
Generates clean Jupyter Notebooks for Phase 2:
- notebooks/01_effort_regressor.ipynb (Model 1)
- notebooks/02_risk_classifier_shap.ipynb (Model 2)
- notebooks/03_qlora_llm_finetuning.ipynb (Model 3 - Colab ready)
"""

import os
import json

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.split("\n")}


def generate_nb1():
    cells = [
        md_cell("# Taskove (Gillo) - Model 1: Task Effort Regressor\n## Architecture: Sentence Embeddings + XGBoost Quantile Regression\nPredicts P25 (optimistic) and P75 (conservative) hours to provide realistic sprint estimates."),
        code_cell("import os\nimport numpy as np\nimport pandas as pd\nimport xgboost as xgb\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nimport matplotlib.pyplot as plt\n%matplotlib inline"),
        code_cell("# Load Processed Agile Dataset\ntrain_df = pd.read_csv('../data/processed/model1_effort_train.csv')\ntest_df = pd.read_csv('../data/processed/model1_effort_test.csv')\nprint(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')\ntrain_df.head()"),
        code_cell("# Run Training Pipeline\nfrom scripts.train_model1 import train\ntrain()"),
        md_cell("### Evaluation & Error Distribution\nWe plot Actual vs Predicted hours and the P25-P75 confidence interval."),
        code_cell("metrics_path = '../models/artifacts/model1_metrics.json'\nwith open(metrics_path) as f:\n    import json\n    metrics = json.load(f)\nprint('Evaluation Metrics:')\nfor k, v in metrics.items():\n    print(f'  {k}: {v}')")
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "01_effort_regressor.ipynb"), "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


def generate_nb2():
    cells = [
        md_cell("# Taskove (Gillo) - Model 2: Early Risk & Slippage Classifier\n## Architecture: LightGBM + SHAP TreeExplainer\nPredicts tasks likely to miss the sprint deadline and translates top SHAP feature contributions into human-readable evidence."),
        code_cell("import os\nimport numpy as np\nimport pandas as pd\nimport lightgbm as lgb\nimport shap\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix\nimport matplotlib.pyplot as plt\n%matplotlib inline"),
        code_cell("# Load Risk Dataset\ntrain_df = pd.read_csv('../data/processed/model2_risk_train.csv')\ntest_df = pd.read_csv('../data/processed/model2_risk_test.csv')\nprint(f'Train samples: {len(train_df)}, Slipped rate: {train_df.is_slipped.mean():.2%}')\ntrain_df.head()"),
        code_cell("# Run Training Pipeline\nfrom scripts.train_model2 import train\ntrain()"),
        md_cell("### SHAP Explainable AI Waterfall Demonstration\nInspect the exact mathematical reasons why a specific task is flagged as high risk."),
        code_cell("import pickle\nwith open('../models/artifacts/model_risk.pkl', 'rb') as f:\n    bundle = pickle.load(f)\n\n# Test on a simulated stalled task\nres = bundle.predict_with_explanation(\n    stalled_days=6,\n    dependency_depth=2,\n    active_blockers=1,\n    assignee_load_ratio=1.35,\n    logged_vs_estimated_ratio=1.40,\n    story_points=5.0\n)\n\nprint(f'Predicted Confidence: {res[\"confidence\"]*100:.1f}%')\nprint(f'Severity: {res[\"severity\"]}')\nprint('Generated Reasoning Steps:')\nfor step in res['reasoning_steps']:\n    print(f'  Step {step[\"order\"]}: {step[\"description\"]} (Evidence: {step[\"evidence\"]})')")
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "02_risk_classifier_shap.ipynb"), "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


def generate_nb3():
    cells = [
        md_cell("# Taskove (Gillo) - Model 3: Agile Summarizer Fine-Tuning\n## Architecture: Qwen2.5-3B-Instruct with 4-bit QLoRA\nFine-tunes a compact 3B LLM on Google Colab (Free T4 GPU) to generate JSON-compliant Standup, Retro, and Release drafts."),
        code_cell("# 1. Install fine-tuning dependencies (Run on Colab)\n# !pip install -q torch transformers datasets peft bitsandbytes trl accelerate"),
        code_cell("import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\nfrom peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n\nmodel_id = 'Qwen/Qwen2.5-3B-Instruct'\nprint(f'Target Base Model: {model_id}')"),
        code_cell("# 2. 4-bit Quantization Configuration\nbnb_config = BitsAndBytesConfig(\n    load_in_4bit=True,\n    bnb_4bit_quant_type='nf4',\n    bnb_4bit_use_double_quant=True,\n    bnb_4bit_compute_dtype=torch.bfloat16\n)\n\n# 3. LoRA PEFT Adapter Setup\nlora_config = LoraConfig(\n    r=16,\n    lora_alpha=32,\n    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],\n    lora_dropout=0.05,\n    bias='none',\n    task_type='CAUSAL_LM'\n)"),
        code_cell("# 4. Load Instruction Dataset\n# dataset = load_dataset('json', data_files='model3_instruction_tuning.jsonl')\nprint('Ready to train with SFTTrainer on Google Colab GPU.')")
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "03_qlora_llm_finetuning.ipynb"), "w") as f:
        json.dump(make_notebook(cells), f, indent=2)


if __name__ == "__main__":
    generate_nb1()
    generate_nb2()
    generate_nb3()
    print("Notebooks 01, 02, and 03 generated in notebooks/")
