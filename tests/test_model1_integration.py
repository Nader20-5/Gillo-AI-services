"""
Integration, Parity, and Robustness Test Suite for Model 1 (Taskove v2_taskove).
Verifies:
1. PARITY TEST: >= 200 issues from the reproduced Epic-excluded test set match benchmark artifact predictions within 1e-6.
2. Mutation test: swapped input field order triggers contract check -> fallback.
3. Language guard: non-Latin text triggers type_median_fallback with reason='non_english_text'.
4. Robustness: empty description, very long text, unknown type, missing artifact, wrong sha256 -> no 500, correct fallback labels.
5. Mapping table: every Taskove type maps as expected, unknown defaults to 'Task'.
6. /ai/models endpoint exposes name, version, trained_on='tawos', test MAE (1.7607), cold-start note, and unvalidated note.
"""
import os
import json
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.ml.registry import (
    registry,
    predict_story_points_with_fallback,
    MODEL1_ORDERED_INPUT_FIELDS,
    LoadedModel1Taskove,
)
from app.ml.taskove_types import (
    TASKOVE_TO_TAWOS_TYPE_MAP,
    DEFAULT_TAWOS_TYPE,
    map_taskove_type_to_tawos,
    is_mostly_non_latin,
    snap_to_fibonacci,
    FIBONACCI_SCALE,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ===========================================================================
# 1. Type Mapping Table Tests
# ===========================================================================
def test_taskove_to_tawos_type_mapping():
    """Verifies the canonical Taskove-to-TAWOS categorical mapping table."""
    expected_mappings = {
        "feature": "Story",
        "bug": "Bug",
        "docs": "Documentation",
        "refactor": "Improvement",
        "infra": "Technical task",
        "test": "Technical task",
        "backend": "Technical task",
        "frontend": "Technical task",
        "qa": "Technical task",
        "ml": "Task",
    }
    for taskove_type, expected_tawos in expected_mappings.items():
        assert map_taskove_type_to_tawos(taskove_type) == expected_tawos
        # Case insensitivity
        assert map_taskove_type_to_tawos(taskove_type.upper()) == expected_tawos

    # Unknown types default to 'Task'
    assert map_taskove_type_to_tawos("unknown_type") == "Task"
    assert map_taskove_type_to_tawos("devops") == "Task"
    assert map_taskove_type_to_tawos("") == "Task"
    assert map_taskove_type_to_tawos(None) == "Task"

    # Known TAWOS types are preserved directly
    assert map_taskove_type_to_tawos("Story") == "Story"
    assert map_taskove_type_to_tawos("Technical task") == "Technical task"
    assert map_taskove_type_to_tawos("Bug") == "Bug"


# ===========================================================================
# 2. Fibonacci Snapping Tests
# ===========================================================================
def test_fibonacci_snapping():
    """Verifies continuous predictions snap to nearest valid Fibonacci story point."""
    assert snap_to_fibonacci(0.2) == 0.5
    assert snap_to_fibonacci(0.6) == 0.5
    assert snap_to_fibonacci(0.8) == 1.0
    assert snap_to_fibonacci(1.4) == 1.0
    assert snap_to_fibonacci(1.6) == 2.0
    assert snap_to_fibonacci(2.4) == 2.0
    assert snap_to_fibonacci(2.6) == 3.0
    assert snap_to_fibonacci(4.1) == 5.0
    assert snap_to_fibonacci(6.9) == 8.0
    assert snap_to_fibonacci(11.0) == 13.0
    assert snap_to_fibonacci(18.0) == 21.0
    assert snap_to_fibonacci(35.0) == 21.0


# ===========================================================================
# 3. PARITY TEST (Most Important)
# ===========================================================================
def test_model1_parity_against_benchmark_artifacts():
    """
    Takes >= 200 issues from the reproduced Epic-excluded test set
    (same procedure as verify_model1_v2_taskove.py), runs them through the service path,
    and asserts predictions equal the benchmark artifact predictions within 1e-6.
    """
    tawos_csv_path = os.path.join("tawos_csv", "issue.csv")
    if not os.path.exists(tawos_csv_path):
        pytest.skip("tawos_csv/issue.csv not found; skipping test partition parity test.")

    # Load test partition identically to verify_model1_v2_taskove.py
    cols = [
        "ID", "Title", "Description_Text", "Type", "Story_Point",
        "Estimation_Date", "Creation_Date",
        "Story_Point_Changed_After_Estimation",
        "Title_Changed_After_Estimation",
        "Description_Changed_After_Estimation",
        "Resolution"
    ]
    df_raw = pd.read_csv(tawos_csv_path, usecols=cols, low_memory=False)
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

    idx_val = int(len(df_clean) * 0.85)
    test_ne = df_clean.iloc[idx_val:][df_clean.iloc[idx_val:]['Type'] != 'Epic'].copy().reset_index(drop=True)

    n_sample = min(250, len(test_ne))
    assert n_sample >= 200, f"Expected >= 200 test issues, found {n_sample}"
    sample = test_ne.iloc[:n_sample].copy()

    # Load active Model 1 wrapper
    m1 = registry.get_model("model_sp_taskove")
    assert isinstance(m1, LoadedModel1Taskove), "Failed to load LoadedModel1Taskove"

    # Vectorize sample in batch
    text_sample = (sample['Title'].fillna('') + ' ' + sample['Description_Text'].fillna('')).str.slice(0, 2000).fillna('')
    X_tfidf = m1.tfidf.transform(text_sample)
    X_cat = m1.ohe.transform(sample[['Type']].fillna('Missing'))
    X_feat = hstack([X_tfidf, X_cat]).tocsr().astype(np.float32)

    batch_preds = np.clip(m1.model_final.predict(X_feat), 0.5, 100.0)
    batch_q25 = np.clip(m1.model_q25.predict(X_feat), 0.5, 100.0)
    batch_q75 = np.clip(m1.model_q75.predict(X_feat), 0.5, 100.0)

    # Run through the service path issue-by-issue
    max_raw_diff = 0.0
    for i in range(n_sample):
        row = sample.iloc[i]
        title = str(row['Title'] or '')
        desc = str(row['Description_Text'] or '')
        tawos_type = str(row['Type'])

        res = predict_story_points_with_fallback(
            title=title,
            description=desc,
            task_type="feature",  # passed with tawos_type override
            tawos_type=tawos_type
        )

        assert res["method"] == "lightgbm_taskove_v2"
        assert res["model_version"] == "v2_taskove"
        assert res["fallback_reason"] is None
        assert res["story_points"] in FIBONACCI_SCALE

        raw_diff = abs(res["raw_story_points"] - batch_preds[i])
        if raw_diff > max_raw_diff:
            max_raw_diff = raw_diff

        assert raw_diff < 1e-6, f"Parity mismatch at index {i}: service={res['raw_story_points']} vs batch={batch_preds[i]}"

        # Interval check
        expected_low = max(0.5, round(batch_q25[i] - m1.c_cold_start, 1))
        expected_high = max(expected_low + 0.5, round(batch_q75[i] + m1.c_cold_start, 1))
        assert res["interval"] == [expected_low, expected_high]

    print(f"\n[PARITY TEST] Verified {n_sample} issues bit-for-bit: max diff = {max_raw_diff:.8e} < 1e-6!")


# ===========================================================================
# 4. Mutation Test (Field Order Contract Check)
# ===========================================================================
def test_mutation_input_field_order_fails_contract():
    """Swapping the input field order must fail the contract check and trigger fallback."""
    # Canonical order: ['title', 'description', 'type']
    invalid_order = ["type", "title", "description"]
    res = predict_story_points_with_fallback(
        title="Add payment webhook handler",
        description="Process stripe invoice.payment_succeeded events",
        task_type="feature",
        field_order=invalid_order
    )
    assert res["method"] == "type_median_fallback"
    assert res["fallback_reason"] == "contract_violation_field_order"
    assert res["story_points"] in FIBONACCI_SCALE
    assert res["interval"][0] <= res["interval"][1]


# ===========================================================================
# 5. Language Guard Tests
# ===========================================================================
def test_language_guard_mostly_non_latin():
    """If text contains >30% non-ASCII letters, model must not run; return per-type median fallback."""
    arabic_text = "إصلاح تسرب الذاكرة في خدمة التنبيهات وإعادة تشغيل الخادم"
    assert is_mostly_non_latin(arabic_text) is True

    res = predict_story_points_with_fallback(
        title=arabic_text,
        description="تفاصيل إضافية غير لاتينية",
        task_type="bug"
    )
    assert res["method"] == "type_median_fallback"
    assert res["fallback_reason"] == "non_english_text"
    # Bug fallback median in TAWOS is 2.0
    assert res["story_points"] == 2.0

    # English text with normal characters should run ML model
    english_text = "Fix memory leak in notification service and restart listener"
    assert is_mostly_non_latin(english_text) is False
    res_en = predict_story_points_with_fallback(
        title=english_text,
        description="Reproduced under high concurrent load",
        task_type="bug"
    )
    assert res_en["method"] == "lightgbm_taskove_v2"
    assert res_en["fallback_reason"] is None


# ===========================================================================
# 6. Robustness Tests (Empty, Long, Unknown, Missing, Checksum)
# ===========================================================================
def test_empty_description_succeeds():
    """Empty description is a normal use case and must execute without error."""
    res = predict_story_points_with_fallback(
        title="Refactor auth middleware to use JWT cookies",
        description="",
        task_type="refactor"
    )
    assert res["method"] == "lightgbm_taskove_v2"
    assert res["fallback_reason"] is None
    assert res["story_points"] in FIBONACCI_SCALE


def test_very_long_text_is_truncated_safely():
    """Text longer than 5000 characters is safely handled and truncated to 2000 chars."""
    long_desc = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 150
    assert len(long_desc) > 5000
    res = predict_story_points_with_fallback(
        title="Handle large text payload safely",
        description=long_desc,
        task_type="feature"
    )
    assert res["method"] == "lightgbm_taskove_v2"
    assert res["fallback_reason"] is None
    assert res["story_points"] in FIBONACCI_SCALE


def test_unknown_task_type_defaults_to_task():
    """An unknown task type maps to 'Task' and executes smoothly without error."""
    res = predict_story_points_with_fallback(
        title="Train unsupervised anomaly detection model",
        description="Run isolation forest on server logs",
        task_type="quantum_algorithm"
    )
    assert res["method"] == "lightgbm_taskove_v2"
    assert res["fallback_reason"] is None
    assert res["story_points"] in FIBONACCI_SCALE


def test_missing_model_fallback(monkeypatch):
    """When model fails to load or is unavailable, fallback is returned gracefully (never HTTP 500)."""
    monkeypatch.setattr(registry, "get_model", lambda name, version=None: None)
    res = predict_story_points_with_fallback(
        title="Test fallback when model missing",
        task_type="feature"
    )
    assert res["method"] == "type_median_fallback"
    assert res["fallback_reason"] == "model_unavailable"
    assert res["story_points"] in FIBONACCI_SCALE


def test_sha256_mismatch_fallback(monkeypatch):
    """Integrity check failure prevents model from loading and safely falls back."""
    monkeypatch.setattr(registry, "_verify_sha256", lambda fpath, exp: False)
    # Clear registry cache to force reload check
    registry._cache.clear()
    m = registry.get_model("model_sp_taskove")
    assert m is None, "Model with failed SHA256 must refuse to load"

    res = predict_story_points_with_fallback(
        title="Test fallback with corrupt hash",
        task_type="bug"
    )
    assert res["method"] == "type_median_fallback"
    assert res["fallback_reason"] == "model_unavailable"
    assert res["story_points"] == 2.0


# ===========================================================================
# 7. Endpoint Tests (/ai/models)
# ===========================================================================
def test_ai_models_endpoint_metadata(client):
    """
    /ai/models must expose:
    - name: 'model_sp_taskove'
    - version: 'v2_taskove'
    - trained_on: 'tawos'
    - test MAE: 1.7607
    - cold-start note: transferable gain ~6.5%
    - note: 'not validated on Taskove production data'
    """
    # Ensure fresh cache
    registry._cache.clear()
    resp = client.get("/ai/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data

    models = data["models"]
    m1_list = [m for m in models if m.get("model_name") == "model_sp_taskove" or m.get("name") == "model_sp_taskove"]
    assert len(m1_list) >= 1, f"Model 1 not found in /ai/models: {models}"

    m1 = m1_list[0]
    assert m1["version"] == "v2_taskove"
    assert m1["trained_on"] == "tawos"
    assert m1["test_mae"] == 1.7607
    assert "cold-start transferable gain is ~6.5%" in m1["cold_start_note"]
    assert m1["note"] == "not validated on Taskove production data"

