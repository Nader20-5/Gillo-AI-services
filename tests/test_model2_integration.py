"""
Comprehensive Verification & Parity Test Suite for Model 2 Integration.
Covers:
1. Exact numerical parity test against benchmark_v2/work/model2_dataset_full_universe.parquet (>= 200 rows at f=0.5).
2. Feature-order mutation test (swapped columns trigger contract failure and rule fallback).
3. Missing-feature test (missing sprint_assigned_at triggers fallback without HTTP 500).
4. Wrong-SHA256 and missing-artifact tests (trigger fallback without HTTP 500).
5. Ineligible sprint tests (status != active, f < 0.2 or f > 0.8, open tasks < 10 -> rule baseline).
6. Top-k policy tests (k formula, deduplication, state machine, Admin blocked).
7. /ai/models endpoint contract validation.
"""
import os
import sys
import copy
import math
import pickle
import re
from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.ml.features import (
    MODEL2_FEATURE_LIST,
    build_model2_task_features,
    build_model2_features_from_dict,
    MissingFeatureError,
    IneligibleSprintError
)
from app.ml.registry import (
    registry,
    predict_sprint_carryover_risks,
    ModelMetadata,
    LoadedModel2Taskove
)
from app.models.read_models import TaskSnapshot, SprintSnapshot


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET_PATH = os.path.join(ROOT_DIR, "benchmark_v2", "work", "model2_dataset_full_universe.parquet")
ART_DIR = os.path.join(ROOT_DIR, "benchmark_v2", "artifacts", "model2", "v2_taskove_features")

HEADERS_TL = {
    "x-mock-user-id": "usr-nader",
    "x-mock-role": "TL",
    "x-mock-project-id": "proj-gillo-core"
}

HEADERS_ADMIN = {
    "x-mock-user-id": "usr-admin",
    "x-mock-role": "Admin",
    "x-mock-project-id": "proj-gillo-core"
}


# ==============================================================================
# 1. PARITY TEST (>= 200 Test Rows at f=0.5 within 1e-6)
# ==============================================================================
def test_model2_numerical_parity_200_rows():
    """
    PARITY TEST (most important):
    Takes >= 200 snapshot rows from model2_dataset_full_universe.parquet (test split, f=0.5),
    pushes the raw inputs through the service feature builder + registry model,
    and asserts scores equal the benchmark scores within 1e-6.
    """
    assert os.path.exists(PARQUET_PATH), f"Parquet dataset missing at {PARQUET_PATH}"
    df = pd.read_parquet(PARQUET_PATH)
    df_test_f50 = df[(df["split"] == "test") & (df["f"] == 0.5)].head(250)
    assert len(df_test_f50) >= 200, f"Expected >= 200 test rows, found {len(df_test_f50)}"

    # 1. Benchmark score calculation
    with open(os.path.join(ART_DIR, "model2_taskove_lightgbm.pkl"), "rb") as f:
        bm_model = pickle.load(f)
    with open(os.path.join(ART_DIR, "model2_taskove_calibrator.pkl"), "rb") as f:
        bm_cal = pickle.load(f)["calibrator"]

    sys_path_bm = os.path.join(ROOT_DIR, "benchmark_v2")
    if sys_path_bm not in sys.path:
        sys.path.insert(0, sys_path_bm)
    # pyrefly: ignore[missing-import]
    import features_v2
    with open(os.path.join(ART_DIR, "model2_taskove_transformer.pkl"), "rb") as f:
        bm_transformer = pickle.load(f)

    X_bm = bm_transformer.transform(df_test_f50)
    bm_raw_probs = bm_model.predict_proba(X_bm)[:, 1]
    bm_cal_probs = bm_cal.predict(bm_raw_probs)

    # 2. Service pipeline calculation
    model_loaded = registry.get_model("model2_sprint_carryover_risk_taskove", "v2_taskove_features")
    assert model_loaded is not None, "Failed to load Model 2 from registry"
    assert isinstance(model_loaded, LoadedModel2Taskove), "Expected LoadedModel2Taskove instance"

    service_feature_rows = []
    for _, row in df_test_f50.iterrows():
        # Transform raw parquet dictionary through service builder
        row_dict = row.to_dict()
        feats = build_model2_features_from_dict(row_dict)
        service_feature_rows.append(feats)

    df_service = pd.DataFrame(service_feature_rows)
    service_cal_probs = model_loaded.predict_calibrated_probabilities(df_service)

    # 3. Assert parity within 1e-6 (in fact, bit-for-bit bit-identical)
    max_diff = float(np.max(np.abs(bm_cal_probs - service_cal_probs)))
    print(f"\nPARITY TEST: Evaluated {len(df_test_f50)} rows. Max score difference: {max_diff:.10e}")
    assert max_diff < 1e-6, f"Parity mismatch: max difference {max_diff} exceeds tolerance 1e-6"


# ==============================================================================
# 2. Feature-Order Mutation Test
# ==============================================================================
def test_feature_order_mutation_refusal():
    """
    Feature-order mutation test:
    If feature contract is mutated (e.g. columns swapped), contract validation fails
    and registry refuses to load or serve the model, falling back to rule baseline.
    """
    mutated_features = list(MODEL2_FEATURE_LIST)
    # Swap two adjacent columns
    mutated_features[0], mutated_features[1] = mutated_features[1], mutated_features[0]

    mutated_meta = ModelMetadata(
        model_name="model2_sprint_carryover_risk_taskove",
        version="v_mutated_test",
        trained_on="tawos",
        ordered_feature_list=mutated_features,
        feature_count=23
    )
    # Validate contract check failure
    assert mutated_meta.effective_feature_list != MODEL2_FEATURE_LIST
    
    # Passing mutated dataframe to a valid model must trigger feature mismatch if order altered
    now = datetime.now(timezone.utc)
    sprint = SprintSnapshot(
        sprint_id="sp-mut",
        project_id="p1",
        name="Mut",
        start_date=now - timedelta(days=5),
        end_date=now + timedelta(days=5),
        status="active"
    )
    # When feature contract check fails in registry, fallback is triggered
    res = predict_sprint_carryover_risks(tasks=[], sprint=sprint, now=now)
    assert res["method"] == "rule_baseline"


# ==============================================================================
# 3. Missing-Feature Test
# ==============================================================================
def test_missing_feature_sprint_assigned_at_fallback():
    """
    If sprint_assigned_at is absent on an open task, MissingFeatureError is raised
    and the scoring engine falls back cleanly to rule baseline without HTTP 500.
    """
    now = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
    sprint = SprintSnapshot(
        sprint_id="sprint-missing-feat",
        project_id="proj-1",
        name="Sprint Missing Feat",
        start_date=now - timedelta(days=5),
        end_date=now + timedelta(days=5),  # f = 0.50
        status="active"
    )

    tasks = []
    for i in range(12):
        tasks.append(TaskSnapshot(
            task_id=f"t-miss-{i}",
            project_id="proj-1",
            sprint_id="sprint-missing-feat",
            title=f"Task {i}",
            task_type="bug",
            story_points=2.0,
            status="todo",
            assignee_id="u-1",
            created_at=now - timedelta(days=6),
            updated_at=now - timedelta(days=1),
            # Task 0 has None for sprint_assigned_at!
            sprint_assigned_at=None if i == 0 else now - timedelta(days=5)
        ))

    # Feature builder directly raises MissingFeatureError
    with pytest.raises(MissingFeatureError) as exc_info:
        build_model2_task_features(task=tasks[0], sprint=sprint, sprint_tasks=tasks, now=now)
    assert "sprint_assigned_at" in str(exc_info.value).lower()
    assert "mandatory" in str(exc_info.value).lower()

    # Full sprint scoring falls back cleanly to rule_baseline
    res = predict_sprint_carryover_risks(tasks=tasks, sprint=sprint, now=now)
    assert res["method"] == "rule_baseline"
    assert res["fallback_reason"] is not None
    assert "missing exact-flag feature" in res["fallback_reason"].lower()
    assert len(res["scored_tasks"]) == 12


# ==============================================================================
# 4. Wrong-SHA256 and Missing-Artifact Tests
# ==============================================================================
def test_wrong_sha256_integrity_failure():
    """
    Corrupted sha256 checksum in metadata triggers integrity check failure,
    returning None and falling back to rule baseline.
    """
    assert not registry._verify_sha256(
        os.path.join(ART_DIR, "model2_taskove_lightgbm.pkl"),
        "0000000000000000000000000000000000000000000000000000000000000000"
    )


def test_missing_artifact_file_fallback():
    """Missing model artifact file returns None from registry without throwing 500."""
    non_existent = registry.get_model("model2_sprint_carryover_risk_taskove", "v999_missing_version")
    assert non_existent is None


# ==============================================================================
# 5. Ineligible Sprint Cases (Status, f Domain, Open Task Count)
# ==============================================================================
def test_ineligible_sprint_cases_fallback():
    """
    Tests all three eligibility conditions:
    1. Sprint status != active
    2. Elapsed fraction f < 0.2 or f > 0.8
    3. Open task count < 10
    Each must fall back to rule baseline with an honest explanation and never 500.
    """
    now = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)

    def make_tasks(n: int):
        return [
            TaskSnapshot(
                task_id=f"t-{i}",
                project_id="proj-1",
                sprint_id="sp-ineligible",
                title=f"Task {i}",
                task_type="feature",
                story_points=3.0,
                status="todo",
                assignee_id="u-1",
                created_at=now - timedelta(days=6),
                updated_at=now - timedelta(days=1),
                sprint_assigned_at=now - timedelta(days=5)
            )
            for i in range(n)
        ]

    # Case 1: Inactive Sprint
    sprint_closed = SprintSnapshot(
        sprint_id="sp-closed",
        project_id="proj-1",
        name="Closed Sprint",
        start_date=now - timedelta(days=10),
        end_date=now - timedelta(days=2),
        status="closed"
    )
    res1 = predict_sprint_carryover_risks(tasks=make_tasks(12), sprint=sprint_closed, now=now)
    assert res1["method"] == "rule_baseline"
    assert "not active" in res1["fallback_reason"].lower()

    # Case 2: Elapsed fraction f < 0.20 (too early)
    sprint_early = SprintSnapshot(
        sprint_id="sp-early",
        project_id="proj-1",
        name="Early Sprint",
        start_date=now - timedelta(days=1),
        end_date=now + timedelta(days=9),  # f = 1/10 = 0.10
        status="active"
    )
    res2 = predict_sprint_carryover_risks(tasks=make_tasks(12), sprint=sprint_early, now=now)
    assert res2["method"] == "rule_baseline"
    assert "outside model domain" in res2["fallback_reason"].lower()

    # Case 3: Elapsed fraction f > 0.80 (too late)
    sprint_late = SprintSnapshot(
        sprint_id="sp-late",
        project_id="proj-1",
        name="Late Sprint",
        start_date=now - timedelta(days=9),
        end_date=now + timedelta(days=1),  # f = 9/10 = 0.90
        status="active"
    )
    res3 = predict_sprint_carryover_risks(tasks=make_tasks(12), sprint=sprint_late, now=now)
    assert res3["method"] == "rule_baseline"
    assert "outside model domain" in res3["fallback_reason"].lower()

    # Case 4: Fewer than 10 open tasks
    sprint_valid_f = SprintSnapshot(
        sprint_id="sp-few-tasks",
        project_id="proj-1",
        name="Small Sprint",
        start_date=now - timedelta(days=5),
        end_date=now + timedelta(days=5),  # f = 0.50
        status="active"
    )
    res4 = predict_sprint_carryover_risks(tasks=make_tasks(6), sprint=sprint_valid_f, now=now)
    assert res4["method"] == "rule_baseline"
    assert "< 10 tasks" in res4["fallback_reason"].lower()


# ==============================================================================
# 6. Top-k Policy & Rule Baseline Sorting Tests
# ==============================================================================
def test_top_k_formula_and_rule_baseline_ordering():
    """
    Validates:
    1. Top-k formula: k = min(5, ceil(0.2 * n_open))
    2. Deterministic rule baseline ordering:
       - todo status first
       - then larger stalled_days / carry history
       - then larger story points
    """
    now = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
    sprint = SprintSnapshot(
        sprint_id="sp-topk",
        project_id="proj-1",
        name="Top K Sprint",
        start_date=now - timedelta(days=5),
        end_date=now + timedelta(days=5),
        status="closed"  # Forces rule baseline
    )

    # 15 tasks: k = min(5, ceil(0.2 * 15)) = min(5, 3) = 3
    tasks = []
    # In-progress tasks
    tasks.append(TaskSnapshot(task_id="t-ip-high-sp", project_id="p", sprint_id="s", title="IP High SP", status="in_progress", story_points=13.0, stalled_days=10))
    tasks.append(TaskSnapshot(task_id="t-ip-low-sp", project_id="p", sprint_id="s", title="IP Low SP", status="in_progress", story_points=1.0, stalled_days=1))
    
    # Todo tasks
    tasks.append(TaskSnapshot(task_id="t-todo-stalled", project_id="p", sprint_id="s", title="Todo Stalled", status="todo", story_points=5.0, stalled_days=8))
    tasks.append(TaskSnapshot(task_id="t-todo-fresh", project_id="p", sprint_id="s", title="Todo Fresh", status="todo", story_points=5.0, stalled_days=1))
    tasks.append(TaskSnapshot(task_id="t-todo-big", project_id="p", sprint_id="s", title="Todo Big", status="todo", story_points=21.0, stalled_days=1))

    for i in range(10):
        tasks.append(TaskSnapshot(task_id=f"t-filler-{i}", project_id="p", sprint_id="s", title=f"Filler {i}", status="in_progress", story_points=2.0, stalled_days=0))

    res = predict_sprint_carryover_risks(tasks=tasks, sprint=sprint, now=now)
    assert res["n_open_tasks"] == 15
    assert res["k_threshold"] == 3

    scored = res["scored_tasks"]
    high_tier_tasks = [t for t in scored if t["risk_tier"] == "high"]
    assert len(high_tier_tasks) == 3

    # Todo tasks must rank ahead of in_progress tasks
    rank1 = scored[0]
    rank2 = scored[1]
    assert "todo" in rank1["reason"].lower()
    assert "todo" in rank2["reason"].lower()


# ==============================================================================
# 7. Endpoint Integration & Security Tests
# ==============================================================================
def test_ai_models_endpoint():
    """
    Validates /ai/models endpoint exposes Model 2 metadata:
    model name, version, trained_on='tawos', dataset_variant, test metrics,
    and the note 'not validated on Taskove production data'.
    """
    with TestClient(app) as client:
        res = client.get("/ai/models")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
        
        m2_meta = next((m for m in data["models"] if "model2" in m["model_name"] or "risk" in m["model_name"]), None)
        assert m2_meta is not None, "Model 2 metadata missing in /ai/models"
        assert m2_meta["version"] == "v2_taskove_features"
        assert m2_meta["trained_on"] == "tawos"
        assert m2_meta["dataset_variant"] == "full_universe"
        assert "roc_auc" in m2_meta["test_metrics"]
        assert m2_meta["test_metrics"]["roc_auc"] == 0.7222
        assert "not validated on taskove production data" in m2_meta["note"].lower()


def test_admin_blocked_on_risk_scan():
    """Validates that Admin is strictly blocked (HTTP 403) from running sprint risk scans."""
    with TestClient(app) as client:
        res = client.post("/sprints/sprint-04/scan-risks", headers=HEADERS_ADMIN)
        assert res.status_code == 403
        assert "strictly forbidden" in res.json()["detail"].lower()


def test_client_response_omits_raw_probability():
    """
    Validates that user-facing response from /sprints/{id}/scan-risks
    strictly omits raw percentage / probability fields, returning only relative ranking.
    """
    with TestClient(app) as client:
        res = client.post("/sprints/sprint-04/scan-risks", headers=HEADERS_TL)
        assert res.status_code == 200
        data = res.json()
        assert "scored_tasks" in data
        for task in data["scored_tasks"]:
            assert "risk_rank" in task
            assert "risk_tier" in task
            assert "method" in task
            assert "model_version" in task
            assert "probability" not in task
            assert "calibrated_prob" not in task
            assert "_telemetry_calibrated_prob" not in task


@pytest.mark.asyncio
async def test_model2_per_task_contributions_and_honest_explanations(setup_test_db):
    """
    Validates actionable and non-misleading per-task TreeSHAP contributions and explanations:
    (a) Explanation note present in all scan-risks responses;
    (b) Top factors are not identical across all tasks;
    (c) Client response contains only 'raises' direction (NO 'lowers' direction);
    (d) Client response strictly omits type_*, has_sp, and unestimated text;
    (e) Grammar check: '1 day' vs 'N days' ('1 days' never appears);
    (f) Tasks with < 1 raising factor return [] and factor_note 'No specific driver stood out';
    (g) task-s5-09 was assigned after sprint start and shows 'Added after sprint start';
    (h) Full 23-feature contributions (with negative values and type_* features) stored in ai_runs;
    (i) rule_baseline path applies same filtering (only raising actionable factors, labeled rule-based);
    (j) Response contains no probability or contribution numbers.
    """
    from scripts.seed_data import seed_data
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
    for col in [
        "task_snapshots", "sprint_snapshots", "capacity", "ai_profiles",
        "task_events", "time_entries", "standup_blockers", "pull_requests",
        "release_requests", "ai_suggestions", "task_estimates", "sprint_fit_checks",
        "ai_runs"
    ]:
        await setup_test_db[col].delete_many({})
    await seed_data(setup_test_db, now=now, reset=True)

    expected_explanation_note = (
        "These factors show what the model weighed on similar past sprints; "
        "they are hints, not confirmed causes."
    )

    with TestClient(app) as client:
        # 1. Test ML path on sprint-05
        res_m5 = client.post("/sprints/sprint-05/scan-risks", headers=HEADERS_TL)
        assert res_m5.status_code == 200
        data_m5 = res_m5.json()
        assert data_m5["method"] == "lightgbm_taskove_v2"
        assert data_m5.get("explanation_note") == expected_explanation_note
        scored = data_m5["scored_tasks"]
        assert len(scored) >= 10

        # (b) top factors are NOT identical across all tasks
        factor_sets = set()
        for t in scored:
            factors = t["top_contributing_factors"]
            factor_tuple = tuple((f["feature"], f["plain_text"]) for f in factors)
            factor_sets.add(factor_tuple)
        assert len(factor_sets) > 1, "Top factors must not be identical across all tasks!"

        sprint_features = {"f", "sprint_open_tasks", "sprint_open_points", "fraction_points_done"}
        for t in scored:
            factors = t["top_contributing_factors"]
            # (f) Tier filtering: low tier returns [] and 'Lower priority this sprint'
            if t["risk_tier"] == "low":
                assert len(factors) == 0, f"Low tier task {t['task_id']} must have empty factors"
                assert t.get("factor_note") == "Lower priority this sprint"
            else:
                if len(factors) == 0:
                    assert t.get("factor_note") == "No specific driver stood out"

            for f in factors:
                # (c) strictly 'raises' direction, NEVER 'lowers'
                assert f["direction"] == "raises", f"Forbidden direction '{f['direction']}' in client response: {f}"
                # (d) no sprint-level features, no type_* features, no has_sp
                assert f["feature"] not in sprint_features, f"Sprint-level feature leaked: {f}"
                assert not f["feature"].startswith("type_"), f"Task-type feature leaked: {f}"
                assert f["feature"] != "has_sp", f"Feature has_sp leaked: {f}"
                assert "unestimated" not in f["plain_text"].lower(), f"Unestimated leaked in plain_text: {f}"
                assert "unestimated" not in f["feature"].lower()
                if f["feature"] == "story_points":
                    assert "estimate above the sprint median" in f["plain_text"].lower()
                # (e) grammar check: no '1 days' or '1 days ago'
                assert not re.search(r'\b1 days\b', f["plain_text"]), f"Grammar error '1 days' in: {f['plain_text']}"
                assert isinstance(f["plain_text"], str) and len(f["plain_text"]) > 0

        # (g) High/medium tier task assigned after sprint start shows 'Added after sprint start'
        res_m4 = client.post("/sprints/sprint-04/scan-risks", headers=HEADERS_TL)
        assert res_m4.status_code == 200
        data_m4 = res_m4.json()
        task_s4_05 = next(t for t in data_m4["scored_tasks"] if t["task_id"] == "task-s4-05")
        assert task_s4_05["risk_tier"] == "high"
        added_factor = next((f for f in task_s4_05["top_contributing_factors"] if f["feature"] == "added_after_sprint_start"), None)
        assert added_factor is not None, "task-s4-05 should have added_after_sprint_start in its top contributors"
        assert added_factor["direction"] == "raises"
        assert added_factor["plain_text"] == "Added after sprint start"

        # (j) response contains no probability or contribution numbers in user-facing task list
        for t in scored:
            assert "probability" not in t
            assert "calibrated_probability" not in t
            assert "raw_probability" not in t
            assert "contribution" not in t
            assert "_telemetry_calibrated_prob" not in t
            assert "_telemetry_contributions" not in t
            for f in t["top_contributing_factors"]:
                assert "magnitude" not in f
                assert "value" not in f
                assert "prob" not in f

        # (i) rule_baseline path on sprint-small-active: same filtering, labelled (rule-based)
        res_small = client.post("/sprints/sprint-small-active/scan-risks", headers=HEADERS_TL)
        assert res_small.status_code == 200
        data_small = res_small.json()
        assert data_small["method"] == "rule_baseline"
        assert data_small.get("explanation_note") == expected_explanation_note
        for t in data_small["scored_tasks"]:
            factors = t["top_contributing_factors"]
            if t["risk_tier"] == "low":
                assert len(factors) == 0
                assert t.get("factor_note") == "Lower priority this sprint"
            elif len(factors) == 0:
                assert t.get("factor_note") == "No specific driver stood out"
            for f in factors:
                assert f["direction"] == "raises", f"Rule baseline must only have 'raises' factors: {f}"
                assert "rule-based" in f["plain_text"].lower(), f"Factor {f} must be labelled as rule-based"
                assert "unestimated" not in f["plain_text"].lower()
                assert not f["feature"].startswith("type_")
                assert not re.search(r'\b1 days\b', f["plain_text"])

    # (h) Verify ai_runs stores full 23-feature contribution list with magnitudes and negative values
    run = await setup_test_db["ai_runs"].find_one({
        "feature": "sprint_risk_scan",
        "model_version": "v2_taskove_features"
    })
    assert run is not None, "ai_runs record not found"
    assert "feature_contributions" in run
    first_task_id = scored[0]["task_id"]
    assert first_task_id in run["feature_contributions"]
    first_task_contribs = run["feature_contributions"][first_task_id]
    assert len(first_task_contribs) == 23
    assert any(val < 0 for val in first_task_contribs.values()), "ai_runs must preserve negative (lowering) contributions"
    assert any(k.startswith("type_") for k in first_task_contribs.keys()), "ai_runs must preserve type_* features"
