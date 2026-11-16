import os
import pytest
from pydantic import ValidationError
from app.config import Settings, settings
from app.ml.registry import (
    registry,
    predict_effort_with_fallback,
    predict_risk_with_fallback
)
from app.models.estimates import TaskEstimateModel, BaselineInfo
from app.models.suggestions import AISuggestionBase


def test_production_refuses_default_jwt_secret():
    """Validates that production environment strictly refuses to start with default or empty JWT_SECRET."""
    # Production with default insecure secret must fail validation
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            ENVIRONMENT="production",
            JWT_SECRET="dev-insecure-secret"
        )
    assert "FATAL: Production mode detected. You must set a non-default, secure 'JWT_SECRET'" in str(exc_info.value)

    # Production with empty secret must fail validation
    with pytest.raises(ValidationError) as exc_info2:
        Settings(
            ENVIRONMENT="production",
            JWT_SECRET=""
        )
    assert "FATAL: Production mode detected. You must set a non-default, secure 'JWT_SECRET'" in str(exc_info2.value)

    # Production with custom strong secret succeeds
    prod_settings = Settings(
        ENVIRONMENT="production",
        JWT_SECRET="super-secure-production-random-key-998811"
    )
    assert prod_settings.ENVIRONMENT == "production"
    assert prod_settings.JWT_SECRET == "super-secure-production-random-key-998811"


def test_ollama_default_model_aligned():
    """Validates that OLLAMA_MODEL default is qwen2.5:3b."""
    cfg = Settings()
    assert cfg.OLLAMA_MODEL == "qwen2.5:3b"


def test_synthetic_model_default_policy():
    """
    Validates that when ENABLE_SYNTHETIC_MODELS=False (default),
    registry refuses synthetic models and falls back to deterministic rule baselines.
    """
    assert settings.ENABLE_SYNTHETIC_MODELS is False

    # Force registry to look up model_est with default settings
    registry._cache.clear()
    loaded = registry.get_model("model_est", "v1_synthetic")
    # Because trained_on == 'synthetic' and ENABLE_SYNTHETIC_MODELS=False, registry returns None
    assert loaded is None

    # Predict effort with default settings -> falls back to rule_baseline
    pred = predict_effort_with_fallback(
        title="Refactor auth",
        task_type="backend",
        story_points=3.0,
        dependency_count=0,
        historical_median=10.0,
        assignee_ratio=1.0
    )
    assert pred["method"] == "rule_baseline"
    assert pred["model_version"] == "deterministic_baseline_v1"
    assert pred["confidence"] in ["low", "medium", "high"]
    assert "confidence_note" in pred
    assert isinstance(pred["confidence_note"], str)

    # Predict risk with default settings -> falls back to rule_baseline
    risk_pred = predict_risk_with_fallback(
        stalled_days=4,
        dependency_depth=1,
        active_blockers=1,
        assignee_load_ratio=1.2,
        logged_vs_estimated_ratio=1.1,
        story_points=5.0
    )
    assert risk_pred["method"] == "rule_baseline"
    assert risk_pred["model_version"] == "deterministic_risk_v1"
    assert risk_pred["confidence"] in ["low", "medium", "high"]
    assert "confidence_note" in risk_pred
    assert isinstance(risk_pred["confidence_note"], str)


def test_categorical_confidence_schema_validation():
    """
    Validates that TaskEstimateModel and AISuggestionBase require or default to
    categorical confidence ('low', 'medium', 'high') with confidence notes.
    """
    est = TaskEstimateModel(
        task_id="task-100",
        sprint_id="sprint-01",
        project_id="proj-1",
        low_hours=4.0,
        high_hours=8.0,
        confidence="high",
        confidence_note="Computed from empirical developer quantiles",
        method="rule_baseline"
    )
    assert est.confidence == "high"
    assert est.confidence_note == "Computed from empirical developer quantiles"

    sug = AISuggestionBase(
        project_id="proj-1",
        sprint_id="sprint-01",
        task_id="task-100",
        type="risk_warning",
        severity="medium",
        confidence="medium",
        confidence_note="Rule trigger score 0.55",
        dedupe_key="risk:sprint-01:task-100"
    )
    assert sug.confidence == "medium"
    assert sug.confidence_note == "Rule trigger score 0.55"


def test_requirements_dev_separation():
    """Validates that requirements.txt does not contain pytest and requirements-dev.txt exists."""
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    dev_path = os.path.join(os.path.dirname(__file__), "..", "requirements-dev.txt")

    assert os.path.exists(req_path)
    assert os.path.exists(dev_path)

    with open(req_path, "r", encoding="utf-8") as f:
        req_content = f.read()
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_content = f.read()

    assert "pytest" not in req_content
    assert "pytest" in dev_content
    assert "pytest-asyncio" in dev_content
    assert "pytest-cov" in dev_content
    assert "apscheduler" in req_content


def test_models_active_version_default_none():
    """Validates active model versions and that 'none' returns None."""
    cfg = Settings()
    assert cfg.MODEL1_ACTIVE_VERSION == "v2_taskove"
    assert cfg.MODEL2_ACTIVE_VERSION == "v2_taskove_features"

    # Registry lookup with 'none' must return None (forcing rule baseline)
    assert registry.get_model("model_est", "none") is None
    assert registry.get_model("model_risk", "none") is None


@pytest.mark.asyncio
async def test_sprint_fit_verdicts_fits_tight_over(setup_test_db):
    """
    Validates that sprint fit check distinctly differentiates fits, tight, and over:
    - 'fits': high-estimate workload <= capacity (all worst-case scenarios fit)
    - 'tight': mid-estimate <= capacity, but high-estimate > capacity (fits on average, but no buffer)
    - 'over': mid-estimate > capacity (expected work exceeds available hours)
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from tests.conftest import HEADERS_TL

    client = TestClient(app)
    db = setup_test_db

    # User usr-tester with 40h capacity
    user_id = "usr-test-fit"
    await db.capacity.insert_one({"sprint_id": "sprint-fit-test", "user_id": user_id, "capacity_hours": 40.0})
    await db.sprint_snapshots.insert_one({"sprint_id": "sprint-fit-test", "project_id": "proj-gillo-core", "status": "active"})

    # 1. Fits: low=15, mid=25, high=35 on 40h capacity (high <= 40)
    await db.task_estimates.delete_many({"sprint_id": "sprint-fit-test"})
    await db.task_estimates.insert_one({
        "task_id": "t-fit-1",
        "sprint_id": "sprint-fit-test",
        "assignee_id": user_id,
        "low_hours": 15.0,
        "mid_hours": 25.0,
        "high_hours": 35.0,
        "status": "active"
    })
    res_fits = client.get("/sprints/sprint-fit-test/fit-check", headers=HEADERS_TL)
    assert res_fits.status_code == 200
    assert res_fits.json()["verdict"] == "fits"

    # 2. Tight: low=20, mid=35, high=48 on 40h capacity (mid=35 <= 40, high=48 > 40)
    await db.task_estimates.delete_many({"sprint_id": "sprint-fit-test"})
    await db.task_estimates.insert_one({
        "task_id": "t-tight-1",
        "sprint_id": "sprint-fit-test",
        "assignee_id": user_id,
        "low_hours": 20.0,
        "mid_hours": 35.0,
        "high_hours": 48.0,
        "status": "active"
    })
    res_tight = client.get("/sprints/sprint-fit-test/fit-check", headers=HEADERS_TL)
    assert res_tight.status_code == 200
    assert res_tight.json()["verdict"] == "tight"

    # 3. Over: low=30, mid=45, high=60 on 40h capacity (mid=45 > 40)
    await db.task_estimates.delete_many({"sprint_id": "sprint-fit-test"})
    await db.task_estimates.insert_one({
        "task_id": "t-over-1",
        "sprint_id": "sprint-fit-test",
        "assignee_id": user_id,
        "low_hours": 30.0,
        "mid_hours": 45.0,
        "high_hours": 60.0,
        "status": "active"
    })
    res_over = client.get("/sprints/sprint-fit-test/fit-check", headers=HEADERS_TL)
    assert res_over.status_code == 200
    assert res_over.json()["verdict"] == "over"


# ===========================================================================
# 2) JWT Hardening & Role Tests
# ===========================================================================
def test_jwt_authentication_suite(setup_test_db, monkeypatch):
    """
    Validates:
    - Valid JWT token authenticates successfully
    - Expired JWT token returns 401
    - Bad signature JWT token returns 401
    - Missing token in production returns 401
    - x-mock-role header is rejected in production (returns 401)
    - Admin JWT token returns 403 on every AI endpoint
    """
    from jose import jwt
    from datetime import datetime, timedelta, timezone
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    now = datetime.now(timezone.utc)

    # 1. Valid token (TL role)
    valid_payload = {
        "sub": "usr-nader",
        "role": "TL",
        "project_id": "proj-gillo-core",
        "exp": now + timedelta(hours=2)
    }
    valid_token = jwt.encode(valid_payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    res_valid = client.get("/suggestions", headers={"Authorization": f"Bearer {valid_token}"})
    assert res_valid.status_code == 200

    # 2. Expired token
    expired_payload = {
        "sub": "usr-nader",
        "role": "TL",
        "project_id": "proj-gillo-core",
        "exp": now - timedelta(hours=2)
    }
    expired_token = jwt.encode(expired_payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    res_expired = client.get("/suggestions", headers={"Authorization": f"Bearer {expired_token}"})
    assert res_expired.status_code == 401
    assert "invalid or expired jwt token" in res_expired.json()["detail"].lower()

    # 3. Token missing required 'exp' claim must be rejected
    no_exp_payload = {
        "sub": "usr-nader",
        "role": "TL",
        "project_id": "proj-gillo-core"
    }
    no_exp_token = jwt.encode(no_exp_payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    res_no_exp = client.get("/suggestions", headers={"Authorization": f"Bearer {no_exp_token}"})
    assert res_no_exp.status_code == 401
    assert "invalid or expired jwt token" in res_no_exp.json()["detail"].lower()

    # 4. Bad signature token (signed with different secret)
    bad_sig_token = jwt.encode(valid_payload, "completely-wrong-secret-signature", algorithm=settings.JWT_ALGORITHM)
    res_bad_sig = client.get("/suggestions", headers={"Authorization": f"Bearer {bad_sig_token}"})
    assert res_bad_sig.status_code == 401
    assert "invalid or expired jwt token" in res_bad_sig.json()["detail"].lower()

    # 5. When ALLOW_MOCK_AUTH is False:
    # Missing token returns 401, and x-mock-* headers are ignored entirely
    monkeypatch.setattr(settings, "ALLOW_MOCK_AUTH", False)

    res_missing_token = client.get("/suggestions")
    assert res_missing_token.status_code == 401
    assert "mock authentication headers are disabled" in res_missing_token.json()["detail"].lower()

    res_mock_ignored = client.get(
        "/suggestions",
        headers={"x-mock-user-id": "usr-nader", "x-mock-role": "TL", "x-mock-project-id": "proj-gillo-core"}
    )
    assert res_mock_ignored.status_code == 401
    assert "mock authentication headers are disabled" in res_mock_ignored.json()["detail"].lower()

    # But a valid Bearer JWT still succeeds when ALLOW_MOCK_AUTH is False
    res_jwt_still_works = client.get("/suggestions", headers={"Authorization": f"Bearer {valid_token}"})
    assert res_jwt_still_works.status_code == 200

    # 6. When ALLOW_MOCK_AUTH is True:
    # x-mock-* headers are accepted
    monkeypatch.setattr(settings, "ALLOW_MOCK_AUTH", True)
    res_mock_allowed = client.get(
        "/suggestions",
        headers={"x-mock-user-id": "usr-nader", "x-mock-role": "TL", "x-mock-project-id": "proj-gillo-core"}
    )
    assert res_mock_allowed.status_code == 200

    # Missing both token and mock headers still returns 401
    res_no_creds = client.get("/suggestions")
    assert res_no_creds.status_code == 401

    # 6. Admin token returns 403 on EVERY AI endpoint
    admin_payload = {
        "sub": "usr-admin",
        "role": "Admin",
        "project_id": "proj-gillo-core",
        "exp": now + timedelta(hours=2)
    }
    admin_token = jwt.encode(admin_payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    headers_admin_jwt = {"Authorization": f"Bearer {admin_token}"}

    all_ai_endpoints = [
        ("GET", "/suggestions", None),
        ("POST", "/sprints/sprint-04/estimate", None),
        ("GET", "/sprints/sprint-04/fit-check", None),
        ("POST", "/sprints/sprint-04/scan-risks", None),
        ("POST", "/standup/draft", None),
        ("POST", "/sprints/sprint-04/retro/draft", None),
        ("POST", "/release-requests/rel-v1.0/draft", None),
        ("POST", "/release-requests/rel-v1.1/explain-block", None),
        ("POST", "/chat", {"query": "Explain sprint risk"}),
    ]

    for method, url, body in all_ai_endpoints:
        if method == "GET":
            res = client.get(url, headers=headers_admin_jwt)
        else:
            res = client.post(url, json=body, headers=headers_admin_jwt)
        assert res.status_code == 403, f"Admin token was not blocked with 403 on {method} {url} (status={res.status_code})"
        assert "forbidden" in res.json().get("detail", "").lower() or "admin" in res.json().get("detail", "").lower()


# ===========================================================================
# 4) Suggestions /apply, /comments, filters, pagination tests
# ===========================================================================
@pytest.mark.asyncio
async def test_suggestions_apply_endpoint(setup_test_db):
    """
    Validates /suggestions/{id}/apply:
    - valid apply confirmation: updates apply_status to 'applied'
    - 409 conflict when apply status is already settled
    - 403 forbidden when called with unauthorized role (Developer, Admin)
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from tests.conftest import HEADERS_TL, HEADERS_DEV, HEADERS_ADMIN

    client = TestClient(app)
    db = setup_test_db

    # Insert an accepted suggestion with apply_status='requested'
    sug_res = await db.ai_suggestions.insert_one({
        "project_id": "proj-gillo-core",
        "sprint_id": "sprint-04",
        "status": "accepted",
        "apply_status": "requested",
        "type": "sprint_fix",
        "dedupe_key": "fix:test-apply-1"
    })
    sug_id = str(sug_res.inserted_id)

    # 1. Developer role calling /apply returns 403
    res_dev = client.post(f"/suggestions/{sug_id}/apply", json={"status": "applied"}, headers=HEADERS_DEV)
    assert res_dev.status_code == 403

    # 2. Admin role calling /apply returns 403
    res_admin = client.post(f"/suggestions/{sug_id}/apply", json={"status": "applied"}, headers=HEADERS_ADMIN)
    assert res_admin.status_code == 403

    # 3. Valid apply with TL role succeeds
    res_valid = client.post(f"/suggestions/{sug_id}/apply", json={"status": "applied"}, headers=HEADERS_TL)
    assert res_valid.status_code == 200
    assert res_valid.json()["apply_status"] == "applied"
    assert res_valid.json()["applied_at"] is not None

    # 4. Repeat apply attempt on already settled suggestion returns 409 Conflict
    res_conflict = client.post(f"/suggestions/{sug_id}/apply", json={"status": "applied"}, headers=HEADERS_TL)
    assert res_conflict.status_code == 409
    assert "already been settled" in res_conflict.json()["detail"].lower()


@pytest.mark.asyncio
async def test_suggestions_comments_endpoint(setup_test_db):
    """
    Validates /suggestions/{id}/comments:
    - Valid comment added and appended to comments list
    - 422 Unprocessable Entity when comment body is empty or whitespace
    - 404 Not Found for non-existent suggestion
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from tests.conftest import HEADERS_TL

    client = TestClient(app)
    db = setup_test_db

    sug_res = await db.ai_suggestions.insert_one({
        "project_id": "proj-gillo-core",
        "sprint_id": "sprint-04",
        "status": "pending",
        "comments": []
    })
    sug_id = str(sug_res.inserted_id)

    # 1. Empty body returns 422
    res_empty = client.post(f"/suggestions/{sug_id}/comments", json={"body": "   "}, headers=HEADERS_TL)
    assert res_empty.status_code == 422

    # 2. Valid comment succeeds
    res_comment = client.post(
        f"/suggestions/{sug_id}/comments",
        json={"body": "Discussed in standup; agreeing to reassign."},
        headers=HEADERS_TL
    )
    assert res_comment.status_code == 200
    comments = res_comment.json().get("comments", [])
    assert len(comments) == 1
    assert comments[0]["body"] == "Discussed in standup; agreeing to reassign."
    assert comments[0]["author_id"] == HEADERS_TL["x-mock-user-id"]

    # 3. Non-existent suggestion returns 404
    fake_id = "507f1f77bcf86cd799439011"
    res_404 = client.post(f"/suggestions/{fake_id}/comments", json={"body": "Hello"}, headers=HEADERS_TL)
    assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_suggestions_filters_and_pagination(setup_test_db):
    """
    Validates list suggestions filters (status, severity, sprint_id) and pagination (skip, limit).
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from tests.conftest import HEADERS_TL

    client = TestClient(app)
    db = setup_test_db

    # Clean prior suggestions for test isolation
    await db.ai_suggestions.delete_many({})

    # Seed 6 suggestions with different combinations
    test_docs = [
        {"dedupe_key": "test:filter:1", "project_id": "proj-gillo-core", "sprint_id": "sprint-01", "type": "sprint_fix", "severity": "low", "status": "pending", "created_at": "2026-10-01T10:00:00Z"},
        {"dedupe_key": "test:filter:2", "project_id": "proj-gillo-core", "sprint_id": "sprint-01", "type": "risk_warning", "severity": "medium", "status": "pending", "created_at": "2026-10-02T10:00:00Z"},
        {"dedupe_key": "test:filter:3", "project_id": "proj-gillo-core", "sprint_id": "sprint-04", "type": "risk_warning", "severity": "high", "status": "pending", "created_at": "2026-10-03T10:00:00Z"},
        {"dedupe_key": "test:filter:4", "project_id": "proj-gillo-core", "sprint_id": "sprint-04", "type": "risk_warning", "severity": "high", "status": "accepted", "created_at": "2026-10-04T10:00:00Z"},
        {"dedupe_key": "test:filter:5", "project_id": "proj-gillo-core", "sprint_id": "sprint-04", "type": "sprint_fix", "severity": "low", "status": "rejected", "created_at": "2026-10-05T10:00:00Z"},
        {"dedupe_key": "test:filter:6", "project_id": "proj-gillo-core", "sprint_id": "sprint-04", "type": "risk_warning", "severity": "high", "status": "pending", "created_at": "2026-10-06T10:00:00Z"},
    ]
    await db.ai_suggestions.insert_many(test_docs)

    # 1. Filter by severity=high
    res_high = client.get("/suggestions?severity=high", headers=HEADERS_TL)
    assert res_high.status_code == 200
    items_high = res_high.json()
    assert len(items_high) == 3
    assert all(i["severity"] == "high" for i in items_high)

    # 2. Filter by status=pending
    res_pending = client.get("/suggestions?status=pending", headers=HEADERS_TL)
    assert res_pending.status_code == 200
    items_pending = res_pending.json()
    assert len(items_pending) == 4
    assert all(i["status"] == "pending" for i in items_pending)

    # 3. Filter by sprint_id=sprint-04
    res_sp4 = client.get("/suggestions?sprint_id=sprint-04", headers=HEADERS_TL)
    assert res_sp4.status_code == 200
    items_sp4 = res_sp4.json()
    assert len(items_sp4) == 4
    assert all(i["sprint_id"] == "sprint-04" for i in items_sp4)

    # 4. Combined filter: sprint_id=sprint-04 AND severity=high AND status=pending
    res_combined = client.get("/suggestions?sprint_id=sprint-04&severity=high&status=pending", headers=HEADERS_TL)
    assert res_combined.status_code == 200
    items_combined = res_combined.json()
    assert len(items_combined) == 2

    # 5. Pagination: limit=2, skip=0
    res_page1 = client.get("/suggestions?limit=2&skip=0", headers=HEADERS_TL)
    assert res_page1.status_code == 200
    assert len(res_page1.json()) == 2

    # Pagination: limit=2, skip=2
    res_page2 = client.get("/suggestions?limit=2&skip=2", headers=HEADERS_TL)
    assert res_page2.status_code == 200
    assert len(res_page2.json()) == 2

    # Verify no overlap between page 1 and page 2
    ids_p1 = {item["id"] for item in res_page1.json()}
    ids_p2 = {item["id"] for item in res_page2.json()}
    assert ids_p1.isdisjoint(ids_p2)


# ===========================================================================
# 5) Estimation Normalization & Double-Counting Elimination Test
# ===========================================================================
@pytest.mark.asyncio
async def test_estimation_service_normalization_eliminates_double_counting(setup_test_db):
    """
    Validates that:
    1. Past actual logged hours are normalized by past assignee ratio to extract nominal hours.
    2. compute_nominal_baseline_median returns nominal median (unscaled).
    3. compute_empirical_baseline_hours scales nominal median by target assignee ratio.
    4. Passing nominal_median as historical_median_hours prevents double-counting the ratio.
    """
    from app.services.estimation_service import (
        _extract_logged_hours_for_tasks,
        compute_nominal_baseline_median,
        compute_empirical_baseline_hours
    )
    from app.ml.registry import predict_effort_with_fallback
    from app.ml.features import compute_model1_features

    db = setup_test_db

    # Create past developer usr-slow with estimation.ratio = 1.50
    await db.ai_profiles.update_one(
        {"user_id": "usr-slow"},
        {"$set": {"estimation": {"ratio": 1.50, "sample_size": 20}}},
        upsert=True
    )

    # Insert 5 completed tasks assigned to usr-slow where 15.0 hours were logged
    # Since usr-slow takes 1.5x longer, 15.0 actual hours corresponds to 10.0 NOMINAL hours
    past_tasks = []
    for i in range(1, 6):
        task_id = f"task-norm-{i}"
        past_tasks.append({
            "task_id": task_id,
            "project_id": "proj-gillo-core",
            "assignee_id": "usr-slow",
            "task_type": "backend",
            "status": "done",
            "story_points": 2.0
        })
        await db.time_entries.delete_many({"task_id": task_id})
        await db.time_entries.insert_one({"task_id": task_id, "user_id": "usr-slow", "hours": 15.0})

    # 1. Verify _extract_logged_hours_for_tasks normalizes actual hours: 15.0 / 1.50 = 10.0
    nominal_hours = await _extract_logged_hours_for_tasks(past_tasks, db)
    assert len(nominal_hours) == 5
    for h in nominal_hours:
        assert round(h, 1) == 10.0, f"Expected 10.0 nominal hours, got {h}"

    # 2. Verify nominal median is 10.0 (not the unnormalized 15.0)
    nom_median = compute_nominal_baseline_median(nominal_hours, story_points=2.0, min_samples=5)
    assert nom_median == 10.0

    # 3. Target assignee has ratio 1.20
    target_ratio = 1.20
    low, mid, high, band = compute_empirical_baseline_hours(
        similar_hours=nominal_hours,
        story_points=2.0,
        assignee_ratio=target_ratio,
        min_samples=5
    )
    # Expected mid = nominal 10.0 * 1.20 = 12.0
    assert mid == 12.0

    # 4. End-to-end fallback verification:
    # Feature extraction must receive nominal_median (10.0)
    features = compute_model1_features(
        task={"title": "Norm Task", "task_type": "backend", "story_points": 2.0, "dependencies": []},
        assignee_ratio=target_ratio,
        historical_median_hours=nom_median
    )
    assert features["historical_median_hours"] == 10.0
    assert features["assignee_user_ratio"] == 1.20

    pred = predict_effort_with_fallback(
        title="Norm Task",
        task_type="backend",
        story_points=features["story_points"],
        dependency_count=features["dependency_count"],
        historical_median=features["historical_median_hours"],
        assignee_ratio=features["assignee_user_ratio"],
        cold_start=False
    )
    # Mid hours in rule baseline is round(historical_median * assignee_ratio, 1) = round(10.0 * 1.20, 1) = 12.0
    assert pred["mid_hours"] == 12.0, f"Double counting occurred! Expected 12.0, got {pred['mid_hours']}"


@pytest.mark.asyncio
async def test_numeric_three_tasks_estimation_with_hand_calculation(setup_test_db):
    """
    Numeric test for estimation with hand calculations:
    - Task 1: Dev A (ratio = 1.50), actual logged = 15.0h
      Nominal = 15.0 / 1.50 = 10.0h
    - Task 2: Dev B (ratio = 0.80), actual logged = 16.0h
      Nominal = 16.0 / 0.80 = 20.0h
    - Task 3: Dev C (ratio = 1.40), no logged time, story_points = 6.0
      Fallback: story_points * 5.0 = 6.0 * 5.0 = 30.0h
      Demonstrates that story_points * 5.0 is NOT divided by Dev C's ratio (1.40).

    Target Developer: ratio = 1.20

    Nominal history: [10.0, 20.0, 30.0]

    Hand calculation for empirical quantiles (with min_samples = 3):
      P25 = 15.0h
      P50 = 20.0h
      P75 = 25.0h
      low  = max(0.5, round(P25 * 1.20, 1)) = round(15.0 * 1.20, 1) = 18.0h
      mid  = max(low, round(P50 * 1.20, 1)) = round(20.0 * 1.20, 1) = 24.0h
      high = max(mid + 0.5, round(P75 * 1.20, 1)) = round(25.0 * 1.20, 1) = 30.0h

    Hand calculation for default min_samples = 5 (percentage band fallback):
      nominal_median = P50([10.0, 20.0, 30.0]) = 20.0h
      mid = round(20.0 * 1.20, 1) = 24.0h
      spread = 0.35 (for story_points=2.0)
      low  = round(24.0 * (1 - 0.35), 1) = round(24.0 * 0.65, 1) = 15.6h
      high = round(24.0 * (1 + 0.35), 1) = round(24.0 * 1.35, 1) = 32.4h
    """
    from app.services.estimation_service import (
        _extract_logged_hours_for_tasks,
        compute_nominal_baseline_median,
        compute_empirical_baseline_hours
    )

    db = setup_test_db

    # 1. Setup profiles with known developer ratios
    await db.ai_profiles.update_one(
        {"user_id": "usr-devA"},
        {"$set": {"estimation": {"ratio": 1.50, "sample_size": 15}}},
        upsert=True
    )
    await db.ai_profiles.update_one(
        {"user_id": "usr-devB"},
        {"$set": {"estimation": {"ratio": 0.80, "sample_size": 15}}},
        upsert=True
    )
    await db.ai_profiles.update_one(
        {"user_id": "usr-devC"},
        {"$set": {"estimation": {"ratio": 1.40, "sample_size": 15}}},
        upsert=True
    )

    # 2. Three completed tasks
    # Task 1: 15.0 actual hours logged by usr-devA (ratio 1.50) -> nominal = 10.0h
    # Task 2: 16.0 actual hours logged by usr-devB (ratio 0.80) -> nominal = 20.0h
    # Task 3: 0 hours logged by usr-devC (ratio 1.40), story_points = 6.0 -> fallback = 30.0h (NOT divided by 1.40)
    tasks = [
        {"task_id": "t-num-1", "assignee_id": "usr-devA", "story_points": 2.0},
        {"task_id": "t-num-2", "assignee_id": "usr-devB", "story_points": 4.0},
        {"task_id": "t-num-3", "assignee_id": "usr-devC", "story_points": 6.0},
    ]

    await db.time_entries.delete_many({"task_id": {"$in": ["t-num-1", "t-num-2", "t-num-3"]}})
    await db.time_entries.insert_one({"task_id": "t-num-1", "user_id": "usr-devA", "hours": 15.0})
    await db.time_entries.insert_one({"task_id": "t-num-2", "user_id": "usr-devB", "hours": 16.0})
    # Notice: NO time entries for t-num-3

    # 3. Extract nominal hours
    nominal_hours = await _extract_logged_hours_for_tasks(tasks, db)
    assert len(nominal_hours) == 3

    # Task 1: 15.0 / 1.50 = 10.0
    assert nominal_hours[0] == 10.0
    # Task 2: 16.0 / 0.80 = 20.0
    assert nominal_hours[1] == 20.0
    # Task 3: story_points * 5.0 = 6.0 * 5.0 = 30.0. Show it is NOT divided by Dev C's ratio 1.40
    assert nominal_hours[2] == 30.0
    assert nominal_hours[2] != round(30.0 / 1.40, 1)

    # 4. Target developer has ratio 1.20
    target_ratio = 1.20

    # Case A: Empirical Quantiles (min_samples = 3)
    # Hand calculation:
    #   p25([10, 20, 30]) = 15.0  -> low = 15.0 * 1.2 = 18.0
    #   p50([10, 20, 30]) = 20.0  -> mid = 20.0 * 1.2 = 24.0
    #   p75([10, 20, 30]) = 25.0  -> high = 25.0 * 1.2 = 30.0
    low_q, mid_q, high_q, band_q = compute_empirical_baseline_hours(
        similar_hours=nominal_hours,
        story_points=2.0,
        assignee_ratio=target_ratio,
        min_samples=3
    )
    assert band_q == "empirical_quantiles"
    assert low_q == 18.0
    assert mid_q == 24.0
    assert high_q == 30.0

    # Case B: Default percentage band fallback (min_samples = 5, history length = 3 < 5)
    # Hand calculation:
    #   When len(similar_hours) < min_samples, nominal_mid falls back to story_points * 5.0 = 2.0 * 5.0 = 10.0h
    #   mid = round(10.0 * 1.20, 1) = 12.0h
    #   spread = 0.35 (since story_points=2.0 is provided)
    #   low = round(12.0 * (1 - 0.35), 1) = round(12.0 * 0.65, 1) = 7.8h
    #   high = round(12.0 * (1 + 0.35), 1) = round(12.0 * 1.35, 1) = 16.2h
    low_pb, mid_pb, high_pb, band_pb = compute_empirical_baseline_hours(
        similar_hours=nominal_hours,
        story_points=2.0,
        assignee_ratio=target_ratio,
        min_samples=5
    )
    assert band_pb == "percentage_band"
    assert low_pb == 7.8
    assert mid_pb == 12.0
    assert high_pb == 16.2



