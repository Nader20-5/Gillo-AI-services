"""
End-to-End Test Suite for Demo Seed Dataset & Validated Model Operations.

Validates that scripts/seed_data.py generates a realistic demo dataset that exercises
Model 1 (Story Points) and Model 2 (Sprint Carry-Over Risk) inside their validated operating ranges:
1. Seed idempotency and event history mirror consistency.
2. Active sprint in range (f in [0.30, 0.70], open tasks >= 10):
   method == "lightgbm_taskove_v2", k == min(5, ceil(0.2*n_open)) suggestions,
   no raw probability percentage in the user-facing response.
3. Missing sprint_assigned_at timestamp triggers rule_baseline with an explanatory reason (no HTTP 500).
4. Sprint with < 10 open tasks or f outside [0.2, 0.8] falls back to rule_baseline with reason.
5. Idempotent re-run: deduplication prevents duplicate pending suggestions.
6. Estimation endpoint (POST /sprints/{id}/estimate):
   - Unestimated tasks receive continuous + Fibonacci-snapped SP, [low, high] interval, confidence note.
   - No hours conversion in story point proposals (hours_conversion_note="unvalidated heuristic").
   - Empty description evaluates safely via LightGBM.
   - Non-English task falls back to type_median_fallback with reason="non_english_text".
"""

import math
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import app
from scripts.seed_data import seed_data, ensure_utc

HEADERS_PM = {
    "x-mock-user-id": "usr-alex",
    "x-mock-role": "PM",
    "x-mock-project-id": "proj-gillo-core"
}

HEADERS_TL = {
    "x-mock-user-id": "usr-nader",
    "x-mock-role": "TL",
    "x-mock-project-id": "proj-gillo-core"
}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
async def seed_db(setup_test_db):
    """Provides a fresh isolated database for demo seed execution tests."""
    collections = [
        "ai_profiles", "sprint_snapshots", "capacity", "task_snapshots",
        "task_events", "time_entries", "standup_blockers", "pull_requests",
        "release_requests", "ai_suggestions", "task_estimates", "sprint_fit_checks"
    ]
    for col in collections:
        await setup_test_db[col].delete_many({})
    return setup_test_db


@pytest.mark.asyncio
async def test_seed_idempotency_and_event_mirror(seed_db):
    """
    Verifies seed idempotency and mathematical consistency between task snapshots
    and the task_events collection (which mirrors Kafka event streams).
    """
    now = datetime.now(timezone.utc)
    
    # 1. First seed execution
    summary1 = await seed_data(seed_db, now=now, reset=True)
    assert summary1["tasks"] > 0
    assert summary1["sprints"] >= 5
    assert summary1["task_events"] > 0

    task_count_1 = await seed_db.task_snapshots.count_documents({})
    event_count_1 = await seed_db.task_events.count_documents({})
    sprint_count_1 = await seed_db.sprint_snapshots.count_documents({})

    # 2. Re-run without reset (must be strictly idempotent)
    summary2 = await seed_data(seed_db, now=now, reset=False)
    task_count_2 = await seed_db.task_snapshots.count_documents({})
    event_count_2 = await seed_db.task_events.count_documents({})
    sprint_count_2 = await seed_db.sprint_snapshots.count_documents({})

    assert task_count_1 == task_count_2, "Idempotent re-run duplicated task snapshots!"
    assert event_count_1 == event_count_2, "Idempotent re-run duplicated task events!"
    assert sprint_count_1 == sprint_count_2, "Idempotent re-run duplicated sprint snapshots!"

    # 3. Verify event mirror consistency for every seeded task
    tasks = await seed_db.task_snapshots.find({}).to_list(length=200)
    for t in tasks:
        tid = t["task_id"]
        # Must have a sprint_assigned event
        assign_evt = await seed_db.task_events.find_one({
            "task_id": tid,
            "type": "sprint_assigned"
        })
        assert assign_evt is not None, f"Task {tid} missing sprint_assigned event"
        assert ensure_utc(t["sprint_assigned_at"]) == ensure_utc(assign_evt["at"]), f"Timestamp mismatch for {tid} sprint_assigned_at"

        # Must have a status_change event
        status_evts = await seed_db.task_events.find({
            "task_id": tid,
            "type": "status_change"
        }).sort("at", 1).to_list(length=10)
        assert len(status_evts) >= 1, f"Task {tid} missing status_change events"
        latest_status_evt = status_evts[-1]
        assert ensure_utc(t["status_changed_at"]) == ensure_utc(latest_status_evt["at"]), f"Timestamp mismatch for {tid} status_changed_at"

        # stalled_days consistency check
        if t["status"] == "done":
            assert t["stalled_days"] == 0, f"Done task {tid} should have 0 stalled days"
        else:
            status_changed_at = ensure_utc(t["status_changed_at"])
            expected_stalled = max(0, int((now - status_changed_at).total_seconds() / 86400))
            assert abs(t["stalled_days"] - expected_stalled) <= 1, f"Stalled days drift on task {tid}"


@pytest.mark.asyncio
async def test_e2e_scan_risks_active_in_range(client, seed_db):
    """
    Scenario 1: Active sprint in model's validated range (f in [0.30, 0.70], open tasks >= 10).
    Asserts:
    - method == 'lightgbm_taskove_v2'
    - model_version == 'v2_taskove_features'
    - k == min(5, ceil(0.20 * n_open))
    - new_risk_warnings_generated == k
    - NO raw probabilities leaked in user-facing client response.
    """
    now = datetime.now(timezone.utc)
    await seed_data(seed_db, now=now, reset=True)

    res = client.post("/sprints/sprint-04/scan-risks", headers=HEADERS_PM)
    assert res.status_code == 200, f"Scan risks failed: {res.text}"
    data = res.json()

    assert data["sprint_id"] == "sprint-04"
    assert data["method"] == "lightgbm_taskove_v2"
    assert data["model_version"] == "v2_taskove_features"
    assert data["fallback_reason"] is None

    n_open = data["active_tasks_scanned"]
    assert n_open >= 10, f"Expected >= 10 open tasks, found {n_open}"
    expected_k = min(5, math.ceil(0.20 * n_open))
    assert data["k_threshold"] == expected_k
    assert data["new_risk_warnings_generated"] == expected_k
    assert len(data["warnings"]) == expected_k

    # Strict Privacy & Guardrail: NO raw probability percentage in user-facing client response
    for st in data["scored_tasks"]:
        assert "probability" not in st
        assert "raw_probability" not in st
        assert "calibrated_probability" not in st
        assert "_telemetry_calibrated_prob" not in st
        assert "risk_rank" in st
        assert "risk_tier" in st
        assert st["risk_tier"] in ["high", "medium", "low"]

    for w in data["warnings"]:
        assert "probability" not in w["proposal"]
        assert "message" in w["proposal"]
        assert f"top-{expected_k} focus" in w["proposal"]["message"].lower()


@pytest.mark.asyncio
async def test_e2e_scan_risks_missing_sprint_assigned_at_fallback(client, seed_db):
    """
    Scenario 2: Removing sprint_assigned_at from a single task must gracefully trigger
    rule_baseline fallback with an explicit reason and NEVER return HTTP 500.
    """
    now = datetime.now(timezone.utc)
    await seed_data(seed_db, now=now, reset=True)

    # Remove sprint_assigned_at from task-s5-01
    upd = await seed_db.task_snapshots.update_one(
        {"task_id": "task-s5-01"},
        {"$unset": {"sprint_assigned_at": ""}}
    )
    assert upd.modified_count == 1

    res = client.post("/sprints/sprint-05/scan-risks", headers=HEADERS_PM)
    assert res.status_code == 200, f"Expected 200 fallback, got: {res.text}"
    data = res.json()

    assert data["sprint_id"] == "sprint-05"
    assert data["method"] == "rule_baseline"
    assert data["fallback_reason"] is not None
    assert "missing exact-flag feature" in data["fallback_reason"].lower()
    assert "sprint_assigned_at" in data["fallback_reason"].lower()
    assert len(data["scored_tasks"]) >= 10


@pytest.mark.asyncio
async def test_e2e_scan_risks_ineligible_sprints(client, seed_db):
    """
    Scenario 3: Sprints with < 10 open tasks or f outside [0.2, 0.8] fall back
    cleanly to rule_baseline with an explanatory reason.
    """
    now = datetime.now(timezone.utc)
    await seed_data(seed_db, now=now, reset=True)

    # 1. Sprint with < 10 open tasks (sprint-small-active)
    res_small = client.post("/sprints/sprint-small-active/scan-risks", headers=HEADERS_PM)
    assert res_small.status_code == 200
    data_small = res_small.json()
    assert data_small["method"] == "rule_baseline"
    assert data_small["fallback_reason"] is not None
    assert "< 10 tasks required" in data_small["fallback_reason"].lower()

    # 2. Sprint with f < 0.20 (sprint-early-active, f ~ 0.0714)
    res_early = client.post("/sprints/sprint-early-active/scan-risks", headers=HEADERS_PM)
    assert res_early.status_code == 200
    data_early = res_early.json()
    assert data_early["method"] == "rule_baseline"
    assert data_early["fallback_reason"] is not None
    assert "outside model domain" in data_early["fallback_reason"].lower()


@pytest.mark.asyncio
async def test_e2e_scan_risks_idempotency_no_duplicates(client, seed_db):
    """
    Scenario 4: Re-running risk scan on an active sprint must not generate duplicate
    suggestions in ai_suggestions (deduplication against pending suggestions).
    """
    now = datetime.now(timezone.utc)
    await seed_data(seed_db, now=now, reset=True)

    # Run 1: generates k suggestions
    res1 = client.post("/sprints/sprint-04/scan-risks", headers=HEADERS_PM)
    assert res1.status_code == 200
    data1 = res1.json()
    k = data1["k_threshold"]
    assert data1["new_risk_warnings_generated"] == k

    # Run 2: identical scan immediately afterwards
    res2 = client.post("/sprints/sprint-04/scan-risks", headers=HEADERS_PM)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["new_risk_warnings_generated"] == 0, "Duplicate suggestions were created!"

    # Database count check
    pending_count = await seed_db.ai_suggestions.count_documents({
        "sprint_id": "sprint-04",
        "type": "risk_warning",
        "status": "pending"
    })
    assert pending_count == k, f"Expected {k} pending suggestions, found {pending_count}"


@pytest.mark.asyncio
async def test_e2e_estimate_mix_of_tasks(client, seed_db):
    """
    Scenario 5: Calling POST /sprints/{id}/estimate exercises Model 1 across a realistic task mix:
    - Unestimated tasks (story_points=None) are estimated with method, interval, and confidence note.
    - No story-point-to-hours conversion (hours_conversion_note="unvalidated heuristic").
    - Empty-description tasks evaluate safely via LightGBM.
    - Non-English task (Arabic) falls back to type_median_fallback with fallback_reason="non_english_text".
    """
    now = datetime.now(timezone.utc)
    await seed_data(seed_db, now=now, reset=True)

    res = client.post("/sprints/sprint-04/estimate", headers=HEADERS_PM)
    assert res.status_code == 200, f"Estimation failed: {res.text}"
    data = res.json()

    assert data["sprint_id"] == "sprint-04"
    assert data["tasks_estimated"] >= 15
    estimates = {e["task_id"]: e for e in data["estimates"]}

    # 1. Unestimated task check (task-s4-05 had story_points=None)
    t_unest = estimates.get("task-s4-05")
    assert t_unest is not None
    assert t_unest["story_points"] is not None
    assert t_unest["story_points"] in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
    assert t_unest["sp_method"] == "lightgbm_taskove_v2"
    assert t_unest["sp_model_version"] == "v2_taskove"
    assert len(t_unest["story_points_interval"]) == 2
    assert t_unest["story_points_interval"][0] <= t_unest["story_points_interval"][1]
    assert "approximate range" in t_unest["story_points_confidence_note"].lower() or "rough guide" in t_unest["story_points_confidence_note"].lower()
    assert "unvalidated heuristic" in t_unest["hours_conversion_note"].lower()

    # 2. Empty description task check (task-s4-04 had description="")
    t_empty = estimates.get("task-s4-04")
    assert t_empty is not None
    assert t_empty["sp_method"] == "lightgbm_taskove_v2"
    assert t_empty["story_points"] is not None

    # 3. Non-English task check (task-s4-13 had Arabic text)
    t_ar = estimates.get("task-s4-13")
    assert t_ar is not None
    assert t_ar["sp_method"] == "type_median_fallback"
    assert t_ar["sp_fallback_reason"] == "non_english_text"
    assert t_ar["story_points"] is not None
    assert "unvalidated heuristic" in t_ar["hours_conversion_note"].lower()
