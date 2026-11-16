import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from app.main import app
from tests.conftest import HEADERS_TL, HEADERS_ADMIN


@pytest.mark.asyncio
async def test_inbox_transitions_and_atomic_conflict(setup_test_db):
    """
    State machine test:
    - Valid transitions: pending -> accepted | rejected | dismissed
    - Atomic conditional updates: 409 Conflict when attempting to decide an already-decided suggestion
    - Apply confirmation transitions
    """
    client = TestClient(app)

    # 1. Insert a pending suggestion into the test DB
    sug_doc = {
        "project_id": "proj-gillo-core",
        "sprint_id": "sprint-04",
        "task_id": "task-act-03",
        "type": "risk_warning",
        "severity": "high",
        "confidence": "high",
        "status": "pending",
        "method": "rule_baseline",
        "dedupe_key": "risk:blocker:sprint-04:task-act-03",
        "apply_status": "none",
    }
    insert_res = await setup_test_db.ai_suggestions.insert_one(sug_doc)
    sug_id = str(insert_res.inserted_id)

    # 2. Accept the suggestion (pending -> accepted)
    accept_res = client.post(f"/suggestions/{sug_id}/accept", headers=HEADERS_TL)
    assert accept_res.status_code == 200, accept_res.text
    accepted_data = accept_res.json()
    assert accepted_data["status"] == "accepted"
    assert accepted_data["apply_status"] == "requested"
    assert accepted_data["decided_by"] == HEADERS_TL["x-mock-user-id"]

    # 3. Concurrent / Duplicate decision attempt must return 409 Conflict
    conflict_res = client.post(f"/suggestions/{sug_id}/accept", headers=HEADERS_TL)
    assert conflict_res.status_code == 409, "Second accept must return 409 Conflict"
    assert "already decided" in conflict_res.json()["detail"].lower()

    reject_conflict = client.post(
        f"/suggestions/{sug_id}/reject",
        json={"decision_reason": "Attempting to reject an already-accepted item"},
        headers=HEADERS_TL
    )
    assert reject_conflict.status_code == 409, "Reject on accepted item must return 409 Conflict"

    # 4. Apply confirmation on accepted suggestion
    apply_res = client.post(
        f"/suggestions/{sug_id}/apply",
        json={"status": "applied"},
        headers=HEADERS_TL
    )
    assert apply_res.status_code == 200, apply_res.text
    assert apply_res.json()["apply_status"] == "applied"

    # 5. Subsequent apply confirmation must return 409 Conflict
    apply_again = client.post(
        f"/suggestions/{sug_id}/apply",
        json={"status": "applied"},
        headers=HEADERS_TL
    )
    assert apply_again.status_code == 409, "Second apply confirmation must return 409 Conflict"


@pytest.mark.asyncio
async def test_decision_reason_length_validation(setup_test_db):
    """Rejecting and dismissing strictly require at least 5 characters for decision_reason."""
    client = TestClient(app)

    sug = await setup_test_db.ai_suggestions.insert_one({
        "project_id": "proj-gillo-core",
        "sprint_id": "sprint-04",
        "type": "risk_warning",
        "status": "pending",
        "dedupe_key": "test:reason:sprint-04",
    })
    sug_id = str(sug.inserted_id)

    # Empty reason
    res_empty = client.post(f"/suggestions/{sug_id}/reject", json={"decision_reason": ""}, headers=HEADERS_TL)
    assert res_empty.status_code == 422

    # Short reason (< 5 characters)
    res_short = client.post(f"/suggestions/{sug_id}/reject", json={"decision_reason": "no"}, headers=HEADERS_TL)
    assert res_short.status_code == 422
    assert "at least 5 characters" in res_short.json()["detail"].lower()

    # Valid reason
    res_valid = client.post(f"/suggestions/{sug_id}/reject", json={"decision_reason": "False positive, blocker already cleared"}, headers=HEADERS_TL)
    assert res_valid.status_code == 200
    assert res_valid.json()["status"] == "rejected"
    assert res_valid.json()["decision_reason"] == "False positive, blocker already cleared"
