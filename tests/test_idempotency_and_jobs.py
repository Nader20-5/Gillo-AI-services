from datetime import datetime, timedelta, timezone
import pytest
from app.jobs.sprint_jobs import (
    run_daily_risk_scan,
    expire_and_supersede_suggestions,
    refresh_developer_ratios,
)
from app.events.consumer import EventConsumer


@pytest.mark.asyncio
async def test_daily_risk_scan_idempotency(setup_test_db):
    """
    Validates that running the daily risk scan twice in succession is idempotent
    and does not duplicate pending risk suggestions.
    """
    # First execution
    res1 = await run_daily_risk_scan(setup_test_db)
    assert res1["tasks_scanned"] >= 1
    warnings_first_run = res1["warnings_generated"]

    # Second execution immediately after
    res2 = await run_daily_risk_scan(setup_test_db)
    assert res2["tasks_scanned"] >= 1
    # Since suggestions are already pending with matching dedupe_key, warnings_generated should be 0
    assert res2["warnings_generated"] == 0, "Duplicate risk warnings were created on re-scan!"

    # Verify unique pending suggestions in DB
    pending_count = await setup_test_db.ai_suggestions.count_documents({"status": "pending"})
    assert pending_count == warnings_first_run


@pytest.mark.asyncio
async def test_expire_and_supersede_job(setup_test_db):
    """
    Validates the suggestion housekeeping job:
    1. Expiring pending suggestions whose expires_at is in the past.
    2. Superseding pending suggestions whose referenced task has changed to 'done'.
    """
    now = datetime.now(timezone.utc)

    # 1. Past-expiration suggestion
    await setup_test_db.ai_suggestions.insert_one({
        "project_id": "proj-gillo-core",
        "sprint_id": "sprint-04",
        "task_id": "task-act-01",
        "type": "risk_warning",
        "status": "pending",
        "expires_at": now - timedelta(hours=2),
        "dedupe_key": "exp:test:1",
    })

    # 2. Suggestion for a task that is now done
    await setup_test_db.ai_suggestions.insert_one({
        "project_id": "proj-gillo-core",
        "sprint_id": "sprint-01",
        "task_id": "task-hist-01",  # status is 'done' in seed
        "type": "risk_warning",
        "status": "pending",
        "dedupe_key": "sup:test:2",
    })

    expired_count, superseded_count = await expire_and_supersede_suggestions(setup_test_db)
    assert expired_count >= 1
    assert superseded_count >= 1

    # Verify status in DB
    s1 = await setup_test_db.ai_suggestions.find_one({"dedupe_key": "exp:test:1"})
    assert s1["status"] == "expired"

    s2 = await setup_test_db.ai_suggestions.find_one({"dedupe_key": "sup:test:2"})
    assert s2["status"] == "superseded"


@pytest.mark.asyncio
async def test_kafka_duplicate_and_dead_letter_delivery(setup_test_db):
    """
    Validates that:
    1. Duplicate event delivery is skipped gracefully (idempotency via processed_events).
    2. Malformed event is trapped into dead_letter_events without crashing.
    """
    consumer = EventConsumer(setup_test_db)

    # 1. Valid event
    evt = {
        "event_id": "evt-idempotent-01",
        "event_type": "StandupSubmitted",
        "payload": {"developer": "usr-sama", "blockers": []}
    }
    res1 = await consumer.process_raw_event(evt)
    assert res1["status"] == "processed"

    # 2. Duplicate delivery of same event_id
    res_dup = await consumer.process_raw_event(evt)
    assert res_dup["status"] == "duplicate_skipped"

    # 3. Malformed event (missing required event_id or event_type)
    bad_evt = {"aggregate_id": "agg-1"}  # Missing event_id and event_type
    res_bad = await consumer.process_raw_event(bad_evt)
    assert res_bad["status"] == "dead_letter"

    # Verify dead_letter_events collection
    dl = await setup_test_db.dead_letter_events.find_one({"raw_event.aggregate_id": "agg-1"})
    assert dl is not None
    assert "error" in dl
