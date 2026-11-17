from pathlib import Path
import pytest
from app.utils.privacy import FORBIDDEN_RANKING_TERMS, strip_mood_fields, contains_mood
from app.events.consumer import EventConsumer, AgileEvent
from app.models.suggestions import AISuggestionBase
from app.models.estimates import SprintFitCheckModel, TaskEstimateModel, BaselineInfo
from fastapi.testclient import TestClient
from app.main import app

PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


def test_no_ranking_in_prompt_templates():
    """
    Ethical Guardrail:
    Prompt templates must never instruct the model to rank, score, or compare developers.
    """
    assert PROMPTS_ROOT.exists(), f"Prompts directory {PROMPTS_ROOT} must exist"
    for md_file in PROMPTS_ROOT.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_RANKING_TERMS:
            assert term not in content, (
                f"Ethical violation: prohibited term '{term}' found in prompt template {md_file}!"
            )


def test_no_ranking_fields_in_response_schemas():
    """
    Ethical Guardrail:
    Response models must never include developer score or ranking fields.
    """
    schemas_to_check = [AISuggestionBase, SprintFitCheckModel]
    for schema in schemas_to_check:
        fields = schema.model_fields.keys()
        for term in FORBIDDEN_RANKING_TERMS:
            assert term not in fields, f"Ethical violation: prohibited field '{term}' in {schema.__name__}"


@pytest.mark.asyncio
async def test_mood_fields_stripped_by_consumer(setup_test_db):
    """
    Privacy Guardrail:
    Event consumer must strip all mood, morale, and sentiment fields before writing to DB.
    """
    consumer = EventConsumer(setup_test_db)

    raw_event = {
        "event_id": "evt-privacy-01",
        "event_type": "StandupSubmitted",
        "payload": {
            "developer": "usr-kareem",
            "mood": "frustrated",
            "happiness_score": 2,
            "blockers": [
                {
                    "blocker_id": "blk-priv-01",
                    "description": "Waiting on credentials",
                    "user_mood": "stressed",
                    "sentiment": "negative"
                }
            ]
        }
    }

    res = await consumer.process_raw_event(raw_event)
    assert res["status"] == "processed"

    # Verify standup_blockers collection
    saved_blocker = await setup_test_db.standup_blockers.find_one({"blocker_id": "blk-priv-01"})
    assert saved_blocker is not None
    assert "user_mood" not in saved_blocker
    assert "sentiment" not in saved_blocker
    assert not contains_mood(saved_blocker), "Database document contains forbidden mood keys!"


@pytest.mark.asyncio
async def test_no_response_contains_per_developer_ratios(setup_test_db):
    """
    Ethical & Privacy Guardrail:
    Developer estimation ratios (refresh_developer_ratios / compute_dynamic_assignee_ratio)
    are strictly internal operational values.
    No client-facing API response must ever expose per-developer ratios, velocity multipliers,
    or developer performance comparisons.
    """
    # 1. Schema Contract: TaskEstimateModel & BaselineInfo must not expose user_ratio
    assert "user_ratio" not in TaskEstimateModel.model_fields
    assert "user_ratio" not in BaselineInfo.model_fields
    assert "developer_ratio" not in TaskEstimateModel.model_fields

    # Seed test sprint and task snapshots into in-memory mock db
    db = setup_test_db
    sprint_id = "sprint-ratio-test"
    project_id = "proj-gillo-core"

    from datetime import datetime, timezone
    t_start = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 11, 15, 0, 0, tzinfo=timezone.utc)

    await db.sprint_snapshots.insert_one({
        "sprint_id": sprint_id,
        "project_id": project_id,
        "name": "Sprint Ratio Test",
        "status": "active",
        "start_date": t_start,
        "end_date": t_end,
        "total_tasks": 2,
        "completed_tasks": 0,
        "blocked_tasks": 0
    })

    await db.task_snapshots.insert_many([
        {
            "task_id": "task-rt-01",
            "sprint_id": sprint_id,
            "project_id": project_id,
            "title": "Build caching layer",
            "task_type": "backend",
            "status": "in_progress",
            "assignee_id": "usr-alice",
            "story_points": 3.0,
            "stalled_days": 1,
            "sprint_assigned_at": t_start
        },
        {
            "task_id": "task-rt-02",
            "sprint_id": sprint_id,
            "project_id": project_id,
            "title": "Fix database connection leak",
            "task_type": "bug",
            "status": "todo",
            "assignee_id": "usr-bob",
            "story_points": 2.0,
            "stalled_days": 0,
            "sprint_assigned_at": t_start
        }
    ])

    # Also seed developer profile with an internal ratio
    await db.ai_profiles.insert_one({
        "user_id": "usr-alice",
        "name": "Alice Developer",
        "role": "Developer",
        "estimation": {
            "ratio": 1.45,
            "sample_size": 8
        }
    })

    headers = {
        "x-mock-user-id": "usr-lead",
        "x-mock-role": "TL",
        "x-mock-project-id": project_id
    }

    client = TestClient(app)

    # 2. Call POST /sprints/{id}/estimate
    resp_est = client.post(f"/sprints/{sprint_id}/estimate", headers=headers)
    assert resp_est.status_code == 200, f"Estimate failed: {resp_est.text}"
    est_data = resp_est.json()

    # 3. Call GET /sprints/{id}/fit-check
    resp_fit = client.get(f"/sprints/{sprint_id}/fit-check", headers=headers)
    assert resp_fit.status_code == 200, f"Fit check failed: {resp_fit.text}"
    fit_data = resp_fit.json()

    # 4. Call POST /sprints/{id}/scan-risks
    resp_risk = client.post(f"/sprints/{sprint_id}/scan-risks", headers=headers)
    assert resp_risk.status_code == 200, f"Scan risks failed: {resp_risk.text}"
    risk_data = resp_risk.json()

    # Recursive check helper
    def assert_no_developer_ratios(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                curr_path = f"{path}.{k}" if path else k
                # Forbidden ratio keys in any client-facing response
                assert k not in ["user_ratio", "developer_ratio", "assignee_ratio", "estimation_ratio", "velocity_ratio"], (
                    f"Privacy violation: prohibited developer ratio key '{k}' found at '{curr_path}'"
                )
                assert_no_developer_ratios(v, curr_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                assert_no_developer_ratios(item, f"{path}[{i}]")

    assert_no_developer_ratios(est_data, path="estimate_response")
    assert_no_developer_ratios(fit_data, path="fit_check_response")
    assert_no_developer_ratios(risk_data, path="scan_risks_response")
