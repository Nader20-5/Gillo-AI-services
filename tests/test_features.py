from datetime import datetime, timedelta, timezone
import pytest
from app.ml.features import (
    compute_model1_features,
    compute_model2_features,
    build_model2_task_features,
    MODEL1_FEATURE_LIST,
    MODEL2_FEATURE_LIST,
    MODEL2_SYNTHETIC_FEATURE_LIST,
)
from app.models.read_models import TaskSnapshot, SprintSnapshot


def test_model1_feature_contract():
    """Validates Model 1 feature ordering and values."""
    task = {
        "task_id": "t-1",
        "story_points": 5.0,
        "dependencies": ["t-0", "t-prev"]
    }
    feats = compute_model1_features(
        task=task,
        historical_median_hours=20.0,
        assignee_ratio=1.1,
    )
    assert list(feats.keys()) == MODEL1_FEATURE_LIST
    assert feats["story_points"] == 5.0
    assert feats["dependency_count"] == 2
    assert feats["historical_median_hours"] == 20.0
    assert feats["assignee_user_ratio"] == 1.1


def test_model1_missing_story_points():
    """Ensures missing story points is not silently defaulted to 1.0."""
    task = {"task_id": "t-2", "dependencies": []}
    feats = compute_model1_features(task=task, historical_median_hours=10.0, assignee_ratio=1.0)
    assert feats["story_points"] is None, "Missing story points must be represented as None, not defaulted to 1.0"


def test_model2_legacy_synthetic_feature_contract():
    """Validates legacy Model 2 synthetic prototype feature ordering and values."""
    task = {
        "task_id": "t-risk-1",
        "story_points": 8.0,
        "stalled_days": 5,
        "dependencies": ["t-dep-1", "t-dep-2"]
    }
    feats = compute_model2_features(
        task=task,
        active_blockers_count=2,
        logged_hours=16.0,
        estimated_hours=20.0,
        assignee_capacity_hours=40.0,
        assignee_total_assigned_hours=50.0,
    )
    assert list(feats.keys()) == MODEL2_SYNTHETIC_FEATURE_LIST
    assert feats["stalled_days"] == 5
    assert feats["dependency_depth"] == 2
    assert feats["active_blockers"] == 2
    assert feats["assignee_load_ratio"] == 1.25  # 50.0 / 40.0
    assert feats["logged_vs_estimated_ratio"] == 0.8  # 16.0 / 20.0
    assert feats["story_points"] == 8.0


def test_model2_canonical_23_feature_contract():
    """Validates the canonical 23-feature contract for Model 2 Taskove from TaskSnapshot."""
    now = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
    sprint_start = now - timedelta(days=5)
    sprint_end = now + timedelta(days=5)  # f = 5/10 = 0.50

    sprint = SprintSnapshot(
        sprint_id="sprint-test-1",
        project_id="proj-1",
        name="Sprint Test 1",
        start_date=sprint_start,
        end_date=sprint_end,
        status="active"
    )

    # Create 10 tasks to meet eligibility (>= 10 open tasks)
    tasks = []
    for i in range(12):
        tasks.append(TaskSnapshot(
            task_id=f"t-{i}",
            project_id="proj-1",
            sprint_id="sprint-test-1",
            title=f"Task {i}",
            task_type="bug" if i == 0 else ("feature" if i == 1 else "task"),
            story_points=3.0 if i < 10 else None,
            status="in_progress" if i < 8 else ("todo" if i < 11 else "done"),
            assignee_id="user-1" if i < 5 else "user-2",
            stalled_days=2,
            dependencies=["t-dep"] if i == 0 else [],
            created_at=sprint_start - timedelta(days=2),
            updated_at=now - timedelta(days=1),
            sprint_assigned_at=sprint_start + timedelta(hours=1) if i == 0 else sprint_start - timedelta(hours=1)
        ))

    target_task = tasks[0]
    feats = build_model2_task_features(
        task=target_task,
        sprint=sprint,
        sprint_tasks=tasks,
        now=now
    )

    assert len(feats) == 23
    assert list(feats.keys()) == MODEL2_FEATURE_LIST
    assert feats["f"] == 0.5
    assert feats["sprint_open_tasks"] == 11.0  # 12 total, 1 done
    assert feats["added_after_sprint_start"] == 1.0
    assert feats["type_Bug"] == 1.0
    assert feats["type_Story"] == 0.0
    assert feats["status_category_is_todo"] == 0.0  # target_task is 'in_progress'
    assert feats["assignee_present"] == 1.0
    assert feats["links_depends_on"] == 1.0


def test_temporal_isolation_features_at_time_t():
    """
    Critical temporal test:
    Features computed at time t must ignore events logged after time t.
    """
    now = datetime.now(timezone.utc)
    t_cutoff = now - timedelta(days=5)

    time_entries = [
        {"hours": 10.0, "logged_at": now - timedelta(days=10)},  # Before cutoff
        {"hours": 5.0, "logged_at": now - timedelta(days=7)},    # Before cutoff
        {"hours": 20.0, "logged_at": now - timedelta(days=2)},   # After cutoff (should be ignored!)
    ]

    # Filter entries strictly strictly <= t_cutoff
    valid_entries_at_t = [e for e in time_entries if e["logged_at"] <= t_cutoff]
    logged_h_at_t = sum(e["hours"] for e in valid_entries_at_t)

    assert logged_h_at_t == 15.0

    task = {"task_id": "t-temporal", "story_points": 3.0, "stalled_days": 1, "dependencies": []}
    feats = compute_model2_features(
        task=task,
        active_blockers_count=0,
        logged_hours=logged_h_at_t,
        estimated_hours=15.0,
        assignee_capacity_hours=40.0,
        assignee_total_assigned_hours=30.0,
    )
    # 15.0 / 15.0 == 1.0 (if post-cutoff entry of 20h had leaked, ratio would be 35/15 = 2.33)
    assert feats["logged_vs_estimated_ratio"] == 1.0
