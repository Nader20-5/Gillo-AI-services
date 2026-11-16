"""
Canonical Feature Engineering Contract for Gillo AI Service.
This module is the SINGLE source of truth for computing Model 1 and Model 2 features.
Both online inference (serving) and offline training scripts must import these definitions.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np

from app.models.read_models import TaskSnapshot, SprintSnapshot


class MissingFeatureError(ValueError):
    """Raised when an exact-flag feature (e.g. sprint_assigned_at) is missing."""
    pass


class IneligibleSprintError(ValueError):
    """Raised when sprint is outside the valid model operational domain."""
    pass


# Canonical feature lists required by the models
MODEL1_FEATURE_LIST: List[str] = [
    "story_points",
    "dependency_count",
    "historical_median_hours",
    "assignee_user_ratio"
]

# Legacy 6-feature list for synthetic prototype
MODEL2_SYNTHETIC_FEATURE_LIST: List[str] = [
    "stalled_days",
    "dependency_depth",
    "active_blockers",
    "assignee_load_ratio",
    "logged_vs_estimated_ratio",
    "story_points"
]

# Model 2 Canonical 23-Feature Contract (matching FEATURE_CONTRACT.md & metadata.json)
MODEL2_FEATURE_LIST: List[str] = [
    "f",
    "sprint_open_tasks",
    "sprint_open_points",
    "fraction_points_done",
    "assignee_open_points",
    "assignee_open_tasks",
    "days_in_current_status",
    "days_since_last_event",
    "story_points",
    "has_sp",
    "added_after_sprint_start",
    "assignee_present",
    "links_depends_on",
    "status_category_is_todo",
    "type_Bug",
    "type_Story",
    "type_Improvement",
    "type_Task",
    "type_New Feature",
    "type_Suggestion",
    "type_Enhancement Request",
    "type_Epic",
    "type_other"
]

TOP_N_TASK_TYPES = [
    "Bug",
    "Story",
    "Improvement",
    "Task",
    "New Feature",
    "Suggestion",
    "Enhancement Request",
    "Epic"
]

TASK_TYPE_MAPPING: Dict[str, str] = {
    "bug": "Bug",
    "defect": "Bug",
    "story": "Story",
    "feature": "Story",
    "user_story": "Story",
    "improvement": "Improvement",
    "task": "Task",
    "chore": "Task",
    "refactor": "Task",
    "backend": "Task",
    "frontend": "Task",
    "infra": "Task",
    "test": "Task",
    "qa": "Task",
    "new_feature": "New Feature",
    "new feature": "New Feature",
    "suggestion": "Suggestion",
    "enhancement_request": "Enhancement Request",
    "enhancement request": "Enhancement Request",
    "epic": "Epic",
}


def compute_model1_features(
    task: Dict[str, Any],
    assignee_ratio: float,
    historical_median_hours: float,
    prediction_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Extracts ordered features for Model 1 (Task Effort Regressor).
    
    Guarantees:
    - Only accesses data available at prediction time.
    - If story_points is missing (None), returns None so caller can lower confidence
      and apply fallback rather than silently fabricating 1.0.
    """
    if prediction_time:
        created_at = task.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if created_at and created_at > prediction_time:
            raise ValueError(f"Temporal violation: task {task.get('task_id')} created after prediction_time {prediction_time}")

    raw_sp = task.get("story_points")
    story_points = float(raw_sp) if raw_sp is not None else None

    dependencies = task.get("dependencies", [])
    dependency_count = int(len(dependencies))

    return {
        "story_points": story_points,
        "dependency_count": dependency_count,
        "historical_median_hours": float(historical_median_hours),
        "assignee_user_ratio": float(assignee_ratio)
    }


def compute_model2_features(
    task: Dict[str, Any],
    active_blockers_count: int,
    logged_hours: float,
    estimated_hours: float,
    assignee_capacity_hours: float,
    assignee_total_assigned_hours: float,
    prediction_time: Optional[datetime] = None
) -> Dict[str, float]:
    """
    Legacy 6-feature extractor for synthetic prototype.
    """
    if prediction_time:
        created_at = task.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if created_at and created_at > prediction_time:
            raise ValueError(f"Temporal violation: task {task.get('task_id')} created after prediction_time {prediction_time}")

    stalled_days = int(task.get("stalled_days", 0))
    dependencies = task.get("dependencies", [])
    dependency_depth = int(len(dependencies))

    if assignee_capacity_hours and assignee_capacity_hours > 0:
        assignee_load_ratio = round(float(assignee_total_assigned_hours) / float(assignee_capacity_hours), 2)
    else:
        assignee_load_ratio = 1.0

    logged_ratio = round(float(logged_hours) / max(float(estimated_hours), 1.0), 2)

    raw_sp = task.get("story_points")
    story_points = float(raw_sp) if raw_sp is not None else 1.0

    return {
        "stalled_days": float(stalled_days),
        "dependency_depth": float(dependency_depth),
        "active_blockers": float(active_blockers_count),
        "assignee_load_ratio": float(assignee_load_ratio),
        "logged_vs_estimated_ratio": float(logged_ratio),
        "story_points": float(story_points)
    }


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensures a datetime is timezone-aware and set to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_model2_task_features(
    task: TaskSnapshot,
    sprint: SprintSnapshot,
    sprint_tasks: List[TaskSnapshot],
    now: Optional[datetime] = None
) -> Dict[str, float]:
    """
    Primary Feature Builder for Model 2 (Sprint Task Carry-Over Risk Classifier).
    Extracts the exact 23 features defined in FEATURE_CONTRACT.md from Taskove live snapshots.
    
    Guarantees & Eligibility:
    - Sprint must be active, elapsed fraction f in [0.2, 0.8], open tasks >= 10.
    - If sprint_assigned_at is absent on task, raises MissingFeatureError (no silent 0 imputation).
    - Story points defaults to None (handled as 0.0 for calculations, has_sp=0.0).
    - Priority is verified absent from the 23 features.
    - Types mapped via canonical Jira one-hot slots; unmapped -> type_other.
    """
    # 0. Check priority absent
    if "priority" in MODEL2_FEATURE_LIST:
        raise ValueError("Fatal contract violation: 'priority' must not be in MODEL2_FEATURE_LIST.")

    now = ensure_utc(now) or datetime.now(timezone.utc)
    sprint_start = ensure_utc(sprint.start_date)
    sprint_end = ensure_utc(sprint.end_date)

    # 1. Eligibility Check: Active Sprint
    if sprint.status.lower() != "active":
        raise IneligibleSprintError(f"Sprint '{sprint.sprint_id}' is not active (status='{sprint.status}').")

    # 2. Eligibility Check: Elapsed fraction f in [0.2, 0.8]
    total_sec = (sprint_end - sprint_start).total_seconds()
    if total_sec <= 0:
        raise IneligibleSprintError(f"Sprint '{sprint.sprint_id}' has invalid duration ({total_sec}s).")
    
    elapsed_sec = (now - sprint_start).total_seconds()
    f = elapsed_sec / total_sec
    if not (0.20 <= f <= 0.80):
        raise IneligibleSprintError(
            f"Sprint elapsed fraction f={f:.4f} is outside model domain [0.2, 0.8]."
        )

    # 3. Eligibility Check: Open tasks >= 10
    open_tasks = [t for t in sprint_tasks if t.status.lower() != "done"]
    if len(open_tasks) < 10:
        raise IneligibleSprintError(
            f"Sprint has {len(open_tasks)} open tasks (< 10 tasks required for statistical ranking)."
        )

    # 4. Mandatory exact-flag check: sprint_assigned_at
    if task.sprint_assigned_at is None:
        raise MissingFeatureError(
            f"Task '{task.task_id}' lacks mandatory sprint_assigned_at timestamp. "
            f"Exact flag added_after_sprint_start cannot be computed."
        )

    assigned_at = ensure_utc(task.sprint_assigned_at)
    added_after_start = 1.0 if assigned_at > sprint_start else 0.0

    # Aggregations
    sprint_open_tasks_count = float(len(open_tasks))
    sprint_open_points = sum(float(t.story_points or 0.0) for t in open_tasks)
    
    done_tasks = [t for t in sprint_tasks if t.status.lower() == "done"]
    done_points = sum(float(t.story_points or 0.0) for t in done_tasks)
    total_points = done_points + sprint_open_points
    fraction_points_done = (done_points / total_points) if total_points > 0 else 0.0

    # Assignee aggregations
    if task.assignee_id is not None:
        assignee_open_points = sum(
            float(t.story_points or 0.0) for t in open_tasks if t.assignee_id == task.assignee_id
        )
        assignee_open_tasks_count = float(
            len([t for t in open_tasks if t.assignee_id == task.assignee_id])
        )
        assignee_present = 1.0
    else:
        assignee_open_points = 0.0
        assignee_open_tasks_count = 0.0
        assignee_present = 0.0

    # Dwell time
    st_changed = ensure_utc(task.status_changed_at)
    upd_at = ensure_utc(task.updated_at)

    if st_changed is not None:
        days_in_current_status = max(0.0, (now - st_changed).total_seconds() / 86400.0)
    elif upd_at is not None:
        days_in_current_status = max(0.0, (now - upd_at).total_seconds() / 86400.0)
    else:
        days_in_current_status = float(task.stalled_days)

    if upd_at is not None:
        days_since_last_event = max(0.0, (now - upd_at).total_seconds() / 86400.0)
    else:
        days_since_last_event = float(task.stalled_days)

    # Story points
    sp_val = float(task.story_points) if task.story_points is not None else 0.0
    has_sp = 1.0 if (task.story_points is not None and task.story_points > 0) else 0.0

    # Blockers & Status
    links_depends_on = float(len(task.dependencies))
    status_category_is_todo = 1.0 if task.status.lower() == "todo" else 0.0

    # Type Mapping
    mapped_type = TASK_TYPE_MAPPING.get(task.task_type.lower(), "other")

    features: Dict[str, float] = {
        "f": float(f),
        "sprint_open_tasks": float(sprint_open_tasks_count),
        "sprint_open_points": float(sprint_open_points),
        "fraction_points_done": float(fraction_points_done),
        "assignee_open_points": float(assignee_open_points),
        "assignee_open_tasks": float(assignee_open_tasks_count),
        "days_in_current_status": float(days_in_current_status),
        "days_since_last_event": float(days_since_last_event),
        "story_points": float(sp_val),
        "has_sp": float(has_sp),
        "added_after_sprint_start": float(added_after_start),
        "assignee_present": float(assignee_present),
        "links_depends_on": float(links_depends_on),
        "status_category_is_todo": float(status_category_is_todo),
    }

    for t_name in TOP_N_TASK_TYPES:
        features[f"type_{t_name}"] = 1.0 if mapped_type == t_name else 0.0
    
    features["type_other"] = 1.0 if mapped_type not in TOP_N_TASK_TYPES else 0.0

    # Verify key order against canonical contract
    assert list(features.keys()) == MODEL2_FEATURE_LIST, "Internal feature order mismatch!"

    return features


def build_model2_features_from_dict(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Transforms a raw row dictionary (e.g. from benchmark dataframe or testing suite)
    into the canonical 23 ordered features.
    """
    num_cols = [
        "f", "sprint_open_tasks", "sprint_open_points", "fraction_points_done",
        "assignee_open_points", "assignee_open_tasks", "days_in_current_status",
        "days_since_last_event", "story_points", "has_sp", "added_after_sprint_start",
        "assignee_present", "links_depends_on"
    ]
    feats = {}
    for c in num_cols:
        val = row.get(c, 0.0)
        feats[c] = float(val) if val is not None else 0.0

    sc = str(row.get("status_category", "")).lower()
    feats["status_category_is_todo"] = 1.0 if sc == "todo" else 0.0

    raw_type = str(row.get("type", ""))
    mapped_type = TASK_TYPE_MAPPING.get(raw_type.lower(), raw_type)

    for t_name in TOP_N_TASK_TYPES:
        feats[f"type_{t_name}"] = 1.0 if mapped_type == t_name else 0.0
    
    feats["type_other"] = 1.0 if mapped_type not in TOP_N_TASK_TYPES else 0.0

    assert list(feats.keys()) == MODEL2_FEATURE_LIST
    return feats
