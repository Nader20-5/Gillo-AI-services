from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskSnapshot(BaseModel):
    task_id: str
    project_id: str
    sprint_id: Optional[str] = None
    title: str
    description: str = ""
    task_type: str = "feature"
    story_points: Optional[float] = None
    status: str = "todo"
    assignee_id: Optional[str] = None
    stalled_days: int = 0
    dependencies: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    sprint_assigned_at: Optional[datetime] = None
    status_changed_at: Optional[datetime] = None


class TimeEntry(BaseModel):
    entry_id: str
    task_id: str
    user_id: str
    hours: float
    logged_at: datetime = Field(default_factory=utc_now)


class SprintSnapshot(BaseModel):
    sprint_id: str
    project_id: str
    name: str
    start_date: datetime
    end_date: datetime
    status: str = "active"


class UserCapacity(BaseModel):
    user_id: str
    sprint_id: str
    capacity_hours: float


class StandupBlocker(BaseModel):
    """
    CRITICAL PRIVACY GUARDRAIL:
    User mood is strictly excluded from this model and database collection.
    """
    blocker_id: str
    user_id: str
    task_id: Optional[str] = None
    description: str
    reported_at: datetime = Field(default_factory=utc_now)
    status: str = "active"  # 'active', 'resolved'


class PullRequestSnapshot(BaseModel):
    pr_id: str
    task_ids: List[str] = Field(default_factory=list)
    title: str
    status: str = "open"  # 'open', 'merged', 'closed'
    branch: str = "main"
    created_at: datetime = Field(default_factory=utc_now)


class ReleaseRequestSnapshot(BaseModel):
    release_request_id: str
    project_id: str
    version: str
    target_date: datetime
    status: str = "draft"
    task_ids: List[str] = Field(default_factory=list)


class AIProfile(BaseModel):
    user_id: str
    estimation: Dict[str, Any] = Field(default_factory=dict)
    skills: Optional[List[str]] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class TaskEvent(BaseModel):
    """
    Task Event Stream Record (mirrors events emitted via Kafka from core service).
    Used to derive audit trails, dwell times, and assignment timestamps.
    """
    event_id: str
    task_id: str
    sprint_id: Optional[str] = None
    type: str  # 'status_change', 'sprint_assigned', 'assignee_change'
    from_val: Optional[str] = Field(default=None, alias="from")
    to_val: Optional[str] = Field(default=None, alias="to")
    at: datetime = Field(default_factory=utc_now)
    source: str = "seed_demo"

    model_config = {
        "populate_by_name": True
    }

